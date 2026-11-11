"""Version 2 wire protocol: @ASCII-JSON*CRC16-CCITT-FALSE\n."""

from __future__ import annotations

from collections import OrderedDict
import json
import math
from typing import Any

CATEGORY_CODE = {"有害垃圾": "H", "可回收垃圾": "R", "厨余垃圾": "K", "其他垃圾": "O"}
CATEGORY_NAME = {code: name for name, code in CATEGORY_CODE.items()}
MAX_FRAME_BYTES = 16384


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_wire_frame(record: dict[str, Any], session: str, tx_seq: int) -> bytes:
    payload = {
        "v": 2, "sid": session, "s": tx_seq, "q": record["seq"],
        "t": record["timestamp_utc"], "wh": record["frame_wh_px"],
        "f": record["fps"], "ms": record["inference_ms"], "u": record["speed_unit"],
        "src": record.get("data_source", "live_inference"),
        "d": [[d["track_id"], d["class_id"], CATEGORY_CODE[d["category"]],
               *d["bbox_xywh_px"], d["confidence"], d["speed_instant"], d["speed_avg_15"]]
              for d in record["detections"]],
    }
    validate_payload(payload)
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    frame = b"@" + body + f"*{crc16(body):04X}\n".encode("ascii")
    if len(frame) > MAX_FRAME_BYTES:
        raise ValueError("通信数据帧超过 16384 字节")
    return frame


def _number(value: Any, minimum: float = 0) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def validate_payload(p: Any) -> None:
    if not isinstance(p, dict) or p.get("v") != 2:
        raise ValueError("不支持的协议版本")
    if not isinstance(p.get("sid"), str) or not 1 <= len(p["sid"]) <= 64:
        raise ValueError("会话编号无效")
    if any(type(p.get(k)) is not int or p[k] < 1 for k in ("s", "q")):
        raise ValueError("通信或识别帧编号无效")
    if not isinstance(p.get("t"), str) or not p["t"]:
        raise ValueError("缺少采样时间")
    wh = p.get("wh")
    if not isinstance(wh, list) or len(wh) != 2 or any(type(v) is not int or v < 1 for v in wh):
        raise ValueError("画面尺寸无效")
    if not _number(p.get("f")) or not _number(p.get("ms")) or p.get("u") not in (None, "px/s", "mm/s", "cm/s"):
        raise ValueError("帧率、推理耗时或速度单位无效")
    if not isinstance(p.get("d"), list) or len(p["d"]) > 128:
        raise ValueError("目标列表无效")
    if p.get("src") not in ("live_inference", "recording_replay"):
        raise ValueError("数据来源无效")
    for d in p["d"]:
        if not isinstance(d, list) or len(d) != 10:
            raise ValueError("目标字段数量应为 10")
        if d[0] is not None and (type(d[0]) is not int or d[0] < 1):
            raise ValueError("目标 ID 无效")
        if type(d[1]) is not int or d[1] < 0 or d[2] not in CATEGORY_NAME:
            raise ValueError("目标类别无效")
        if any(not _number(v) for v in d[3:7]) or d[5] <= 0 or d[6] <= 0 or d[3]+d[5] > wh[0] or d[4]+d[6] > wh[1]:
            raise ValueError("目标框超出画面或尺寸无效")
        if not _number(d[7]) or d[7] > 1 or any(v is not None and not _number(v) for v in d[8:10]):
            raise ValueError("置信度或速度无效")
        if p["u"] is None and any(v is not None for v in d[8:10]):
            raise ValueError("有速度数值时必须指定单位")


class FrameDecoder:
    """Incremental decoder, preserving partial frames and recovering at the next @."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.bad_crc = 0
        self.bad_format = 0
        self.discarded_bytes = 0

    def reset_buffer(self) -> None:
        self.buffer.clear()

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self.buffer.extend(chunk)
        result = []
        while self.buffer:
            start = self.buffer.find(b"@")
            if start < 0:
                self.discarded_bytes += len(self.buffer)
                self.buffer.clear()
                break
            if start:
                self.discarded_bytes += start
                del self.buffer[:start]
            end = self.buffer.find(b"\n")
            next_start = self.buffer.find(b"@", 1)
            if next_start >= 0 and (end < 0 or next_start < end):
                self.bad_format += 1
                del self.buffer[:next_start]
                continue
            if end < 0:
                if len(self.buffer) > MAX_FRAME_BYTES:
                    self.bad_format += 1
                    self.buffer.clear()
                break
            line = bytes(self.buffer[:end]).rstrip(b"\r")
            del self.buffer[:end+1]
            if len(line)+1 > MAX_FRAME_BYTES:
                self.bad_format += 1
                continue
            try:
                body, check = line[1:].rsplit(b"*", 1)
                if len(check) != 4 or any(c not in b"0123456789abcdefABCDEF" for c in check):
                    raise ValueError("CRC 字段无效")
                if crc16(body) != int(check, 16):
                    self.bad_crc += 1
                    continue
                payload = json.loads(body)
                validate_payload(payload)
            except (ValueError, TypeError, KeyError, UnicodeDecodeError, OverflowError, RecursionError):
                self.bad_format += 1
                continue
            result.append(payload)
        return result


class SequenceStats:
    """Count gaps in actual wire sequences, independently of skipped capture frames."""

    def __init__(self) -> None:
        self.sessions: OrderedDict[str, int] = OrderedDict()
        self.received = self.missing = self.duplicates = self.out_of_order = 0

    def accept(self, payload: dict[str, Any]) -> str:
        sid, seq = payload["sid"], payload["s"]
        previous = self.sessions.get(sid)
        self.received += 1
        if previous is None:
            state = "new_session"
        elif seq == previous:
            self.duplicates += 1
            return "duplicate"
        elif seq < previous:
            self.out_of_order += 1
            return "out_of_order"
        else:
            self.missing += seq-previous-1
            state = "ok"
        self.sessions[sid] = seq
        self.sessions.move_to_end(sid)
        if len(self.sessions) > 64:
            self.sessions.popitem(last=False)
        return state
