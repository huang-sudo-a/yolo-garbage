"""Replay genuine recorded detector results through the production sender."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision.telemetry import Telemetry

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "outputs/tracking_validation_20261103/detections.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="回放已有识别日志到通信助手，不使用摄像头，不代表实时识别")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--port", default="socket://127.0.0.1:8765")
    parser.add_argument("--baudrate", type=int, default=460800)
    parser.add_argument("--hz", type=float, default=10)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("hz 必须为正")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        parser.error("日志没有数据")
    print(f"历史识别结果回放（不是当前摄像头）：{args.input}", flush=True)
    seq = 0
    try:
        with Telemetry(args.log, args.port, args.baudrate, args.hz) as output:
            while True:
                for original in records:
                    if args.stop_file and args.stop_file.exists():
                        return 0
                    seq += 1
                    record = dict(original, seq=seq, data_source="recording_replay", original_seq=original["seq"])
                    output.write(record)
                    time.sleep(1/args.hz)
                if not args.loop:
                    break
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
