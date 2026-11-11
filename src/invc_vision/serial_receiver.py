"""Shared serial/TCP receiver used by the CLI and Chinese desktop assistant."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import socket
import threading
from typing import Callable

import serial

from .protocol import FrameDecoder, SequenceStats


class SerialReceiver:
    def __init__(self, log_dir: Path, callback: Callable, *, listen_port: int | None = 8765,
                 port: str | None = None, baudrate: int = 460800) -> None:
        if (listen_port is None) == (port is None):
            raise ValueError("选择本机监听或 COM 接收中的一种")
        self.log_dir, self.callback = log_dir, callback
        self.listen_port, self.port, self.baudrate = listen_port, port, baudrate
        self.decoder, self.stats = FrameDecoder(), SequenceStats()
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, name="invc-serial-receiver", daemon=True)
        self.error = None
        self.connections = 0
        self.last_frame = None
        self.last_receive = None

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self._thread.join(timeout=6)
        if self._thread.is_alive():
            raise RuntimeError("接收线程未能按时退出")

    def snapshot(self) -> dict:
        return {"received": self.stats.received, "missing": self.stats.missing,
                "duplicates": self.stats.duplicates, "out_of_order": self.stats.out_of_order,
                "bad_crc": self.decoder.bad_crc, "bad_format": self.decoder.bad_format,
                "discarded_bytes": self.decoder.discarded_bytes, "connections": self.connections,
                "last_frame": self.last_frame, "last_received_local": self.last_receive}

    def _feed(self, chunk: bytes, log) -> None:
        before = self.decoder.bad_crc+self.decoder.bad_format
        frames = self.decoder.feed(chunk)
        for payload in frames:
            state = self.stats.accept(payload)
            self.last_receive = datetime.now().astimezone().isoformat(timespec="milliseconds")
            log.write(json.dumps({"received_local": self.last_receive, "sequence_state": state,
                                  "payload": payload}, ensure_ascii=False)+"\n")
            if state not in ("duplicate", "out_of_order"):
                self.last_frame = payload
                self.callback("frame", payload)
        if self.decoder.bad_crc+self.decoder.bad_format != before:
            self.callback("status", "收到损坏数据，已丢弃；正在等待下一完整帧")
        if frames or self.decoder.bad_crc+self.decoder.bad_format != before:
            self.save_summary()

    def _read_connection(self, connection, log, tcp: bool) -> None:
        self.decoder.reset_buffer()
        self.connections += 1
        self.callback("status", "已连接，等待数据")
        while not self.stop_event.is_set():
            try:
                chunk = connection.recv(4096) if tcp else connection.read(4096)
            except socket.timeout:
                continue
            if not chunk:
                if tcp:
                    break
                continue
            self._feed(chunk, log)

    def _run(self) -> None:
        server = None
        try:
            with (self.log_dir / "received.jsonl").open("a", encoding="utf-8", buffering=1) as log:
                if self.listen_port is not None:
                    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                        server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    else:
                        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    server.bind(("127.0.0.1", self.listen_port))
                    self.listen_port = server.getsockname()[1]
                    server.listen(1)
                    server.settimeout(0.2)
                    self.ready.set()
                    self.callback("status", f"本机模拟接收已开启：127.0.0.1:{self.listen_port}，等待发送端")
                while not self.stop_event.is_set():
                    connection = None
                    try:
                        if server is not None:
                            try:
                                connection, _ = server.accept()
                            except socket.timeout:
                                continue
                            connection.settimeout(0.2)
                        else:
                            connection = serial.serial_for_url(self.port, baudrate=self.baudrate, timeout=0.2,
                                                               write_timeout=0.2, bytesize=8, parity="N", stopbits=1)
                            self.ready.set()
                        self._read_connection(connection, log, server is not None)
                        self.callback("status", "发送端已断开，等待重新连接")
                    except (OSError, ValueError) as exc:
                        self.error = str(exc)
                        self.callback("status", f"连接异常，正在等待重连：{exc}")
                        self.stop_event.wait(0.5)
                    finally:
                        if connection is not None:
                            connection.close()
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            self.callback("error", str(exc))
        finally:
            if server is not None:
                server.close()
            self.ready.set()
            self.save_summary()

    def save_summary(self) -> None:
        self.log_dir.joinpath("summary.json").write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
