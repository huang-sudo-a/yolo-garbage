"""Newline-delimited JSON for replayable logs and optional serial transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import encode_wire_frame
from .serial_sender import SerialSender


def encode_frame(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class Telemetry:
    def __init__(self, log_path: Path, port: str | None = None, baudrate: int = 460800,
                 serial_hz: float = 10) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self._log = log_path.open("a", encoding="utf-8", buffering=1)
        try:
            self._sender = SerialSender(port, baudrate, serial_hz) if port else None
        except Exception:
            self._log.close()
            raise
        self._closed = False

    def status(self) -> dict[str, Any] | None:
        return self._sender.status() if self._sender else None

    def write(self, record: dict[str, Any]) -> None:
        packet = encode_frame(record)
        self._log.write(packet.decode("utf-8"))
        if self._sender is not None:
            self._sender.submit(record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._sender is not None:
                summary = self._sender.close()
                self.log_path.with_suffix(".serial.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            self._log.close()

    def __enter__(self) -> "Telemetry":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
