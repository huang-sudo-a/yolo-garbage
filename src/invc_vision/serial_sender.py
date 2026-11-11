"""Nonblocking latest-result sender. Local logs are independent of the connection."""

from __future__ import annotations

from copy import deepcopy
import math
import threading
import time
from typing import Any, Callable
import uuid

import serial

from .protocol import encode_wire_frame


class SerialSender:
    def __init__(self, port: str, baudrate: int = 460800, hz: float = 10,
                 reconnect_seconds: float = 1, opener: Callable | None = None) -> None:
        if not math.isfinite(hz) or hz <= 0 or baudrate <= 0 or reconnect_seconds <= 0:
            raise ValueError("串口波特率、发送频率和重连间隔必须为正")
        self.port, self.baudrate, self.hz = port, baudrate, hz
        self.session = uuid.uuid4().hex[:16]
        self._open = opener or serial.serial_for_url
        self._reconnect = reconnect_seconds
        self._condition = threading.Condition()
        self._pending = None
        self._closing = False
        self._state = {"port": port, "session": self.session, "connected": False,
                       "submitted": 0, "sent": 0, "replaced": 0, "failed": 0,
                       "connect_errors": 0, "tx_seq": 0, "bytes_sent": 0, "error": None}
        self._state["discarded_on_close"] = 0
        self._thread = threading.Thread(target=self._run, name="invc-serial-sender", daemon=True)
        self._thread.start()

    def submit(self, record: dict[str, Any]) -> None:
        snapshot = deepcopy(record)
        with self._condition:
            if self._closing:
                raise RuntimeError("发送器已关闭")
            self._state["submitted"] += 1
            if self._pending is not None:
                self._state["replaced"] += 1
            self._pending = snapshot
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return dict(self._state)

    def _run(self) -> None:
        connection = None
        next_send = next_connect = 0.0
        try:
            while True:
                with self._condition:
                    if self._closing and self._pending is None:
                        break
                    now = time.monotonic()
                    due = max(next_send, next_connect if connection is None else 0)
                    if self._pending is None or now < due:
                        if self._closing:
                            # Closing does not wait for a disconnected endpoint to return.
                            if self._pending is not None:
                                self._state["discarded_on_close"] += 1
                            self._pending = None
                            break
                        self._condition.wait(min(max(due-now, 0.01), 0.1) if self._pending is not None else 0.1)
                        continue
                if connection is None:
                    try:
                        connection = self._open(self.port, baudrate=self.baudrate, timeout=0.1,
                                                write_timeout=0.2, bytesize=8, parity="N", stopbits=1,
                                                xonxoff=False, rtscts=False, dsrdtr=False)
                        with self._condition:
                            self._state.update(connected=True, error=None)
                    except (OSError, ValueError) as exc:
                        with self._condition:
                            self._state["connect_errors"] += 1
                            self._state.update(connected=False, error=str(exc))
                        next_connect = time.monotonic()+self._reconnect
                        continue
                with self._condition:
                    record, self._pending = self._pending, None
                    if record is None:
                        continue
                    self._state["tx_seq"] += 1
                    seq = self._state["tx_seq"]
                try:
                    packet = encode_wire_frame(record, self.session, seq)
                    offset = 0
                    deadline = time.monotonic()+0.3
                    while offset < len(packet):
                        count = connection.write(packet[offset:])
                        if not count or time.monotonic() > deadline:
                            raise serial.SerialTimeoutException("发送超时或未写入完整帧")
                        offset += count
                    with self._condition:
                        self._state["sent"] += 1
                        self._state["bytes_sent"] += len(packet)
                        self._state["error"] = None
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    with self._condition:
                        self._state["failed"] += 1
                        self._state.update(connected=False, error=str(exc))
                    connection.close()
                    connection = None
                    next_connect = time.monotonic()+self._reconnect
                next_send = time.monotonic()+1/self.hz
        finally:
            if connection is not None:
                connection.close()
            with self._condition:
                self._state["connected"] = False

    def close(self) -> dict[str, Any]:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        # pySerial socket connection establishment has a fixed 5-second timeout.
        self._thread.join(timeout=6)
        if self._thread.is_alive():
            raise RuntimeError("串口发送线程未能按时退出")
        return self.status()
