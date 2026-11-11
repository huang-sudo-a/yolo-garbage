import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision.protocol import FrameDecoder, SequenceStats, crc16, encode_wire_frame, MAX_FRAME_BYTES
from invc_vision.serial_receiver import SerialReceiver
from invc_vision.serial_sender import SerialSender
from invc_vision.telemetry import Telemetry


def record(seq=1, count=1):
    return {"version": 1, "seq": seq, "timestamp_utc": "2026-11-08T00:00:00.000Z",
            "frame_wh_px": [1280, 720], "fps": 10, "inference_ms": 5.2, "speed_unit": None,
            "detections": [{"track_id": i+1, "class_id": i % 6, "category": "其他垃圾",
                            "bbox_xywh_px": [i*20, 100, 18, 30], "confidence": 0.91,
                            "speed_instant": None, "speed_avg_15": None} for i in range(count)]}


def until(predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for asynchronous serial result")


class ProtocolTests(unittest.TestCase):
    def test_crc_reference_and_every_byte_fragmentation(self):
        self.assertEqual(crc16(b"123456789"), 0x29B1)
        decoder = FrameDecoder()
        frames = []
        packet = encode_wire_frame(record(count=15), "session", 1)
        for byte in packet:
            frames += decoder.feed(bytes([byte]))
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0]["d"]), 15)
        self.assertEqual(frames[0]["d"][14][3:7], [280, 100, 18, 30])
        self.assertEqual(frames[0]["d"][0][8:], [None, None])

    def test_corruption_partial_restart_noise_and_joined_frames(self):
        decoder = FrameDecoder()
        first = encode_wire_frame(record(), "session", 1)
        corrupt = first.replace(b'"f":10', b'"f":11')
        final = encode_wire_frame(record(3, 0), "session", 3)
        output = decoder.feed(b"noise\n"+corrupt+first[:30]+first+final)
        self.assertEqual([p["s"] for p in output], [1, 3])
        self.assertEqual(output[-1]["d"], [])
        self.assertEqual(decoder.bad_crc, 1)
        self.assertGreaterEqual(decoder.bad_format, 1)
        self.assertEqual(decoder.discarded_bytes, 6)

    def test_schema_validation_even_with_correct_crc_and_oversize_recovery(self):
        decoder = FrameDecoder()
        invalid = b'{"v":2,"sid":"x","s":1}'
        decoder.feed(b"@"+invalid+f"*{crc16(invalid):04X}\n".encode())
        decoder.feed(b"@"+b"x"*MAX_FRAME_BYTES)
        self.assertEqual(decoder.bad_format, 2)
        self.assertEqual(len(decoder.feed(encode_wire_frame(record(), "x", 1))), 1)
        bad = record()
        bad["detections"][0]["bbox_xywh_px"] = [1275, 100, 20, 20]
        with self.assertRaises(ValueError):
            encode_wire_frame(bad, "x", 1)

    def test_gap_stats_ignore_capture_skips_session_restart_and_duplicates(self):
        decoder, stats = FrameDecoder(), SequenceStats()
        def accept(tx, capture, session="one"):
            return stats.accept(decoder.feed(encode_wire_frame(record(capture), session, tx))[0])
        accept(1, 1)
        accept(2, 50)  # Deliberately throttled detector frames are not wire losses.
        self.assertEqual(stats.missing, 0)
        accept(4, 100)
        self.assertEqual(stats.missing, 1)
        self.assertEqual(accept(4, 100), "duplicate")
        self.assertEqual(accept(3, 70), "out_of_order")
        self.assertEqual(accept(1, 1, "two"), "new_session")
        self.assertEqual((stats.missing, stats.duplicates, stats.out_of_order), (1, 1, 1))


class SenderTests(unittest.TestCase):
    def test_latest_queue_does_not_block_capture_and_copies_input(self):
        gate, entered = threading.Event(), threading.Event()
        class SlowConnection:
            packets = []
            def write(self, packet):
                entered.set()
                gate.wait(2)
                self.packets.append(packet)
                return len(packet)
            def close(self):
                pass
        connection = SlowConnection()
        sender = SerialSender("test", hz=100, opener=lambda *_a, **_kw: connection)
        try:
            sender.submit(record())
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            for i in range(2, 102):
                data = record(i)
                sender.submit(data)
                data["detections"][0]["class_id"] = 999
            self.assertLess(time.monotonic()-started, 0.5)
            gate.set()
            until(lambda: sender.status()["sent"] == 2)
        finally:
            gate.set()
            sender.close()
        packets = FrameDecoder().feed(b"".join(connection.packets))
        self.assertEqual([p["q"] for p in packets], [1, 101])
        self.assertEqual(packets[-1]["d"][0][1], 0)
        self.assertEqual(sender.status()["replaced"], 99)

    def test_partial_write_failure_reconnect_and_newest_frame(self):
        class Connection:
            def __init__(self, fail=False):
                self.data, self.fail, self.closed = bytearray(), fail, False
            def write(self, data):
                if self.fail and self.data:
                    raise OSError("unplugged")
                count = min(7, len(data))
                self.data.extend(data[:count])
                return count
            def close(self):
                self.closed = True
        first, second = Connection(True), Connection()
        connections = iter([first, second])
        sender = SerialSender("test", hz=100, reconnect_seconds=0.02, opener=lambda *_a, **_kw: next(connections))
        try:
            sender.submit(record())
            until(lambda: sender.status()["failed"] == 1)
            sender.submit(record(2))
            until(lambda: sender.status()["sent"] == 1)
        finally:
            sender.close()
        self.assertTrue(first.closed and second.closed)
        self.assertEqual(FrameDecoder().feed(bytes(second.data))[0]["s"], 2)
        self.assertEqual(FrameDecoder().feed(bytes(second.data))[0]["q"], 2)

    def test_absent_receiver_does_not_stop_log_and_late_open_recovers(self):
        class Connection:
            data = bytearray()
            def write(self, data):
                self.data.extend(data)
                return len(data)
            def close(self):
                pass
        connection, available = Connection(), threading.Event()
        def opener(*_a, **_kw):
            if not available.is_set():
                raise OSError("no device")
            return connection
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            telemetry = Telemetry(path)
            telemetry._sender = SerialSender("missing", hz=100, reconnect_seconds=0.02, opener=opener)
            try:
                telemetry.write(record())
                until(lambda: telemetry.status()["connect_errors"] >= 1)
                telemetry.write(record(2))
                available.set()
                until(lambda: telemetry.status()["sent"] == 1)
            finally:
                telemetry.close()
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            summary = json.loads(path.with_suffix(".serial.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["connect_errors"], 1)
            self.assertEqual(FrameDecoder().feed(bytes(connection.data))[0]["q"], 2)


class ReceiverIntegrationTests(unittest.TestCase):
    def test_real_tcp_fragment_corruption_reconnect_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver = SerialReceiver(Path(directory), lambda *_: None, listen_port=0)
            receiver.start()
            self.assertTrue(receiver.ready.wait(2))
            self.assertIsNone(receiver.error)
            try:
                with socket.create_connection(("127.0.0.1", receiver.listen_port)) as client:
                    packet = encode_wire_frame(record(count=15), "first", 1)
                    client.sendall(packet[:10])
                    time.sleep(0.03)
                    client.sendall(packet[10:]+packet.replace(b'"f":10', b'"f":11'))
                    client.sendall(encode_wire_frame(record(5, 0), "first", 2))
                until(lambda: receiver.stats.received == 2)
                with socket.create_connection(("127.0.0.1", receiver.listen_port)) as client:
                    client.sendall(encode_wire_frame(record(10), "first", 3))
                    client.sendall(encode_wire_frame(record(), "new", 1))
                until(lambda: receiver.stats.received == 4)
            finally:
                receiver.close()
            summary = json.loads(Path(directory, "summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["received"], summary["missing"], summary["bad_crc"], summary["connections"]), (4, 0, 1, 2))
            self.assertEqual(len(Path(directory, "received.jsonl").read_text(encoding="utf-8").splitlines()), 4)

    def test_production_pyserial_sender_to_receiver(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver = SerialReceiver(Path(directory) / "rx", lambda *_: None, listen_port=0)
            receiver.start()
            self.assertTrue(receiver.ready.wait(2))
            sender = SerialSender(f"socket://127.0.0.1:{receiver.listen_port}", hz=20)
            try:
                sender.submit(record(1, 15))
                until(lambda: receiver.stats.received == 1)
                sender.submit(record(4, 0))
                until(lambda: receiver.stats.received == 2)
                self.assertEqual(receiver.stats.missing, 0)
                self.assertEqual(receiver.last_frame["d"], [])
            finally:
                sender.close()
                receiver.close()


if __name__ == "__main__":
    unittest.main()
