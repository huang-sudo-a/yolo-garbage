import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision.motion import MotionTracker
from invc_vision.telemetry import encode_frame, encode_wire_frame
from invc_vision.protocol import FrameDecoder


class MotionTests(unittest.TestCase):
    def test_window_average_uses_actual_time_and_travelled_path(self):
        tracker = MotionTracker(window=3)
        tracker.update(7, (0, 0), 1.0)
        tracker.update(7, (3, 0), 1.2)
        speed = tracker.update(7, (3, 4), 1.5)
        self.assertAlmostEqual(speed.instant, 4 / 0.3)
        self.assertAlmostEqual(speed.average_15, 7 / 0.5)
        self.assertEqual(speed.samples, 3)

    def test_stale_track_resets_history(self):
        tracker = MotionTracker(window=15, stale_seconds=1)
        tracker.update(1, (0, 0), 1)
        speed = tracker.update(1, (30, 0), 3)
        self.assertEqual(speed.samples, 1)
        self.assertEqual(speed.instant, 0)

    def test_track_histories_are_separate(self):
        tracker = MotionTracker()
        tracker.update(1, (0, 0), 1)
        tracker.update(2, (100, 0), 1)
        self.assertEqual(tracker.update(1, (10, 0), 2).instant, 10)
        self.assertEqual(tracker.update(2, (100, 10), 2).instant, 10)


class ProtocolTests(unittest.TestCase):
    def test_json_line_is_utf8_and_parseable(self):
        encoded = encode_frame({"seq": 5, "category": "厨余垃圾"})
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(json.loads(encoded)["category"], "厨余垃圾")

    def test_compact_serial_packet_retains_measurement_fields(self):
        record = {"version": 1, "seq": 9, "timestamp_utc": "2026-10-14T00:00:00Z", "fps": 30, "speed_unit": "cm/s", "frame_wh_px": [640, 480], "inference_ms": 4,
                  "detections": [{"track_id": 7, "class_id": 12, "category": "厨余垃圾",
                                  "bbox_xywh_px": [1, 2, 30, 40], "confidence": 0.9,
                                  "speed_instant": 2.1, "speed_avg_15": 1.8, "speed_unit": "cm/s"}]}
        packet = FrameDecoder().feed(encode_wire_frame(record, "test_session", 3))[0]
        self.assertEqual(packet["s"], 3)
        self.assertEqual(packet["q"], 9)
        self.assertEqual(packet["u"], "cm/s")
        self.assertEqual(packet["d"][0], [7, 12, "K", 1, 2, 30, 40, 0.9, 2.1, 1.8])


if __name__ == "__main__":
    unittest.main()
