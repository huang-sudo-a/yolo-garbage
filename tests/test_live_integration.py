"""Check the pure-detection path, real rendering, logs, and resource cleanup."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision import app


class FakeCapture:
    def __init__(self):
        self.released = False

    def get(self, property_id):
        return {3: 640, 4: 480, 5: 30}.get(property_id, 0)

    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeModel:
    names = {0: "土豆"}
    calls = []

    def __init__(self, _path):
        pass

    def predict(self, *_args, **kwargs):
        self.calls.append(kwargs)
        boxes = type("Boxes", (), {
            "xyxy": torch.tensor([[30.0, 80.0, 160.0, 200.0]]),
            "cls": torch.tensor([0]), "conf": torch.tensor([0.9]),
        })()
        return [type("Result", (), {"boxes": boxes, "speed": {"preprocess": 1.0, "inference": 4.0, "postprocess": 1.0}})()]

    def track(self, *_args, **_kwargs):
        raise AssertionError("The detection preview must not use built-in tracking")


class LiveIntegrationTest(unittest.TestCase):
    def make_args(self, root):
        with patch.object(sys, "argv", ["run_live.py", "--no-tracking", "--headless", "--max-frames", "16"]):
            args = app.parse_args()
        args.model = root / "best.pt"
        args.model.touch()
        args.categories = root / "categories.yaml"
        args.categories.write_text("categories:\n  厨余垃圾:\n    - 土豆\n", encoding="utf-8")
        args.log = root / "session.jsonl"
        args.snapshot = root / "中文截图.jpg"
        args.width, args.height = 640, 480
        return args

    def test_detection_filters_chinese_render_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            capture = FakeCapture()
            FakeModel.calls = []
            with patch.object(app, "YOLO", FakeModel), patch.object(app, "open_capture", return_value=capture), patch.object(app.torch.cuda, "is_available", return_value=True), patch.object(app.torch.cuda, "get_device_name", return_value="Test GPU"):
                self.assertEqual(app.run(args), 0)
            records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 16)
            self.assertTrue(capture.released)
            self.assertTrue(args.snapshot.is_file())
            detection = records[-1]["detections"][0]
            self.assertEqual((detection["item"], detection["category"]), ("土豆", "厨余垃圾"))
            self.assertEqual(detection["bbox_xywh_px"], [30, 80, 130, 120])
            self.assertIsNone(detection["track_id"])
            self.assertIsNone(detection["speed_avg_15"])
            self.assertEqual(records[-1]["mode"], "detect")
            self.assertEqual(records[-1]["inference_ms"], 4.0)
            self.assertEqual(records[-1]["pipeline_ms"], 6.0)
            self.assertTrue(all(call["conf"] == 0.5 and call["iou"] == 0.5 and call["agnostic_nms"] and not call["rect"] for call in FakeModel.calls))

    def test_independent_tracking_ids_and_media_clock_are_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))
            args.tracking = True
            args.source = "video.mp4"
            capture = FakeCapture()
            with patch.object(app, "YOLO", FakeModel), patch.object(app, "open_capture", return_value=capture), patch.object(app.torch.cuda, "is_available", return_value=True), patch.object(app.torch.cuda, "get_device_name", return_value="Test GPU"):
                self.assertEqual(app.run(args), 0)
            records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["detections"][0]["track_id"] for r in records}, {1})
            self.assertEqual(records[-1]["mode"], "detect_track")
            self.assertEqual(records[-1]["tracker"]["algorithm"], "kalman_hungarian")
            self.assertEqual(records[-1]["detections"][0]["track_state"], "confirmed")
            self.assertEqual(len(records[-1]["detections"][0]["trail_px"]), 15)
            self.assertAlmostEqual(records[-1]["tracking_time_s"], 16/30)
            self.assertIsNone(records[-1]["detections"][0]["speed_avg_15"])

    def test_camera_disconnect_reports_failure_and_releases_device(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory))
            capture = FakeCapture()
            capture.read = lambda: (False, None)
            with patch.object(app, "YOLO", FakeModel), patch.object(app, "open_capture", return_value=capture), patch.object(app.torch.cuda, "is_available", return_value=True), patch.object(app.torch.cuda, "get_device_name", return_value="Test GPU"):
                with self.assertRaisesRegex(RuntimeError, "摄像头读帧失败"):
                    app.run(args)
            self.assertTrue(capture.released)
            self.assertFalse(args.snapshot.exists())


if __name__ == "__main__":
    unittest.main()
