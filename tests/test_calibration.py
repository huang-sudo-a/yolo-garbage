from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision.calibration import PlaneCalibration


class CalibrationTest(unittest.TestCase):
    def test_chessboard_spacing_and_saved_transform(self):
        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        for row in range(7):
            for column in range(10):
                color = (255, 255, 255) if (row + column) % 2 else (0, 0, 0)
                cv2.rectangle(frame, (100 + column * 40, 80 + row * 40),
                              (100 + (column + 1) * 40, 80 + (row + 1) * 40), color, -1)
        calibration, preview = PlaneCalibration.from_chessboard(frame, 9, 6, 2.5)
        self.assertEqual(preview.shape, frame.shape)
        first = calibration.point_cm((140, 120))
        next_corner = calibration.point_cm((180, 120))
        distance = np.hypot(next_corner[0] - first[0], next_corner[1] - first[1])
        self.assertAlmostEqual(distance, 2.5, delta=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plane.json"
            calibration.save(path)
            restored = PlaneCalibration.load(path)
            self.assertEqual(restored.image_size, (640, 480))
            self.assertTrue(np.allclose(restored.matrix, calibration.matrix))


if __name__ == "__main__":
    unittest.main()
