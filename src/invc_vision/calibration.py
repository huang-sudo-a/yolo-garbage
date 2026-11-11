"""Planar chessboard calibration for speed in centimetres per second."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class PlaneCalibration:
    def __init__(self, matrix: np.ndarray, image_size: tuple[int, int]) -> None:
        if matrix.shape != (3, 3):
            raise ValueError("homography must be 3x3")
        self.matrix = matrix.astype(np.float64)
        self.image_size = tuple(image_size)

    def point_cm(self, point_px: tuple[float, float]) -> tuple[float, float]:
        point = np.array([[[point_px[0], point_px[1]]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(point, self.matrix)[0, 0]
        return float(projected[0]), float(projected[1])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"homography_px_to_cm": self.matrix.tolist(), "image_size": list(self.image_size)}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PlaneCalibration":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.array(data["homography_px_to_cm"], dtype=np.float64), tuple(data["image_size"]))

    @classmethod
    def from_chessboard(cls, frame: np.ndarray, columns: int, rows: int, square_cm: float) -> tuple["PlaneCalibration", np.ndarray]:
        if columns < 3 or rows < 3 or square_cm <= 0:
            raise ValueError("棋盘格内角点和方格边长必须为正且合理")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, (columns, rows))
        if not found:
            raise RuntimeError("未检测到完整棋盘格；请使其完全入镜并保持清晰")
        image_points = corners.reshape(-1, 2).astype(np.float32)
        plane_points = np.array([(c * square_cm, r * square_cm) for r in range(rows) for c in range(columns)], dtype=np.float32)
        matrix, inliers = cv2.findHomography(image_points, plane_points, method=cv2.RANSAC, ransacReprojThreshold=0.5)
        if matrix is None or inliers is None or int(inliers.sum()) < columns * rows * 0.8:
            raise RuntimeError("标定角点拟合失败，请重拍棋盘格")
        preview = frame.copy()
        cv2.drawChessboardCorners(preview, (columns, rows), corners, found)
        return cls(matrix, (frame.shape[1], frame.shape[0])), preview
