"""Independent constant-velocity Kalman boxes and Hungarian data association.

No YOLO tracking API or third-party tracker is used. Only observed detections
are returned; unmatched predictions remain internal for short-gap recovery.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost rectangular assignment by Hungarian primal/dual augmentation."""
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Assignment costs must be a finite 2-D matrix")
    if not all(matrix.shape):
        return []
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    n, m = matrix.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner, previous = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for row in range(1, n + 1):
        owner[0] = row
        column = 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[column] = True
            current_row = owner[column]
            delta, next_column = np.inf, 0
            for candidate in range(1, m + 1):
                if used[candidate]:
                    continue
                reduced = matrix[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate], previous[candidate] = reduced, column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[owner[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            parent = previous[column]
            owner[column] = owner[parent]
            column = parent
    pairs = [(int(owner[column] - 1), column - 1) for column in range(1, m + 1) if owner[column]]
    return sorted((column, row) if transposed else (row, column) for row, column in pairs)


def associate(cost: np.ndarray) -> list[tuple[int, int]]:
    """Match as many gated pairs as possible, then minimize their total cost.

    A private unmatched column per track avoids forced invalid matches. Its
    penalty exceeds the entire feasible assignment's cost range, preserving
    valid-match cardinality before optimizing cost.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or np.isnan(cost).any() or np.isneginf(cost).any():
        raise ValueError("Invalid association matrix")
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    valid = np.isfinite(cost)
    if not valid.any():
        return []
    minimum, maximum = cost[valid].min(), cost[valid].max()
    normalized = (cost - minimum) / max(maximum - minimum, 1.0)
    penalty = float(n + 1)
    forbidden = penalty * (n + m + 2)
    augmented = np.full((n, m + n), forbidden)
    augmented[:, :m] = np.where(valid, normalized, forbidden)
    for row in range(n):
        augmented[row, m + row] = penalty
    return [(row, column) for row, column in hungarian(augmented) if column < m and valid[row, column]]


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # left, top, width, height, pixels
    class_id: int
    confidence: float

    def __post_init__(self) -> None:
        if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox) or min(self.bbox[2:]) <= 0:
            raise ValueError("Detection box must have finite coordinates and positive size")
        if self.class_id < 0 or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid detection class or confidence")


class KalmanBox:
    """Eight-state [cx, cy, w, h, vx, vy, vw, vh] linear Kalman filter."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        x, y, w, h = bbox
        self.state = np.array([x + w / 2, y + h / 2, w, h, 0, 0, 0, 0], dtype=float)
        self.covariance = np.diag([100.0] * 4 + [10000.0] * 4)
        self.measurement = np.concatenate((np.eye(4), np.zeros((4, 4))), axis=1)
        self.noise = np.diag([25.0, 25.0, 36.0, 36.0])

    def predict(self, dt: float) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Prediction dt must be positive")
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4) * dt
        process = np.zeros((8, 8))
        for index, acceleration in enumerate((80.0, 80.0, 40.0, 40.0)):
            q = acceleration**2
            process[index, index] = q * dt**4 / 4
            process[index, index + 4] = process[index + 4, index] = q * dt**3 / 2
            process[index + 4, index + 4] = q * dt**2
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)

    def correct(self, bbox: tuple[float, float, float, float]) -> None:
        x, y, w, h = bbox
        observed = np.array([x + w / 2, y + h / 2, w, h])
        residual = observed - self.measurement @ self.state
        innovation = self.measurement @ self.covariance @ self.measurement.T + self.noise
        gain = np.linalg.solve(innovation, self.measurement @ self.covariance).T
        self.state += gain @ residual
        # Joseph form maintains symmetry/positive semidefiniteness with rounding.
        remainder = np.eye(8) - gain @ self.measurement
        self.covariance = remainder @ self.covariance @ remainder.T + gain @ self.noise @ gain.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        cx, cy, w, h = self.state[:4]
        return float(cx-w/2), float(cy-h/2), float(w), float(h)


def box_iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0.0, min(ax+aw, bx+bw)-max(ax, bx)) * max(0.0, min(ay+ah, by+bh)-max(ay, by))
    return intersection / max(aw*ah + bw*bh - intersection, 1e-9)


@dataclass
class _Track:
    id: int
    filter: KalmanBox
    last_seen: float
    hits: int = 1
    class_scores: dict[int, float] = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=15))

    @property
    def class_id(self) -> int:
        return max(self.class_scores, key=self.class_scores.get)


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    bbox: tuple[float, float, float, float]
    confirmed: bool
    hits: int
    trail: tuple[tuple[float, float], ...]


class MultiObjectTracker:
    def __init__(self, max_lost_seconds: float = 1.0, min_hits: int = 2, distance_factor: float = 1.5) -> None:
        if not math.isfinite(max_lost_seconds) or max_lost_seconds <= 0 or min_hits < 1 or not math.isfinite(distance_factor) or distance_factor <= 0:
            raise ValueError("Invalid tracker configuration")
        self.max_lost_seconds = max_lost_seconds
        self.min_hits = min_hits
        self.distance_factor = distance_factor
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._timestamp: float | None = None

    @property
    def retained_tracks(self) -> int:
        return len(self._tracks)

    def update(self, detections: Sequence[Detection], timestamp: float) -> list[TrackObservation]:
        if not math.isfinite(timestamp) or (self._timestamp is not None and timestamp <= self._timestamp):
            raise ValueError("Tracking timestamps must be finite and strictly increasing")
        dt = timestamp - self._timestamp if self._timestamp is not None else 0.0
        self._tracks = [track for track in self._tracks if timestamp-track.last_seen <= self.max_lost_seconds]
        for track in self._tracks:
            track.filter.predict(dt)
        costs = np.full((len(self._tracks), len(detections)), np.inf)
        for row, track in enumerate(self._tracks):
            predicted = track.filter.bbox
            px, py, pw, ph = predicted
            for column, detection in enumerate(detections):
                x, y, w, h = detection.bbox
                distance = math.hypot(x+w/2-px-pw/2, y+h/2-py-ph/2)
                gate = max(40.0, self.distance_factor * math.hypot(max(pw, w), max(ph, h)))
                size_ratio = max(w/pw, pw/w, h/ph, ph/h)
                if distance > gate or size_ratio > 4:
                    continue
                costs[row, column] = 0.55*(1-box_iou(predicted, detection.bbox)) + 0.3*distance/gate + 0.15*(track.class_id != detection.class_id)
        assigned: dict[int, _Track] = {}
        for row, column in associate(costs):
            track, detection = self._tracks[row], detections[column]
            track.filter.correct(detection.bbox)
            track.last_seen = timestamp
            track.hits += 1
            track.class_scores = {c: score*0.95 for c, score in track.class_scores.items()}
            track.class_scores[detection.class_id] = track.class_scores.get(detection.class_id, 0) + detection.confidence
            assigned[column] = track
        observations = []
        for column, detection in enumerate(detections):
            track = assigned.get(column)
            if track is None:
                track = _Track(self._next_id, KalmanBox(detection.bbox), timestamp, class_scores={detection.class_id: detection.confidence})
                self._next_id += 1
                self._tracks.append(track)
            x, y, w, h = track.filter.bbox
            track.history.append((x+w/2, y+h/2))
            observations.append(TrackObservation(track.id, track.filter.bbox, track.hits >= self.min_hits, track.hits, tuple(track.history)))
        self._timestamp = timestamp
        return observations
