"""Timestamp-based 15-observation speed estimation per stable tracker ID."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Speed:
    instant: float
    average_15: float
    samples: int


class MotionTracker:
    def __init__(self, window: int = 15, stale_seconds: float = 1.0) -> None:
        if window < 2 or stale_seconds <= 0:
            raise ValueError("window >= 2 and stale_seconds > 0 required")
        self.window = window
        self.stale_seconds = stale_seconds
        self._history: dict[int, deque[tuple[float, float, float]]] = {}

    def update(self, track_id: int, point: tuple[float, float], timestamp: float) -> Speed:
        history = self._history.get(track_id)
        if history is None or (history and timestamp - history[-1][0] > self.stale_seconds):
            history = deque(maxlen=self.window)
            self._history[track_id] = history
        if history and timestamp <= history[-1][0]:
            raise ValueError("timestamps must increase for each track")
        history.append((timestamp, float(point[0]), float(point[1])))
        if len(history) == 1:
            return Speed(0.0, 0.0, 1)
        previous, current = history[-2], history[-1]
        first = history[0]
        instant = hypot(current[1] - previous[1], current[2] - previous[2]) / (current[0] - previous[0])
        distance = sum(hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(history, list(history)[1:]))
        average = distance / (current[0] - first[0])
        return Speed(instant, average, len(history))

    def prune(self, timestamp: float) -> None:
        for track_id, history in list(self._history.items()):
            if timestamp - history[-1][0] > self.stale_seconds:
                del self._history[track_id]

    def trail(self, track_id: int) -> list[tuple[int, int]]:
        return [(round(x), round(y)) for _, x, y in self._history.get(track_id, ())]
