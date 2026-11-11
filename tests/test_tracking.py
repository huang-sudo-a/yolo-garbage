"""Assignment optimality and identity behavior under motion and missed detections."""

from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision.tracking import Detection, KalmanBox, MultiObjectTracker, associate, hungarian


class AssignmentTests(unittest.TestCase):
    def test_rectangular_assignment_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(7)
        for n in range(1, 5):
            for m in range(1, 5):
                for _ in range(8):
                    cost = rng.integers(-5, 12, size=(n, m)).astype(float)
                    pairs = hungarian(cost)
                    if n <= m:
                        optimum = min(sum(cost[row, col] for row, col in enumerate(choice)) for choice in permutations(range(m), n))
                    else:
                        optimum = min(sum(cost[row, col] for col, row in enumerate(choice)) for choice in permutations(range(n), m))
                    self.assertEqual(len(pairs), min(n, m))
                    self.assertEqual(len({p[0] for p in pairs}), len(pairs))
                    self.assertEqual(len({p[1] for p in pairs}), len(pairs))
                    self.assertAlmostEqual(sum(cost[row, col] for row, col in pairs), optimum)

    def test_gates_do_not_force_invalid_pairs_or_reduce_valid_cardinality(self):
        self.assertEqual(associate(np.array([[0.1, 1.0], [0.9, np.inf]])), [(0, 1), (1, 0)])
        self.assertEqual(associate(np.full((3, 2), np.inf)), [])
        self.assertEqual(hungarian(np.empty((0, 4))), [])
        with self.assertRaises(ValueError):
            hungarian(np.array([[np.nan]]))


class TrackingTests(unittest.TestCase):
    def detection(self, x, y=50, cls=0):
        return Detection((float(x), float(y), 40.0, 30.0), cls, 0.9)

    def test_static_jitter_and_input_reordering_preserve_ids(self):
        tracker = MultiObjectTracker()
        rng = np.random.default_rng(14)
        expected = None
        raw, filtered = [], []
        for frame in range(80):
            a = self.detection(50 + rng.normal(0, 2))
            b = self.detection(250 + rng.normal(0, 2))
            detections = [a, b] if frame % 2 == 0 else [b, a]
            observations = tracker.update(detections, frame*0.1)
            ids = {"left" if det.bbox[0] < 150 else "right": obs.track_id for det, obs in zip(detections, observations)}
            expected = ids if expected is None else expected
            self.assertEqual(ids, expected)
            index = 0 if frame % 2 == 0 else 1
            raw.append(a.bbox[0])
            filtered.append(observations[index].bbox[0])
        self.assertLess(np.std(filtered[15:]), np.std(raw[15:]))
        self.assertEqual(len(observations[0].trail), 15)

    def test_opposite_slow_motion_irregular_time_and_ordering(self):
        tracker = MultiObjectTracker()
        timestamp, expected = 0.0, None
        for frame in range(100):
            timestamp += (0.07, 0.1, 0.16)[frame % 3]
            a = self.detection(30 + 28*timestamp, 60)
            b = self.detection(470 - 28*timestamp, 110)
            inputs = [a, b] if frame % 2 else [b, a]
            outputs = tracker.update(inputs, timestamp)
            ids = {"upper" if d.bbox[1] == 60 else "lower": o.track_id for d, o in zip(inputs, outputs)}
            expected = ids if expected is None else expected
            self.assertEqual(ids, expected)

    def test_fifteen_targets_remain_distinct_under_repeated_reordering(self):
        tracker = MultiObjectTracker()
        rng = np.random.default_rng(23)
        expected = None
        for frame in range(45):
            order = rng.permutation(15)
            inputs = [self.detection(20+(index % 5)*150+frame*0.3+rng.normal(0, 0.5), 50+(index // 5)*150, int(index % 6)) for index in order]
            outputs = tracker.update(inputs, frame*0.1)
            identities = {int(index): observation.track_id for index, observation in zip(order, outputs)}
            expected = identities if expected is None else expected
            self.assertEqual(identities, expected)
            self.assertEqual(len({observation.track_id for observation in outputs}), 15)

    def test_short_gap_recovers_without_fabricated_detections(self):
        tracker = MultiObjectTracker(max_lost_seconds=1.0)
        first = tracker.update([self.detection(50)], 0)[0].track_id
        tracker.update([self.detection(54)], 0.1)
        for step in range(2, 8):
            self.assertEqual(tracker.update([], step*0.1), [])
            self.assertEqual(tracker.retained_tracks, 1)
        restored = tracker.update([self.detection(82)], 0.8)[0]
        self.assertEqual(restored.track_id, first)
        self.assertTrue(restored.confirmed)

    def test_expired_or_distant_objects_receive_new_ids(self):
        tracker = MultiObjectTracker(max_lost_seconds=0.5)
        first = tracker.update([self.detection(50)], 0)[0].track_id
        second = tracker.update([self.detection(50)], 0.6)[0].track_id
        third = tracker.update([self.detection(1000)], 0.7)[0].track_id
        self.assertEqual((first, second, third), (1, 2, 3))

    def test_brief_classification_flicker_does_not_reset_identity(self):
        tracker = MultiObjectTracker()
        ids = [tracker.update([self.detection(50 + frame, cls=cls)], frame*0.1)[0].track_id for frame, cls in enumerate([0, 0, 0, 2, 0, 0])]
        self.assertEqual(set(ids), {1})

    def test_invalid_inputs_and_duplicate_timestamps_are_rejected(self):
        with self.assertRaises(ValueError):
            self.detection(float("nan"))
        tracker = MultiObjectTracker()
        tracker.update([], 1)
        with self.assertRaises(ValueError):
            tracker.update([], 1)
        with self.assertRaises(ValueError):
            MultiObjectTracker(max_lost_seconds=-1)

    def test_kalman_covariance_remains_finite_symmetric_and_positive(self):
        box = KalmanBox((20, 30, 40, 50))
        for frame in range(100):
            box.predict(0.03 + 0.01*(frame % 3))
            box.correct((20 + frame*0.5, 30, 40, 50))
        self.assertTrue(np.isfinite(box.state).all())
        np.testing.assert_allclose(box.covariance, box.covariance.T, atol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(box.covariance).min(), -1e-10)


if __name__ == "__main__":
    unittest.main()
