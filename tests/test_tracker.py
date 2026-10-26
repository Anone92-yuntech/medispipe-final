"""Model-free integration checks against the pinned original OC-SORT."""

from __future__ import annotations

import unittest

import numpy as np

from ocsort_adapter import OCSortTracker


class TrackerTests(unittest.TestCase):
    @staticmethod
    def detections():
        return np.array([
            [10.0, 10.0, 100.0, 200.0, 0.9],
            [300.0, 10.0, 500.0, 300.0, 0.8],
        ])

    @staticmethod
    def empty():
        return np.empty((0, 5), dtype=np.float64)

    def test_black_frames_advance_official_tracker_without_target(self):
        tracker = OCSortTracker(fps=10)
        for timestamp in (0, 100, 200):
            self.assertEqual(tracker.update(self.empty(), (480, 640), timestamp), [])
        self.assertEqual(tracker.frame_count, 3)
        self.assertEqual(tracker.tracker.frame_count, 3)
        self.assertEqual(tracker.tracker.trackers, [])

    def test_first_detection_counts_and_missed_prediction_is_not_observed(self):
        tracker = OCSortTracker(fps=10)
        first = tracker.update(self.detections(), (480, 640), 0)
        self.assertEqual(len(first), 2)
        for track in first:
            self.assertTrue(track.observed)
            self.assertEqual(track.observed_count, 1)
            self.assertEqual(track.first_seen_ms, 0)
            self.assertEqual(track.last_seen_ms, 0)
            self.assertIsInstance(track.bbox, np.ndarray)
        second = tracker.update(self.detections(), (480, 640), 100)
        self.assertEqual([track.track_id for track in first],
                         [track.track_id for track in second])
        for track in second:
            self.assertEqual(track.observed_count, 2)
            self.assertEqual(track.last_seen_ms, 100)
        missed = tracker.update(self.empty(), (480, 640), 200)
        self.assertEqual(len(missed), 2)
        for track in missed:
            self.assertFalse(track.observed)
            self.assertEqual(track.observed_count, 2)
            self.assertEqual(track.last_seen_ms, 100)
        recovered = tracker.update(self.detections(), (480, 640), 300)
        self.assertEqual([track.track_id for track in first],
                         [track.track_id for track in recovered])
        for track in recovered:
            self.assertTrue(track.observed)
            self.assertEqual(track.observed_count, 3)
            self.assertEqual(track.first_seen_ms, 0)
            self.assertEqual(track.last_seen_ms, 300)

    def test_empty_frames_expire_ids_and_new_people_have_new_evidence(self):
        tracker = OCSortTracker(fps=10, max_age_seconds=0.2)
        first = tracker.update(self.detections(), (480, 640), 0)
        self.assertEqual(tracker.tracker.max_age, 2)
        for timestamp in (100, 200):
            predicted = tracker.update(self.empty(), (480, 640), timestamp)
            self.assertEqual(len(predicted), 2)
            self.assertTrue(all(not track.observed for track in predicted))
        self.assertEqual(tracker.update(self.empty(), (480, 640), 300), [])
        new = tracker.update(self.detections(), (480, 640), 400)
        self.assertGreater(min(track.track_id for track in new),
                           max(track.track_id for track in first))
        self.assertTrue(all(track.observed_count == 1 for track in new))
        self.assertTrue(all(track.first_seen_ms == 400 for track in new))
        self.assertEqual(tracker.frame_count, 5)

    def test_detector_input_and_previous_output_do_not_change(self):
        tracker = OCSortTracker(fps=25)
        detections = self.detections()
        original = detections.copy()
        first = tracker.update(detections, (480, 640, 3), 0)
        saved = [track.bbox.copy() for track in first]
        next_detections = detections.copy()
        next_detections[:, [0, 2]] += 4
        tracker.update(next_detections, (480, 640), 40)
        np.testing.assert_array_equal(detections, original)
        for track, previous in zip(first, saved):
            np.testing.assert_array_equal(track.bbox, previous)

    def test_reset_clears_counters_ids_and_timestamp_state(self):
        tracker = OCSortTracker(fps=25)
        tracker.update(self.detections(), (480, 640), 1000)
        tracker.reset()
        self.assertEqual(tracker.frame_count, 0)
        self.assertEqual(tracker.update(self.empty(), (480, 640), 0), [])
        fresh = tracker.update(self.detections(), (480, 640), 40)
        self.assertEqual([track.track_id for track in fresh], [1, 2])
        self.assertTrue(all(track.observed_count == 1 for track in fresh))
        self.assertTrue(all(track.first_seen_ms == 40 for track in fresh))

    def test_invalid_detections_do_not_become_people(self):
        tracker = OCSortTracker(fps=25)
        invalid = np.array([
            [0, 0, 0, 50, 0.9],
            [30, 30, 10, 50, 0.9],
            [10, 10, 50, 50, float("nan")],
            [10, 10, 50, 50, 2.0],
        ])
        self.assertEqual(tracker.update(invalid, (480, 640), 0), [])
        self.assertEqual(tracker.frame_count, 1)

    def test_duplicate_timestamp_does_not_add_observations_or_advance(self):
        tracker = OCSortTracker(fps=25)
        tracker.update(self.detections(), (480, 640), 100)
        for timestamp in (100, 50):
            with self.assertRaises(ValueError):
                tracker.update(self.detections(), (480, 640), timestamp)
        self.assertEqual(tracker.frame_count, 1)
        next_frame = tracker.update(self.detections(), (480, 640), 140)
        self.assertTrue(all(track.observed_count == 2 for track in next_frame))


if __name__ == "__main__":
    unittest.main()
