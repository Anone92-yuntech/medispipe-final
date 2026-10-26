"""Regression tests for geometry/identity failure cases, without model downloads."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from target_selection import (  # noqa: E402
    ShoulderSample, TargetSelector, TrackObservation, crop_person,
    evaluate_shoulders, map_pose_to_original,
)


def observation(track_id, bbox=(0, 0, 100, 200), observed=True):
    return TrackObservation(track_id, np.array(bbox, dtype=float), observed, 100, 0, 0)


def sample(width):
    return ShoulderSample(width, np.array([[0, 0], [width, 0]], dtype=float))


def landmarks(width_fraction=0.5):
    result = [SimpleNamespace(x=0.5, y=0.2, visibility=0.99, presence=0.99)
              for _ in range(33)]
    result[11] = SimpleNamespace(x=0.5 - width_fraction / 2, y=0.5,
                                 visibility=0.99, presence=0.99)
    result[12] = SimpleNamespace(x=0.5 + width_fraction / 2, y=0.5,
                                 visibility=0.99, presence=0.99)
    return result


class ShoulderGeometryTests(unittest.TestCase):
    def test_crop_uses_clipped_rectangle_for_mapping(self):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        cropped, rect = crop_person(frame, (-10, -20, 40, 30), padding=0.1)
        self.assertEqual(rect, (0, 0, 45, 35))
        self.assertEqual(cropped.shape, (35, 45, 3))
        mapped = map_pose_to_original(landmarks(), rect)
        np.testing.assert_allclose(mapped.points[11], [11.25, 17.5])
        self.assertIsNone(crop_person(frame, (90, 90, 110, 120)))
        self.assertIsNone(crop_person(frame, (10, 10, 5, 20)))

    def test_different_crop_scales_have_same_original_shoulder_width(self):
        first, reason = evaluate_shoulders(landmarks(0.5), (10, 20, 100, 200),
                                          (10, 20, 110, 220))
        self.assertIsNotNone(first, reason)
        second, reason = evaluate_shoulders(landmarks(0.25), (30, 40, 200, 400),
                                           (30, 40, 230, 440))
        self.assertIsNotNone(second, reason)
        self.assertAlmostEqual(first.width, 50)
        self.assertAlmostEqual(second.width, 50)

    def test_missing_low_confidence_nonfinite_and_off_person_are_rejected(self):
        for case in ("missing", "visibility", "presence", "nan", "outside"):
            with self.subTest(case=case):
                points = landmarks()
                if case == "missing":
                    points = points[:12]
                elif case == "visibility":
                    points[11].visibility = 0.1
                elif case == "presence":
                    points[12].presence = 0.1
                elif case == "nan":
                    points[12].x = float("nan")
                else:
                    points[11].x = -0.1
                value, reason = evaluate_shoulders(points, (0, 0, 100, 200), (0, 0, 100, 200))
                self.assertIsNone(value)
                self.assertTrue(reason)

    def test_missing_confidence_is_not_assumed_to_be_one(self):
        points = [{"x": point.x, "y": point.y} for point in landmarks()]
        self.assertIsNone(evaluate_shoulders(points, (0, 0, 100, 200), (0, 0, 100, 200))[0])

    def test_wrong_person_inside_large_crop_is_rejected(self):
        # Large adult crop accidentally resolves the child's head and shoulders.
        adult = observation(1, (0, 0, 300, 500))
        child = observation(2, (140, 100, 240, 300))
        points = landmarks()
        for item in points:
            item.x, item.y = 190 / 300, 130 / 500
        points[11].x, points[11].y = 160 / 300, 160 / 500
        points[12].x, points[12].y = 220 / 300, 160 / 500
        value, reason = evaluate_shoulders(points, (0, 0, 300, 500), adult.bbox,
                                          [adult, child], source_track_id=1)
        self.assertIsNone(value)
        self.assertEqual(reason, "ambiguous_with_id_2")
        # Lower-body overlap alone must not suppress otherwise valid shoulders.
        other = observation(3, (20, 150, 90, 350))
        value, reason = evaluate_shoulders(landmarks(), (0, 0, 100, 200),
                                          (0, 0, 100, 200), [other])
        self.assertIsNotNone(value, reason)

    def test_lowered_head_is_allowed_for_tracking_but_not_initial_selection(self):
        points = landmarks()
        for point in points[:11]:
            point.y = 0.8
        value, reason = evaluate_shoulders(points, (0, 0, 100, 200), (0, 0, 100, 200))
        self.assertIsNone(value)
        self.assertEqual(reason, "head_shoulder_geometry")
        value, reason = evaluate_shoulders(
            points, (0, 0, 100, 200), (0, 0, 100, 200), for_selection=False,
        )
        self.assertIsNotNone(value, reason)
        self.assertAlmostEqual(value.width, 50)
        np.testing.assert_allclose(value.head_center, [50, 160])

    def test_tracking_accepts_shoulder_angles_unsuitable_for_size_ranking(self):
        for case in ("nearly_vertical", "large_vertical_difference"):
            with self.subTest(case=case):
                points = landmarks()
                if case == "nearly_vertical":
                    rectangle, box = (0, 0, 100, 200), (0, 0, 100, 200)
                    points[11].x, points[11].y = 0.45, 0.4
                    points[12].x, points[12].y = 0.55, 0.6
                else:
                    rectangle, box = (0, 0, 300, 200), (0, 0, 300, 200)
                    points[11].x, points[11].y = 0.35, 0.2
                    points[12].x, points[12].y = 0.65, 0.8
                value, reason = evaluate_shoulders(points, rectangle, box)
                self.assertIsNone(value)
                self.assertEqual(reason, "shoulders_not_comparable")
                value, reason = evaluate_shoulders(points, rectangle, box, for_selection=False)
                self.assertIsNotNone(value, reason)
                self.assertGreater(value.width, 4)

    def test_tracking_still_rejects_missing_unreliable_or_outside_shoulders(self):
        for case in ("missing", "visibility", "presence", "nan", "outside", "collapsed"):
            with self.subTest(case=case):
                points = landmarks()
                if case == "missing":
                    points = points[:12]
                elif case == "visibility":
                    points[11].visibility = 0.1
                elif case == "presence":
                    points[12].presence = 0.1
                elif case == "nan":
                    points[12].x = float("nan")
                elif case == "outside":
                    points[11].x = -0.1
                else:
                    points[11].x = points[12].x
                value, reason = evaluate_shoulders(
                    points, (0, 0, 100, 200), (0, 0, 100, 200), for_selection=False,
                )
                self.assertIsNone(value)
                self.assertTrue(reason)

    def test_tracking_still_rejects_head_outside_source_person(self):
        points = landmarks()
        for point in points[:11]:
            point.y = 1.2
        value, reason = evaluate_shoulders(
            points, (0, 0, 100, 200), (0, 0, 100, 200), for_selection=False,
        )
        self.assertIsNone(value)
        self.assertEqual(reason, "head_outside_person")

    def test_tracking_still_rejects_a_lowered_head_pose_fitting_another_person(self):
        adult = observation(1, (0, 0, 300, 500))
        child = observation(2, (140, 100, 240, 300))
        points = landmarks()
        for point in points:
            point.x, point.y = 190 / 300, 250 / 500
        points[11].x, points[11].y = 160 / 300, 160 / 500
        points[12].x, points[12].y = 220 / 300, 160 / 500
        value, reason = evaluate_shoulders(
            points, (0, 0, 300, 500), adult.bbox, [adult, child],
            source_track_id=1, for_selection=False,
        )
        self.assertIsNone(value)
        self.assertEqual(reason, "ambiguous_with_id_2")


class SelectorTests(unittest.TestCase):
    def make_selector(self, **kwargs):
        return TargetSelector(init_seconds=2.0, min_samples=10, **kwargs)

    @staticmethod
    def fill(selector, first_width=40, second_width=80, third=False):
        for index in range(11):
            timestamp = index * 200
            tracks = [observation(1), observation(2)]
            if third:
                tracks.append(observation(3))
            selector.observe(tracks, timestamp)
            selector.add_sample(1, sample(first_width), timestamp)
            selector.add_sample(2, sample(second_width), timestamp)

    def test_black_screen_cannot_lock(self):
        selector = self.make_selector()
        for timestamp in (0, 1000, 10000):
            selector.observe([], timestamp)
            self.assertIsNone(selector.decide())

    def test_single_person_waits_for_valid_evidence_then_locks(self):
        selector = self.make_selector()
        for timestamp in range(0, 2000, 200):
            selector.observe([observation(1)], timestamp)
            selector.add_sample(1, sample(40), timestamp)
            self.assertIsNone(selector.decide())
        selector.observe([observation(1)], 2000)
        selector.add_sample(1, sample(40), 2000)
        self.assertEqual(selector.decide(), 1)
        self.assertEqual(selector.selection_reason, "only_qualified_candidate")
        self.assertEqual(selector.statistics()[1]["valid_seconds"], 2)

    def test_single_person_requires_minimum_sample_count_and_stability(self):
        selector = self.make_selector()
        for timestamp in (0, 1000, 2000):
            selector.observe([observation(1)], timestamp)
            selector.add_sample(1, sample(40), timestamp)
        self.assertEqual(selector.statistics()[1]["valid_seconds"], 2)
        self.assertIsNone(selector.decide())
        self.assertIsNone(selector.target_id)
        selector = self.make_selector(stable_observations=20)
        for index in range(11):
            timestamp = index * 200
            selector.observe([observation(1)], timestamp)
            selector.add_sample(1, sample(40), timestamp)
        self.assertGreaterEqual(selector.statistics()[1]["sample_count"], 10)
        self.assertEqual(selector.statistics()[1]["valid_seconds"], 2)
        self.assertFalse(selector.statistics()[1]["stable"])
        self.assertIsNone(selector.decide())

    def test_first_single_frame_does_not_lock(self):
        selector = self.make_selector()
        selector.observe([observation(1)], 0)
        selector.add_sample(1, sample(40), 0)
        self.assertIsNone(selector.decide())
        self.assertFalse(selector.statistics()[1]["ready"])

    def test_multiframe_median_selects_and_locks_across_ranking_changes(self):
        selector = self.make_selector()
        self.fill(selector)
        self.assertEqual(selector.decide(), 1)
        self.assertEqual(selector.selection_reason, "smallest_qualified_candidate")
        self.assertEqual(selector.statistics()[1]["valid_seconds"], 2)
        for timestamp in range(2200, 5000, 200):
            selector.observe([observation(1), observation(2)], timestamp)
            selector.add_sample(1, sample(200), timestamp)
            selector.add_sample(2, sample(20), timestamp)
            self.assertEqual(selector.decide(), 1)

    def test_invalid_other_shoulders_do_not_block_the_only_ready_person(self):
        selector = self.make_selector()
        for index in range(11):
            timestamp = index * 200
            selector.observe([observation(1), observation(2)], timestamp)
            selector.add_sample(1, None, timestamp, "missing_shoulders")
            selector.add_sample(2, sample(80), timestamp)
        self.assertEqual(selector.decide(), 2)
        stats = selector.statistics()[1]
        self.assertEqual(stats["sample_count"], 0)
        self.assertIsNone(stats["median_width"])
        self.assertEqual(stats["rejection_reason"], "missing_shoulders")

    def test_unassessed_or_unstable_third_person_does_not_block_selection(self):
        selector = self.make_selector()
        self.fill(selector, third=True)
        self.assertEqual(selector.decide(), 1)
        self.assertEqual(selector.statistics()[3]["sample_count"], 0)
        selector = self.make_selector()
        self.fill(selector)
        selector.observe([observation(1), observation(2), observation(3)], 2200)
        self.assertFalse(selector.statistics()[3]["stable"])
        self.assertEqual(selector.decide(), 1)

    def test_ready_smaller_person_is_not_blocked_by_insufficient_other_samples(self):
        selector = self.make_selector()
        for index in range(11):
            timestamp = index * 200
            selector.observe([observation(1), observation(2)], timestamp)
            selector.add_sample(1, sample(40), timestamp)
            if index >= 8:
                selector.add_sample(2, sample(80), timestamp)
        self.assertFalse(selector.statistics()[2]["ready"])
        self.assertEqual(selector.decide(), 1)

    def test_similar_widths_need_clearer_evidence(self):
        selector = self.make_selector(min_gap=0.1)
        self.fill(selector, first_width=76, second_width=80)
        self.assertIsNone(selector.decide())
        self.assertEqual(selector.status, "shoulder_sizes_too_similar")

    def test_ready_history_cannot_lock_missing_prediction_or_stale_person(self):
        for case in ("missing", "prediction", "stale"):
            with self.subTest(case=case):
                selector = self.make_selector()
                for index in range(11):
                    timestamp = index * 200
                    selector.observe([observation(1)], timestamp)
                    selector.add_sample(1, sample(40), timestamp)
                self.assertTrue(selector.statistics()[1]["ready"])
                if case == "missing":
                    selector.observe([], 2200)
                elif case == "prediction":
                    selector.observe([observation(1, observed=False)], 2200)
                else:
                    selector.observe([observation(1)], 4100)
                self.assertFalse(selector.statistics()[1]["ready"])
                self.assertIsNone(selector.decide())

    def test_wall_time_and_sparse_good_samples_are_not_valid_duration(self):
        selector = self.make_selector()
        for timestamp in (0, 20000):
            selector.observe([observation(1), observation(2)], timestamp)
            selector.add_sample(1, sample(40), timestamp)
            selector.add_sample(2, sample(80), timestamp)
        self.assertIsNone(selector.decide())
        self.assertEqual(selector.statistics()[1]["valid_seconds"], 0)

    def test_rejected_intervals_do_not_count_as_valid_observation_time(self):
        selector = self.make_selector()
        for index in range(25):
            timestamp = index * 100
            selector.observe([observation(1), observation(2)], timestamp)
            selector.add_sample(1, sample(40) if index % 2 == 0 else None, timestamp)
            selector.add_sample(2, sample(80), timestamp)
        self.assertGreaterEqual(selector.statistics()[1]["sample_count"], 10)
        self.assertEqual(selector.statistics()[1]["valid_seconds"], 0)
        self.assertEqual(selector.decide(), 2)

    def test_prediction_and_changed_id_never_replace_lost_target(self):
        selector = self.make_selector()
        self.fill(selector)
        self.assertEqual(selector.decide(), 1)
        selector.observe([observation(1, observed=False), observation(2)], 2200)
        self.assertIsNone(selector.target_observation()[0])
        selector.observe([observation(7), observation(2)], 2400)
        self.assertIsNone(selector.target_observation()[0])
        self.assertEqual(selector.decide(), 1)

    def test_implausible_same_id_jump_is_latched_until_reset(self):
        selector = self.make_selector()
        self.fill(selector)
        selector.decide()
        selector.observe([observation(1, (2000, 1000, 2100, 1200)), observation(2)], 2200)
        self.assertIsNone(selector.target_observation()[0])
        selector.observe([observation(1), observation(2)], 2400)
        self.assertIsNone(selector.target_observation()[0])
        self.assertEqual(selector.target_id, 1)
        selector.reset()
        self.assertIsNone(selector.target_id)
        self.assertEqual(selector.selection_reason, "")
        self.assertEqual(selector.statistics(), {})

    def test_long_unobserved_gap_requires_confirmation_instead_of_trusting_id(self):
        selector = self.make_selector()
        self.fill(selector)
        selector.decide()
        selector.observe([], 2200)
        selector.observe([observation(1), observation(2)], 5000)
        self.assertIsNone(selector.target_observation()[0])
        self.assertIn("uncertain", selector.status)

    def test_invalid_timing_configuration_and_backwards_time(self):
        with self.assertRaises(ValueError):
            TargetSelector(init_seconds=float("nan"))
        selector = self.make_selector()
        selector.observe([], 1000)
        with self.assertRaises(ValueError):
            selector.observe([], 500)


if __name__ == "__main__":
    unittest.main()
