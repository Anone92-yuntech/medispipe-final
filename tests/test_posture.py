"""Source-time and occlusion regressions for suspected table-rest geometry."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posture_detection import PostureConfig, TableRestDetector  # noqa: E402
from target_selection import MappedPose, map_pose_to_original  # noqa: E402


def pose(head_y=50, head_x=100, arm_y=200, shoulder_width=100):
    points = np.full((33, 2), np.nan)
    points[:11] = (head_x, head_y)
    points[11] = (100 - shoulder_width / 2, 150)
    points[12] = (100 + shoulder_width / 2, 150)
    points[13], points[15] = (50, arm_y), (150, arm_y)
    points[14], points[16] = (150, arm_y + 20), (50, arm_y + 20)
    return MappedPose(points, np.ones(33), np.ones(33))


def detector(**overrides):
    config = PostureConfig(baseline_seconds=0.2, baseline_min_samples=3,
                           hold_seconds=0.3, **overrides)
    return TableRestDetector(config)


def calibrate(item, reference=None):
    reference = reference or pose()
    result = None
    for timestamp in (0, 100, 200):
        result = item.update(reference, timestamp)
    return result


class PostureGeometryTests(unittest.TestCase):
    def test_baseline_requires_both_time_and_sample_count(self):
        item = TableRestDetector(PostureConfig(baseline_seconds=0.2, baseline_min_samples=5))
        self.assertEqual(item.update(pose(), 0).state, "calibrating")
        self.assertEqual(item.update(pose(), 100).state, "calibrating")
        self.assertEqual(item.update(pose(), 200).state, "calibrating")
        self.assertEqual(item.update(pose(), 210).state, "calibrating")
        final = item.update(pose(), 220)
        self.assertEqual(final.state, "normal")
        self.assertEqual(final.baseline_samples, 5)
        self.assertEqual(final.baseline_head_y, 50)
        self.assertEqual(final.baseline_shoulder_width, 100)

    def test_baseline_can_be_built_with_no_visible_arms(self):
        reference = pose()
        reference.presence[[13, 14, 15, 16]] = 0
        item = detector()
        result = calibrate(item, reference)
        self.assertTrue(item.baseline_ready)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.detail, "forearm_unavailable")
        self.assertEqual(item.update(pose(head_y=180), 300).state, "candidate")

    def test_upright_reference_is_required_and_invalid_samples_are_not_zero(self):
        item = detector()
        for timestamp in (0, 100, 200, 300):
            result = item.update(pose(head_y=150), timestamp)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.detail, "upright_baseline_required")
        self.assertIsNone(result.baseline_head_y)
        self.assertEqual(result.baseline_samples, 0)
        invalid = pose()
        invalid.presence[12] = 0
        self.assertEqual(item.update(invalid, 400).detail, "baseline_shoulders_unavailable")
        self.assertFalse(item.baseline_ready)

    def test_head_lowering_alone_does_not_qualify(self):
        item = detector()
        calibrate(item)
        result = item.update(pose(head_y=100), 300)
        self.assertEqual(result.state, "normal")
        self.assertAlmostEqual(result.head_drop_ratio, 0.5)
        self.assertAlmostEqual(result.forearm_distance_ratio, 1.0)

    def test_forearm_near_head_without_head_lowering_does_not_qualify(self):
        item = detector()
        calibrate(item)
        result = item.update(pose(arm_y=60), 300)
        self.assertEqual(result.state, "normal")
        self.assertAlmostEqual(result.head_drop_ratio, 0)
        self.assertAlmostEqual(result.forearm_distance_ratio, 0.1)

    def test_head_distance_is_to_closed_segment_not_extended_line(self):
        item = detector()
        calibrate(item)
        result = item.update(pose(head_y=180, head_x=200), 300)
        self.assertEqual(result.state, "normal")
        self.assertGreater(result.forearm_distance_ratio, 0.5)

    def test_single_reliable_nose_and_one_visible_forearm_are_sufficient(self):
        reference = pose()
        reference.visibility[1:11] = 0
        item = detector()
        calibrate(item, reference)
        current = pose(head_y=180)
        current.visibility[1:11] = 0
        current.visibility[[14, 16]] = 0
        result = item.update(current, 300)
        self.assertEqual(result.state, "candidate")
        self.assertAlmostEqual(result.forearm_distance_ratio, 0.2)

    def test_thresholds_accept_exact_boundary_with_reliable_points(self):
        item = detector()
        reference = pose()
        reference.visibility[:] = 0.7
        reference.presence[:] = 0.7
        calibrate(item, reference)
        current = pose(head_y=80, arm_y=115)
        current.visibility[:] = 0.7
        current.presence[:] = 0.7
        result = item.update(current, 300)
        self.assertEqual(result.state, "candidate")
        self.assertAlmostEqual(result.head_drop_ratio, 0.3)
        self.assertAlmostEqual(result.forearm_distance_ratio, 0.35)

    def test_missing_low_confidence_nonfinite_or_degenerate_arms_are_unknown(self):
        for invalid in ("visibility", "presence", "nonfinite", "degenerate", "missing"):
            with self.subTest(invalid=invalid):
                item = detector()
                calibrate(item)
                current = pose(head_y=180)
                if invalid == "visibility":
                    current.visibility[[13, 14]] = 0.1
                elif invalid == "presence":
                    current.presence[[15, 16]] = 0.1
                elif invalid == "nonfinite":
                    current.points[[15, 16], 0] = np.inf
                elif invalid == "degenerate":
                    current.points[15] = current.points[13]
                    current.points[16] = current.points[14]
                else:
                    current.points = current.points[:13]
                result = item.update(current, 300)
                self.assertEqual(result.state, "unknown")
                self.assertEqual(result.detail, "forearm_unavailable")

    def test_frozen_head_anchor_cannot_change_to_other_face_points(self):
        item = detector()
        calibrate(item)
        current = pose(head_y=180)
        current.visibility[0] = 0
        result = item.update(current, 300)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.detail, "head_unavailable")
        self.assertTrue(item.baseline_ready)

    def test_face_fallback_requires_three_points_and_keeps_exact_subset(self):
        reference = pose()
        reference.visibility[:11] = 0
        reference.visibility[[1, 4, 7]] = 1
        item = detector()
        self.assertEqual(calibrate(item, reference).state, "normal")
        current = pose(head_y=180)
        current.visibility[:11] = 0
        current.visibility[[1, 4, 7]] = 1
        self.assertEqual(item.update(current, 300).state, "candidate")
        current.visibility[7] = 0
        current.visibility[8] = 1
        self.assertEqual(item.update(current, 400).detail, "head_unavailable")
        second = detector()
        reference.visibility[7] = 0
        self.assertEqual(second.update(reference, 0).detail, "head_unavailable")

    def test_geometry_must_stay_with_source_person_box(self):
        item = detector()
        calibrate(item)
        current = pose(head_y=180)
        self.assertEqual(item.update(current, 300, bbox=(0, 0, 40, 300)).detail,
                         "head_outside_person")
        self.assertEqual(item.update(current, 400, bbox=(0, 0, 0, 300)).detail,
                         "invalid_person_box")
        current.points[[13, 14, 15, 16], 0] += 400
        self.assertEqual(item.update(current, 500, bbox=(0, 0, 200, 300)).detail,
                         "forearm_unavailable")

    def test_scale_and_translation_preserve_ratios(self):
        first, second = detector(), detector()
        reference = pose()
        transformed = pose()
        transformed.points = transformed.points * 2 + [400, 300]
        calibrate(first, reference)
        calibrate(second, transformed)
        current = pose(head_y=180)
        transformed = pose(head_y=180)
        transformed.points = transformed.points * 2 + [400, 300]
        a, b = first.update(current, 300), second.update(transformed, 300)
        self.assertEqual(a.state, b.state)
        self.assertAlmostEqual(a.head_drop_ratio, b.head_drop_ratio)
        self.assertAlmostEqual(a.forearm_distance_ratio, b.forearm_distance_ratio)

    def test_different_crops_map_to_identical_original_pixel_evidence(self):
        def through_crop(source, rect):
            left, top, width, height = rect
            rows = [[(point[0] - left) / width, (point[1] - top) / height, visibility, presence]
                    for point, visibility, presence in
                    zip(source.points, source.visibility, source.presence)]
            return map_pose_to_original(rows, rect)
        first, second = detector(), detector()
        calibrate(first, through_crop(pose(), (0, 0, 200, 300)))
        calibrate(second, through_crop(pose(), (-100, -100, 500, 600)))
        a = first.update(through_crop(pose(head_y=180), (0, 0, 200, 300)), 300)
        b = second.update(through_crop(pose(head_y=180), (-100, -100, 500, 600)), 300)
        self.assertEqual(a.state, b.state)
        self.assertAlmostEqual(a.head_drop_ratio, b.head_drop_ratio)
        self.assertAlmostEqual(a.forearm_distance_ratio, b.forearm_distance_ratio)

    def test_baseline_is_robust_and_does_not_drift_after_lock(self):
        item = detector()
        for timestamp, y in zip((0, 100, 200), (50, 45, 55)):
            result = item.update(pose(head_y=y), timestamp)
        self.assertEqual(result.baseline_head_y, 50)
        for timestamp in (300, 400, 500, 600):
            result = item.update(pose(head_y=180, shoulder_width=20), timestamp)
        self.assertEqual(result.state, "resting")
        self.assertEqual(result.baseline_head_y, 50)
        self.assertEqual(result.baseline_shoulder_width, 100)
        self.assertAlmostEqual(result.head_drop_ratio, 1.3)


class PostureTimeTests(unittest.TestCase):
    def test_exact_hold_boundary_uses_source_time(self):
        item = detector()
        calibrate(item)
        current = pose(head_y=180)
        self.assertEqual(item.update(current, 300).held_seconds, 0)
        self.assertEqual(item.update(current, 400).state, "candidate")
        self.assertEqual(item.update(current, 599).state, "candidate")
        result = item.update(current, 600)
        self.assertEqual(result.state, "resting")
        self.assertAlmostEqual(result.held_seconds, 0.3)

    def test_missing_frame_breaks_hold_and_clears_positive_output(self):
        item = detector()
        calibrate(item)
        current = pose(head_y=180)
        for timestamp in (300, 400, 500, 600):
            result = item.update(current, timestamp)
        self.assertEqual(result.state, "resting")
        result = item.update(None, 700)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.held_seconds, 0)
        self.assertTrue(item.baseline_ready)
        result = item.update(current, 800)
        self.assertEqual(result.state, "candidate")
        self.assertEqual(result.held_seconds, 0)

    def test_long_timestamp_gap_cannot_instantly_confirm(self):
        item = detector()
        calibrate(item)
        current = pose(head_y=180)
        self.assertEqual(item.update(current, 300).state, "candidate")
        result = item.update(current, 8000)
        self.assertEqual(result.state, "candidate")
        self.assertEqual(result.held_seconds, 0)
        self.assertTrue(item.baseline_ready)

    def test_missing_and_gapped_baseline_restarts_contiguous_evidence(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                item = detector()
                item.update(pose(), 0)
                item.update(pose(), 100)
                if missing:
                    item.update(None, 200)
                    start = 300
                else:
                    start = 2000
                result = item.update(pose(), start)
                self.assertEqual(result.state, "calibrating")
                self.assertEqual(result.baseline_samples, 1)
                self.assertEqual(item.update(pose(), start + 100).state, "calibrating")
                self.assertEqual(item.update(pose(), start + 200).state, "normal")

    def test_normal_condition_breaks_hold_and_exits_resting(self):
        item = detector()
        calibrate(item)
        for timestamp in (300, 400, 500, 600):
            result = item.update(pose(head_y=180), timestamp)
        self.assertEqual(result.state, "resting")
        self.assertEqual(item.update(pose(), 700).state, "normal")
        result = item.update(pose(head_y=180), 800)
        self.assertEqual(result.state, "candidate")
        self.assertEqual(result.held_seconds, 0)

    def test_reset_clears_baseline_and_allows_new_video_time(self):
        item = detector()
        calibrate(item)
        item.update(pose(head_y=180), 300)
        item.reset()
        self.assertFalse(item.baseline_ready)
        result = item.update(pose(head_y=80), 0)
        self.assertEqual(result.state, "calibrating")
        self.assertIsNone(result.baseline_head_y)
        self.assertEqual(result.baseline_samples, 1)

    def test_timestamps_must_be_nonnegative_integer_and_strictly_increasing(self):
        item = detector()
        for timestamp in (-1, 0.5, True, float("nan")):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                item.update(pose(), timestamp)
        item.update(pose(), np.int64(0))
        for timestamp in (0, -1):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                item.update(pose(), timestamp)

    def test_config_rejects_nonfinite_or_nonsensical_thresholds(self):
        for name in ("baseline_seconds", "head_drop_ratio", "forearm_distance_ratio",
                     "hold_seconds", "max_gap_seconds"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(PostureConfig(), **{name: value})
        for value in (0, -1, 1.5, True):
            with self.subTest(samples=value), self.assertRaises(ValueError):
                PostureConfig(baseline_min_samples=value)
        for value in (-0.1, 1.1, float("nan")):
            with self.subTest(confidence=value), self.assertRaises(ValueError):
                PostureConfig(confidence=value)


if __name__ == "__main__":
    unittest.main()
