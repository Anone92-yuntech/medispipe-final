"""Pose-qualified seat baseline, then observed-box IoU departure rules."""
import unittest

import numpy as np

from seat_detection import SeatConfig, SeatDetector
from target_selection import MappedPose, TrackObservation, map_pose_to_original


def pose_at(x=300.0, y=300.0, shoulder_width=100.0):
    points = np.full((33, 2), [x, y - 60.0], dtype=float)
    points[11] = [x - shoulder_width / 2, y]
    points[12] = [x + shoulder_width / 2, y]
    return MappedPose(points, np.ones(33), np.ones(33))


def track_at(x=300.0, y=300.0, observed=True, track_id=1,
             width=300.0, height=300.0):
    return track_box([x - width / 2, y - height / 2,
                      x + width / 2, y + height / 2], observed, track_id)


def track_box(box, observed=True, track_id=1):
    return TrackObservation(track_id, np.asarray(box, dtype=float), observed, 100, 0, 0)


class SeatTests(unittest.TestCase):
    def setUp(self):
        # Old hold arguments remain accepted, but the new rule must not use them.
        self.detector = SeatDetector(SeatConfig(baseline_seconds=0.3, min_samples=4,
                                                hold_seconds=10, return_seconds=10))

    def calibrate(self, detector=None, offset=0, pose=None):
        detector = detector or self.detector
        pose = pose_at() if pose is None else pose
        result = None
        for ms in (0, 100, 200, 300):
            result = detector.update(pose, track_at(), offset + ms,
                                     frame_shape=(1080, 1920))
        self.assertEqual(result.state, "seated")
        self.assertTrue(detector.baseline_ready)
        return result

    def make_away(self, detector=None, timestamp=400):
        detector = detector or self.detector
        result = detector.update(None, track_at(900), timestamp)
        self.assertEqual(result.state, "away")
        self.assertEqual(result.iou, 0)
        return result

    def test_default_reference_requires_two_source_seconds_and_ten_samples(self):
        detector = SeatDetector()
        self.assertEqual(detector.config.iou_threshold, 0.5)
        for ms in range(0, 2000, 100):
            result = detector.update(pose_at(), track_at(), ms)
            self.assertEqual(result.state, "calibrating")
            self.assertFalse(detector.baseline_ready)
        result = detector.update(pose_at(), track_at(), 1999)
        self.assertEqual(result.state, "calibrating")
        result = detector.update(pose_at(), track_at(), 2000)
        self.assertEqual(result.state, "seated")
        self.assertEqual(result.seat_roi, (150.0, 150.0, 450.0, 450.0))
        self.assertEqual(result.iou, 1)

    def test_minimum_observations_cannot_be_replaced_by_elapsed_time(self):
        detector = SeatDetector(SeatConfig(baseline_seconds=0.1, min_samples=4))
        for ms in (0, 100, 200):
            result = detector.update(pose_at(), track_at(), ms)
            self.assertEqual(result.state, "calibrating")
        self.assertEqual(detector.update(pose_at(), track_at(), 300).state, "seated")

    def test_reference_is_coordinate_median_and_frozen(self):
        boxes = ([150, 150, 450, 450], [160, 140, 460, 440],
                 [140, 160, 440, 460], [152, 148, 454, 452])
        for ms, box in zip((0, 100, 200, 300), boxes):
            result = self.detector.update(pose_at(), track_box(box), ms)
        expected = (151.0, 149.0, 452.0, 451.0)
        self.assertEqual(result.state, "seated")
        self.assertEqual(result.seat_roi, expected)
        for ms in range(400, 1800, 100):
            result = self.detector.update(None, track_at(340, width=350), ms)
            self.assertEqual(result.seat_roi, expected)
            self.assertEqual(result.state, "seated")

    def test_qualified_baseline_does_not_use_external_shoulder_scale(self):
        result = self.calibrate()
        self.assertEqual(result.seat_roi, (150.0, 150.0, 450.0, 450.0))
        for ms, scale in ((400, 10), (500, 1000), (600, float("nan"))):
            result = self.detector.update(None, track_at(), ms, baseline_scale=scale)
            self.assertEqual(result.state, "seated")
            self.assertEqual(result.iou, 1)
            self.assertEqual(result.seat_roi, (150.0, 150.0, 450.0, 450.0))

    def test_visible_lowered_head_still_supplies_the_required_pose_evidence(self):
        pose = pose_at()
        pose.points[:11, 1] = 350
        self.calibrate(pose=pose)

    def test_unobserved_pose_cannot_establish_a_box_baseline(self):
        for ms in range(0, 2101, 100):
            result = self.detector.update(None, track_at(), ms)
            self.assertEqual(result.state, "unknown")
            self.assertEqual(result.detail, "baseline_head_shoulders_unavailable")
            self.assertFalse(self.detector.baseline_ready)
            self.assertIsNone(result.seat_roi)

    def test_face_and_each_shoulder_must_be_valid_during_the_whole_baseline(self):
        invalid = []
        for index in (11, 12):
            for confidence in ("visibility", "presence"):
                pose = pose_at()
                getattr(pose, confidence)[index] = 0.69
                invalid.append(pose)
        hidden_face = pose_at()
        hidden_face.visibility[:11] = 0.1
        invalid.append(hidden_face)
        nan_shoulder = pose_at()
        nan_shoulder.points[12, 0] = np.nan
        invalid.append(nan_shoulder)
        nonfinite_face = pose_at()
        nonfinite_face.points[:11, 0] = np.inf
        invalid.append(nonfinite_face)
        nan_presence = pose_at()
        nan_presence.presence[11] = np.nan
        invalid.append(nan_presence)
        invalid.append(pose_at(700))  # Valid-looking pose belongs outside the source box.
        for index, invalid_pose in enumerate(invalid):
            with self.subTest(case=index):
                detector = SeatDetector(self.detector.config)
                for ms in (0, 100, 200, 300, 400, 500):
                    result = detector.update(invalid_pose, track_at(), ms)
                    self.assertEqual(result.state, "unknown")
                    self.assertEqual(result.detail, "baseline_head_shoulders_unavailable")
                    self.assertIsNone(result.seat_roi)
                self.calibrate(detector, offset=600)

    def test_pose_loss_restarts_the_entire_continuous_baseline_window(self):
        self.detector.update(pose_at(), track_at(), 0)
        self.detector.update(pose_at(), track_at(), 100)
        broken = self.detector.update(None, track_at(900), 200)
        self.assertEqual(broken.state, "unknown")
        for ms in (300, 400, 500, 599):
            result = self.detector.update(pose_at(), track_at(), ms)
            self.assertEqual(result.state, "calibrating")
            self.assertIsNone(result.seat_roi)
        result = self.detector.update(pose_at(), track_at(), 600)
        self.assertEqual(result.state, "seated")
        self.assertEqual(result.seat_roi, (150.0, 150.0, 450.0, 450.0))

    def test_a_last_good_pose_does_not_qualify_prior_pose_free_observations(self):
        detector = SeatDetector()
        for ms in range(0, 2000, 100):
            detector.update(None, track_at(), ms)
        result = detector.update(pose_at(), track_at(), 2000)
        self.assertEqual(result.state, "calibrating")
        self.assertFalse(detector.baseline_ready)
        for ms in range(2100, 4000, 100):
            result = detector.update(pose_at(), track_at(), ms)
            self.assertFalse(detector.baseline_ready)
        result = detector.update(pose_at(), track_at(), 4000)
        self.assertEqual(result.state, "seated")

    def test_baseline_accepts_nose_or_three_reliable_fallback_face_points(self):
        nose_only = pose_at()
        nose_only.visibility[1:11] = 0
        self.calibrate(pose=nose_only)
        fallback = pose_at()
        fallback.visibility[:11] = 0
        fallback.visibility[[2, 5, 7]] = 0.7
        fallback.presence[[2, 5, 7]] = 0.7
        detector = SeatDetector(self.detector.config)
        self.calibrate(detector, pose=fallback)
        only_two = pose_at()
        only_two.visibility[:11] = 0
        only_two.visibility[[2, 5]] = 1
        detector = SeatDetector(self.detector.config)
        for ms in (0, 100, 200, 300):
            result = detector.update(only_two, track_at(), ms)
            self.assertEqual(result.state, "unknown")
            self.assertFalse(detector.baseline_ready)

    def test_manual_reference_cannot_bypass_head_and_shoulder_qualification(self):
        detector = SeatDetector(self.detector.config, roi=(100, 100, 500, 500))
        for ms in range(0, 1100, 100):
            result = detector.update(None, track_at(), ms)
            self.assertEqual(result.state, "unknown")
            self.assertFalse(detector.baseline_ready)
        result = self.calibrate(detector, offset=1100)
        self.assertEqual(result.seat_roi, (100.0, 100.0, 500.0, 500.0))

    def test_iou_half_boundary_is_inclusive_for_remaining_seated(self):
        self.calibrate()
        result = self.detector.update(None, track_at(400), 400)
        self.assertEqual(result.iou, 0.5)
        self.assertEqual(result.state, "seated")
        result = self.detector.update(None, track_at(399.9), 500)
        self.assertGreater(result.iou, 0.5)
        self.assertEqual(result.state, "seated")
        result = self.detector.update(None, track_at(400.1), 501)
        self.assertLess(result.iou, 0.5)
        self.assertEqual(result.state, "away")

    def test_departure_is_immediate_without_pose_or_old_hold_timer(self):
        self.calibrate()
        result = self.detector.update(None, track_at(500), 301)
        self.assertEqual(result.state, "away")
        self.assertAlmostEqual(result.iou, 0.2)
        self.assertEqual(result.held_seconds, 0)

    def test_vertical_displacement_can_also_reduce_iou_below_threshold(self):
        self.calibrate()
        result = self.detector.update(None, track_at(300, 400), 400)
        self.assertEqual(result.iou, 0.5)
        self.assertEqual(result.state, "seated")
        result = self.detector.update(None, track_at(300, 410), 401)
        self.assertLess(result.iou, 0.5)
        self.assertEqual(result.state, "away")

    def test_box_size_changes_count_even_when_center_does_not_move(self):
        for width in (150, 600):
            with self.subTest(width=width):
                detector = SeatDetector(self.detector.config)
                self.calibrate(detector)
                result = detector.update(None, track_at(width=width, height=width), 400)
                self.assertAlmostEqual(result.iou, 0.25)
                self.assertEqual(result.state, "away")

    def test_return_requires_pose_but_has_no_old_return_hold(self):
        self.calibrate()
        self.make_away()
        for ms in (401, 500, 5000):
            result = self.detector.update(None, track_at(), ms)
            self.assertEqual(result.state, "away")
            self.assertEqual(result.detail, "awaiting_return_pose")
            self.assertEqual(result.iou, 1)
        result = self.detector.update(pose_at(), track_at(), 5001)
        self.assertEqual(result.state, "seated")
        self.assertEqual(result.iou, 1)
        self.assertEqual(result.held_seconds, 0)

    def test_return_at_exact_half_iou_and_inside_center_is_allowed(self):
        self.calibrate()
        self.make_away()
        result = self.detector.update(pose_at(400), track_at(400), 401)
        self.assertEqual(result.iou, 0.5)
        self.assertEqual(result.state, "seated")

    def test_return_requires_box_center_inside_reference(self):
        config = SeatConfig(baseline_seconds=0.3, min_samples=4, iou_threshold=0.1)
        detector = SeatDetector(config)
        self.calibrate(detector)
        self.make_away(detector)
        # IoU is 0.2 >= this test threshold, but x=500 exceeds reference right=450.
        result = detector.update(pose_at(500), track_at(500), 500)
        self.assertAlmostEqual(result.iou, 0.2)
        self.assertEqual(result.state, "away")
        result = detector.update(pose_at(440), track_at(440), 501)
        self.assertGreater(result.iou, 0.1)
        self.assertEqual(result.state, "seated")

    def test_return_rejects_invalid_or_other_person_upper_body_pose(self):
        missing_head = pose_at()
        missing_head.visibility[:11] = 0.1
        missing_shoulder = pose_at()
        missing_shoulder.presence[11] = 0.1
        nan_shoulder = pose_at()
        nan_shoulder.points[12, 0] = np.nan
        for pose in (None, missing_head, missing_shoulder, nan_shoulder, pose_at(700)):
            with self.subTest(pose=pose):
                detector = SeatDetector(self.detector.config)
                self.calibrate(detector)
                self.make_away(detector)
                result = detector.update(pose, track_at(), 500)
                self.assertEqual(result.state, "away")
                self.assertEqual(result.detail, "awaiting_return_pose")
                self.assertEqual(detector.update(pose_at(), track_at(), 501).state, "seated")

    def test_return_accepts_valid_head_shoulders_without_forearms(self):
        self.calibrate()
        self.make_away()
        pose = pose_at()
        pose.visibility[13:17] = 0
        pose.presence[13:17] = 0
        self.assertEqual(self.detector.update(pose, track_at(), 401).state, "seated")

    def test_return_can_use_three_reliable_face_points_when_nose_is_missing(self):
        self.calibrate()
        self.make_away()
        pose = pose_at()
        pose.visibility[0] = 0
        self.assertEqual(self.detector.update(pose, track_at(), 401).state, "seated")

    def test_missing_pose_does_not_change_a_never_departed_seated_state(self):
        self.calibrate()
        pose = pose_at()
        pose.visibility[:] = 0
        for ms, current_pose in ((400, None), (500, pose)):
            result = self.detector.update(current_pose, track_at(330), ms)
            self.assertEqual(result.state, "seated")
            self.assertGreater(result.iou, 0.5)

    def test_missing_or_predicted_tracks_are_unknown_and_not_departure_evidence(self):
        self.calibrate()
        for ms, track in ((400, None), (5000, track_at(900, observed=False)), (5100, None)):
            result = self.detector.update(pose_at(900), track, ms)
            self.assertEqual(result.state, "unknown")
            self.assertEqual(result.detail, "target_not_observed")
            self.assertIsNone(result.iou)
        result = self.detector.update(None, track_at(), 5200)
        self.assertEqual(result.state, "seated")
        self.assertEqual(result.seat_roi, (150.0, 150.0, 450.0, 450.0))

    def test_missing_track_while_away_is_unknown_and_does_not_confirm_return(self):
        self.calibrate()
        self.make_away()
        result = self.detector.update(None, None, 500)
        self.assertEqual(result.state, "unknown")
        result = self.detector.update(None, track_at(), 600)
        self.assertEqual(result.state, "away")
        self.assertEqual(result.detail, "awaiting_return_pose")
        self.assertEqual(self.detector.update(pose_at(), track_at(), 601).state, "seated")

    def test_gap_does_not_clear_an_established_reference_or_delay_departure(self):
        self.calibrate()
        result = self.detector.update(None, track_at(900), 100000)
        self.assertEqual(result.state, "away")
        self.assertEqual(result.seat_roi, (150.0, 150.0, 450.0, 450.0))

    def test_missing_or_large_gap_breaks_unfinished_baseline_observations(self):
        for interruption in ("missing", "gap"):
            with self.subTest(interruption=interruption):
                detector = SeatDetector(self.detector.config)
                detector.update(pose_at(), track_at(), 0)
                detector.update(pose_at(), track_at(), 100)
                if interruption == "missing":
                    detector.update(None, track_at(observed=False), 200)
                    restart = 300
                else:
                    restart = 1000
                for ms in (restart, restart + 100, restart + 200):
                    result = detector.update(pose_at(), track_at(), ms)
                    self.assertEqual(result.state, "calibrating")
                self.assertEqual(detector.update(pose_at(), track_at(), restart + 300).state, "seated")

    def test_nonzero_source_start_uses_relative_baseline_elapsed_time(self):
        self.calibrate(offset=100000)
        self.assertEqual(self.detector.update(None, track_at(900), 100301).state, "away")

    def test_pose_crop_mapping_does_not_change_box_iou_or_return_alignment(self):
        self.calibrate()
        self.make_away()
        original = pose_at()
        for index, rect in enumerate(((100, 80, 400, 500), (0, 0, 1000, 800))):
            left, top, width, height = rect
            rows = [((x - left) / width, (y - top) / height, 1, 1)
                    for x, y in original.points]
            mapped = map_pose_to_original(rows, rect)
            result = self.detector.update(mapped, track_at(), 500 + index)
            self.assertEqual(result.state, "seated")
            self.assertEqual(result.iou, 1)

    def test_manual_roi_is_the_fixed_iou_reference(self):
        detector = SeatDetector(self.detector.config, roi=(100, 100, 500, 500))
        result = self.calibrate(detector)
        self.assertEqual(result.seat_roi, (100.0, 100.0, 500.0, 500.0))
        self.assertAlmostEqual(result.iou, 0.5625)
        result = detector.update(None, track_at(700), 400)
        self.assertEqual(result.state, "away")
        self.assertEqual(result.iou, 0)
        self.assertEqual(result.seat_roi, (100.0, 100.0, 500.0, 500.0))

    def test_id_change_and_manual_reset_discard_reference_and_away_state(self):
        self.calibrate()
        self.make_away()
        result = self.detector.update(None, track_at(track_id=2), 500)
        self.assertIsNone(result.seat_roi)
        self.assertFalse(self.detector.baseline_ready)
        self.assertNotEqual(result.state, "away")
        self.detector.reset()
        result = self.detector.update(pose_at(), track_at(), 0)
        self.assertEqual(result.state, "calibrating")
        self.assertIsNone(result.seat_roi)
        self.assertFalse(self.detector.baseline_ready)

    def test_nan_infinite_zero_area_and_inverted_boxes_are_unknown(self):
        self.calibrate()
        bad_boxes = ([np.nan, 150, 450, 450], [150, 150, np.inf, 450],
                     [450, 150, 150, 450], [150, 150, 150, 450], [1, 2, 3])
        for index, box in enumerate(bad_boxes):
            result = self.detector.update(pose_at(), track_box(box), 400 + index)
            self.assertEqual(result.state, "unknown")
            self.assertEqual(result.detail, "invalid_person_box")
            self.assertIsNone(result.iou)
        self.assertTrue(self.detector.baseline_ready)
        self.assertEqual(self.detector.update(None, track_at(), 500).state, "seated")

    def test_time_and_configuration_validation(self):
        invalid_configs = ({"baseline_seconds": 0}, {"departure_ratio": float("nan")},
                           {"hold_seconds": -1}, {"min_samples": True}, {"confidence": 1.1},
                           {"iou_threshold": 0}, {"iou_threshold": -0.1},
                           {"iou_threshold": 1.1}, {"iou_threshold": float("nan")},
                           {"iou_threshold": float("inf")})
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                SeatConfig(**config)
        for box in ((1, 2, 1, 5), (1, 2, float("nan"), 5)):
            with self.assertRaises(ValueError):
                SeatDetector(roi=box)
        self.detector.update(None, None, 0)
        for ms in (0, -1, 1.5, True):
            with self.subTest(timestamp=ms), self.assertRaises(ValueError):
                self.detector.update(None, None, ms)


if __name__ == "__main__":
    unittest.main()
