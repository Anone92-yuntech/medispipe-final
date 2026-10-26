import unittest

import numpy as np

from seat_detection import SeatConfig, SeatDetector
from target_selection import MappedPose, TrackObservation, map_pose_to_original


def pose_at(x=300.0, y=240.0, shoulder_width=100.0):
    points = np.full((33, 2), [x, y - 60.0], dtype=float)
    points[11] = [x - shoulder_width / 2, y]
    points[12] = [x + shoulder_width / 2, y]
    return MappedPose(points, np.ones(33), np.ones(33))


def track_at(x=300.0, y=240.0, observed=True, track_id=1):
    return TrackObservation(track_id, np.array([x - 100, y - 140, x + 100, y + 220]),
                            observed, 100, 0, 0)


class SeatTests(unittest.TestCase):
    def setUp(self):
        self.detector = SeatDetector(SeatConfig(baseline_seconds=0.3, min_samples=4,
                                                hold_seconds=0.5, return_seconds=0.3))

    def calibrate(self, detector=None, offset=0):
        detector = detector or self.detector
        result = None
        for ms in (0, 100, 200, 300):
            result = detector.update(pose_at(), track_at(), offset + ms, frame_shape=(1080, 1920))
        self.assertEqual(result.state, "seated")
        return result

    def test_static_baseline_is_median_and_frozen(self):
        result = self.calibrate()
        self.assertEqual(result.baseline_anchor, (300.0, 240.0))
        self.assertEqual(result.seat_roi, (200.0, 0.0, 400.0, 1080.0))
        result = self.detector.update(pose_at(330, 260, 150), track_at(330, 260), 400)
        self.assertEqual(result.state, "seated")
        self.assertEqual(result.baseline_anchor, (300.0, 240.0))
        self.assertEqual(self.detector.baseline_scale, 100.0)

    def test_normal_lean_inside_region_never_becomes_departure(self):
        self.calibrate()
        for ms in range(400, 2000, 100):
            result = self.detector.update(pose_at(390, 295), track_at(390, 295), ms)
            self.assertEqual(result.state, "seated")

    def test_standing_in_place_or_vertical_lean_is_not_departure(self):
        self.calibrate()
        for index, y in enumerate((100, 80, 60, 100, 310, 350, 400, 400, 400, 400)):
            result = self.detector.update(pose_at(300, y), track_at(300, y), 400 + index * 100)
            self.assertEqual(result.state, "seated")
            self.assertEqual(result.displacement_ratio, 0)

    def test_departure_requires_continuous_observed_hold(self):
        self.calibrate()
        for ms in range(400, 900, 100):
            result = self.detector.update(pose_at(450), track_at(450), ms)
            self.assertEqual(result.state, "candidate")
        result = self.detector.update(pose_at(450), track_at(450), 900)
        self.assertEqual(result.state, "away")
        self.assertEqual(result.held_seconds, 0.5)
        self.assertAlmostEqual(result.displacement_ratio, 1.5)

    def test_missing_and_predicted_tracks_are_unknown_even_after_long_wait(self):
        self.calibrate()
        for timestamp, track in ((400, None), (5000, track_at(observed=False)), (5100, None)):
            result = self.detector.update(pose_at(450), track, timestamp)
            self.assertEqual(result.state, "unknown")
            self.assertEqual(result.detail, "target_not_observed")
        result = self.detector.update(pose_at(450), track_at(450), 5200)
        self.assertEqual(result.state, "candidate")
        self.assertEqual(result.held_seconds, 0)

    def test_missing_pose_does_not_use_bbox_as_departure_evidence(self):
        self.calibrate()
        result = self.detector.update(None, track_at(600), 400)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.detail, "pose_unavailable")

    def test_low_head_or_shoulder_confidence_is_unknown(self):
        self.calibrate()
        pose = pose_at(450)
        pose.visibility[:11] = 0.1
        result = self.detector.update(pose, track_at(450), 400)
        self.assertEqual(result.detail, "head_unavailable")
        pose = pose_at(450)
        pose.presence[11] = 0.1
        result = self.detector.update(pose, track_at(450), 500)
        self.assertEqual(result.detail, "shoulders_unavailable")

    def test_missing_head_can_use_three_reliable_face_points(self):
        self.calibrate()
        pose = pose_at()
        pose.visibility[0] = 0
        self.assertEqual(self.detector.update(pose, track_at(), 400).state, "seated")

    def test_return_requires_hold_and_unknown_breaks_return_evidence(self):
        self.calibrate()
        for ms in range(400, 1000, 100):
            result = self.detector.update(pose_at(450), track_at(450), ms)
        self.assertEqual(result.state, "away")
        result = self.detector.update(pose_at(), track_at(), 1000)
        self.assertEqual(result.detail, "holding_seat_return")
        result = self.detector.update(None, track_at(), 1100)
        self.assertEqual(result.state, "unknown")
        for ms in (1200, 1300, 1400):
            result = self.detector.update(pose_at(), track_at(), ms)
            self.assertEqual(result.state, "away")
        result = self.detector.update(pose_at(), track_at(), 1500)
        self.assertEqual(result.state, "seated")

    def test_source_gap_breaks_departure_hold_but_keeps_reference(self):
        self.calibrate()
        self.detector.update(pose_at(450), track_at(450), 400)
        self.detector.update(pose_at(450), track_at(450), 500)
        result = self.detector.update(pose_at(450), track_at(450), 5000)
        self.assertEqual(result.state, "candidate")
        self.assertEqual(result.held_seconds, 0)
        self.assertEqual(result.baseline_anchor, (300.0, 240.0))

    def test_nonzero_source_start_uses_relative_elapsed_time(self):
        self.calibrate(offset=100000)
        for ms in range(100400, 100900, 100):
            self.assertEqual(self.detector.update(pose_at(450), track_at(450), ms).state, "candidate")
        self.assertEqual(self.detector.update(pose_at(450), track_at(450), 100900).state, "away")

    def test_coordinates_mapped_from_different_crops_agree(self):
        self.calibrate()
        original = pose_at(450)
        for index, rect in enumerate(((200, 50, 400, 500), (0, 0, 1000, 800))):
            left, top, width, height = rect
            rows = [((x - left) / width, (y - top) / height, 1, 1) for x, y in original.points]
            mapped = map_pose_to_original(rows, rect)
            result = self.detector.update(mapped, track_at(450), 400 + index * 100)
            self.assertEqual(result.state, "candidate")
            self.assertAlmostEqual(result.displacement_ratio, 1.5)

    def test_manual_roi_overrides_automatic_geometry(self):
        detector = SeatDetector(self.detector.config, roi=(250, 150, 350, 300))
        result = self.calibrate(detector)
        self.assertEqual(result.seat_roi, (250, 150, 350, 300))
        result = detector.update(pose_at(375), track_at(375), 400)
        self.assertEqual(result.state, "candidate")

    def test_manual_roi_requires_an_initial_inside_reference(self):
        detector = SeatDetector(self.detector.config, roi=(100, 100, 200, 300))
        result = detector.update(pose_at(), track_at(), 0)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.detail, "awaiting_seated_baseline")

    def test_external_baseline_scale_used_then_frozen(self):
        for ms in (0, 100, 200, 300):
            result = self.detector.update(pose_at(), track_at(), ms, baseline_scale=200)
        self.assertEqual(result.seat_roi, (100, 0, 500, 1040))
        result = self.detector.update(pose_at(450), track_at(450), 400, baseline_scale=10)
        self.assertEqual(result.state, "seated")
        self.assertAlmostEqual(result.displacement_ratio, 0.75)

    def test_reset_and_id_change_discard_previous_reference(self):
        self.calibrate()
        result = self.detector.update(pose_at(), track_at(track_id=2), 400)
        self.assertEqual(result.detail, "target_identity_changed")
        self.assertIsNone(result.baseline_anchor)
        self.detector.reset()
        result = self.detector.update(pose_at(), track_at(), 0)
        self.assertEqual(result.state, "calibrating")

    def test_missing_frames_while_away_are_unknown(self):
        self.calibrate()
        for ms in range(400, 1000, 100):
            result = self.detector.update(pose_at(450), track_at(450), ms)
        self.assertEqual(result.state, "away")
        result = self.detector.update(None, None, 1000)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.held_seconds, 0)

    def test_lowered_head_cannot_be_used_as_initial_sitting_reference(self):
        pose = pose_at()
        pose.points[:11, 1] = 250
        result = self.detector.update(pose, track_at(), 0)
        self.assertEqual(result.detail, "upright_baseline_required")

    def test_invalid_or_outside_geometry_is_unknown(self):
        self.calibrate()
        pose = pose_at(450)
        pose.points[11, 0] = np.nan
        self.assertEqual(self.detector.update(pose, track_at(450), 400).state, "unknown")
        self.assertEqual(self.detector.update(pose_at(450), track_at(), 500).state, "unknown")
        self.assertEqual(self.detector.update(pose_at(), track_at(), 600,
                                             frame_shape=(200, 200)).state, "unknown")

    def test_time_and_config_validation(self):
        for config in ({"baseline_seconds": 0}, {"departure_ratio": float("nan")},
                       {"hold_seconds": -1}, {"min_samples": True}, {"confidence": 1.1}):
            with self.assertRaises(ValueError):
                SeatConfig(**config)
        with self.assertRaises(ValueError):
            SeatDetector(roi=(1, 2, 1, 5))
        self.detector.update(None, None, 0)
        for ms in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                self.detector.update(None, None, ms)


if __name__ == "__main__":
    unittest.main()
