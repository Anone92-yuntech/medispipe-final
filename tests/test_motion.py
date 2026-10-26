"""Synthetic movement contracts; these do not establish real-video accuracy."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from motion_detection import MotionConfig, MotionDetector


def pose(nose=0.0, left=0.0, right=0.0, translation=(0.0, 0.0),
         left_conf=1.0, right_conf=1.0, face_conf=1.0, shoulders_conf=1.0,
         scale=100.0):
    points = np.full((33, 2), [500.0, 400.0])
    points[11], points[12] = [450, 400], [550, 400]
    points[7], points[8] = [470, 300], [530, 300]
    points[2], points[5] = [485, 285], [515, 285]
    points[0] = [500 + nose * 60, 310]
    points[15], points[16] = [460 + left * 100, 500], [540 + right * 100, 500]
    points = (points - [500, 400]) * (scale / 100) + [500, 400] + translation
    confidence = np.ones(33)
    confidence[15], confidence[16] = left_conf, right_conf
    confidence[[0, 2, 5, 7, 8]] = face_conf
    confidence[[11, 12]] = shoulders_conf
    return SimpleNamespace(points=points, visibility=confidence.copy(), presence=confidence.copy())


class HeadMotionTests(unittest.TestCase):
    def setUp(self):
        self.detector = MotionDetector()
        for ms in range(0, 501, 50):
            self.detector.update(pose(), ms)

    def test_four_alternating_arrivals_required(self):
        for ms, value, expected in ((600, .2, 1), (800, -.2, 2), (1000, .2, 3)):
            result = self.detector.update(pose(nose=value), ms)
            self.assertEqual(result.turn_count, expected)
            self.assertFalse(result.head_shake)
        result = self.detector.update(pose(nose=-.2), 1200)
        self.assertEqual(result.turn_count, 4)
        self.assertTrue(result.head_shake)

    def test_both_initial_directions_work(self):
        for i, value in enumerate((-.2, .2, -.2, .2)):
            result = self.detector.update(pose(nose=value), 600 + i * 200)
        self.assertTrue(result.head_shake)

    def test_same_direction_reentries_do_not_count(self):
        for i, value in enumerate((.2, 0, .2, 0, .2, 0, .2)):
            result = self.detector.update(pose(nose=value), 600 + i * 150)
        self.assertEqual(result.turn_count, 1)
        self.assertFalse(result.head_shake)

    def test_rigid_face_translation_is_not_head_turn(self):
        for ms in range(550, 2600, 50):
            item = pose()
            item.points[:11] += [30 * math.sin(ms / 100), 8 * math.cos(ms / 100)]
            result = self.detector.update(item, ms)
            self.assertFalse(result.head_shake)
        self.assertEqual(result.turn_count, 0)

    def test_turns_outside_window_expire(self):
        for i, value in enumerate((.2, -.2, .2, -.2)):
            self.detector.update(pose(nose=value), 600 + i * 200)
        for ms in range(1250, 3301, 50):
            result = self.detector.update(pose(nose=-.2), ms)
        self.assertFalse(result.head_shake)
        self.assertEqual(result.turn_count, 0)

    def test_low_confidence_clears_events_and_reference(self):
        self.detector.update(pose(nose=.2), 600)
        result = self.detector.update(pose(face_conf=.2), 700)
        self.assertEqual(result.head_state, "unknown")
        result = self.detector.update(pose(nose=-.2), 800)
        self.assertEqual(result.head_state, "calibrating")
        self.assertEqual(result.turn_count, 0)

    def test_duplicate_and_backwards_timestamp_rejected(self):
        for ms in (500, 499):
            with self.assertRaises(ValueError):
                self.detector.update(pose(), ms)

    def test_gap_clears_calibration_and_events(self):
        self.detector.update(pose(nose=.2), 600)
        result = self.detector.update(pose(nose=-.2), 1000)
        self.assertEqual(result.head_state, "calibrating")
        self.assertEqual(result.turn_count, 0)

    def test_face_pair_fixed_until_confidence_loss(self):
        result = self.detector.update(pose(nose=.2), 600)
        self.assertEqual(result.metrics["face_pair"], "ears")
        item = pose()
        item.visibility[[7, 8]] = .2
        result = self.detector.update(item, 700)
        self.assertEqual(result.head_state, "unknown")
        result = self.detector.update(item, 800)
        self.assertEqual(result.metrics["face_pair"], "eyes")
        self.assertEqual(result.head_state, "calibrating")

    def test_noise_and_unstable_neutral_cannot_shake(self):
        detector = MotionDetector()
        for i, ms in enumerate(range(0, 2000, 50)):
            result = detector.update(pose(nose=.2 if i % 6 < 3 else -.2), ms)
            self.assertFalse(result.head_shake)
        self.assertEqual(result.head_state, "calibrating")

    def test_reset_clears_all_history(self):
        for i, value in enumerate((.2, -.2, .2, -.2)):
            self.detector.update(pose(nose=value), 600 + i * 200)
        self.detector.reset()
        result = self.detector.update(pose(nose=-.2), 0)
        self.assertEqual(result.head_state, "calibrating")
        self.assertEqual(result.turn_count, 0)

    def test_new_sequence_after_neutral_idle_can_start_same_side(self):
        self.detector.update(pose(nose=.2), 600)
        for ms in range(650, 2801, 50):
            self.detector.update(pose(), ms)
        for i, value in enumerate((.2, -.2, .2, -.2)):
            result = self.detector.update(pose(nose=value), 2900 + i * 200)
        self.assertTrue(result.head_shake)


class WristMotionTests(unittest.TestCase):
    def sequence(self, *, hz=3.0, amplitude=.08, seconds=2.0, fps=30,
                 left=True, right=False, scale=100, translation=False,
                 right_conf=1.0, vertical=False, jitter=False):
        detector = MotionDetector()
        outputs = []
        rng = np.random.default_rng(42)
        ms = 0
        while ms <= seconds * 1000 + 1e-5:
            t = ms / 1000
            value = amplitude * math.sin(2 * math.pi * hz * t)
            item = pose(left=value if left else 0, right=-value if right else 0,
                        scale=scale, right_conf=right_conf,
                        translation=(40 * math.sin(t * 3), 20 * math.cos(t * 3)) if translation else (0, 0))
            if vertical:
                for index, enabled, sign in ((15, left, 1), (16, right, -1)):
                    if enabled:
                        item.points[index] += [-sign * value * scale, sign * value * scale]
            outputs.append(detector.update(item, ms, scale=scale))
            ms += int(round(1000 / fps + (rng.uniform(-4, 4) if jitter else 0)))
        return detector, outputs

    def test_single_hand_does_not_require_other_wrist(self):
        _, results = self.sequence(right_conf=.1)
        self.assertTrue(results[-1].left_wave)
        self.assertEqual(results[-1].right_state, "unknown")
        self.assertEqual(results[-1].wave_state, "left")

    def test_both_hands_can_wave_in_opposite_directions(self):
        _, results = self.sequence(right=True)
        self.assertEqual(results[-1].wave_state, "both")

    def test_vertical_waving_uses_dominant_axis(self):
        _, results = self.sequence(vertical=True)
        self.assertTrue(results[-1].left_wave)

    def test_slow_large_small_and_stationary_are_rejected(self):
        for args in ({"hz": .7}, {"amplitude": .25}, {"amplitude": .008}, {"amplitude": 0}):
            with self.subTest(args=args):
                _, results = self.sequence(**args)
                self.assertFalse(any(item.left_wave for item in results))

    def test_scale_and_body_translation_do_not_change_result(self):
        for scale in (60, 100, 300):
            with self.subTest(scale=scale):
                _, results = self.sequence(scale=scale, translation=True)
                self.assertTrue(results[-1].left_wave)
                self.assertAlmostEqual(results[-1].metrics["left_amplitude"], .14, delta=.02)

    def test_variable_video_sampling_is_supported(self):
        _, results = self.sequence(jitter=True)
        self.assertTrue(results[-1].left_wave)

    def test_source_sampling_too_slow_is_unknown(self):
        _, results = self.sequence(fps=7)
        self.assertEqual(results[-1].left_state, "unknown")
        self.assertEqual(results[-1].left_detail, "sampling_too_slow")

    def test_missing_wrist_clears_only_affected_hand(self):
        detector, results = self.sequence(right=True)
        self.assertEqual(results[-1].wave_state, "both")
        result = detector.update(pose(left_conf=.1, right=.08 * math.sin(2 * math.pi * 3 * 2.05)), 2050)
        self.assertEqual(result.left_state, "unknown")
        self.assertTrue(result.right_wave)
        result = detector.update(pose(), 2100)
        self.assertEqual(result.left_state, "calibrating")

    def test_pose_loss_invalidates_both_hands(self):
        detector, _ = self.sequence(right=True)
        result = detector.update(None, 2050)
        self.assertEqual(result.left_state, "unknown")
        self.assertEqual(result.right_state, "unknown")
        result = detector.update(pose(), 2100)
        self.assertEqual(result.left_state, "calibrating")

    def test_short_fast_burst_after_idle_not_sustained_wave(self):
        detector = MotionDetector()
        for ms in range(0, 2000, 33):
            value = 0 if ms < 1000 else .08 * math.sin(2 * math.pi * 4 * ms / 1000)
            result = detector.update(pose(left=value), ms)
            self.assertFalse(result.left_wave)

    def test_one_way_displacement_has_no_cycles(self):
        detector = MotionDetector()
        for ms in range(0, 2000, 33):
            result = detector.update(pose(left=.2 * ms / 2000), ms)
            self.assertFalse(result.left_wave)

    def test_noisy_stationary_wrist_does_not_wave(self):
        detector = MotionDetector()
        rng = np.random.default_rng(7)
        for ms in range(0, 2000, 33):
            result = detector.update(pose(left=rng.uniform(-.008, .008)), ms)
            self.assertFalse(result.left_wave)

    def test_low_confidence_shoulders_and_nan_are_unknown(self):
        detector, _ = self.sequence()
        result = detector.update(pose(shoulders_conf=.1), 2050)
        self.assertEqual(result.left_state, "unknown")
        item = pose()
        item.points[15] = [float("nan"), 400]
        result = detector.update(item, 2100)
        self.assertEqual(result.left_state, "unknown")

    def test_fixed_scale_does_not_follow_later_shoulder_width(self):
        detector = MotionDetector()
        detector.update(pose(scale=100), 0)
        detector.update(pose(scale=200), 50)
        self.assertEqual(detector.scale, 100)
        detector.update(pose(scale=200), 100, scale=150)
        self.assertEqual(detector.scale, 150)
        detector.update(pose(scale=200), 150, scale=180)
        self.assertEqual(detector.scale, 150)

    def test_box_rejects_implausible_person_coordinates(self):
        detector = MotionDetector()
        result = detector.update(pose(), 0, bbox=[0, 0, 100, 100])
        self.assertEqual(result.left_state, "unknown")
        self.assertEqual(result.head_state, "unknown")

    def test_validation_rejects_invalid_parameters(self):
        for args in ({"window_seconds": float("nan")}, {"confidence": 1.1},
                     {"wave_min_amplitude": .3}, {"wave_min_seconds": 3},
                     {"wave_min_cycles": 0}, {"shake_min_turns": 1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                MotionConfig(**args)


if __name__ == "__main__":
    unittest.main()
