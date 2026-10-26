"""Integration contracts for independent behaviors, identity and CSV timing.

Model inference is not validated here.  Real OC-SORT is retained in pipeline
tests; synthetic original-pixel poses exercise motion and seat orchestration.
"""

import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from behavior_csv import TimeCsvLogger
from behavior_engine import BehaviorEngine
from motion_detection import MotionResult
from posture_detection import PostureResult
from seat_detection import SeatResult
from target_selection import MappedPose, TrackObservation
from tracking_pipeline import FrameAnalysis, PipelineConfig, TrackingPipeline
from test_pipeline import FakeDetector, FakePoseFactory


def mapped_pose(timestamp_ms=0, *, left_wave=False, right_wave=False,
                hidden_right=False, scale=100.0, offset=0.0):
    points = np.full((33, 2), [500.0, 400.0])
    points[0] = [500, 310]
    points[7], points[8] = [470, 300], [530, 300]
    points[2], points[5] = [485, 285], [515, 285]
    points[11], points[12] = [450, 400], [550, 400]
    value = 8 * math.sin(2 * math.pi * 3 * timestamp_ms / 1000)
    points[15] = [460 + (value if left_wave else 0), 500]
    points[16] = [540 - (value if right_wave else 0), 500]
    points = (points - [500, 400]) * scale / 100 + [500 + offset, 400]
    confidence = np.ones(33)
    if hidden_right:
        confidence[16] = .1
    return MappedPose(points, confidence.copy(), confidence.copy())


def analysis(timestamp_ms, *, target_id=7, generation=1, observed=True,
             state=None, posture_state='normal', recovery=0, scale=100,
             left_wave=False, right_wave=False, hidden_right=False, offset=0):
    pose = mapped_pose(timestamp_ms, left_wave=left_wave, right_wave=right_wave,
                       hidden_right=hidden_right, scale=scale, offset=offset)
    track = TrackObservation(target_id, np.array([300 + offset, 150, 700 + offset, 700]),
                             bool(observed), 100, 0, timestamp_ms)
    tracks = [] if observed is None or target_id is None else [track]
    posture = PostureResult(posture_state, 'fixture_posture', baseline_samples=20,
                            baseline_head_y=310, baseline_shoulder_width=scale)
    return FrameAnalysis(tracks, target_id, pose,
                         state or ('locked' if observed else 'lost'),
                         'fixture', {target_id: {'median_width': scale}}, timestamp_ms,
                         selection_generation=generation, posture=posture,
                         pose_recovery_count=recovery)


class BehaviorEngineTests(unittest.TestCase):
    shape = (720, 1280, 3)

    def waving_engine(self, **kwargs):
        engine = BehaviorEngine()
        result = None
        for ms in range(0, 2113, 33):
            result = engine.update(analysis(ms, left_wave=True, **kwargs), self.shape)
        self.assertEqual(result.statuses['left_wave'], 'active')
        return engine, result, 2145

    def test_single_visible_left_hand_active_while_both_hands_unknown(self):
        _, result, _ = self.waving_engine(hidden_right=True)
        self.assertEqual(result.statuses['right_wave'], 'unknown')
        self.assertEqual(result.statuses['both_wave'], 'unknown')
        self.assertEqual(result.motion.wave_state, 'left')

    def test_both_visible_waving_hands_active(self):
        _, result, _ = self.waving_engine(right_wave=True)
        self.assertEqual(result.statuses['both_wave'], 'active')

    def test_generation_change_resets_motion_and_seat_even_same_id(self):
        engine, _, ms = self.waving_engine()
        self.assertTrue(engine.seat.baseline_ready)
        result = engine.update(analysis(ms, generation=2, left_wave=True, scale=160), self.shape)
        self.assertEqual(result.statuses['left_wave'], 'calibrating')
        self.assertEqual(result.metrics['left_valid_seconds'], 0)
        self.assertFalse(engine.seat.baseline_ready)
        self.assertEqual(engine.motion.scale, 160)

    def test_target_id_change_resets_history_even_generation_unchanged(self):
        engine, _, ms = self.waving_engine()
        result = engine.update(analysis(ms, target_id=8, left_wave=True), self.shape)
        self.assertEqual(result.statuses['left_wave'], 'calibrating')
        self.assertEqual(result.metrics['left_valid_seconds'], 0)
        self.assertFalse(engine.seat.baseline_ready)

    def test_missing_and_predicted_only_targets_are_all_unknown(self):
        for observed in (False, None):
            with self.subTest(observed=observed):
                engine, _, ms = self.waving_engine()
                # Deliberately supply stale pose/posture to enforce the engine's
                # observed-evidence boundary instead of relying on its caller.
                result = engine.update(analysis(ms, observed=observed,
                                                posture_state='resting'), self.shape)
                self.assertTrue(all(s == 'unknown' for s in result.statuses.values()), result.statuses)
                self.assertEqual(result.seat.detail, 'target_not_observed')
                self.assertNotEqual(result.seat.state, 'away')
                self.assertTrue(engine.seat.baseline_ready)

    def test_uncertain_identity_is_not_valid_behavior_evidence(self):
        engine, _, ms = self.waving_engine()
        result = engine.update(analysis(ms, state='uncertain', posture_state='resting'), self.shape)
        self.assertTrue(all(s == 'unknown' for s in result.statuses.values()), result.statuses)

    def test_no_target_is_unknown_even_when_input_carries_stale_pose(self):
        engine = BehaviorEngine()
        result = engine.update(analysis(0, target_id=None, posture_state='resting'), self.shape)
        self.assertTrue(all(s == 'unknown' for s in result.statuses.values()), result.statuses)

    def test_missing_pose_does_not_reuse_stale_positive_posture(self):
        engine = BehaviorEngine()
        frame = replace(analysis(0, posture_state='resting'), pose=None,
                        state='pose_uncertain')
        result = engine.update(frame, self.shape)
        self.assertTrue(all(s == 'unknown' for s in result.statuses.values()), result.statuses)

    def test_pose_recovery_clears_motion_but_preserves_seat_and_posture_reference(self):
        engine, before, ms = self.waving_engine()
        anchor = before.seat.baseline_anchor
        self.assertIsNotNone(anchor)
        result = engine.update(analysis(ms, left_wave=True, recovery=1), self.shape)
        self.assertEqual(result.statuses['left_wave'], 'calibrating')
        self.assertEqual(result.metrics['left_valid_seconds'], 0)
        self.assertEqual(result.statuses['head_shake'], 'calibrating')
        self.assertTrue(engine.seat.baseline_ready)
        self.assertEqual(result.seat.baseline_anchor, anchor)
        self.assertEqual(result.metrics['posture']['baseline_head_y'], 310)
        self.assertEqual(result.metrics['posture']['baseline_shoulder_width'], 100)

    def test_manual_roi_change_resets_only_seat(self):
        engine, _, ms = self.waving_engine()
        motion = engine.motion
        engine.set_seat_roi((400, 200, 600, 600))
        self.assertIs(engine.motion, motion)
        self.assertFalse(engine.seat.baseline_ready)
        result = engine.update(analysis(ms, left_wave=True), self.shape)
        self.assertEqual(result.statuses['left_wave'], 'active')
        self.assertEqual(result.statuses['seat_leave'], 'calibrating')

    def test_explicit_reset_can_start_video_time_at_zero_again(self):
        engine, _, _ = self.waving_engine()
        engine.reset()
        result = engine.update(analysis(0), self.shape)
        self.assertEqual(result.statuses['head_shake'], 'calibrating')
        self.assertEqual(result.statuses['left_wave'], 'calibrating')
        self.assertEqual(result.statuses['seat_leave'], 'calibrating')

    def test_concurrent_behaviors_do_not_suppress_each_other(self):
        engine = BehaviorEngine()
        motion = MotionResult('active', 'active', 'normal', 'head_turns',
                              'left_repetition', 'right_still', 4, {'test': 1})
        with patch.object(engine.motion, 'update', return_value=motion), \
                patch.object(engine.seat, 'update', return_value=SeatResult('away', 'outside_seat')):
            result = engine.update(analysis(0, posture_state='resting'), self.shape)
        self.assertEqual(result.statuses, {
            'head_shake': 'active', 'left_wave': 'active', 'right_wave': 'normal',
            'both_wave': 'normal', 'table_rest': 'active', 'seat_leave': 'active'})
        self.assertEqual(result.metrics['head_turn_count'], 4)
        self.assertEqual(result.details['seat_leave'], 'outside_seat')

    def test_events_preserve_short_activity_between_one_second_point_samples(self):
        engine = BehaviorEngine()
        normal = MotionResult('normal', 'normal', 'normal')
        active = MotionResult('active', 'active', 'normal', turn_count=4)
        with tempfile.TemporaryDirectory() as directory:
            logger = TimeCsvLogger(Path(directory), Path(directory) / '來源.mp4')
            try:
                for ms, motion in ((0, normal), (100, active), (300, active), (400, normal), (1000, normal)):
                    with patch.object(engine.motion, 'update', return_value=motion):
                        frame = analysis(ms)
                        result = engine.update(frame, self.shape)
                    logger.update(ms, frame.target_id, frame.selection_generation,
                                  result.statuses, result.metrics)
            finally:
                logger.close()
            with logger.timeline_path.open(encoding='utf-8-sig', newline='') as file:
                points = list(csv.DictReader(file))
            with logger.events_path.open(encoding='utf-8-sig', newline='') as file:
                events = list(csv.DictReader(file))
        self.assertEqual([float(row['sample_time_seconds']) for row in points], [0, 1])
        self.assertTrue(all(row['head_shake'] == 'normal' for row in points))
        self.assertEqual({row['behavior'] for row in events}, {'head_shake', 'left_wave'})
        self.assertTrue(all(float(row['start_seconds']) == .1 for row in events))
        self.assertTrue(all(float(row['end_seconds']) == .4 for row in events))

    def test_csv_boundary_closes_old_identity_without_mixing_new_generation(self):
        engine = BehaviorEngine()
        active = MotionResult('active', 'normal', 'normal', turn_count=4)
        with tempfile.TemporaryDirectory() as directory:
            logger = TimeCsvLogger(Path(directory), Path(directory) / '來源.mp4')
            try:
                with patch.object(engine.motion, 'update', return_value=active):
                    result = engine.update(analysis(0), self.shape)
                logger.update(0, 7, 1, result.statuses, result.metrics)
                logger.boundary(100, 'manual_reset')
                engine.reset()
                result = engine.update(analysis(100, generation=2), self.shape)
                logger.update(100, 7, 2, result.statuses, result.metrics)
            finally:
                logger.close()
            with logger.events_path.open(encoding='utf-8-sig', newline='') as file:
                events = list(csv.DictReader(file))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['selection_generation'], '1')
        self.assertEqual(events[0]['end_reason'], 'manual_reset')
        self.assertEqual(events[0]['end_observed'], '0')


class RetainedTargetPipelineTests(unittest.TestCase):
    def make_pipeline(self, retain):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        model = Path(directory.name) / 'fixture.task'
        model.write_bytes(b'Fake pose factory replaces construction')
        factory = FakePoseFactory()
        pose_patch = patch.object(TrackingPipeline, '_create_pose', lambda _pipeline, mode: factory.create(mode))
        pose_patch.start()
        self.addCleanup(pose_patch.stop)
        detector = FakeDetector()
        pipeline = TrackingPipeline(detector, model, fps=10,
            config=PipelineConfig(init_seconds=.2, min_samples=2, sample_stride=1,
                                  stable_observations=2, retain_target_on_loss=retain))
        self.addCleanup(pipeline.close)
        frame = np.zeros((400, 800, 3), np.uint8)
        for index in range(30):
            result = pipeline.process(frame, index * 100, index)
            if result.target_id is not None:
                return pipeline, detector, frame, index, result
        self.fail('fake integration did not lock a target')

    def test_retained_missing_id_survives_tracker_expiry_without_auto_reselection(self):
        pipeline, detector, frame, index, locked = self.make_pipeline(True)
        detector.detections = np.empty((0, 5), dtype=float)
        for offset in range(1, 23):
            result = pipeline.process(frame, (index + offset) * 100, index + offset)
            self.assertEqual(result.target_id, locked.target_id)
            self.assertEqual(result.state, 'lost')
            self.assertEqual(result.selection_generation, locked.selection_generation)
            self.assertEqual(result.auto_reset_count, 0)
            self.assertIsNone(result.pose)
        self.assertFalse(any(t.track_id == locked.target_id for t in result.tracks))

    def test_legacy_false_still_reinitializes_after_loss_threshold(self):
        pipeline, detector, frame, index, locked = self.make_pipeline(False)
        detector.detections = np.empty((0, 5), dtype=float)
        for offset in range(1, 8):
            result = pipeline.process(frame, (index + offset) * 100, index + offset)
        self.assertIsNone(result.target_id)
        self.assertEqual(result.auto_reset_count, 1)
        self.assertGreater(result.selection_generation, locked.selection_generation)


if __name__ == '__main__':
    unittest.main()
