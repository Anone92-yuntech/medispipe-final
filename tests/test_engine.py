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
        for offset, seat_state in ((0, 'normal'), (200, 'active')):
            with self.subTest(offset=offset):
                engine, _, ms = self.waving_engine()
                result = engine.update(analysis(ms, state='uncertain',
                                                posture_state='resting', offset=offset), self.shape)
                self.assertTrue(all(result.statuses[name] == 'unknown' for name in
                                    ('head_shake', 'left_wave', 'right_wave', 'both_wave', 'table_rest')),
                                result.statuses)
                # An observed OC-SORT bbox still supports the user's independent
                # departure rule even when MediaPipe's pose anchor is rejected.
                self.assertEqual(result.statuses['seat_leave'], seat_state)

    def test_no_target_is_unknown_even_when_input_carries_stale_pose(self):
        engine = BehaviorEngine()
        result = engine.update(analysis(0, target_id=None, posture_state='resting'), self.shape)
        self.assertTrue(all(s == 'unknown' for s in result.statuses.values()), result.statuses)

    def test_missing_pose_does_not_reuse_stale_positive_posture(self):
        engine = BehaviorEngine()
        frame = replace(analysis(0, posture_state='resting'), pose=None,
                        state='pose_uncertain')
        result = engine.update(frame, self.shape)
        self.assertTrue(all(result.statuses[name] == 'unknown' for name in
                            ('head_shake', 'left_wave', 'right_wave', 'both_wave', 'table_rest')),
                        result.statuses)
        # A reliable face/shoulder pose is now required while calibrating.
        self.assertEqual(result.statuses['seat_leave'], 'unknown')
        self.assertEqual(result.seat.detail, 'baseline_head_shoulders_unavailable')

    def test_box_departure_is_pose_independent_but_return_requires_pose(self):
        engine = BehaviorEngine()
        for ms in range(0, 2001, 100):
            result = engine.update(analysis(ms), self.shape)
        self.assertEqual(result.statuses['seat_leave'], 'normal')
        baseline = result.seat.seat_roi
        departed = engine.update(replace(analysis(2100, offset=200), pose=None,
                                         state='pose_uncertain'), self.shape)
        self.assertEqual(departed.statuses['seat_leave'], 'active')
        self.assertAlmostEqual(departed.seat.iou, 1 / 3)
        self.assertEqual(departed.seat.seat_roi, baseline)
        pending = engine.update(replace(analysis(2200), pose=None,
                                        state='pose_uncertain'), self.shape)
        self.assertEqual(pending.statuses['seat_leave'], 'active')
        self.assertEqual(pending.seat.detail, 'awaiting_return_pose')
        returned = engine.update(analysis(2300), self.shape)
        self.assertEqual(returned.statuses['seat_leave'], 'normal')

    def test_unavailable_or_rejected_pose_prevents_initial_seat_reference(self):
        for failure in ('missing', 'rejected'):
            with self.subTest(failure=failure):
                engine = BehaviorEngine()
                for ms in range(0, 3001, 100):
                    frame = analysis(ms)
                    if failure == 'missing':
                        frame = replace(frame, pose=None, state='pose_uncertain')
                    else:
                        frame = replace(frame, state='uncertain')
                    result = engine.update(frame, self.shape)
                    self.assertEqual(result.statuses['seat_leave'], 'unknown')
                    self.assertIsNone(result.seat.seat_roi)
                for ms in range(3100, 5100, 100):
                    result = engine.update(analysis(ms), self.shape)
                    self.assertEqual(result.statuses['seat_leave'], 'calibrating')
                result = engine.update(analysis(5100), self.shape)
                self.assertEqual(result.statuses['seat_leave'], 'normal')
                self.assertEqual(result.seat.seat_roi, (300, 150, 700, 700))

    def test_pose_recovery_before_calibration_does_not_reuse_invalid_samples(self):
        engine = BehaviorEngine()
        for ms in range(0, 1501, 100):
            engine.update(analysis(ms), self.shape)
        result = engine.update(replace(analysis(1600), pose=None,
                                        state='pose_uncertain'), self.shape)
        self.assertIsNone(result.seat.seat_roi)
        for ms in range(1700, 3700, 100):
            result = engine.update(analysis(ms, recovery=1), self.shape)
            self.assertEqual(result.statuses['seat_leave'], 'calibrating')
        result = engine.update(analysis(3700, recovery=1), self.shape)
        self.assertEqual(result.statuses['seat_leave'], 'normal')

    def test_full_reselection_cancels_manual_reference_and_uses_new_boxes(self):
        engine = BehaviorEngine(seat_roi=(300, 150, 700, 700))
        for ms in range(0, 2001, 100):
            before = engine.update(analysis(ms), self.shape)
        self.assertEqual(before.seat.seat_roi, (300, 150, 700, 700))
        reset_frame = replace(analysis(2100, target_id=None, generation=2),
                              pose=None, posture=None, state='reinitializing',
                              auto_reset_count=1, auto_reset_reason='auto_reset_pose_failed')
        reset = engine.update(reset_frame, self.shape)
        self.assertIsNone(engine.seat_roi)
        self.assertIsNone(reset.seat.seat_roi)
        self.assertTrue(all(value == 'unknown' for value in reset.statuses.values()))
        self.assertEqual(reset.metrics['auto_reset_reason'], 'auto_reset_pose_failed')
        for ms in range(2200, 4201, 100):
            after = engine.update(analysis(ms, target_id=8, generation=2, offset=300), self.shape)
        self.assertEqual(after.seat.seat_roi, (600, 150, 1000, 700))
        self.assertEqual(after.statuses['seat_leave'], 'normal')

    def test_pose_only_recovery_preserves_manual_reference_but_r_cancels_it(self):
        engine = BehaviorEngine(seat_roi=(300, 150, 700, 700))
        # The first target acquired after initial waiting keeps the supplied box.
        engine.update(replace(analysis(0, target_id=None), pose=None), self.shape)
        for ms in range(100, 2101, 100):
            before = engine.update(analysis(ms), self.shape)
        after = engine.update(analysis(2200, recovery=1), self.shape)
        self.assertEqual(after.seat.seat_roi, before.seat.seat_roi)
        self.assertIsNotNone(engine.seat_roi)
        engine.reset()
        self.assertIsNone(engine.seat_roi)
        self.assertFalse(engine.seat.baseline_ready)

    def test_seat_csv_keeps_departure_open_until_pose_confirmed_return(self):
        engine = BehaviorEngine()
        with tempfile.TemporaryDirectory() as directory:
            logger = TimeCsvLogger(Path(directory), Path(directory) / '來源.mp4')
            try:
                for ms in range(0, 2001, 100):
                    frame = analysis(ms)
                    result = engine.update(frame, self.shape)
                    logger.update(ms, frame.target_id, frame.selection_generation,
                                  result.statuses, result.metrics)
                for ms, offset, valid_pose in ((2100, 200, False), (2200, 0, False),
                                               (2300, 0, True)):
                    frame = analysis(ms, offset=offset)
                    if not valid_pose:
                        frame = replace(frame, pose=None, state='pose_uncertain')
                    result = engine.update(frame, self.shape)
                    logger.update(ms, frame.target_id, frame.selection_generation,
                                  result.statuses, result.metrics)
            finally:
                logger.close(2300)
            with logger.events_path.open(encoding='utf-8-sig', newline='') as file:
                events = [event for event in csv.DictReader(file) if event['behavior'] == 'seat_leave']
        self.assertEqual(len(events), 1)
        self.assertEqual(float(events[0]['start_seconds']), 2.1)
        self.assertEqual(float(events[0]['end_seconds']), 2.3)
        self.assertEqual(events[0]['end_observed'], '1')

    def test_full_reselection_closes_seat_csv_as_unknown_not_returned(self):
        engine = BehaviorEngine()
        with tempfile.TemporaryDirectory() as directory:
            logger = TimeCsvLogger(Path(directory), Path(directory) / '來源.mp4')
            try:
                for ms in range(0, 2001, 100):
                    result = engine.update(analysis(ms), self.shape)
                    logger.update(ms, 7, 1, result.statuses, result.metrics)
                result = engine.update(replace(analysis(2100, offset=200), pose=None), self.shape)
                logger.update(2100, 7, 1, result.statuses, result.metrics)
                frame = replace(analysis(2200, target_id=None, generation=2),
                                pose=None, posture=None, state='reinitializing')
                result = engine.update(frame, self.shape)
                logger.update(2200, None, 2, result.statuses, result.metrics)
            finally:
                logger.close(2200)
            with logger.events_path.open(encoding='utf-8-sig', newline='') as file:
                events = list(csv.DictReader(file))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['behavior'], 'seat_leave')
        self.assertEqual(events[0]['selection_generation'], '1')
        self.assertEqual(events[0]['end_observed'], '0')
        self.assertIsNone(result.seat.seat_roi)

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
