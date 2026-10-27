"""Real OC-SORT + selector integration; only detector/pose inference are fakes.

These tests check orchestration and identity safety, not image-model accuracy.
No weights are downloaded and no processed frames or videos are saved.
"""
from pathlib import Path
from dataclasses import replace
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mediapipe.tasks.python import vision  # noqa: E402
from tracking_pipeline import PipelineConfig, TrackingPipeline  # noqa: E402
from vendor.ocsort.trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402


CHILD = [50, 60, 170, 180, 0.99]
ADULT = [300, 40, 540, 340, 0.99]


class FakeDetector:
    def __init__(self):
        self.detections = np.array([CHILD, ADULT], dtype=float)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self.detections.copy()


class FakeLandmarker:
    def __init__(self, mode, factory):
        self.mode = mode
        self.factory = factory
        self.calls = []
        self.closed = False
        self.last_timestamp = -1

    def _result(self, image, timestamp):
        if self.closed:
            raise AssertionError("closed pose instance was reused")
        height, width = image.numpy_view().shape[:2]
        self.calls.append((width, height, timestamp))
        if self.factory.video_error and self.mode == vision.RunningMode.VIDEO:
            raise RuntimeError("simulated VIDEO inference failure")
        if self.factory.no_pose or (
            self.factory.video_no_pose and self.mode == vision.RunningMode.VIDEO
        ):
            return SimpleNamespace(pose_landmarks=[])
        x = self.factory.center_x
        half_width = self.factory.shoulder_fraction / 2
        points = [SimpleNamespace(x=x, y=0.2, visibility=0.99, presence=0.99)
                  for _ in range(33)]
        points[11] = SimpleNamespace(x=x-half_width, y=0.5,
                                     visibility=0.99, presence=0.99)
        points[12] = SimpleNamespace(x=x+half_width, y=0.5,
                                     visibility=0.99, presence=0.99)
        # Normal upper arms/forearms stay well away from the upright head.
        for index, arm_x, arm_y in ((13, 0.25, 0.75), (15, 0.45, 0.8),
                                    (14, 0.75, 0.75), (16, 0.55, 0.8)):
            points[index].x = arm_x
            points[index].y = arm_y
        if (self.factory.invalid_shoulders_above_width is not None
                and width > self.factory.invalid_shoulders_above_width):
            # Simulate a cropped adult with only one reliably visible shoulder.
            points[12].presence = 0.1
        if self.factory.invalid_video_shoulders and self.mode == vision.RunningMode.VIDEO:
            points[12].visibility = 0.1
        if self.mode == vision.RunningMode.IMAGE and self.factory.image_head_y is not None:
            for point in points[:11]:
                point.y = self.factory.image_head_y
        if self.mode == vision.RunningMode.VIDEO:
            if self.factory.video_head_y is not None:
                for point in points[:11]:
                    point.y = self.factory.video_head_y
            if self.factory.video_head_hidden:
                for point in points[:11]:
                    point.visibility = 0.1
            if self.factory.video_forearm_y is not None:
                for index, arm_x in ((13, 0.3), (15, 0.6), (14, 0.7), (16, 0.4)):
                    points[index].x = arm_x
                    points[index].y = self.factory.video_forearm_y
        return SimpleNamespace(pose_landmarks=[points])

    def detect(self, image):
        if self.mode != vision.RunningMode.IMAGE:
            raise AssertionError("initialization used a stateful VIDEO instance")
        return self._result(image, None)

    def detect_for_video(self, image, timestamp_ms):
        if self.mode != vision.RunningMode.VIDEO:
            raise AssertionError("locked target used the IMAGE instance")
        if timestamp_ms <= self.last_timestamp:
            raise AssertionError("VIDEO timestamps are not strictly increasing")
        self.last_timestamp = timestamp_ms
        return self._result(image, timestamp_ms)

    def close(self):
        self.closed = True


class FakePoseFactory:
    def __init__(self):
        self.instances = []
        self.center_x = 0.5
        self.shoulder_fraction = 0.4
        self.invalid_shoulders_above_width = None
        self.no_pose = False
        self.video_no_pose = False
        self.invalid_video_shoulders = False
        self.video_error = False
        self.video_head_y = None
        self.image_head_y = None
        self.video_head_hidden = False
        self.video_forearm_y = None

    def create(self, mode):
        instance = FakeLandmarker(mode, self)
        self.instances.append(instance)
        return instance

    def image_instances(self):
        return [instance for instance in self.instances
                if instance.mode == vision.RunningMode.IMAGE]

    def video_instances(self):
        return [instance for instance in self.instances
                if instance.mode == vision.RunningMode.VIDEO]

    def video_call_count(self):
        return sum(len(instance.calls) for instance in self.video_instances())


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        model = Path(self.directory.name) / "fixture.task"
        model.write_bytes(b"not used: pose construction is replaced by a fake")
        self.factory = FakePoseFactory()
        factory = self.factory
        self.pose_patch = patch.object(
            TrackingPipeline, "_create_pose", lambda _pipeline, mode: factory.create(mode)
        )
        self.pose_patch.start()
        self.detector = FakeDetector()
        self.pipeline = TrackingPipeline(
            self.detector, model, fps=10,
            config=PipelineConfig(init_seconds=2, min_samples=10,
                                  sample_stride=1, stable_observations=3,
                                  auto_reset_seconds=0.5),
        )
        self.frame = np.zeros((400, 800, 3), dtype=np.uint8)
        self.index = 0
        self.timestamp = 0

    def tearDown(self):
        try:
            self.pipeline.close()
        finally:
            self.pose_patch.stop()
            self.directory.cleanup()

    def step(self):
        result = self.pipeline.process(self.frame, self.timestamp, self.index)
        self.index += 1
        self.timestamp += 100
        return result

    def lock(self):
        for _ in range(40):
            result = self.step()
            if result.target_id is not None:
                self.assertEqual(result.state, "locked", result.detail)
                self.assertIsNotNone(result.pose, result.detail)
                return result
        self.fail(f"failed to lock: {result.detail}, statistics={result.statistics}")

    def calibrate_posture(self):
        self.lock()
        for _ in range(25):
            result = self.step()
            self.assertEqual(result.state, "locked")
        self.assertIsNotNone(result.posture)
        self.assertEqual(result.posture.state, "normal")
        self.assertGreaterEqual(result.posture.baseline_samples, 10)
        self.assertIsNotNone(result.posture.baseline_head_y)
        return result

    def test_auto_reset_timeout_requires_finite_nonnegative_duration(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PipelineConfig(auto_reset_seconds=value)
        self.assertEqual(PipelineConfig(auto_reset_seconds=0).auto_reset_seconds, 0)

    def test_pose_recovery_timeout_requires_finite_nonnegative_duration(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PipelineConfig(pose_recovery_seconds=value)
        self.assertEqual(PipelineConfig(pose_recovery_seconds=0).pose_recovery_seconds, 0)

    def test_pose_reselection_timeout_is_independent_and_validated(self):
        self.assertEqual(PipelineConfig().pose_reselect_seconds, 10.0)
        self.assertEqual(PipelineConfig().auto_reset_seconds, 0)
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PipelineConfig(pose_reselect_seconds=value)
        self.assertEqual(PipelineConfig(pose_reselect_seconds=0).pose_reselect_seconds, 0)

    def test_black_prefix_advances_real_tracker_without_pose_or_false_lock(self):
        self.assertIsInstance(self.pipeline.tracker.tracker, OCSort)
        self.detector.detections = np.empty((0, 5), dtype=float)
        for _ in range(15):
            result = self.step()
            self.assertEqual(result.state, "waiting")
            self.assertIsNone(result.target_id)
            self.assertIsNone(result.pose)
        self.assertEqual(self.pipeline.tracker.frame_count, 15)
        self.assertEqual(self.detector.calls, 15)
        self.assertTrue(all(not instance.calls for instance in self.factory.instances))
        self.detector.detections = np.array([CHILD, ADULT], dtype=float)
        result = self.lock()
        self.assertGreaterEqual(result.timestamp_ms, 1500 + 2200)

    def test_two_second_lock_uses_image_for_all_people_then_video_only_target(self):
        result = self.lock()
        child_track = next(track for track in result.tracks
                           if abs((track.bbox[2] - track.bbox[0]) - 120) < 1)
        self.assertEqual(result.target_id, child_track.track_id)
        self.assertGreaterEqual(result.timestamp_ms, 2200)
        self.assertGreaterEqual(result.statistics[result.target_id]["valid_seconds"], 2)
        image_instances = self.factory.image_instances()
        self.assertEqual(len(image_instances), 1)
        image_calls = len(image_instances[0].calls)
        self.assertGreaterEqual(image_calls, 40)
        image_widths = {call[0] for call in image_instances[0].calls}
        self.assertEqual(len(image_widths), 2)  # separate child/adult crops
        for _ in range(5):
            result = self.step()
            self.assertEqual(result.target_id, child_track.track_id)
            self.assertEqual(result.state, "locked")
        self.assertEqual(len(image_instances[0].calls), image_calls)
        self.assertEqual(len(self.factory.video_instances()), 1)
        video = self.factory.video_instances()[0]
        self.assertEqual(len({call[0] for call in video.calls}), 1)
        self.assertEqual(video.calls[0][0], min(image_widths))
        times = [call[2] for call in video.calls]
        self.assertEqual(times, sorted(set(times)))

    def test_only_one_visible_person_locks_after_evidence_not_first_frame(self):
        self.detector.detections = np.array([CHILD], dtype=float)
        first = self.step()
        self.assertIsNone(first.target_id)
        self.assertIsNone(first.pose)
        result = self.lock()
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.target_id, result.tracks[0].track_id)
        self.assertEqual(result.detail, "only_qualified_candidate")
        self.assertGreaterEqual(result.timestamp_ms, 2200)
        self.assertGreaterEqual(result.statistics[result.target_id]["valid_seconds"], 2)
        self.assertEqual(len(self.factory.video_instances()), 1)

    def test_partial_adult_shoulders_do_not_block_valid_person_video_tracking(self):
        self.factory.invalid_shoulders_above_width = 200
        result = self.lock()
        self.assertEqual(len(result.tracks), 2)
        child = next(track for track in result.tracks
                     if abs((track.bbox[2] - track.bbox[0]) - 120) < 1)
        adult = next(track for track in result.tracks
                     if track.track_id != child.track_id)
        self.assertEqual(result.target_id, child.track_id)
        self.assertEqual(result.detail, "only_qualified_candidate")
        self.assertEqual(result.statistics[adult.track_id]["sample_count"], 0)
        self.assertIsNone(result.statistics[adult.track_id]["median_width"])
        self.assertEqual(result.statistics[adult.track_id]["rejection_reason"],
                         "low_shoulder_confidence")
        image_calls = len(self.factory.image_instances()[0].calls)
        self.assertEqual(len(self.factory.video_instances()), 1)
        for _ in range(5):
            result = self.step()
            self.assertEqual(result.target_id, child.track_id)
            self.assertEqual(result.state, "locked")
            self.assertIsNotNone(result.pose)
        self.assertEqual(len(self.factory.image_instances()[0].calls), image_calls)
        self.assertTrue(all(call[0] < 200
                            for call in self.factory.video_instances()[0].calls))

    def test_missing_target_resets_then_new_id_requires_fresh_evidence(self):
        result = self.lock()
        locked_id = result.target_id
        video_calls = self.factory.video_call_count()
        image_calls = len(self.factory.image_instances()[0].calls)
        initial_generation = self.pipeline.selection_generation
        self.detector.detections = np.empty((0, 5), dtype=float)
        result = self.step()
        first_loss_ms = result.timestamp_ms
        target = next(track for track in result.tracks if track.track_id == locked_id)
        self.assertFalse(target.observed)
        self.assertIsNone(result.pose)
        self.assertEqual(result.target_id, locked_id)
        self.assertEqual(result.state, "lost")
        self.assertTrue(self.factory.video_instances()[0].closed)
        reset = None
        # Allow the reset timeout and removal of the stale OC-SORT tracklets.
        for _ in range(15):
            result = self.step()
            if result.state == "reinitializing":
                reset = result
        self.assertIsNotNone(reset)
        self.assertEqual(reset.timestamp_ms - first_loss_ms, 500)
        self.assertEqual(reset.detail, "auto_reset_tracking_lost")
        self.assertIsNone(reset.target_id)
        self.assertIsNone(reset.pose)
        self.assertEqual(reset.statistics, {})
        self.assertEqual(result.state, "waiting")
        self.assertEqual(self.pipeline.auto_reset_count, 1)
        self.assertEqual(self.pipeline.selection_generation, initial_generation + 1)
        self.assertEqual(self.pipeline.last_auto_reset_reason, "auto_reset_tracking_lost")
        self.assertFalse(any(track.track_id == locked_id for track in result.tracks))
        self.assertEqual(self.factory.video_call_count(), video_calls)
        self.assertEqual(len(self.factory.image_instances()[0].calls), image_calls)
        self.detector.detections = np.array([CHILD, ADULT], dtype=float)
        restart_ms = self.timestamp
        for _ in range(8):
            result = self.step()
            self.assertIsNone(result.target_id)
            self.assertIsNone(result.pose)
        result = self.lock()
        new_child = next(track for track in result.tracks
                         if abs((track.bbox[2]-track.bbox[0])-120) < 1)
        self.assertNotEqual(new_child.track_id, locked_id)
        self.assertEqual(result.target_id, new_child.track_id)
        self.assertGreaterEqual(result.timestamp_ms - restart_ms, 2000)
        self.assertGreaterEqual(result.statistics[result.target_id]["valid_seconds"], 2)
        self.assertEqual(self.detector.calls, self.pipeline.tracker.frame_count)

    def test_auto_reset_threshold_uses_source_timestamp_not_processing_duration(self):
        self.lock()
        self.detector.detections = np.empty((0, 5), dtype=float)
        first_loss = self.step()
        self.timestamp = first_loss.timestamp_ms + 499
        result = self.step()
        self.assertEqual(result.state, "lost")
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.timestamp = first_loss.timestamp_ms + 500
        result = self.step()
        self.assertEqual(result.state, "reinitializing")
        self.assertEqual(self.pipeline.auto_reset_count, 1)
        self.assertEqual(result.statistics, {})
        self.assertEqual(self.detector.calls, self.pipeline.tracker.frame_count)

    def test_brief_tracking_loss_recovers_same_id_without_reset(self):
        first = self.lock()
        old_video = self.factory.video_instances()[0]
        self.detector.detections = np.array([ADULT], dtype=float)
        for _ in range(4):
            result = self.step()
            self.assertEqual(result.state, "lost")
            self.assertEqual(result.target_id, first.target_id)
        self.detector.detections = np.array([CHILD, ADULT], dtype=float)
        recovered = self.step()
        self.assertEqual(recovered.state, "locked")
        self.assertEqual(recovered.target_id, first.target_id)
        self.assertIsNotNone(recovered.pose)
        self.assertTrue(old_video.closed)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        # A later short outage starts a new interval, rather than reusing the old one.
        self.detector.detections = np.array([ADULT], dtype=float)
        for _ in range(4):
            result = self.step()
            self.assertNotEqual(result.state, "reinitializing")
        self.assertEqual(self.pipeline.auto_reset_count, 0)

    def test_observed_pose_failure_refreshes_video_only_and_keeps_target_evidence(self):
        first = self.lock()
        initial_generation = self.pipeline.selection_generation
        original_selector = self.pipeline.selector
        original_video = self.factory.video_instances()[0]
        image_calls = len(self.factory.image_instances()[0].calls)
        self.factory.video_no_pose = True
        first_failure = self.step()
        self.assertEqual(first_failure.state, "pose_uncertain")
        self.assertIsNone(first_failure.pose)
        self.assertEqual(first_failure.target_id, first.target_id)
        self.assertFalse(original_video.closed)
        for _ in range(19):
            result = self.step()
            self.assertEqual(result.state, "pose_uncertain")
            self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(result.timestamp_ms - first_failure.timestamp_ms, 1900)
        self.assertEqual(self.pipeline.pose_recovery_count, 0)
        result = self.step()
        self.assertEqual(result.timestamp_ms - first_failure.timestamp_ms, 2000)
        self.assertEqual(result.state, "pose_uncertain")
        self.assertEqual(result.target_id, first.target_id)
        self.assertIsNone(result.pose)
        self.assertTrue(original_video.closed)
        self.assertIs(self.pipeline.selector, original_selector)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.selection_generation, initial_generation)
        self.assertEqual(self.pipeline.pose_recovery_count, 1)
        self.assertEqual(result.pose_recovery_count, 1)
        self.assertIn(first.target_id, result.statistics)
        self.assertGreaterEqual(result.statistics[first.target_id]["sample_count"], 10)
        self.assertEqual(len(self.factory.image_instances()[0].calls), image_calls)
        for _ in range(19):
            result = self.step()
            self.assertEqual(self.pipeline.pose_recovery_count, 1)
            self.assertEqual(result.target_id, first.target_id)
        result = self.step()
        self.assertEqual(self.pipeline.pose_recovery_count, 2)
        self.assertEqual(result.target_id, first.target_id)
        self.factory.video_no_pose = False
        result = self.step()
        self.assertEqual(result.state, "locked")
        self.assertIsNotNone(result.pose)
        self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(result.pose_failure_seconds, 0)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(len(self.factory.image_instances()[0].calls), image_calls)

    def test_intermittent_pose_failure_does_not_accumulate_across_healthy_frames(self):
        first = self.lock()
        for _ in range(3):
            self.factory.video_no_pose = True
            for _ in range(15):
                result = self.step()
                self.assertIsNone(result.pose)
                self.assertEqual(result.target_id, first.target_id)
                self.assertNotEqual(result.state, "reinitializing")
            self.factory.video_no_pose = False
            result = self.step()
            self.assertEqual(result.state, "locked")
            self.assertIsNotNone(result.pose)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.pose_recovery_count, 0)

    def test_two_second_pose_retries_do_not_extend_ten_second_reselection_deadline(self):
        first = self.calibrate_posture()
        generation = self.pipeline.selection_generation
        tracker = self.pipeline.tracker
        old_selector = self.pipeline.selector
        old_posture = self.pipeline.posture_detector
        image_calls = sum(len(instance.calls) for instance in self.factory.image_instances())
        self.factory.video_no_pose = True
        first_failure = self.step()
        # Retries at 2, 4, 6 and 8 seconds must retain the original deadline.
        for _ in range(99):
            result = self.step()
            self.assertEqual(result.target_id, first.target_id)
            self.assertEqual(result.state, "pose_uncertain")
            self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.pose_recovery_count, 4)
        self.timestamp = first_failure.timestamp_ms + 9999
        result = self.step()
        self.assertEqual(result.target_id, first.target_id)
        self.assertAlmostEqual(result.pose_failure_seconds, 9.999)
        self.assertEqual(self.pipeline.selection_generation, generation)
        self.timestamp = first_failure.timestamp_ms + 10000
        reset = self.step()
        self.assertEqual(reset.state, "reinitializing")
        self.assertEqual(reset.detail, "auto_reset_pose_failed")
        self.assertEqual(reset.auto_reset_reason, "auto_reset_pose_failed")
        self.assertEqual(reset.auto_reset_count, 1)
        self.assertIsNone(reset.target_id)
        self.assertIsNone(reset.pose)
        self.assertIsNone(reset.posture)
        self.assertEqual(reset.statistics, {})
        self.assertEqual(self.pipeline.selection_generation, generation + 1)
        self.assertIs(self.pipeline.tracker, tracker)
        self.assertIsNot(self.pipeline.selector, old_selector)
        self.assertIsNot(self.pipeline.posture_detector, old_posture)
        self.assertFalse(self.pipeline.posture_detector.baseline_ready)
        self.assertEqual(self.pipeline.pose_recovery_count, 4)  # Full reset takes priority.
        self.assertTrue(all(instance.closed for instance in self.factory.video_instances()))
        self.assertEqual(sum(len(instance.calls) for instance in self.factory.image_instances()),
                         image_calls)  # New IMAGE evidence starts on the following frame.
        self.assertEqual(self.detector.calls, tracker.frame_count)
        self.assertTrue(next(track for track in reset.tracks
                             if track.track_id == first.target_id).observed)

        self.factory.video_no_pose = False
        first_fresh = self.step()
        self.assertIsNone(first_fresh.target_id)
        self.assertEqual(sum(len(instance.calls) for instance in self.factory.image_instances()),
                         image_calls + 2)  # Re-evaluate both currently observed IDs.
        self.assertEqual(len(first_fresh.statistics), 2)
        self.assertTrue(all(data["sample_count"] == 1
                            for data in first_fresh.statistics.values()))
        for _ in range(19):
            result = self.step()
            self.assertIsNone(result.target_id)
        second = self.lock()
        self.assertGreaterEqual(second.timestamp_ms - first_fresh.timestamp_ms, 2000)
        self.assertEqual(second.target_id, first.target_id)  # Same ID, new evidence generation.
        self.assertEqual(second.selection_generation, generation + 1)
        self.assertEqual(second.pose_failure_seconds, 0)
        self.assertEqual(self.pipeline.auto_reset_count, 1)
        self.assertIsNone(second.posture.baseline_head_y)
        self.assertEqual(self.detector.calls, tracker.frame_count)

    def test_healthy_pose_restarts_full_reselection_clock(self):
        first = self.lock()
        self.factory.video_no_pose = True
        first_failure = self.step()
        for _ in range(99):
            result = self.step()
        self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(result.pose_failure_seconds, 9.9)
        self.factory.video_no_pose = False
        healthy = self.step()
        self.assertEqual(healthy.state, "locked")
        self.assertEqual(healthy.pose_failure_seconds, 0)
        self.factory.video_no_pose = True
        second_failure = self.step()
        self.assertEqual(second_failure.pose_failure_seconds, 0)
        for _ in range(99):
            self.step()
        self.timestamp = second_failure.timestamp_ms + 9999
        result = self.step()
        self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.timestamp = second_failure.timestamp_ms + 10000
        result = self.step()
        self.assertEqual(result.state, "reinitializing")
        self.assertEqual(result.detail, "auto_reset_pose_failed")

    def test_pose_timeout_still_reselects_when_tracking_loss_retention_is_enabled(self):
        self.pipeline.config = replace(self.pipeline.config, retain_target_on_loss=True,
                                       pose_reselect_seconds=0.3)
        self.lock()
        self.factory.video_no_pose = True
        first_failure = self.step()
        for _ in range(2):
            result = self.step()
            self.assertIsNotNone(result.target_id)
        result = self.step()
        self.assertEqual(result.timestamp_ms - first_failure.timestamp_ms, 300)
        self.assertEqual(result.state, "reinitializing")
        self.assertEqual(result.detail, "auto_reset_pose_failed")
        self.assertIsNone(result.target_id)
        self.assertEqual(self.pipeline.pose_recovery_count, 0)

    def test_retained_tracking_loss_clears_pose_timeout_before_observed_failure_restarts(self):
        self.pipeline.config = replace(self.pipeline.config, retain_target_on_loss=True)
        first = self.lock()
        self.factory.video_no_pose = True
        self.step()
        for _ in range(99):
            self.step()
        self.detector.detections = np.array([ADULT], dtype=float)
        lost = self.step()
        self.assertEqual(lost.state, "lost")
        self.assertEqual(lost.target_id, first.target_id)
        self.assertEqual(lost.pose_failure_seconds, 0)
        self.detector.detections = np.array([CHILD, ADULT], dtype=float)
        resumed = self.step()
        self.assertEqual(resumed.state, "pose_uncertain")
        self.assertEqual(resumed.target_id, first.target_id)
        self.assertEqual(resumed.pose_failure_seconds, 0)
        for _ in range(99):
            self.step()
        self.timestamp = resumed.timestamp_ms + 9999
        result = self.step()
        self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(self.pipeline.auto_reset_count, 0)

    def test_no_people_after_pose_timeout_waits_without_repeated_resets(self):
        self.pipeline.config = replace(self.pipeline.config, pose_reselect_seconds=0.2)
        self.lock()
        self.factory.video_no_pose = True
        self.step()
        self.step()
        reset = self.step()
        self.assertEqual(reset.state, "reinitializing")
        generation = reset.selection_generation
        self.detector.detections = np.empty((0, 5), dtype=float)
        self.factory.no_pose = True
        for _ in range(120):
            result = self.step()
            self.assertEqual(result.state, "waiting")
            self.assertIsNone(result.target_id)
            self.assertIsNone(result.pose)
        self.assertEqual(self.pipeline.auto_reset_count, 1)
        self.assertEqual(self.pipeline.selection_generation, generation)
        self.assertEqual(self.detector.calls, self.pipeline.tracker.frame_count)

    def test_zero_pose_reselection_timeout_resets_before_video_retry(self):
        self.pipeline.config = replace(self.pipeline.config, pose_reselect_seconds=0,
                                       pose_recovery_seconds=0)
        self.lock()
        frames_before = self.pipeline.tracker.frame_count
        image_calls = sum(len(instance.calls) for instance in self.factory.image_instances())
        self.factory.video_no_pose = True
        result = self.step()
        self.assertEqual(result.state, "reinitializing")
        self.assertEqual(result.detail, "auto_reset_pose_failed")
        self.assertIsNone(result.target_id)
        self.assertEqual(result.statistics, {})
        self.assertEqual(self.pipeline.auto_reset_count, 1)
        self.assertEqual(self.pipeline.pose_recovery_count, 0)
        self.assertEqual(self.pipeline.tracker.frame_count, frames_before + 1)
        self.assertEqual(sum(len(instance.calls) for instance in self.factory.image_instances()),
                         image_calls)

    def test_invalid_video_shoulders_refresh_pose_without_reselecting(self):
        first = self.lock()
        self.factory.invalid_video_shoulders = True
        for _ in range(21):
            result = self.step()
        self.assertEqual(result.state, "pose_uncertain")
        self.assertIsNone(result.pose)
        self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(self.pipeline.pose_recovery_count, 1)
        self.assertEqual(self.pipeline.auto_reset_count, 0)

    def test_video_inference_exception_discards_handle_but_keeps_observed_target(self):
        first = self.lock()
        old_video = self.factory.video_instances()[0]
        self.factory.video_error = True
        for _ in range(21):
            result = self.step()
            self.assertIsNone(result.pose)
            self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(result.state, "pose_uncertain")
        self.assertTrue(old_video.closed)
        self.assertEqual(self.pipeline.pose_recovery_count, 1)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.factory.video_error = False
        result = self.lock()
        self.assertEqual(result.target_id, first.target_id)
        self.assertIsNotNone(result.pose)

    def test_invalid_target_crop_keeps_observed_target_and_refreshes_pose(self):
        first = self.lock()
        old_video = self.factory.video_instances()[0]
        with patch("tracking_pipeline.crop_person", return_value=None):
            result = self.step()
            self.assertEqual(result.state, "pose_uncertain")
            self.assertIsNone(result.pose)
            self.assertTrue(old_video.closed)
            for _ in range(20):
                result = self.step()
        self.assertEqual(result.state, "pose_uncertain")
        self.assertTrue(old_video.closed)
        self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(self.pipeline.pose_recovery_count, 1)
        self.assertEqual(self.pipeline.auto_reset_count, 0)

    def test_waiting_with_no_poses_does_not_repeatedly_reset(self):
        self.lock()
        self.factory.no_pose = True
        self.detector.detections = np.empty((0, 5), dtype=float)
        for _ in range(6):
            result = self.step()
        self.assertEqual(result.state, "reinitializing")
        reset_generation = self.pipeline.selection_generation
        for _ in range(30):
            result = self.step()
            self.assertEqual(result.state, "waiting")
            self.assertIsNone(result.target_id)
        self.assertEqual(self.pipeline.auto_reset_count, 1)
        self.assertEqual(self.pipeline.selection_generation, reset_generation)

    def test_unlocked_pose_failures_keep_collecting_without_auto_reset(self):
        self.factory.no_pose = True
        initial_generation = self.pipeline.selection_generation
        for _ in range(35):
            result = self.step()
            self.assertIsNone(result.target_id)
            self.assertIsNone(result.pose)
            self.assertEqual(result.state, "collecting")
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.selection_generation, initial_generation)
        self.assertTrue(all(data["sample_count"] == 0
                            for data in result.statistics.values()))

    def test_tracking_loss_timer_starts_fresh_after_observed_pose_failure(self):
        self.lock()
        self.factory.video_no_pose = True
        first_failure = self.step()
        for _ in range(22):
            self.step()
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.detector.detections = np.empty((0, 5), dtype=float)
        first_loss = self.step()
        for _ in range(4):
            result = self.step()
            self.assertEqual(result.state, "lost")
            self.assertIsNotNone(result.target_id)
        result = self.step()
        self.assertGreater(result.timestamp_ms - first_failure.timestamp_ms, 2000)
        self.assertEqual(result.timestamp_ms - first_loss.timestamp_ms, 500)
        self.assertEqual(result.state, "reinitializing")
        self.assertEqual(result.detail, "auto_reset_tracking_lost")
        self.assertEqual(self.pipeline.auto_reset_count, 1)

    def test_observed_box_clears_tracking_loss_timer_even_when_pose_is_invalid(self):
        first = self.lock()
        self.detector.detections = np.array([ADULT], dtype=float)
        for _ in range(3):
            result = self.step()
            self.assertEqual(result.state, "lost")
        self.detector.detections = np.array([CHILD, ADULT], dtype=float)
        self.factory.video_no_pose = True
        result = self.step()
        self.assertEqual(result.state, "pose_uncertain")
        self.assertEqual(result.target_id, first.target_id)
        self.detector.detections = np.array([ADULT], dtype=float)
        for _ in range(3):
            result = self.step()
            self.assertEqual(result.state, "lost")
            self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(self.pipeline.auto_reset_count, 0)

    def test_observed_geometry_quarantine_never_becomes_tracking_loss_reset(self):
        first = self.lock()
        generation = self.pipeline.selection_generation
        original_selector = self.pipeline.selector
        with patch.object(original_selector, "target_observation", return_value=(
            None, "target_identity_uncertain_reset_required"
        )):
            for _ in range(25):
                result = self.step()
                self.assertEqual(result.state, "uncertain")
                self.assertEqual(result.target_id, first.target_id)
                self.assertTrue(next(track for track in result.tracks
                                     if track.track_id == first.target_id).observed)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.selection_generation, generation)
        self.assertIs(self.pipeline.selector, original_selector)

    def test_zero_timeout_resets_on_first_failure_without_duplicate_tracker_update(self):
        self.pipeline.config = replace(self.pipeline.config, auto_reset_seconds=0)
        self.lock()
        self.detector.detections = np.empty((0, 5), dtype=float)
        frames_before = self.pipeline.tracker.frame_count
        detector_calls_before = self.detector.calls
        image_calls = sum(len(instance.calls) for instance in self.factory.image_instances())
        result = self.step()
        self.assertEqual(result.state, "reinitializing")
        self.assertEqual(result.detail, "auto_reset_tracking_lost")
        self.assertIsNone(result.target_id)
        self.assertIsNone(result.pose)
        self.assertEqual(result.statistics, {})
        self.assertEqual(self.pipeline.tracker.frame_count, frames_before + 1)
        self.assertEqual(self.detector.calls, detector_calls_before + 1)
        self.assertEqual(sum(len(instance.calls) for instance in self.factory.image_instances()),
                         image_calls)

    def test_zero_pose_recovery_refreshes_only_video_without_duplicate_tracking(self):
        self.pipeline.config = replace(self.pipeline.config, pose_recovery_seconds=0)
        first = self.lock()
        generation = self.pipeline.selection_generation
        self.factory.video_no_pose = True
        frames_before = self.pipeline.tracker.frame_count
        image_calls = sum(len(instance.calls) for instance in self.factory.image_instances())
        result = self.step()
        self.assertEqual(result.state, "pose_uncertain")
        self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(self.pipeline.selection_generation, generation)
        self.assertEqual(self.pipeline.pose_recovery_count, 1)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.tracker.frame_count, frames_before + 1)
        self.assertEqual(sum(len(instance.calls) for instance in self.factory.image_instances()),
                         image_calls)
        self.pipeline.reset_selection()
        self.assertEqual(self.pipeline.pose_recovery_count, 1)
        self.assertEqual(self.pipeline.selection_generation, generation + 1)

    def test_low_head_remains_invalid_for_initial_shoulder_selection(self):
        self.factory.image_head_y = 0.65
        for _ in range(30):
            result = self.step()
            self.assertIsNone(result.target_id)
            self.assertIsNone(result.pose)
        self.assertTrue(all(data["sample_count"] == 0
                            for data in result.statistics.values()))
        self.assertTrue(all(data["rejection_reason"] == "head_shoulder_geometry"
                            for data in result.statistics.values()))
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.factory.image_head_y = None
        self.lock()

    def test_locked_low_head_and_near_forearms_trigger_only_after_sustained_hold(self):
        first = self.calibrate_posture()
        generation = self.pipeline.selection_generation
        image_calls = len(self.factory.image_instances()[0].calls)
        self.factory.video_head_y = 0.65
        self.factory.video_forearm_y = 0.67
        first_low = self.step()
        self.assertEqual(first_low.state, "locked")
        self.assertIsNotNone(first_low.pose)
        self.assertEqual(first_low.posture.state, "candidate")
        self.assertEqual(first_low.posture.held_seconds, 0)
        for _ in range(14):
            result = self.step()
            self.assertEqual(result.state, "locked")
            self.assertEqual(result.posture.state, "candidate")
            self.assertEqual(result.target_id, first.target_id)
        result = self.step()
        self.assertEqual(result.timestamp_ms - first_low.timestamp_ms, 1500)
        self.assertEqual(result.posture.state, "resting")
        self.assertEqual(result.posture.detail, "suspected_table_rest")
        self.assertGreaterEqual(result.posture.held_seconds, 1.5)
        self.assertEqual(self.pipeline.selection_generation, generation)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.pose_recovery_count, 0)
        self.assertEqual(len(self.factory.image_instances()[0].calls), image_calls)

    def test_low_head_or_near_forearms_alone_does_not_trigger_table_rest(self):
        self.calibrate_posture()
        self.factory.video_head_y = 0.65
        self.factory.video_forearm_y = 0.25  # lowered head stays away from arms
        for _ in range(20):
            result = self.step()
            self.assertEqual(result.state, "locked")
            self.assertEqual(result.posture.state, "normal")
        self.factory.video_head_y = 0.2
        self.factory.video_forearm_y = 0.22  # near arms, but no head lowering
        for _ in range(20):
            result = self.step()
            self.assertEqual(result.posture.state, "normal")

    def test_pose_recovery_preserves_posture_baseline_and_restarts_positive_hold(self):
        first = self.calibrate_posture()
        generation = self.pipeline.selection_generation
        baseline_y = first.posture.baseline_head_y
        baseline_width = first.posture.baseline_shoulder_width
        self.factory.video_head_y = 0.65
        self.factory.video_forearm_y = 0.67
        for _ in range(16):
            result = self.step()
        self.assertEqual(result.posture.state, "resting")
        self.factory.video_no_pose = True
        for _ in range(21):
            result = self.step()
            self.assertEqual(result.state, "pose_uncertain")
            self.assertEqual(result.target_id, first.target_id)
            self.assertEqual(result.posture.state, "unknown")
            self.assertEqual(result.posture.held_seconds, 0)
            self.assertEqual(result.posture.baseline_head_y, baseline_y)
            self.assertEqual(result.posture.baseline_shoulder_width, baseline_width)
        self.assertEqual(self.pipeline.pose_recovery_count, 1)
        self.assertEqual(self.pipeline.selection_generation, generation)
        self.factory.video_no_pose = False
        result = self.step()
        self.assertEqual(result.state, "locked")
        self.assertEqual(result.posture.state, "candidate")
        self.assertEqual(result.posture.held_seconds, 0)
        self.assertEqual(result.posture.baseline_head_y, baseline_y)
        for _ in range(15):
            result = self.step()
        self.assertEqual(result.posture.state, "resting")
        self.assertEqual(self.pipeline.auto_reset_count, 0)

    def test_missing_head_outputs_unknown_without_losing_tracking_or_baseline(self):
        first = self.calibrate_posture()
        self.factory.video_head_hidden = True
        for _ in range(22):
            result = self.step()
            self.assertEqual(result.state, "pose_uncertain")
            self.assertEqual(result.target_id, first.target_id)
            self.assertEqual(result.posture.state, "unknown")
            self.assertEqual(result.posture.baseline_head_y, first.posture.baseline_head_y)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.pose_recovery_count, 1)

    def test_manual_and_tracking_resets_discard_posture_baseline(self):
        self.calibrate_posture()
        self.pipeline.reset_selection()
        result = self.step()
        if result.posture is not None:
            self.assertIsNone(result.posture.baseline_head_y)
            self.assertEqual(result.posture.baseline_samples, 0)
        self.calibrate_posture()
        self.detector.detections = np.empty((0, 5), dtype=float)
        for _ in range(6):
            result = self.step()
        self.assertEqual(result.state, "reinitializing")
        self.assertIsNone(result.target_id)
        if result.posture is not None:
            self.assertIsNone(result.posture.baseline_head_y)
            self.assertEqual(result.posture.baseline_samples, 0)
        self.assertEqual(self.pipeline.auto_reset_count, 1)

    def test_r_resets_evidence_and_video_state_but_retains_source_timeline(self):
        first = self.lock()
        old_video = self.factory.video_instances()[0]
        tracker_frames = self.pipeline.tracker.frame_count
        old_generation = self.pipeline.selection_generation
        self.factory.video_no_pose = True
        self.step()  # manual reset must also clear an active failure interval
        tracker_frames += 1
        self.pipeline.reset_selection()
        self.factory.video_no_pose = False
        self.assertTrue(old_video.closed)
        self.assertIsNone(self.pipeline.selector.target_id)
        self.assertEqual(self.pipeline.selector.selection_reason, "")
        self.assertEqual(self.pipeline.selector.statistics(), {})
        self.assertEqual(self.pipeline.tracker.frame_count, tracker_frames)
        self.assertEqual(self.pipeline.selection_generation, old_generation + 1)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.assertEqual(self.pipeline.pose_failure_seconds, 0)
        self.assertIsNone(self.pipeline._pose_failure_started_ms)
        self.assertIsNone(self.pipeline._pose_retry_ms)
        with self.assertRaises(ValueError):
            self.pipeline.process(self.frame, first.timestamp_ms, self.index)
        result = self.step()
        self.assertIsNone(result.target_id)
        self.assertIsNone(result.pose)
        second = self.lock()
        self.assertEqual(second.target_id, first.target_id)
        self.assertGreaterEqual(second.timestamp_ms-first.timestamp_ms, 2100)
        self.assertEqual(len(self.factory.video_instances()), 2)
        new_video = self.factory.video_instances()[1]
        self.assertGreater(new_video.calls[0][2], old_video.calls[-1][2])

    def test_large_detection_jump_does_not_paint_pose_on_a_new_person(self):
        result = self.lock()
        locked_id = result.target_id
        self.detector.detections = np.array([[650, 60, 770, 180, 0.99], ADULT], dtype=float)
        call_count = self.factory.video_call_count()
        result = self.step()
        self.assertIsNone(result.pose)
        self.assertEqual(result.target_id, locked_id)
        self.assertEqual(result.state, "lost")
        self.assertEqual(self.factory.video_call_count(), call_count)

    def test_pose_jump_inside_observed_box_only_recovers_near_trusted_anchor(self):
        self.factory.center_x = 0.12
        self.factory.shoulder_fraction = 0.12
        first = self.lock()
        self.factory.center_x = 0.88
        result = self.step()
        self.assertEqual(result.target_id, first.target_id)
        self.assertTrue(next(track for track in result.tracks
                             if track.track_id == first.target_id).observed)
        self.assertEqual(result.state, "uncertain")
        self.assertIsNone(result.pose)
        for _ in range(20):
            result = self.step()
            self.assertEqual(result.state, "uncertain")
            self.assertIsNone(result.pose)
            self.assertEqual(result.target_id, first.target_id)
        self.assertEqual(result.state, "uncertain")
        self.assertEqual(result.target_id, first.target_id)
        self.assertIsNone(result.pose)
        self.assertEqual(self.pipeline.pose_recovery_count, 1)
        self.assertEqual(self.pipeline.auto_reset_count, 0)
        self.factory.center_x = 0.12
        result = self.step()
        self.assertEqual(result.state, "locked")
        self.assertEqual(result.target_id, first.target_id)

    def test_close_releases_all_pose_instances(self):
        self.lock()
        self.pipeline.close()
        self.pipeline.close()  # cleanup is idempotent
        self.assertTrue(all(instance.closed for instance in self.factory.instances))


if __name__ == "__main__":
    unittest.main()
