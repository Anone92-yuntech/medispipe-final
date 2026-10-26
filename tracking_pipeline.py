"""Detector -> official OC-SORT -> shoulder initialization -> target-only pose."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import tempfile
import urllib.request

import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks.python import vision

from ocsort_adapter import OCSortTracker
from posture_detection import PostureConfig, PostureResult, TableRestDetector
from target_selection import (
    MappedPose, TargetSelector, TrackObservation, crop_person,
    evaluate_shoulders, map_pose_to_original,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def ensure_model(model_path: Path = DEFAULT_MODEL) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    if model_path != DEFAULT_MODEL:
        raise ValueError(f"Missing or empty custom pose model: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    print("Downloading the official MediaPipe Pose Landmarker Lite model...")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=model_path.parent, suffix=".download", delete=False
            ) as target:
                temporary_path = Path(target.name)
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        if not temporary_path.stat().st_size:
            raise OSError("The downloaded pose model is empty")
        temporary_path.replace(model_path)
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
    return model_path


@dataclass(frozen=True)
class PipelineConfig:
    init_seconds: float = 2.0
    min_samples: int = 10
    min_gap: float = 0.10
    landmark_confidence: float = 0.70
    sample_stride: int = 3
    crop_padding: float = 0.08
    stable_observations: int = 3
    sample_max_age: float = 3.0
    max_age_seconds: float = 1.0
    detection_confidence: float = 0.35
    detector_label: str = ""
    auto_reset_seconds: float = 0.5
    pose_recovery_seconds: float = 2.0
    posture_baseline_seconds: float = 2.0
    posture_head_drop_ratio: float = 0.30
    posture_forearm_distance_ratio: float = 0.35
    posture_hold_seconds: float = 1.5
    posture_max_gap_seconds: float = 0.25
    retain_target_on_loss: bool = False

    def __post_init__(self):
        if not math.isfinite(self.auto_reset_seconds) or self.auto_reset_seconds < 0:
            raise ValueError("auto_reset_seconds must be finite and nonnegative")
        if not math.isfinite(self.pose_recovery_seconds) or self.pose_recovery_seconds < 0:
            raise ValueError("pose_recovery_seconds must be finite and nonnegative")
        self.posture_config()  # Validate before creating any model resources.

    def posture_config(self):
        return PostureConfig(
            baseline_seconds=self.posture_baseline_seconds,
            head_drop_ratio=self.posture_head_drop_ratio,
            forearm_distance_ratio=self.posture_forearm_distance_ratio,
            hold_seconds=self.posture_hold_seconds,
            confidence=self.landmark_confidence,
            max_gap_seconds=self.posture_max_gap_seconds,
        )


@dataclass
class FrameAnalysis:
    tracks: list[TrackObservation]
    target_id: int | None
    pose: MappedPose | None
    state: str
    detail: str
    statistics: dict
    timestamp_ms: int
    selection_generation: int = 0
    auto_reset_count: int = 0
    auto_reset_reason: str = ""
    posture: PostureResult | None = None
    pose_recovery_count: int = 0
    pose_failure_seconds: float = 0.0


class TrackingPipeline:
    def __init__(self, detector, model_path: Path, fps: float,
                 config: PipelineConfig | None = None):
        self.detector = detector
        self.config = config or PipelineConfig()
        self._model_bytes = model_path.read_bytes()
        self.tracker = OCSortTracker(
            fps=fps, max_age_seconds=self.config.max_age_seconds,
            det_threshold=self.config.detection_confidence,
        )
        self._image_pose = self._create_pose(vision.RunningMode.IMAGE)
        self._video_pose = None
        self._last_timestamp = -1
        self.auto_reset_count = 0
        self.last_auto_reset_reason = ""
        self.selection_generation = 0
        self.pose_recovery_count = 0
        self.last_pose_error = ""
        self.reset_selection()

    def _create_pose(self, mode):
        return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=self._model_bytes),
            running_mode=mode, num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        ))

    def _close_video_pose(self):
        if self._video_pose is not None:
            instance = self._video_pose
            self._video_pose = None
            try:
                instance.close()
            except RuntimeError as exc:
                self.last_pose_error = str(exc)

    def reset_selection(self):
        """Clear evidence for manual/automatic recovery; preserve OC-SORT/time."""
        self._close_video_pose()
        self.selection_generation += 1
        config = self.config
        self.selector = TargetSelector(
            init_seconds=config.init_seconds, min_samples=config.min_samples,
            min_gap=config.min_gap, stable_observations=config.stable_observations,
            sample_max_age=config.sample_max_age,
        )
        self.last_anchor = None
        self.last_anchor_time = None
        self.last_anchor_scale = None
        self.rejections = {}
        self.identity_uncertain = False
        self._failure_started_ms = None
        self._pose_failure_started_ms = None
        self._pose_retry_ms = None
        self.pose_failure_seconds = 0.0
        self.posture_detector = TableRestDetector(config.posture_config())

    def _recover_if_needed(self, state, detail, timestamp_ms, target_observed):
        """Keep a visible ID; pose-only recovery and track loss use separate clocks."""
        if target_observed:
            self._failure_started_ms = None
            if state == "locked":
                self._pose_failure_started_ms = None
                self._pose_retry_ms = None
                self.pose_failure_seconds = 0.0
                return state, detail
            if self._pose_failure_started_ms is None:
                self._pose_failure_started_ms = timestamp_ms
            self.pose_failure_seconds = (timestamp_ms - self._pose_failure_started_ms) / 1000
            retry_from = (self._pose_retry_ms if self._pose_retry_ms is not None
                          else self._pose_failure_started_ms)
            if timestamp_ms - retry_from >= self.config.pose_recovery_seconds * 1000:
                self._close_video_pose()
                self.pose_recovery_count += 1
                self._pose_retry_ms = timestamp_ms
                if state != "uncertain":
                    detail = "pose_recovery_retry"
            return state, detail
        self._pose_failure_started_ms = None
        self._pose_retry_ms = None
        self.pose_failure_seconds = 0.0
        if self.config.retain_target_on_loss:
            return "lost", "target_lost_retained"
        if self._failure_started_ms is None:
            self._failure_started_ms = timestamp_ms
        if timestamp_ms - self._failure_started_ms < self.config.auto_reset_seconds * 1000:
            return state, detail
        reason = "auto_reset_tracking_lost"
        self.reset_selection()
        self.auto_reset_count += 1
        self.last_auto_reset_reason = reason
        return "reinitializing", reason

    @staticmethod
    def _mp_image(crop):
        height, width = crop.shape[:2]
        if max(height, width) > 1280:
            scale = 1280 / max(height, width)
            crop = cv2.resize(crop, (max(1, round(width * scale)),
                                     max(1, round(height * scale))))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _evaluate(self, result, rectangle, track, observed, *, for_selection=True):
        if not result.pose_landmarks:
            return None, None, "No pose detected"
        landmarks = result.pose_landmarks[0]
        sample, reason = evaluate_shoulders(
            landmarks, rectangle, track.bbox, all_tracks=observed,
            confidence=self.config.landmark_confidence,
            source_track_id=track.track_id,
            for_selection=for_selection,
        )
        if sample is None:
            return None, None, reason
        return sample, map_pose_to_original(landmarks, rectangle), ""

    def process(self, frame: np.ndarray, timestamp_ms: int,
                frame_index: int) -> FrameAnalysis:
        if timestamp_ms <= self._last_timestamp:
            raise ValueError("Frame timestamps must strictly increase")
        self._last_timestamp = timestamp_ms
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame.shape, timestamp_ms)
        observed = [track for track in tracks if track.observed]
        self.selector.observe(tracks, timestamp_ms)
        pose = None
        state, detail = "waiting", "Waiting for people"

        if self.selector.target_id is None:
            if frame_index % self.config.sample_stride == 0:
                for track in observed:
                    if track.observed_count < self.config.stable_observations:
                        continue
                    cropped = crop_person(frame, track.bbox, self.config.crop_padding)
                    sample, reason = None, "Invalid crop"
                    if cropped is not None:
                        crop, rectangle = cropped
                        if self._image_pose is None:
                            self._image_pose = self._create_pose(vision.RunningMode.IMAGE)
                        try:
                            result = self._image_pose.detect(self._mp_image(crop))
                        except (RuntimeError, ValueError) as exc:
                            self.last_pose_error = str(exc)
                            reason = "pose_inference_error"
                            instance, self._image_pose = self._image_pose, None
                            try:
                                instance.close()
                            except RuntimeError:
                                pass
                        else:
                            sample, _, reason = self._evaluate(result, rectangle, track, observed)
                    self.selector.add_sample(track.track_id, sample, timestamp_ms,
                                             rejection_reason=reason)
                    self.rejections[track.track_id] = reason
            self.selector.decide(timestamp_ms)
            state = "collecting" if observed else "waiting"
            detail = self.selector.status

        target_id = self.selector.target_id
        posture = None
        if target_id is not None:
            raw_target = next((item for item in observed if item.track_id == target_id), None)
            track, reason = self.selector.target_observation(timestamp_ms)
            if raw_target is None:
                state, detail = "lost", "target_lost"
                self._close_video_pose()
            elif track is None:
                state, detail = "uncertain", reason
                self._close_video_pose()
            else:
                cropped = crop_person(frame, track.bbox, self.config.crop_padding)
                state, detail = "pose_uncertain", "Invalid target crop"
                if cropped is None:
                    self._close_video_pose()
                else:
                    if self._video_pose is None:
                        self._video_pose = self._create_pose(vision.RunningMode.VIDEO)
                    crop, rectangle = cropped
                    try:
                        result = self._video_pose.detect_for_video(
                            self._mp_image(crop), timestamp_ms
                        )
                    except (RuntimeError, ValueError) as exc:
                        self.last_pose_error = str(exc)
                        sample, mapped, reason = None, None, "pose_inference_error"
                        self._close_video_pose()
                    else:
                        sample, mapped, reason = self._evaluate(
                            result, rectangle, track, observed, for_selection=False
                        )
                    if mapped is not None:
                        # Track ID alone is not an identity guarantee. Reject a
                        # sudden pose-anchor jump rather than draw another person.
                        anchor = mapped.points[[11, 12]].mean(axis=0)
                        scale = max(1.0, float(np.linalg.norm(track.bbox[2:] - track.bbox[:2])))
                        if self.last_anchor is not None:
                            dt = (timestamp_ms - self.last_anchor_time) / 1000
                            displacement = float(np.linalg.norm(anchor - self.last_anchor))
                            allowance = self.last_anchor_scale * (
                                0.3 if self.identity_uncertain else (0.3 + min(1.5, 2.5 * dt))
                            )
                            if displacement > allowance:
                                self.identity_uncertain = True
                                state, detail = "uncertain", "pose_anchor_jumped"
                                self._close_video_pose()
                            else:
                                pose = mapped
                        else:
                            pose = mapped
                        if pose is not None:
                            self.identity_uncertain = False
                            self.last_anchor, self.last_anchor_time = anchor, timestamp_ms
                            self.last_anchor_scale = scale
                            state, detail = "locked", self.selector.selection_reason
                    else:
                        detail = reason
                        # Missing pose evidence is unknown, not evidence that the
                        # tracked person changed. Retry this ID with its own clock.
            posture = self.posture_detector.update(
                pose, timestamp_ms, bbox=raw_target.bbox if raw_target is not None else None
            )
            if state == "locked" and posture.state == "unknown":
                state, detail = "pose_uncertain", "posture_landmarks_missing"
            state, detail = self._recover_if_needed(
                state, detail, timestamp_ms, target_observed=raw_target is not None
            )
            target_id = self.selector.target_id
            if target_id is None:
                posture = None
        return FrameAnalysis(tracks, target_id, pose, state, detail,
                             self.selector.statistics(), timestamp_ms,
                             self.selection_generation, self.auto_reset_count,
                             self.last_auto_reset_reason, posture,
                             self.pose_recovery_count, self.pose_failure_seconds)

    def close(self):
        try:
            self._close_video_pose()
        finally:
            if self._image_pose is not None:
                self._image_pose.close()
                self._image_pose = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
