"""Original-image geometry and conservative shoulder-based target selection.

This module deliberately has no detector, MediaPipe, or OpenCV dependency.  All
positions and shoulder widths are measured in original video pixels.  A target
is selected once and never reassigned implicitly, including when it disappears.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from statistics import median
from typing import Any, Iterable

import numpy as np


@dataclass
class TrackObservation:
    track_id: int
    bbox: np.ndarray
    observed: bool
    observed_count: int
    first_seen_ms: int
    last_seen_ms: int


@dataclass
class MappedPose:
    points: np.ndarray  # (N, 2), original video pixels
    visibility: np.ndarray
    presence: np.ndarray


@dataclass
class ShoulderSample:
    width: float
    points: np.ndarray  # left shoulder, right shoulder; original video pixels
    head_center: np.ndarray | None = None


def _box(bbox: Any) -> np.ndarray | None:
    try:
        value = np.asarray(bbox, dtype=float).reshape(-1)[:4]
    except (TypeError, ValueError):
        return None
    if value.size != 4 or not np.isfinite(value).all():
        return None
    if value[2] <= value[0] or value[3] <= value[1]:
        return None
    return value


def crop_person(
    frame: np.ndarray, bbox: Any, padding: float = 0.08
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Return a view and its *actual clipped* (left, top, width, height)."""
    box = _box(bbox)
    if box is None or frame is None or frame.ndim < 2:
        return None
    if not isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    frame_h, frame_w = frame.shape[:2]
    bw, bh = box[2:] - box[:2]
    left = max(0, min(frame_w, int(np.floor(box[0] - bw * padding))))
    top = max(0, min(frame_h, int(np.floor(box[1] - bh * padding))))
    right = max(0, min(frame_w, int(np.ceil(box[2] + bw * padding))))
    bottom = max(0, min(frame_h, int(np.ceil(box[3] + bh * padding))))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right], (left, top, right - left, bottom - top)


def _landmark_value(landmark: Any, name: str, index: int) -> float:
    if hasattr(landmark, name):
        value = getattr(landmark, name)
    elif isinstance(landmark, dict):
        value = landmark.get(name)
    else:
        try:
            value = landmark[index]
        except (IndexError, TypeError):
            value = None
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def map_pose_to_original(
    landmarks: Iterable[Any], crop_rect: tuple[int, int, int, int]
) -> MappedPose:
    """Map MediaPipe-style landmarks (or x,y,visibility,presence rows).

    Missing confidence values become NaN, so they cannot silently pass a
    confidence check.  MediaPipe Tasks landmark objects supply both values.
    This helper assumes no caller-added letterboxing of the crop.
    """
    left, top, width, height = crop_rect
    if width <= 0 or height <= 0:
        raise ValueError("crop rectangle must have positive dimensions")
    points, visibility, presence = [], [], []
    for item in landmarks:
        x = _landmark_value(item, "x", 0)
        y = _landmark_value(item, "y", 1)
        points.append((left + x * width, top + y * height))
        visibility.append(_landmark_value(item, "visibility", 2))
        presence.append(_landmark_value(item, "presence", 3))
    return MappedPose(
        np.asarray(points, dtype=float).reshape(-1, 2),
        np.asarray(visibility, dtype=float),
        np.asarray(presence, dtype=float),
    )


def _inside(points: np.ndarray, box: np.ndarray, tolerance: float = 0.0) -> bool:
    size = box[2:] - box[:2]
    low, high = box[:2] - size * tolerance, box[2:] + size * tolerance
    return bool(np.all(points >= low) and np.all(points <= high))


def _signature_fit(points: np.ndarray, box: np.ndarray) -> float:
    """Relative fit to a detector box's upper-body anchor, lower is better."""
    width, height = box[2:] - box[:2]
    shoulder_center = np.mean(points[:2], axis=0)
    anchor = np.array([(box[0] + box[2]) / 2, box[1] + 0.30 * height])
    return float(np.linalg.norm((shoulder_center - anchor) / [width, height]))


def evaluate_shoulders(
    landmarks: Iterable[Any] | MappedPose,
    crop_rect: tuple[int, int, int, int],
    track_bbox: Any,
    all_tracks: Iterable[TrackObservation] | None = None,
    confidence: float = 0.7,
    *,
    source_track_id: int | None = None,
    for_selection: bool = True,
) -> tuple[ShoulderSample | None, str]:
    """Validate both shoulders and return pixel width or a rejection reason.

    If a head-and-shoulder signature also fits another person's box and that
    box fits equally well or better, decline the ambiguous pose.  Mere overlap
    of hands or lower bodies does not trigger this check.

    for_selection=False permits lowered heads and shoulder angles unsuitable
    for size ranking while retaining confidence, bounds and identity checks.
    """
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must lie between zero and one")
    box = _box(track_bbox)
    if box is None:
        return None, "invalid_person_box"
    pose = landmarks if isinstance(landmarks, MappedPose) else map_pose_to_original(landmarks, crop_rect)
    if len(pose.points) < 13 or len(pose.visibility) < 13 or len(pose.presence) < 13:
        return None, "missing_shoulders"
    indices = np.array([11, 12])
    shoulder_points = pose.points[indices]
    values = np.r_[shoulder_points.ravel(), pose.visibility[indices], pose.presence[indices]]
    if not np.isfinite(values).all():
        return None, "nonfinite_shoulder_data"
    if np.any(pose.visibility[indices] < confidence) or np.any(pose.presence[indices] < confidence):
        return None, "low_shoulder_confidence"
    left, top, width, height = crop_rect
    crop_box = np.array([left, top, left + width, top + height], dtype=float)
    if not _inside(shoulder_points, crop_box) or not _inside(shoulder_points, box, 0.06):
        return None, "shoulders_outside_person"
    shoulder_width = float(np.linalg.norm(shoulder_points[0] - shoulder_points[1]))
    box_width, box_height = box[2:] - box[:2]
    if shoulder_width < 4.0 or shoulder_width > 1.15 * box_width:
        return None, "implausible_shoulder_width"
    # Perspective is allowed, but a near-vertical collapsed pair is unsuitable
    # for deciding which person's apparent shoulder width is smaller.
    horizontal = abs(float(shoulder_points[0, 0] - shoulder_points[1, 0]))
    vertical = abs(float(shoulder_points[0, 1] - shoulder_points[1, 1]))
    if for_selection and (horizontal < 0.35 * shoulder_width or vertical > 0.45 * box_height):
        return None, "shoulders_not_comparable"
    face_points = pose.points[:11]
    face_valid = (
        np.isfinite(face_points).all(axis=1)
        & np.isfinite(pose.visibility[:11])
        & np.isfinite(pose.presence[:11])
        & (pose.visibility[:11] >= confidence)
        & (pose.presence[:11] >= confidence)
    )
    head_center = np.median(face_points[face_valid], axis=0) if np.any(face_valid) else None
    if head_center is not None:
        if not _inside(head_center[None, :], box, 0.06):
            return None, "head_outside_person"
        if for_selection and head_center[1] > np.mean(shoulder_points[:, 1]) + 0.12 * box_height:
            return None, "head_shoulder_geometry"
    signature = shoulder_points if head_center is None else np.vstack([shoulder_points, head_center])
    source_fit = _signature_fit(signature, box)
    for other in all_tracks or ():
        if not other.observed or (source_track_id is not None and other.track_id == source_track_id):
            continue
        other_box = _box(other.bbox)
        if other_box is None or np.allclose(other_box, box, atol=0.01):
            continue
        if _inside(signature, other_box) and _signature_fit(signature, other_box) <= source_fit + 0.07:
            return None, f"ambiguous_with_id_{other.track_id}"
    return ShoulderSample(shoulder_width, shoulder_points.copy(), head_center), ""


@dataclass
class _TimedSample:
    timestamp_ms: int
    sample: ShoulderSample
    sequence: int


@dataclass
class _Evidence:
    samples: deque[_TimedSample] = field(default_factory=deque)
    observed_count: int = 0
    observed_duration_ms: int = 0
    last_observed_ms: int | None = None
    observed_previous_frame: bool = False
    sequence: int = 0
    rejection_reason: str = "not_sampled"


class TargetSelector:
    """Collect comparable evidence, select the smallest median, then lock.

    Call observe() every video frame, including frames with no detections.
    At each attempted pose evaluation call add_sample(), passing None for a
    rejection. decide() compares only currently observed candidates with enough
    valid shoulder evidence; one qualified candidate can be selected alone.
    """

    def __init__(
        self,
        init_seconds: float = 2.0,
        min_samples: int = 10,
        min_gap: float = 0.10,
        stable_observations: int = 3,
        sample_max_age: float = 2.0,
        continuity_max_gap: float = 2.0,
    ) -> None:
        values = (init_seconds, min_gap, sample_max_age, continuity_max_gap)
        if not all(isfinite(value) for value in values):
            raise ValueError("selector parameters must be finite")
        if init_seconds <= 0 or min_samples < 2 or stable_observations < 1:
            raise ValueError("initialization duration/counts must be positive (min_samples >= 2)")
        if not 0 <= min_gap < 1 or sample_max_age <= 0 or continuity_max_gap <= 0:
            raise ValueError("invalid gap or sample age")
        self.init_seconds = float(init_seconds)
        self.min_samples = int(min_samples)
        self.min_gap = float(min_gap)
        self.stable_observations = int(stable_observations)
        self.sample_max_age = float(sample_max_age)
        self.continuity_max_gap = float(continuity_max_gap)
        self.history_seconds = max(3 * self.init_seconds, 2 * self.sample_max_age, 4.0)
        self.reset()

    def reset(self) -> None:
        self.target_id: int | None = None
        self.selection_reason = ""
        self.status = "waiting_for_people"
        self.current_tracks: dict[int, TrackObservation] = {}
        self._evidence: dict[int, _Evidence] = {}
        self._timestamp_ms: int | None = None
        self._last_target_box: np.ndarray | None = None
        self._last_target_ms: int | None = None
        self._target_suspect = False

    def observe(self, tracks: Iterable[TrackObservation], timestamp_ms: int) -> None:
        timestamp_ms = int(timestamp_ms)
        if self._timestamp_ms is not None and timestamp_ms < self._timestamp_ms:
            raise ValueError("video time moved backwards; reset selector for a new video")
        self._timestamp_ms = timestamp_ms
        self.current_tracks = {int(track.track_id): track for track in tracks}
        observed_ids: set[int] = set()
        for track_id, track in self.current_tracks.items():
            state = self._evidence.setdefault(track_id, _Evidence())
            if not track.observed or _box(track.bbox) is None:
                continue
            observed_ids.add(track_id)
            if state.last_observed_ms != timestamp_ms:
                state.observed_count += 1
                if state.observed_previous_frame and state.last_observed_ms is not None:
                    state.observed_duration_ms += timestamp_ms - state.last_observed_ms
                state.last_observed_ms = timestamp_ms
            state.observed_previous_frame = True
        for track_id, state in self._evidence.items():
            if track_id not in observed_ids:
                if state.observed_previous_frame:
                    state.sequence += 1
                state.observed_previous_frame = False
            self._prune(state, timestamp_ms)
        if self.target_id is None:
            self.status = "collecting_samples" if observed_ids else "waiting_for_people"

    def _prune(self, state: _Evidence, timestamp_ms: int) -> None:
        cutoff = timestamp_ms - int(self.history_seconds * 1000)
        while state.samples and state.samples[0].timestamp_ms < cutoff:
            state.samples.popleft()

    def add_sample(
        self,
        track_id: int,
        sample: ShoulderSample | None,
        timestamp_ms: int,
        rejection_reason: str = "invalid_shoulders",
    ) -> None:
        track_id, timestamp_ms = int(track_id), int(timestamp_ms)
        track = self.current_tracks.get(track_id)
        if track is None or not track.observed or _box(track.bbox) is None:
            return
        state = self._evidence.setdefault(track_id, _Evidence())
        if sample is None or not isfinite(sample.width) or sample.width <= 0:
            state.sequence += 1
            state.rejection_reason = rejection_reason or "invalid_shoulders"
            return
        if state.samples and timestamp_ms <= state.samples[-1].timestamp_ms:
            return
        state.samples.append(_TimedSample(timestamp_ms, sample, state.sequence))
        state.rejection_reason = ""
        self._prune(state, timestamp_ms)

    def _candidate_statistics(self, track_id: int, timestamp_ms: int) -> dict[str, Any]:
        state = self._evidence[track_id]
        self._prune(state, timestamp_ms)
        samples = list(state.samples)
        valid_ms = 0
        for previous, current in zip(samples, samples[1:]):
            delta = current.timestamp_ms - previous.timestamp_ms
            if previous.sequence == current.sequence and 0 < delta <= self.sample_max_age * 1000:
                valid_ms += delta
        width = float(median(item.sample.width for item in samples)) if samples else None
        fresh = bool(samples and timestamp_ms - samples[-1].timestamp_ms <= self.sample_max_age * 1000)
        observed = self.current_tracks.get(track_id)
        current = bool(observed and observed.observed and _box(observed.bbox) is not None)
        stable = current and state.observed_count >= self.stable_observations
        ready = stable and fresh and len(samples) >= self.min_samples and valid_ms + 1 >= self.init_seconds * 1000
        return {
            "sample_count": len(samples),
            "median_width": width,
            "valid_seconds": valid_ms / 1000,
            "observed_count": state.observed_count,
            "observed_seconds": state.observed_duration_ms / 1000,
            "observed": current,
            "stable": stable,
            "fresh": fresh,
            "ready": ready,
            "rejection_reason": state.rejection_reason,
        }

    def statistics(self, timestamp_ms: int | None = None) -> dict[int, dict[str, Any]]:
        now = int(timestamp_ms if timestamp_ms is not None else (self._timestamp_ms or 0))
        return {track_id: self._candidate_statistics(track_id, now) for track_id in self._evidence}

    def decide(self, timestamp_ms: int | None = None) -> int | None:
        if self.target_id is not None:
            return self.target_id
        now = int(timestamp_ms if timestamp_ms is not None else (self._timestamp_ms or 0))
        stats = self.statistics(now)
        observed = {track_id: data for track_id, data in stats.items() if data["observed"]}
        candidates = {track_id: data for track_id, data in observed.items() if data["ready"]}
        if not observed:
            self.status = "waiting_for_people"
            return None
        if not candidates:
            self.status = "collecting_valid_shoulders"
            return None
        ranked = sorted(candidates, key=lambda track_id: candidates[track_id]["median_width"])
        if len(ranked) > 1:
            smallest, second = (candidates[track_id]["median_width"] for track_id in ranked[:2])
            if (second - smallest) / second < self.min_gap:
                self.status = "shoulder_sizes_too_similar"
                return None
        self.selection_reason = ("only_qualified_candidate" if len(ranked) == 1
                                 else "smallest_qualified_candidate")
        self.target_id = ranked[0]
        target = self.current_tracks[self.target_id]
        self._last_target_box = _box(target.bbox).copy()
        self._last_target_ms = now
        self.status = "target_locked"
        return self.target_id

    def target_observation(
        self, timestamp_ms: int | None = None
    ) -> tuple[TrackObservation | None, str]:
        """Return an observed, continuous target; never return prediction only.

        A large implausible jump quarantines this lock until R/reset.  This is
        a geometry guard rather than an appearance-based identity guarantee.
        """
        now = int(timestamp_ms if timestamp_ms is not None else (self._timestamp_ms or 0))
        if self.target_id is None:
            return None, "no_target"
        if self._target_suspect:
            self.status = "target_identity_uncertain_reset_required"
            return None, self.status
        track = self.current_tracks.get(self.target_id)
        if track is None or not track.observed or _box(track.bbox) is None:
            self.status = "target_lost"
            return None, self.status
        current_box = _box(track.bbox)
        if self._last_target_box is not None and self._last_target_ms is not None:
            delta_s = max(0, now - self._last_target_ms) / 1000
            previous = self._last_target_box
            previous_size = previous[2:] - previous[:2]
            current_size = current_box[2:] - current_box[:2]
            displacement = np.linalg.norm((current_box[:2] + current_box[2:] - previous[:2] - previous[2:]) / 2)
            size_scale = max(float(np.linalg.norm(previous_size)), 1.0)
            area_ratio = float(np.prod(current_size) / np.prod(previous_size))
            allowed_displacement = size_scale * (0.65 + 2.0 * min(delta_s, 0.5))
            if delta_s > self.continuity_max_gap or displacement > allowed_displacement or not 0.15 <= area_ratio <= 6.0:
                self._target_suspect = True
                self.status = "target_identity_uncertain_reset_required"
                return None, self.status
        self._last_target_box = current_box.copy()
        self._last_target_ms = now
        self.status = "target_locked"
        return track, ""
