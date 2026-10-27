"""Seat-box calibration needs a reliable pose; departure uses box IoU alone.

A fixed reference box uses continuous observed boxes with valid head/shoulders.
Missing detections and predicted boxes are unknown, never departure evidence.
Full target reinitialization cancels the reference and any manual seat box.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from target_selection import MappedPose, TrackObservation


@dataclass(frozen=True)
class SeatConfig:
    baseline_seconds: float = 2.0
    min_samples: int = 10
    # Accepted for older settings/CLI; IoU rules do not use these three fields.
    departure_ratio: float = 1.0
    hold_seconds: float = 1.0
    return_seconds: float = 0.7
    max_gap_seconds: float = 0.25
    confidence: float = 0.7
    iou_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in ("baseline_seconds", "departure_ratio", "hold_seconds",
                     "return_seconds", "max_gap_seconds"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not isinstance(self.min_samples, int) or isinstance(self.min_samples, bool)
                or self.min_samples < 1):
            raise ValueError("min_samples must be a positive integer")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must lie between zero and one")
        if not isfinite(self.iou_threshold) or not 0 < self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be finite and in (0,1]")


@dataclass(frozen=True)
class SeatResult:
    state: str
    detail: str
    displacement_ratio: float | None = None  # Legacy: not calculated by IoU rules.
    held_seconds: float = 0.0
    seat_roi: tuple[float, float, float, float] | None = None
    baseline_anchor: tuple[float, float] | None = None
    iou: float | None = None


class SeatDetector:
    """Freeze median observed bbox corners, then compare intersection/union.

    Calibration needs a reliable head/shoulder pose at every observed box.
    After calibration, detecting departure is independent of pose availability.
    After departure, return needs sufficient box overlap, its center inside
    the reference, and a reliable head/shoulder pose belonging to that box.
    A supplied ROI is an initial-generation reference; full resets cancel it.
    """

    def __init__(self, config: SeatConfig | None = None, roi: Any = None) -> None:
        self.config = config or SeatConfig()
        self._manual_roi = None if roi is None else self._box(roi)
        if roi is not None and self._manual_roi is None:
            raise ValueError("roi must be a finite positive original-pixel rectangle")
        self.reset(clear_roi=False)

    def reset(self, *, clear_roi: bool = True) -> None:
        if clear_roi:
            self._manual_roi = None
        self._last_timestamp_ms: int | None = None
        self._track_id: int | None = None
        self._clear_reference()

    def _clear_reference(self) -> None:
        self._seat_roi: np.ndarray | None = None
        self._baseline_anchor: np.ndarray | None = None
        self._baseline_started_ms: int | None = None
        self._box_samples: list[np.ndarray] = []
        self._departure_started_ms: int | None = None
        self._away = False

    @property
    def baseline_ready(self) -> bool:
        return self._seat_roi is not None

    @property
    def baseline_scale(self) -> None:
        """Legacy accessor: the IoU rule has no shoulder scale."""
        return None

    @staticmethod
    def _box(value: Any) -> np.ndarray | None:
        try:
            box = np.asarray(value, dtype=float).reshape(-1)
        except (ValueError, TypeError):
            return None
        if (box.size != 4 or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])):
            return None
        return box

    @staticmethod
    def _inside(point: np.ndarray, box: np.ndarray, tolerance: float = 0.0) -> bool:
        margin = (box[2:] - box[:2]) * tolerance
        return bool(np.all(point >= box[:2] - margin)
                    and np.all(point <= box[2:] + margin))

    @staticmethod
    def _iou(first: np.ndarray, second: np.ndarray) -> float | None:
        overlap = np.maximum(0.0, np.minimum(first[2:], second[2:])
                             - np.maximum(first[:2], second[:2]))
        intersection = float(np.prod(overlap))
        union = float(np.prod(first[2:] - first[:2])
                      + np.prod(second[2:] - second[:2]) - intersection)
        if not isfinite(union) or union <= 0 or not isfinite(intersection):
            return None
        value = intersection / union
        return value if isfinite(value) else None

    def _point(self, pose: MappedPose, index: int) -> np.ndarray | None:
        try:
            point = np.asarray(pose.points[index], dtype=float)
            visibility = float(pose.visibility[index])
            presence = float(pose.presence[index])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if (point.shape != (2,) or not np.isfinite(point).all()
                or not isfinite(visibility) or not isfinite(presence)
                or min(visibility, presence) < self.config.confidence):
            return None
        return point

    def _head_shoulders_valid(self, pose: MappedPose | None, box: np.ndarray,
                              frame_box: np.ndarray | None) -> bool:
        """Shared quality gate for continuous calibration and confirmed return."""
        if pose is None:
            return False
        head = self._point(pose, 0)
        if head is None:
            face = [self._point(pose, i) for i in range(1, 11)]
            face = [point for point in face if point is not None]
            if len(face) < 3:
                return False
            head = np.median(face, axis=0)
        shoulders = [self._point(pose, i) for i in (11, 12)]
        if any(point is None for point in shoulders):
            return False
        width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
        if width < 4 or width > 1.15 * (box[2] - box[0]):
            return False
        signature = [head, *shoulders]
        return (all(self._inside(point, box, 0.06) for point in signature)
                and (frame_box is None or all(self._inside(point, frame_box) for point in signature)))

    def _clear_calibration(self) -> None:
        self._baseline_started_ms = None
        self._box_samples.clear()

    def _result(self, state: str, detail: str, timestamp_ms: int,
                iou: float | None = None) -> SeatResult:
        roi = None if self._seat_roi is None else tuple(map(float, self._seat_roi))
        anchor = None if self._baseline_anchor is None else tuple(map(float, self._baseline_anchor))
        held = (0.0 if self._departure_started_ms is None
                else max(0.0, (timestamp_ms - self._departure_started_ms) / 1000))
        return SeatResult(state, detail, held_seconds=held, seat_roi=roi,
                          baseline_anchor=anchor, iou=iou)

    def _unknown(self, detail: str, timestamp_ms: int) -> SeatResult:
        self._departure_started_ms = None
        if not self.baseline_ready:
            self._clear_calibration()
        return self._result("unknown", detail, timestamp_ms)

    def update(self, pose: MappedPose | None, track: TrackObservation | None,
               timestamp_ms: int, baseline_scale: float | None = None,
               frame_shape: Any = None) -> SeatResult:
        # baseline_scale is accepted for callers of the earlier shoulder rule.
        if (not isinstance(timestamp_ms, (int, np.integer))
                or isinstance(timestamp_ms, bool) or timestamp_ms < 0):
            raise ValueError("timestamp_ms must be a nonnegative integer")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("seat timestamps must increase strictly")
        gap = None if self._last_timestamp_ms is None else timestamp_ms - self._last_timestamp_ms
        self._last_timestamp_ms = int(timestamp_ms)
        if gap is not None and gap > self.config.max_gap_seconds * 1000:
            self._departure_started_ms = None
            if not self.baseline_ready:
                self._clear_calibration()
        if track is None or not track.observed:
            return self._unknown("target_not_observed", timestamp_ms)
        box = self._box(track.bbox)
        if box is None:
            return self._unknown("invalid_person_box", timestamp_ms)
        if self._track_id is not None and track.track_id != self._track_id:
            self._manual_roi = None
            self._clear_reference()
            self._track_id = track.track_id
            return self._unknown("target_identity_changed", timestamp_ms)
        self._track_id = track.track_id
        frame_box = None
        if frame_shape is not None:
            try:
                height, width = float(frame_shape[0]), float(frame_shape[1])
            except (TypeError, IndexError, ValueError):
                return self._unknown("invalid_frame_shape", timestamp_ms)
            if not isfinite(height) or not isfinite(width) or height <= 0 or width <= 0:
                return self._unknown("invalid_frame_shape", timestamp_ms)
            frame_box = np.array([0.0, 0.0, width, height])
            if min(box[2], width) <= max(box[0], 0) or min(box[3], height) <= max(box[1], 0):
                return self._unknown("person_box_outside_frame", timestamp_ms)

        center = (box[:2] + box[2:]) / 2
        if not self.baseline_ready:
            if not self._head_shoulders_valid(pose, box, frame_box):
                return self._unknown("baseline_head_shoulders_unavailable", timestamp_ms)
            if self._manual_roi is not None and not self._inside(center, self._manual_roi):
                return self._unknown("awaiting_seated_baseline", timestamp_ms)
            if self._baseline_started_ms is None:
                self._baseline_started_ms = int(timestamp_ms)
            self._box_samples.append(box.copy())
            elapsed = (timestamp_ms - self._baseline_started_ms) / 1000
            if elapsed < self.config.baseline_seconds or len(self._box_samples) < self.config.min_samples:
                return self._result("calibrating", "collecting_observed_box_baseline", timestamp_ms)
            self._seat_roi = (self._manual_roi.copy() if self._manual_roi is not None
                              else np.median(np.asarray(self._box_samples), axis=0))
            self._baseline_anchor = (self._seat_roi[:2] + self._seat_roi[2:]) / 2
            self._box_samples.clear()
        iou = self._iou(box, self._seat_roi)
        if iou is None:
            return self._unknown("invalid_box_iou", timestamp_ms)
        if iou < self.config.iou_threshold:
            if self._departure_started_ms is None:
                self._departure_started_ms = int(timestamp_ms)
            self._away = True
            return self._result("away", "box_iou_below_threshold", timestamp_ms, iou)
        if self._away:
            if not self._inside(center, self._seat_roi):
                return self._result("away", "awaiting_box_center_return", timestamp_ms, iou)
            if not self._head_shoulders_valid(pose, box, frame_box):
                return self._result("away", "awaiting_return_pose", timestamp_ms, iou)
            self._away = False
        self._departure_started_ms = None
        return self._result("seated", "observed_box_inside_seat", timestamp_ms, iou)
