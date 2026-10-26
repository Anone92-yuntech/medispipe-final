"""Observed upper-body departure from a calibrated seat region.

This is an image-space rule, not a chair-contact sensor.  Missing detections,
Kalman predictions and missing poses are unknown, never evidence of leaving.
All input and returned geometry uses original-video pixels.
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
    departure_ratio: float = 1.0
    hold_seconds: float = 1.0
    return_seconds: float = 0.7
    max_gap_seconds: float = 0.25
    confidence: float = 0.7

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


@dataclass(frozen=True)
class SeatResult:
    state: str
    detail: str
    displacement_ratio: float | None = None
    held_seconds: float = 0.0
    seat_roi: tuple[float, float, float, float] | None = None
    baseline_anchor: tuple[float, float] | None = None


class SeatDetector:
    """Freeze an upright shoulder anchor, then time reliable region departures.

    ``roi`` is an optional original-pixel (left, top, right, bottom) rectangle.
    Otherwise the region extends +/- departure_ratio * S0 horizontally about
    the reference shoulders, and only horizontal departure counts.  Standing
    in place or leaning vertically is not leaving the original seat region.
    The automatic rectangle spans the original frame height for display; when
    no frame dimensions are supplied it uses a generous display-only height.
    Calibration starts with an upright, observed target; a supplied ROI also
    requires that target to be initially inside it.  Neither image resizing
    for display nor a changing crop rectangle affects the frozen reference.

    Call reset() on target selection-generation changes.  An unexpected ID
    change is also guarded here, without reusing the previous person's data.
    """

    def __init__(self, config: SeatConfig | None = None,
                 roi: Any = None) -> None:
        self.config = config or SeatConfig()
        self._manual_roi = None if roi is None else self._box(roi)
        if roi is not None and self._manual_roi is None:
            raise ValueError("roi must be a finite positive original-pixel rectangle")
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_ms: int | None = None
        self._track_id: int | None = None
        self._clear_reference()

    def _clear_reference(self) -> None:
        self._baseline_anchor: np.ndarray | None = None
        self._baseline_scale: float | None = None
        self._seat_roi: np.ndarray | None = None
        self._baseline_started_ms: int | None = None
        self._anchor_samples: list[np.ndarray] = []
        self._scale_samples: list[float] = []
        self._departure_started_ms: int | None = None
        self._return_started_ms: int | None = None
        self._away = False

    @property
    def baseline_ready(self) -> bool:
        return self._baseline_anchor is not None

    @property
    def baseline_scale(self) -> float | None:
        return self._baseline_scale

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
    def _inside(point: np.ndarray, box: np.ndarray,
                tolerance: float = 0.0) -> bool:
        margin = (box[2:] - box[:2]) * tolerance
        return bool(np.all(point >= box[:2] - margin)
                    and np.all(point <= box[2:] + margin))

    def _point(self, pose: MappedPose, index: int) -> np.ndarray | None:
        try:
            point = np.asarray(pose.points[index], dtype=float)
            visibility = float(pose.visibility[index])
            presence = float(pose.presence[index])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if (point.shape != (2,) or not np.isfinite(point).all()
                or not isfinite(visibility) or not isfinite(presence)
                or visibility < self.config.confidence
                or presence < self.config.confidence):
            return None
        return point

    def _head(self, pose: MappedPose) -> np.ndarray | None:
        nose = self._point(pose, 0)
        if nose is not None:
            return nose
        face = [self._point(pose, index) for index in range(1, 11)]
        available = [point for point in face if point is not None]
        return np.median(available, axis=0) if len(available) >= 3 else None

    def _clear_holds(self) -> None:
        self._departure_started_ms = None
        self._return_started_ms = None

    def _clear_calibration(self) -> None:
        self._baseline_started_ms = None
        self._anchor_samples.clear()
        self._scale_samples.clear()

    def _result(self, state: str, detail: str,
                displacement: float | None = None,
                held: float = 0.0) -> SeatResult:
        roi = None if self._seat_roi is None else tuple(map(float, self._seat_roi))
        anchor = (None if self._baseline_anchor is None
                  else tuple(map(float, self._baseline_anchor)))
        return SeatResult(state, detail, displacement, held, roi, anchor)

    def _unknown(self, detail: str) -> SeatResult:
        self._clear_holds()
        if not self.baseline_ready:
            self._clear_calibration()
        return self._result("unknown", detail)

    def update(self, pose: MappedPose | None,
               track: TrackObservation | None, timestamp_ms: int,
               baseline_scale: float | None = None,
               frame_shape: Any = None) -> SeatResult:
        if (not isinstance(timestamp_ms, (int, np.integer))
                or isinstance(timestamp_ms, bool) or timestamp_ms < 0):
            raise ValueError("timestamp_ms must be a nonnegative integer")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("seat timestamps must increase strictly")
        gap = (None if self._last_timestamp_ms is None
               else timestamp_ms - self._last_timestamp_ms)
        self._last_timestamp_ms = int(timestamp_ms)
        if gap is not None and gap > self.config.max_gap_seconds * 1000:
            self._clear_holds()
            if not self.baseline_ready:
                self._clear_calibration()

        if track is None or not track.observed:
            return self._unknown("target_not_observed")
        box = self._box(track.bbox)
        if box is None:
            return self._unknown("invalid_person_box")
        if self._track_id is not None and track.track_id != self._track_id:
            self._clear_reference()
            self._track_id = track.track_id
            return self._unknown("target_identity_changed")
        self._track_id = track.track_id
        if pose is None:
            return self._unknown("pose_unavailable")
        shoulders = [self._point(pose, index) for index in (11, 12)]
        head = self._head(pose)
        if head is None:
            return self._unknown("head_unavailable")
        if any(point is None for point in shoulders):
            return self._unknown("shoulders_unavailable")
        signature = np.vstack([head, *shoulders])
        if not all(self._inside(point, box, 0.06) for point in signature):
            return self._unknown("upper_body_outside_target")
        if frame_shape is not None:
            try:
                height, width = float(frame_shape[0]), float(frame_shape[1])
            except (TypeError, IndexError, ValueError):
                return self._unknown("invalid_frame_shape")
            if (not isfinite(height) or not isfinite(width) or height <= 0 or width <= 0):
                return self._unknown("invalid_frame_shape")
            frame_box = np.array([0.0, 0.0, width, height])
            if not all(self._inside(point, frame_box) for point in signature):
                return self._unknown("upper_body_outside_frame")

        shoulder_points = np.asarray(shoulders)
        shoulder_width = float(np.linalg.norm(shoulder_points[0] - shoulder_points[1]))
        if shoulder_width < 4 or shoulder_width > 1.15 * (box[2] - box[0]):
            return self._unknown("shoulder_geometry_invalid")
        anchor = np.mean(shoulder_points, axis=0)
        if not self.baseline_ready:
            if head[1] > anchor[1] - 0.15 * shoulder_width:
                return self._unknown("upright_baseline_required")
            if self._manual_roi is not None and not self._inside(anchor, self._manual_roi):
                return self._unknown("awaiting_seated_baseline")
            reference_scale = shoulder_width
            if baseline_scale is not None:
                try:
                    reference_scale = float(baseline_scale)
                except (ValueError, TypeError):
                    return self._unknown("baseline_scale_invalid")
                if not isfinite(reference_scale) or reference_scale < 4:
                    return self._unknown("baseline_scale_invalid")
            if self._baseline_started_ms is None:
                self._baseline_started_ms = int(timestamp_ms)
            self._anchor_samples.append(anchor.copy())
            self._scale_samples.append(reference_scale)
            elapsed = (timestamp_ms - self._baseline_started_ms) / 1000
            if (elapsed < self.config.baseline_seconds
                    or len(self._anchor_samples) < self.config.min_samples):
                return self._result("calibrating", "collecting_seated_baseline")
            self._baseline_anchor = np.median(self._anchor_samples, axis=0)
            self._baseline_scale = float(np.median(self._scale_samples))
            if self._manual_roi is not None:
                self._seat_roi = self._manual_roi.copy()
            else:
                extent = self._baseline_scale * self.config.departure_ratio
                display_height = (float(frame_shape[0]) if frame_shape is not None
                                  else float(self._baseline_anchor[1] + 4 * self._baseline_scale))
                self._seat_roi = np.array([self._baseline_anchor[0] - extent, 0.0,
                                           self._baseline_anchor[0] + extent,
                                           max(1.0, display_height)])

        displacement = abs(float(anchor[0] - self._baseline_anchor[0])) / self._baseline_scale
        inside = (self._inside(anchor, self._seat_roi) if self._manual_roi is not None
                  else bool(self._seat_roi[0] <= anchor[0] <= self._seat_roi[2]))
        if not inside:
            self._return_started_ms = None
            if self._departure_started_ms is None:
                self._departure_started_ms = int(timestamp_ms)
            held = (timestamp_ms - self._departure_started_ms) / 1000
            if self._away or held >= self.config.hold_seconds:
                self._away = True
                return self._result("away", "observed_upper_body_outside_seat", displacement, held)
            return self._result("candidate", "holding_seat_departure", displacement, held)

        self._departure_started_ms = None
        if self._away:
            if self._return_started_ms is None:
                self._return_started_ms = int(timestamp_ms)
            held = (timestamp_ms - self._return_started_ms) / 1000
            if held < self.config.return_seconds:
                return self._result("away", "holding_seat_return", displacement, held)
            self._away = False
        self._return_started_ms = None
        return self._result("seated", "observed_inside_seat_region", displacement)
