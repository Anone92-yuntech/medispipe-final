"""Conservative, original-pixel geometry for a suspected table-rest posture.

The event means that a lowered head remains near a visible forearm.  It does
not establish physical contact, crossed hands, sleep, or a medical condition.
This module is independent of MediaPipe/OpenCV and only consumes mapped poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Any

import numpy as np

from target_selection import MappedPose


@dataclass(frozen=True)
class PostureConfig:
    baseline_seconds: float = 2.0
    baseline_min_samples: int = 10
    head_drop_ratio: float = 0.30
    forearm_distance_ratio: float = 0.35
    hold_seconds: float = 1.5
    confidence: float = 0.7
    max_gap_seconds: float = 0.25

    def __post_init__(self) -> None:
        for name in ("baseline_seconds", "head_drop_ratio", "forearm_distance_ratio",
                     "hold_seconds", "max_gap_seconds"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not isinstance(self.baseline_min_samples, int)
                or isinstance(self.baseline_min_samples, bool)
                or self.baseline_min_samples < 1):
            raise ValueError("baseline_min_samples must be a positive integer")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True)
class PostureResult:
    state: str
    detail: str
    head_drop_ratio: float | None = None
    forearm_distance_ratio: float | None = None
    held_seconds: float = 0.0
    baseline_samples: int = 0
    baseline_head_y: float | None = None
    baseline_shoulder_width: float | None = None


class TableRestDetector:
    """Calibrate once per target, then require continuous source-time evidence.

    The caller must reset when selecting a new target, including the same OC-
    SORT ID in a new selection generation.  Temporary pose recovery must keep
    this instance: an established baseline survives missing/occluded poses.

    A reliable nose is the preferred head anchor.  If it is unavailable at the
    start of calibration, a median of at least three reliable face landmarks
    is used.  That exact landmark subset is retained, preventing a change of
    visible face points from appearing as head movement.
    """

    def __init__(self, config: PostureConfig | None = None) -> None:
        self.config = config or PostureConfig()
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_ms: int | None = None
        self._baseline_head_y: float | None = None
        self._baseline_shoulder_width: float | None = None
        self._positive_started_ms: int | None = None
        self._head_indices: tuple[int, ...] | None = None
        self._calibration_started_ms: int | None = None
        self._head_samples: list[float] = []
        self._width_samples: list[float] = []

    @property
    def baseline_ready(self) -> bool:
        return self._baseline_head_y is not None

    def _clear_calibration(self) -> None:
        self._calibration_started_ms = None
        self._head_samples.clear()
        self._width_samples.clear()
        self._head_indices = None

    def _result(self, state: str, detail: str,
                drop: float | None = None, distance: float | None = None,
                held: float = 0.0) -> PostureResult:
        return PostureResult(state, detail, drop, distance, held,
                             len(self._head_samples), self._baseline_head_y,
                             self._baseline_shoulder_width)

    def _unknown(self, detail: str) -> PostureResult:
        self._positive_started_ms = None
        if not self.baseline_ready:
            self._clear_calibration()
        return self._result("unknown", detail)

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
        if self._head_indices is None:
            if self._point(pose, 0) is not None:
                self._head_indices = (0,)
            else:
                visible = tuple(i for i in range(1, 11)
                                if self._point(pose, i) is not None)
                if len(visible) < 3:
                    return None
                self._head_indices = visible
        points = [self._point(pose, i) for i in self._head_indices]
        if any(point is None for point in points):
            return None
        return np.median(np.asarray(points), axis=0)

    @staticmethod
    def _box(bbox: Any) -> np.ndarray | None:
        try:
            box = np.asarray(bbox, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if (box.size != 4 or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])):
            return None
        return box

    @staticmethod
    def _inside(point: np.ndarray, box: np.ndarray | None) -> bool:
        if box is None:
            return True
        margin = (box[2:] - box[:2]) * 0.10
        return bool(np.all(point >= box[:2] - margin)
                    and np.all(point <= box[2:] + margin))

    @staticmethod
    def _segment_distance(head: np.ndarray, elbow: np.ndarray,
                          wrist: np.ndarray) -> float | None:
        segment = wrist - elbow
        length_squared = float(segment @ segment)
        if not isfinite(length_squared) or length_squared <= 1e-12:
            return None
        fraction = float(np.clip((head - elbow) @ segment / length_squared, 0, 1))
        distance = float(np.linalg.norm(head - (elbow + fraction * segment)))
        return distance if isfinite(distance) else None

    def update(self, pose: MappedPose | None, timestamp_ms: int,
               bbox: Any = None) -> PostureResult:
        if (not isinstance(timestamp_ms, (int, np.integer))
                or isinstance(timestamp_ms, bool) or timestamp_ms < 0):
            raise ValueError("timestamp_ms must be a nonnegative integer")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("posture timestamps must increase strictly")
        gap = (None if self._last_timestamp_ms is None
               else timestamp_ms - self._last_timestamp_ms)
        self._last_timestamp_ms = int(timestamp_ms)
        if gap is not None and gap > self.config.max_gap_seconds * 1000:
            self._positive_started_ms = None
            if not self.baseline_ready:
                self._clear_calibration()
        if pose is None:
            return self._unknown("pose_unavailable")
        box = None if bbox is None else self._box(bbox)
        if bbox is not None and box is None:
            return self._unknown("invalid_person_box")
        head = self._head(pose)
        if head is None:
            return self._unknown("head_unavailable")
        if not self._inside(head, box):
            return self._unknown("head_outside_person")

        if not self.baseline_ready:
            shoulders = [self._point(pose, i) for i in (11, 12)]
            if any(point is None for point in shoulders):
                return self._unknown("baseline_shoulders_unavailable")
            if not all(self._inside(point, box) for point in shoulders):
                return self._unknown("baseline_shoulders_outside_person")
            shoulder_points = np.asarray(shoulders)
            width = float(np.linalg.norm(shoulder_points[0] - shoulder_points[1]))
            if not isfinite(width) or width <= 1e-6:
                return self._unknown("baseline_shoulder_width_invalid")
            # Calibration needs an upright reference, never a resting pose.
            if head[1] > float(np.mean(shoulder_points[:, 1])) - 0.15 * width:
                return self._unknown("upright_baseline_required")
            if self._calibration_started_ms is None:
                self._calibration_started_ms = int(timestamp_ms)
            self._head_samples.append(float(head[1]))
            self._width_samples.append(width)
            elapsed = (timestamp_ms - self._calibration_started_ms) / 1000
            if (elapsed < self.config.baseline_seconds
                    or len(self._head_samples) < self.config.baseline_min_samples):
                return self._result("calibrating", "collecting_upright_baseline")
            self._baseline_head_y = float(median(self._head_samples))
            self._baseline_shoulder_width = float(median(self._width_samples))

        drop = (float(head[1]) - self._baseline_head_y) / self._baseline_shoulder_width
        if not isfinite(drop):
            return self._unknown("head_geometry_invalid")
        distances = []
        for elbow_index, wrist_index in ((13, 15), (14, 16)):
            elbow, wrist = self._point(pose, elbow_index), self._point(pose, wrist_index)
            if (elbow is None or wrist is None
                    or not self._inside(elbow, box) or not self._inside(wrist, box)):
                continue
            distance = self._segment_distance(head, elbow, wrist)
            if distance is not None:
                distances.append(distance)
        if not distances:
            return self._unknown("forearm_unavailable")
        distance_ratio = min(distances) / self._baseline_shoulder_width
        if drop < self.config.head_drop_ratio or distance_ratio > self.config.forearm_distance_ratio:
            self._positive_started_ms = None
            return self._result("normal", "posture_conditions_not_met", drop, distance_ratio)
        if self._positive_started_ms is None:
            self._positive_started_ms = int(timestamp_ms)
        held = (timestamp_ms - self._positive_started_ms) / 1000
        if held >= self.config.hold_seconds:
            return self._result("resting", "suspected_table_rest", drop, distance_ratio, held)
        return self._result("candidate", "holding_posture", drop, distance_ratio, held)
