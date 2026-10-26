"""Source-time motion rules for the selected person's head and wrists.

These rules describe visible movement, not emotions or age.  Geometry is in
original video pixels.  Wrist positions are relative to the shoulder midpoint
and divided by a fixed reference shoulder width.  Head turning is a 2-D proxy:
the nose's signed projection onto a fixed eye/ear pair, relative to a short
initial neutral reference.  A rigid translation of the head is not a turn.

``shake_min_turns=4`` means four alternating arrivals at left/right extremes
inside two seconds (L,R,L,R or R,L,R,L).  The first extreme counts as one.
Missing observations never interpolate across the gap or contribute events.
All thresholds are initial experimental settings requiring video validation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class MotionConfig:
    window_seconds: float = 2.0
    shake_min_turns: int = 4
    shake_threshold: float = 0.12
    shake_min_interval: float = 0.12
    wave_min_hz: float = 2.0
    wave_min_amplitude: float = 0.03
    wave_max_amplitude: float = 0.25
    wave_min_cycles: int = 3
    wave_min_seconds: float = 1.5
    confidence: float = 0.7
    max_gap_seconds: float = 0.2
    head_calibration_seconds: float = 0.4
    wave_min_samples: int = 12

    def __post_init__(self) -> None:
        positive = (
            "window_seconds", "shake_threshold", "shake_min_interval",
            "wave_min_hz", "wave_min_amplitude", "wave_max_amplitude",
            "wave_min_seconds", "max_gap_seconds", "head_calibration_seconds",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        if self.shake_min_turns < 2 or self.wave_min_cycles < 1 or self.wave_min_samples < 4:
            raise ValueError("turns, cycles, and sample counts are too small")
        if self.wave_min_amplitude >= self.wave_max_amplitude:
            raise ValueError("minimum wave amplitude must be below maximum")
        if self.wave_min_seconds > self.window_seconds:
            raise ValueError("wave_min_seconds cannot exceed window_seconds")


@dataclass
class MotionResult:
    head_state: str = "unknown"
    left_state: str = "unknown"
    right_state: str = "unknown"
    head_detail: str = "head_unavailable"
    left_detail: str = "wrist_unavailable"
    right_detail: str = "wrist_unavailable"
    turn_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def head_shake(self) -> bool:
        return self.head_state == "active"

    @property
    def left_wave(self) -> bool:
        return self.left_state == "active"

    @property
    def right_wave(self) -> bool:
        return self.right_state == "active"

    @property
    def wave_state(self) -> str:
        if self.left_wave and self.right_wave:
            return "both"
        if self.left_wave:
            return "left"
        if self.right_wave:
            return "right"
        if self.left_state == "unknown" or self.right_state == "unknown":
            return "unknown"
        if self.left_state == "calibrating" or self.right_state == "calibrating":
            return "calibrating"
        return "none"


class MotionDetector:
    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self.reset()

    def reset(self) -> None:
        """Call when changing the selected person, video, or initialization."""
        self._last_timestamp: int | None = None
        self._scale: float | None = None
        self._external_scale = False
        self._reset_head()
        self._hands: dict[str, deque] = {"left": deque(), "right": deque()}

    @property
    def scale(self) -> float | None:
        return self._scale

    def _reset_head(self) -> None:
        self._face_pair: tuple[int, int] | None = None
        self._head_calibration: list[tuple[float, float]] = []
        self._neutral: float | None = None
        self._head_last: tuple[float, float] | None = None
        self._head_side = 0
        self._turns: deque[tuple[float, int]] = deque()

    def _point(self, pose: Any, index: int, bbox: Any) -> np.ndarray | None:
        if pose is None:
            return None
        try:
            p = np.asarray(pose.points[index], dtype=float)
            visibility = float(pose.visibility[index])
            presence = float(pose.presence[index])
            if p.shape != (2,) or not np.isfinite(p).all():
                return None
            if not math.isfinite(visibility) or not math.isfinite(presence):
                return None
            if min(visibility, presence) < self.config.confidence:
                return None
            if bbox is not None:
                box = np.asarray(bbox, dtype=float).reshape(-1)[:4]
                if box.size != 4 or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                    return None
                margin = (box[2:] - box[:2]) * 0.25
                if np.any(p < box[:2] - margin) or np.any(p > box[2:] + margin):
                    return None
            return p
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    def update(self, pose: Any, timestamp_ms: int, scale: float | None = None,
               bbox: Any = None) -> MotionResult:
        timestamp = float(timestamp_ms)
        if not math.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            raise ValueError("motion timestamps must be finite and strictly increasing")
        t = timestamp / 1000.0
        if self._last_timestamp is not None and (timestamp - self._last_timestamp) / 1000 > self.config.max_gap_seconds + 1e-9:
            self._reset_head()
            for history in self._hands.values():
                history.clear()
        self._last_timestamp = timestamp
        shoulders = [self._point(pose, i, bbox) for i in (11, 12)]
        shoulder_center = None
        if all(p is not None for p in shoulders):
            width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
            if width > 1:
                shoulder_center = (shoulders[0] + shoulders[1]) / 2
                if self._scale is None:
                    self._scale = width
        try:
            supplied_scale = float(scale) if scale is not None else float("nan")
        except (TypeError, ValueError):
            supplied_scale = float("nan")
        if math.isfinite(supplied_scale) and supplied_scale > 1 and not self._external_scale:
            if self._scale is not None and not math.isclose(self._scale, supplied_scale, rel_tol=0.01):
                for history in self._hands.values():
                    history.clear()
            self._scale = supplied_scale
            self._external_scale = True
        result = MotionResult(metrics={"reference_shoulder_width_px": self._scale})
        result.head_state, result.head_detail, result.turn_count, head_metrics = self._head(pose, t, bbox)
        result.metrics.update(head_metrics)
        for name, index in (("left", 15), ("right", 16)):
            wrist = self._point(pose, index, bbox)
            point = None
            if wrist is not None and shoulder_center is not None and self._scale is not None:
                point = (wrist - shoulder_center) / self._scale
            state, detail, metrics = self._hand(name, point, t)
            setattr(result, f"{name}_state", state)
            setattr(result, f"{name}_detail", detail)
            result.metrics.update({f"{name}_{key}": value for key, value in metrics.items()})
        return result

    def _head(self, pose: Any, t: float, bbox: Any) -> tuple[str, str, int, dict]:
        nose = self._point(pose, 0, bbox)
        if self._face_pair is None:
            # Prefer the wider ear pair, then fall back to the eye centers.  Do
            # not switch pairs while a valid history is accumulating.
            for pair in ((7, 8), (2, 5)):
                if all(self._point(pose, i, bbox) is not None for i in pair):
                    self._face_pair = pair
                    break
        pair_points = [self._point(pose, i, bbox) for i in self._face_pair] if self._face_pair else []
        if nose is None or len(pair_points) != 2 or any(p is None for p in pair_points):
            self._reset_head()
            return "unknown", "head_unavailable", 0, {}
        axis = pair_points[1] - pair_points[0]
        width = float(np.linalg.norm(axis))
        if width < 2 or (self._scale and not 0.04 * self._scale <= width <= 1.3 * self._scale):
            self._reset_head()
            return "unknown", "head_geometry_invalid", 0, {}
        ratio = float(np.dot(nose - (pair_points[0] + pair_points[1]) / 2, axis) / (width * width))
        if abs(ratio) > 1.5:
            self._reset_head()
            return "unknown", "head_geometry_invalid", 0, {}
        if self._head_last is not None:
            dt, distance = t - self._head_last[0], abs(ratio - self._head_last[1])
            if dt > 0 and distance > 0.35 and distance / dt > 8:
                self._reset_head()
                return "unknown", "head_position_jump", 0, {}
        self._head_last = (t, ratio)
        metrics = {"head_turn_ratio": ratio, "face_pair": "ears" if self._face_pair == (7, 8) else "eyes"}
        if self._neutral is None:
            self._head_calibration.append((t, ratio))
            if t - self._head_calibration[0][0] + 1e-9 < self.config.head_calibration_seconds:
                return "calibrating", "head_neutral_calibration", 0, metrics
            values = [v for _, v in self._head_calibration]
            # A moving initial head does not establish a reliable neutral.
            if np.ptp(values) > self.config.shake_threshold:
                self._head_calibration = [(t, ratio)]
                return "calibrating", "head_neutral_not_stable", 0, metrics
            self._neutral = float(np.median(values))
            self._head_calibration.clear()
        delta = ratio - self._neutral
        metrics["head_turn_from_neutral"] = delta
        side = 1 if delta >= self.config.shake_threshold else -1 if delta <= -self.config.shake_threshold else 0
        while self._turns and t - self._turns[0][0] > self.config.window_seconds + 1e-9:
            self._turns.popleft()
        if not self._turns and side == 0:
            # After an old episode has expired and the head returns to neutral,
            # a new sequence may start on either side.  Holding one extreme
            # does not manufacture a new turn when the time window advances.
            self._head_side = 0
        if side and side != self._head_side:
            if not self._turns or t - self._turns[-1][0] + 1e-9 >= self.config.shake_min_interval:
                self._turns.append((t, side))
                self._head_side = side
        count = len(self._turns)
        state = "active" if count >= self.config.shake_min_turns else "normal"
        return state, "alternating_head_turns" if state == "active" else "insufficient_head_turns", count, metrics

    def _hand(self, name: str, point: np.ndarray | None, t: float) -> tuple[str, str, dict]:
        history = self._hands[name]
        if point is None:
            history.clear()
            return "unknown", "wrist_or_shoulders_unavailable", {}
        if history:
            dt = t - history[-1][0]
            step = float(np.linalg.norm(point - history[-1][1]))
            if dt > self.config.max_gap_seconds + 1e-9 or (step > 0.35 and step / dt > 6):
                history.clear()
                return "unknown", "wrist_position_jump_or_gap", {}
        history.append((t, point.copy()))
        while history and t - history[0][0] > self.config.window_seconds + 1e-9:
            history.popleft()
        duration = t - history[0][0]
        metrics = {"valid_seconds": duration, "valid_samples": len(history)}
        if duration + 1e-9 < self.config.wave_min_seconds or len(history) < self.config.wave_min_samples:
            return "calibrating", "collecting_wrist_motion", metrics
        times = np.asarray([entry[0] for entry in history])
        points = np.asarray([entry[1] for entry in history])
        differences = np.diff(times)
        interval = float(np.median(differences))
        # A frequency near Nyquist is indistinguishable from aliasing.  Require
        # at least four source observations per requested minimum cycle.
        if interval <= 0 or np.max(differences) > 1 / (4 * self.config.wave_min_hz) + 1e-9:
            return "unknown", "sampling_too_slow", metrics
        uniform_times = np.linspace(times[0], times[-1], max(4, int(round(duration / interval)) + 1))
        uniform_points = np.column_stack([np.interp(uniform_times, times, points[:, i]) for i in range(2)])
        # Symmetric three-sample smoothing, without delayed timestamps or any
        # interpolation over missing observations (those clear the history).
        if len(uniform_times) >= 5:
            uniform_points = (uniform_points[:-2] + uniform_points[1:-1] + uniform_points[2:]) / 3
            uniform_times = uniform_times[1:-1]
        centered = uniform_points - np.mean(uniform_points, axis=0)
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
        if not len(singular) or singular[0] < 1e-8:
            metrics.update(amplitude=0.0, frequency_hz=0.0, cycles=0)
            return "normal", "wrist_stationary", metrics
        signal = centered @ axes[0]
        amplitude = float(np.percentile(signal, 95) - np.percentile(signal, 5))
        metrics["amplitude"] = amplitude
        if not self.config.wave_min_amplitude <= amplitude <= self.config.wave_max_amplitude:
            return "normal", "wrist_amplitude_outside_range", metrics
        sample_interval = float(np.median(np.diff(uniform_times)))
        prominence = self.config.wave_min_amplitude * 0.65
        # Suppress isolated wiggles; do not impose a maximum movement frequency.
        distance = max(1, int(round(0.06 / sample_interval)))
        peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
        troughs, _ = find_peaks(-signal, prominence=prominence, distance=distance)
        extrema = sorted([(int(i), 1) for i in peaks] + [(int(i), -1) for i in troughs])
        alternating: list[tuple[int, int]] = []
        for index, sign in extrema:
            if alternating and alternating[-1][1] == sign:
                previous = alternating[-1][0]
                if sign * signal[index] > sign * signal[previous]:
                    alternating[-1] = (index, sign)
            else:
                alternating.append((index, sign))
        cycles = max(0, (len(alternating) - 1) // 2)
        hz = 0.0
        elapsed = 0.0
        if len(alternating) >= 3:
            elapsed = uniform_times[alternating[-1][0]] - uniform_times[alternating[0][0]]
            hz = (len(alternating) - 1) / (2 * elapsed) if elapsed > 0 else 0.0
        metrics.update(frequency_hz=float(hz), cycles=cycles, activity_seconds=float(elapsed))
        active = (cycles >= self.config.wave_min_cycles
                  and hz + 1e-9 >= self.config.wave_min_hz
                  and elapsed + 1e-9 >= self.config.wave_min_seconds)
        return ("active", "fast_small_repeated_wrist_motion", metrics) if active else ("normal", "insufficient_wrist_cycles_or_frequency", metrics)
