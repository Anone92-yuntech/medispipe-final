"""Original OC-SORT integration with explicit observed/predicted track metadata.

The vendored tracker is the official noahcao/OC_SORT implementation. It is
deliberately left unchanged; provenance and hashes are in vendor/ocsort.
Only the adapter sanitises detector output and adds application metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Sequence

import numpy as np

from target_selection import TrackObservation
from vendor.ocsort.trackers.ocsort_tracker.ocsort import OCSort


OCSORT_REPOSITORY = "https://github.com/noahcao/OC_SORT"
OCSORT_COMMIT = "8462e7e729a93ccd3bd995c0a79a890336cb3a0b"
OCSORT_VERSION = f"official OC-SORT {OCSORT_COMMIT}"


@dataclass
class _ObservationHistory:
    observed_count: int
    first_seen_ms: int
    last_seen_ms: int


class OCSortTracker:
    """Track original-pixel ``[x1,y1,x2,y2,confidence]`` person detections.

    Returned records include *all live tracklets*, even unconfirmed and missed
    ones, so the UI can distinguish waiting, observed, and predicted states.
    ``observed`` is true only when the current frame supplied a detection to
    that tracklet. Predictions must never be sent to pose/selection as new
    person observations. Confirmation for selection is handled independently
    with ``observed_count`` and elapsed source-video time.

    A separate instance is created for each video. Upstream OC-SORT uses a
    class-wide ID counter; instances should not process videos concurrently.
    """

    def __init__(
        self,
        fps: float,
        max_age_seconds: float = 1.0,
        min_hits: int = 3,
        det_threshold: float = 0.35,
        iou_threshold: float = 0.3,
    ) -> None:
        if not isfinite(float(fps)) or fps <= 0:
            raise ValueError("fps must be finite and greater than zero")
        if not isfinite(float(max_age_seconds)) or max_age_seconds < 0:
            raise ValueError("max_age_seconds must be finite and non-negative")
        if isinstance(min_hits, bool) or int(min_hits) != min_hits or min_hits < 1:
            raise ValueError("min_hits must be a positive integer")
        if not isfinite(float(det_threshold)) or not 0 <= det_threshold < 1:
            raise ValueError("det_threshold must be in [0, 1)")
        if not isfinite(float(iou_threshold)) or not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.fps = float(fps)
        self.max_age_seconds = float(max_age_seconds)
        self.min_hits = int(min_hits)
        self.det_threshold = float(det_threshold)
        self.iou_threshold = float(iou_threshold)
        self.reset()

    def reset(self) -> None:
        """Clear tracklets, per-ID evidence, and timestamp history."""
        self.tracker = OCSort(
            det_thresh=self.det_threshold,
            max_age=max(0, ceil(self.max_age_seconds * self.fps)),
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            use_byte=False,
        )
        self._history: dict[int, _ObservationHistory] = {}
        self._last_timestamp_ms: int | None = None

    @property
    def frame_count(self) -> int:
        return int(self.tracker.frame_count)

    def update(
        self,
        detections: np.ndarray,
        frame_shape: Sequence[int],
        timestamp_ms: int,
    ) -> list[TrackObservation]:
        """Advance exactly one frame, including frames with zero detections.

        ``frame_shape`` accepts ``(height,width)`` or an image's full shape.
        Source-video timestamps must be strictly increasing within this run.
        This guards duplicate observations during UI pauses. ``None`` is not
        forwarded upstream because upstream returns early without aging.
        """
        if len(frame_shape) < 2:
            raise ValueError("frame_shape requires height and width")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("frame dimensions must be positive")
        timestamp_value = float(timestamp_ms)
        if not isfinite(timestamp_value) or timestamp_value < 0:
            raise ValueError("timestamp_ms must be finite and non-negative")
        timestamp = int(timestamp_value)
        if self._last_timestamp_ms is not None and timestamp <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must strictly increase for each processed frame")

        if detections is None:
            values = np.empty((0, 5), dtype=np.float64)
        else:
            values = np.asarray(detections, dtype=np.float64)
            if values.size == 0:
                values = np.empty((0, 5), dtype=np.float64)
            elif values.ndim != 2 or values.shape[1] != 5:
                raise ValueError("detections must have shape (N,5): xyxy + confidence")
            else:
                # Official update rescales its input array in place; do not
                # modify the detector's data or any external cached boxes.
                values = values.copy()
                valid = np.isfinite(values).all(axis=1)
                values = values[valid]
                values[:, [0, 2]] = np.clip(values[:, [0, 2]], 0, width)
                values[:, [1, 3]] = np.clip(values[:, [1, 3]], 0, height)
                valid = (
                    (values[:, 2] > values[:, 0])
                    & (values[:, 3] > values[:, 1])
                    & (values[:, 4] >= 0)
                    & (values[:, 4] <= 1)
                )
                values = values[valid]

        # Equal source/inference sizes make upstream's scale exactly 1.
        # This call MUST also occur on black/no-person frames.
        self.tracker.update(values, (height, width), (height, width))
        self._last_timestamp_ms = timestamp

        result: list[TrackObservation] = []
        live_ids: set[int] = set()
        for tracklet in self.tracker.trackers:
            track_id = int(tracklet.id) + 1  # official returned IDs are 1-based
            live_ids.add(track_id)
            observed = tracklet.time_since_update == 0
            history = self._history.get(track_id)
            if observed:
                if history is None:
                    history = _ObservationHistory(1, timestamp, timestamp)
                    self._history[track_id] = history
                else:
                    history.observed_count += 1
                    history.last_seen_ms = timestamp
            if history is None:
                # A tracklet is created from an actual detection, hence this
                # branch should never be reached in the official implementation.
                continue

            last_observation = np.asarray(tracklet.last_observation)
            if observed and last_observation.sum() >= 0:
                bbox = last_observation[:4]
            else:
                # New tracklets have a placeholder last_observation even
                # though their initial Kalman state comes from a real box.
                # Missed tracks use a prediction for diagnostics only.
                bbox = np.asarray(tracklet.get_state()[0])[:4]
            if not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            result.append(
                TrackObservation(
                    track_id=track_id,
                    bbox=np.asarray(bbox, dtype=np.float64).copy(),
                    observed=bool(observed),
                    observed_count=history.observed_count,
                    first_seen_ms=history.first_seen_ms,
                    last_seen_ms=history.last_seen_ms,
                )
            )

        # OC-SORT permanently removes stale tracklets. Counters can be freed;
        # an unrelated future detection receives a new ID rather than inheriting
        # the vanished target's evidence.
        for expired_id in self._history.keys() - live_ids:
            del self._history[expired_id]
        return sorted(result, key=lambda item: item.track_id)
