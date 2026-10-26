"""Stream source-time behavior samples and intervals without saving video frames.

The timeline is *point sampled*: the first actual update in each source-time
bucket is written. Its state describes ``sample_time_seconds``, not every instant
of the surrounding bucket. Seeking over a bucket never invents a row. The event
file complements these samples by recording every observed active interval.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import operator
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


BEHAVIORS = (
    "head_shake", "left_wave", "right_wave", "both_wave", "table_rest", "seat_leave"
)
VALID_STATES = frozenset({"active", "normal", "unknown", "calibrating", "candidate"})
TIMELINE_FIELDS = (
    "video", "interval_start_seconds", "interval_end_seconds", "sample_time_seconds",
    "target_id", "selection_generation", *BEHAVIORS, "metrics_json",
)
EVENT_FIELDS = (
    "video", "behavior", "target_id", "selection_generation", "start_seconds",
    "confirmation_seconds", "end_seconds", "duration_seconds", "last_active_seconds",
    "end_reason", "end_observed", "terminal_state", "start_metrics_json",
    "end_metrics_json",
)


def _json_clean(value: Any) -> Any:
    """Make diagnostics JSON compatible; nonfinite measurements stay missing."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    # NumPy scalars are accepted without making NumPy a logger dependency.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_clean(item_method())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(_json_clean(value), ensure_ascii=False, allow_nan=False, sort_keys=True)


def _safe_stem(stem: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")[:100]
    if not stem:
        stem = "video"
    if stem.split(".")[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        stem = "_" + stem
    return stem


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _seconds(timestamp_ms: float) -> str:
    return f"{timestamp_ms / 1000.0:.6f}"


class TimeCsvLogger:
    """Write one run's timeline, concurrent events and interpretation metadata.

    ``update`` and ``boundary`` require nondecreasing original-video timestamps.
    Active events start at their first confirmed ``active`` observation; candidate
    observations are never silently backdated. An event ends at the first other
    state. ``end_observed`` is 1 only when that state is a valid ``normal`` result.
    Unknowns, resets and termination provide an observation boundary, not proof of
    the true action ending. ``last_active_seconds`` preserves the last evidence.

    A source-time gap greater than ``max_gap_seconds`` closes existing events at
    the preceding actual observation and starts new events if activity resumes.
    The default gap tolerance is twice the timeline interval.
    """

    def __init__(
        self,
        output_dir: Path,
        video_path: Path,
        interval_seconds: float = 1.0,
        metadata: dict | None = None,
        *,
        max_gap_seconds: float | None = None,
    ) -> None:
        self.interval_seconds = float(interval_seconds)
        if not math.isfinite(self.interval_seconds) or self.interval_seconds < 0.001:
            raise ValueError("interval_seconds must be finite and at least 0.001")
        self.max_gap_seconds = float(
            max_gap_seconds if max_gap_seconds is not None else 2 * self.interval_seconds
        )
        if not math.isfinite(self.max_gap_seconds) or self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be finite and positive")
        self.output_dir = Path(output_dir)
        self.video_path = Path(video_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._last_timestamp_ms: int | None = None
        self._last_update_ms: int | None = None
        self._last_bucket: int | None = None
        self._last_identity: tuple[int | None, int] | None = None
        self._last_metrics: dict = {}
        self._active: dict[str, dict] = {}
        self.timeline_row_count = 0
        self.event_count = 0
        self._created_utc = datetime.now(timezone.utc)
        self._metadata = _json_clean(metadata or {})
        self._open_run_files()
        self._timeline_writer = csv.DictWriter(self._timeline_file, fieldnames=TIMELINE_FIELDS)
        self._event_writer = csv.DictWriter(self._events_file, fieldnames=EVENT_FIELDS)
        try:
            self._timeline_writer.writeheader()
            self._event_writer.writeheader()
            self._timeline_file.flush()
            self._events_file.flush()
            self._write_metadata()
        except BaseException:
            self._timeline_file.close()
            self._events_file.close()
            self._closed = True
            raise

    def _open_run_files(self) -> None:
        """Exclusive creation prevents overwriting a preceding run, even on collision."""
        stamp = self._created_utc.strftime("%Y%m%dT%H%M%S_%fZ")
        for _ in range(100):
            prefix = f"{_safe_stem(self.video_path.stem)}_{stamp}_{uuid4().hex[:8]}"
            self.timeline_path = self.output_dir / f"{prefix}_timeline.csv"
            self.events_path = self.output_dir / f"{prefix}_events.csv"
            self.metadata_path = self.output_dir / f"{prefix}_metadata.json"
            created: list[Path] = []
            handles = []
            try:
                self._timeline_file = self.timeline_path.open("x", encoding="utf-8-sig", newline="")
                created.append(self.timeline_path)
                handles.append(self._timeline_file)
                self._events_file = self.events_path.open("x", encoding="utf-8-sig", newline="")
                created.append(self.events_path)
                handles.append(self._events_file)
                with self.metadata_path.open("x", encoding="utf-8"):
                    created.append(self.metadata_path)
                return
            except BaseException as error:
                for handle in handles:
                    handle.close()
                # Only these exact files, created by this attempt, may be removed.
                for path in created:
                    path.unlink(missing_ok=True)
                if not isinstance(error, FileExistsError):
                    raise
        raise FileExistsError("Could not reserve a unique CSV run filename")

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def paths(self) -> dict[str, Path]:
        return {"timeline": self.timeline_path, "events": self.events_path, "metadata": self.metadata_path}

    def _write_metadata(self, *, reason: str | None = None) -> None:
        content = {
            "schema_version": 1,
            "video_path": str(self.video_path.resolve()),
            "created_utc": self._created_utc.isoformat(),
            "time_basis": "original source-video seconds; no wall-clock duration",
            "timeline_semantics": "First actual observation per source-time bucket; states apply only at sample_time_seconds. Missing buckets are not invented. Short actions may occur between samples; see events.csv.",
            "event_semantics": "Start and confirmation are first observed active. End is the first non-active observation or a boundary; end_observed=1 only for a normal observation. last_active_seconds records the last positive evidence. Durations are observation intervals, not validated ground truth.",
            "interval_seconds": self.interval_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "behavior_keys": BEHAVIORS,
            "states": sorted(VALID_STATES),
            "timeline_rows": self.timeline_row_count,
            "events": self.event_count,
            "last_source_seconds": None if self._last_timestamp_ms is None else self._last_timestamp_ms / 1000,
            "closed": reason is not None,
            "close_reason": reason,
            "settings": self._metadata,
        }
        with self.metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(content, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")

    def _timestamp(self, timestamp_ms: int) -> int:
        value = _integer(timestamp_ms, "timestamp_ms")
        if self._last_timestamp_ms is not None and value < self._last_timestamp_ms:
            raise ValueError("CSV timestamps must not move backwards")
        return value

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CSV logger is closed")

    def _end_event(
        self, behavior: str, timestamp_ms: int, reason: str,
        terminal_state: str, end_observed: bool, metrics: dict,
    ) -> None:
        event = self._active.pop(behavior)
        self._event_writer.writerow({
            "video": self.video_path.name,
            "behavior": behavior,
            "target_id": "" if event["target_id"] is None else event["target_id"],
            "selection_generation": event["generation"],
            "start_seconds": _seconds(event["start_ms"]),
            "confirmation_seconds": _seconds(event["start_ms"]),
            "end_seconds": _seconds(timestamp_ms),
            "duration_seconds": _seconds(max(0, timestamp_ms - event["start_ms"])),
            "last_active_seconds": _seconds(event["last_active_ms"]),
            "end_reason": reason,
            "end_observed": int(end_observed),
            "terminal_state": terminal_state,
            "start_metrics_json": event["start_metrics"],
            "end_metrics_json": _json(metrics),
        })
        self.event_count += 1
        self._events_file.flush()

    def _end_all(self, timestamp_ms: int, reason: str, metrics: dict) -> None:
        for behavior in tuple(self._active):
            self._end_event(behavior, timestamp_ms, reason, "unknown", False, metrics)

    def update(
        self,
        timestamp_ms: int,
        target_id: int | None,
        selection_generation: int,
        statuses: dict[str, str],
        metrics: dict | None = None,
    ) -> None:
        self._require_open()
        timestamp_ms = self._timestamp(timestamp_ms)
        generation = _integer(selection_generation, "selection_generation")
        if target_id is not None:
            target_id = _integer(target_id, "target_id")
        if set(statuses) - set(BEHAVIORS):
            raise ValueError(f"Unknown behavior keys: {sorted(set(statuses) - set(BEHAVIORS))}")
        current = {behavior: statuses.get(behavior, "unknown") for behavior in BEHAVIORS}
        if any(state not in VALID_STATES for state in current.values()):
            raise ValueError("Behavior states must be active, normal, unknown, calibrating or candidate")
        clean_metrics = _json_clean(metrics or {})
        # A jump in source time cannot prove continuous behavior across unseen frames.
        if self._last_update_ms is not None and timestamp_ms - self._last_update_ms > self.max_gap_seconds * 1000:
            self._end_all(self._last_update_ms, "source_gap", self._last_metrics)
        identity = (target_id, generation)
        if self._last_identity is not None and identity != self._last_identity:
            self._end_all(timestamp_ms, "target_change", clean_metrics)
        for behavior, state in current.items():
            if state == "active":
                if behavior not in self._active:
                    self._active[behavior] = {
                        "target_id": target_id,
                        "generation": generation,
                        "start_ms": timestamp_ms,
                        "last_active_ms": timestamp_ms,
                        "start_metrics": _json(clean_metrics),
                    }
                else:
                    self._active[behavior]["last_active_ms"] = timestamp_ms
            elif behavior in self._active:
                self._end_event(behavior, timestamp_ms, state, state, state == "normal", clean_metrics)

        bucket_ms = self.interval_seconds * 1000
        bucket = math.floor(timestamp_ms / bucket_ms)
        if bucket != self._last_bucket:
            self._timeline_writer.writerow({
                "video": self.video_path.name,
                "interval_start_seconds": _seconds(bucket * bucket_ms),
                "interval_end_seconds": _seconds((bucket + 1) * bucket_ms),
                "sample_time_seconds": _seconds(timestamp_ms),
                "target_id": "" if target_id is None else target_id,
                "selection_generation": generation,
                **current,
                "metrics_json": _json(clean_metrics),
            })
            self.timeline_row_count += 1
            self._timeline_file.flush()
            self._last_bucket = bucket
        self._last_identity = identity
        self._last_timestamp_ms = timestamp_ms
        self._last_update_ms = timestamp_ms
        self._last_metrics = clean_metrics

    def boundary(self, timestamp_ms: int, reason: str) -> None:
        """Close events before manual reset, target change, seek or other boundary."""
        self._require_open()
        timestamp_ms = self._timestamp(timestamp_ms)
        self._end_all(timestamp_ms, str(reason), self._last_metrics)
        self._last_identity = None
        self._last_timestamp_ms = timestamp_ms

    def close(self, timestamp_ms: int | None = None, reason: str = "end_video") -> None:
        if self._closed:
            return
        if timestamp_ms is None:
            timestamp_ms = self._last_timestamp_ms if self._last_timestamp_ms is not None else 0
        timestamp_ms = self._timestamp(timestamp_ms)
        try:
            self._end_all(timestamp_ms, str(reason), self._last_metrics)
            self._last_timestamp_ms = timestamp_ms
            self._write_metadata(reason=str(reason))
        finally:
            self._timeline_file.close()
            self._events_file.close()
            self._closed = True

    def __enter__(self) -> "TimeCsvLogger":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        reason = "interrupted" if exc_type is not None and issubclass(exc_type, KeyboardInterrupt) else "error" if exc_type else "end_video"
        self.close(reason=reason)
        return False
