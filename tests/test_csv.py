from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from behavior_csv import BEHAVIORS, TimeCsvLogger


class TimeCsvLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "受試者中文影片.mp4"
        self.video.write_bytes(b"immutable test source")
        self.logs = self.root / "output" / "logs"

    def logger(self, **kwargs):
        logger = TimeCsvLogger(self.logs, self.video, **kwargs)
        self.addCleanup(logger.close)
        return logger

    @staticmethod
    def read(path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def states(**overrides):
        return {**{behavior: "normal" for behavior in BEHAVIORS}, **overrides}

    def test_black_prefix_logs_unknown_at_time_zero_without_events(self):
        logger = self.logger()
        for timestamp in range(0, 3001, 100):
            logger.update(timestamp, None, 0, {})
        logger.close()
        rows = self.read(logger.timeline_path)
        self.assertEqual([float(row["sample_time_seconds"]) for row in rows], [0, 1, 2, 3])
        self.assertEqual(rows[0]["target_id"], "")
        self.assertTrue(all(row[key] == "unknown" for row in rows for key in BEHAVIORS))
        self.assertEqual(self.read(logger.events_path), [])

    def test_30_fps_writes_one_actual_point_sample_per_source_second(self):
        logger = self.logger()
        timestamps = [round(frame * 1000 / 30) for frame in range(91)]
        for timestamp in timestamps:
            logger.update(timestamp, 7, 1, self.states())
        logger.close()
        rows = self.read(logger.timeline_path)
        self.assertEqual(len(rows), 4)
        self.assertEqual([float(row["interval_start_seconds"]) for row in rows], [0, 1, 2, 3])
        self.assertTrue(all(round(float(row["sample_time_seconds"]) * 1000) in timestamps for row in rows))

    def test_nonzero_start_and_seek_do_not_fabricate_rows(self):
        logger = self.logger()
        for timestamp in [8450, 8480, 9120, 20100]:
            logger.update(timestamp, 1, 1, self.states())
        logger.close()
        rows = self.read(logger.timeline_path)
        self.assertEqual([float(row["interval_start_seconds"]) for row in rows], [8, 9, 20])
        self.assertEqual([float(row["sample_time_seconds"]) for row in rows], [8.45, 9.12, 20.1])

    def test_short_event_between_timeline_samples_is_preserved(self):
        logger = self.logger()
        logger.update(0, 1, 1, self.states())
        logger.update(100, 1, 1, self.states(head_shake="candidate"))
        logger.update(300, 1, 1, self.states(head_shake="active"))
        logger.update(500, 1, 1, self.states(head_shake="active"))
        logger.update(700, 1, 1, self.states(head_shake="normal"))
        logger.close()
        self.assertEqual(len(self.read(logger.timeline_path)), 1)
        event = self.read(logger.events_path)[0]
        self.assertEqual(float(event["start_seconds"]), 0.3)
        self.assertEqual(float(event["confirmation_seconds"]), 0.3)
        self.assertEqual(float(event["duration_seconds"]), 0.4)
        self.assertEqual(float(event["last_active_seconds"]), 0.5)
        self.assertEqual(event["end_reason"], "normal")
        self.assertEqual(event["end_observed"], "1")

    def test_concurrent_behaviors_have_independent_intervals(self):
        logger = self.logger()
        logger.update(0, 1, 1, self.states(left_wave="active", right_wave="active", both_wave="active"))
        logger.update(500, 1, 1, self.states(left_wave="active", table_rest="active"))
        logger.update(1500, 1, 1, self.states(seat_leave="active"))
        logger.close(1800)
        events = {row["behavior"]: row for row in self.read(logger.events_path)}
        self.assertEqual(set(events), {"left_wave", "right_wave", "both_wave", "table_rest", "seat_leave"})
        self.assertEqual(float(events["left_wave"]["duration_seconds"]), 1.5)
        self.assertEqual(float(events["right_wave"]["duration_seconds"]), 0.5)
        self.assertEqual(float(events["table_rest"]["start_seconds"]), 0.5)
        self.assertEqual(events["seat_leave"]["end_reason"], "end_video")

    def test_unknown_ends_event_without_claiming_observed_end(self):
        logger = self.logger()
        logger.update(100, 1, 1, self.states(table_rest="active"))
        logger.update(250, 1, 1, self.states(table_rest="active"))
        logger.update(300, 1, 1, self.states(table_rest="unknown"))
        logger.update(400, 1, 1, self.states(table_rest="unknown"))
        logger.close()
        rows = self.read(logger.events_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["end_reason"], "unknown")
        self.assertEqual(rows[0]["end_observed"], "0")
        self.assertEqual(float(rows[0]["last_active_seconds"]), 0.25)

    def test_source_gap_ends_at_last_evidence_then_new_active_event(self):
        logger = self.logger()
        logger.update(0, 1, 1, self.states(head_shake="active"))
        logger.update(1000, 1, 1, self.states(head_shake="active"))
        logger.update(6000, 1, 1, self.states(head_shake="active"))
        logger.update(6500, 1, 1, self.states())
        logger.close()
        rows = self.read(logger.events_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["end_reason"], "source_gap")
        self.assertEqual(float(rows[0]["end_seconds"]), 1)
        self.assertEqual(float(rows[1]["start_seconds"]), 6)
        self.assertEqual(float(rows[1]["duration_seconds"]), 0.5)

    def test_target_and_generation_changes_cannot_join_events(self):
        logger = self.logger()
        logger.update(0, 1, 1, self.states(left_wave="active"))
        logger.update(500, 2, 1, self.states(left_wave="active"))
        logger.update(1000, 2, 2, self.states(left_wave="active"))
        logger.close(1500)
        rows = self.read(logger.events_path)
        self.assertEqual([row["target_id"] for row in rows], ["1", "2", "2"])
        self.assertEqual([row["selection_generation"] for row in rows], ["1", "1", "2"])
        self.assertEqual([row["end_reason"] for row in rows], ["target_change", "target_change", "end_video"])
        self.assertTrue(all(float(row["duration_seconds"]) == 0.5 for row in rows))

    def test_manual_boundary_closes_before_reacquisition_even_same_id(self):
        logger = self.logger()
        logger.update(100, 1, 1, self.states(right_wave="active"))
        logger.boundary(500, "manual_reset")
        logger.update(500, 1, 1, self.states(right_wave="active"))
        logger.close(800, "quit")
        rows = self.read(logger.events_path)
        self.assertEqual([row["end_reason"] for row in rows], ["manual_reset", "quit"])
        self.assertEqual([float(row["start_seconds"]) for row in rows], [0.1, 0.5])
        self.assertEqual(len(self.read(logger.timeline_path)), 1)

    def test_context_error_flushes_and_propagates_original_error(self):
        logger = self.logger()
        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            with logger:
                logger.update(0, 1, 1, self.states(seat_leave="active"))
                raise RuntimeError("expected failure")
        self.assertTrue(logger.closed)
        self.assertEqual(self.read(logger.events_path)[0]["end_reason"], "error")
        self.assertEqual(json.loads(logger.metadata_path.read_text(encoding="utf-8"))["close_reason"], "error")

    def test_keyboard_interrupt_closes_and_propagates(self):
        logger = self.logger()
        with self.assertRaises(KeyboardInterrupt):
            with logger:
                logger.update(100, 1, 1, self.states(head_shake="active"))
                raise KeyboardInterrupt()
        self.assertEqual(self.read(logger.events_path)[0]["end_reason"], "interrupted")

    def test_chinese_names_bom_metrics_and_source_unchanged(self):
        before = hashlib.sha256(self.video.read_bytes()).hexdigest()
        logger = self.logger(metadata={"說明": "測試", "infinite": float("inf")})
        logger.update(0, 1, 1, self.states(left_wave="active"), {"距離": float("nan"), "方向": "左右"})
        logger.close(100)
        self.assertTrue(logger.timeline_path.name.startswith("受試者中文影片_"))
        self.assertTrue(logger.timeline_path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertTrue(logger.events_path.read_bytes().startswith(b"\xef\xbb\xbf"))
        row = self.read(logger.timeline_path)[0]
        self.assertEqual(json.loads(row["metrics_json"]), {"距離": None, "方向": "左右"})
        self.assertEqual(json.loads(logger.metadata_path.read_text(encoding="utf-8"))["settings"], {"說明": "測試", "infinite": None})
        self.assertEqual(hashlib.sha256(self.video.read_bytes()).hexdigest(), before)
        self.assertEqual({path.suffix for path in self.logs.iterdir()}, {".csv", ".json"})

    def test_same_source_creates_distinct_exclusive_run_files(self):
        first, second = self.logger(), self.logger()
        first.close()
        original = first.timeline_path.read_bytes()
        second.update(0, 1, 1, self.states())
        second.close()
        self.assertNotEqual(first.timeline_path, second.timeline_path)
        self.assertEqual(first.timeline_path.read_bytes(), original)
        self.assertEqual(len(list(self.logs.iterdir())), 6)

    def test_close_is_idempotent_and_closed_logger_rejects_updates(self):
        logger = self.logger()
        logger.update(100, 1, 1, self.states(head_shake="active"))
        logger.close()
        original = logger.events_path.read_bytes()
        logger.close(99999, "error")
        self.assertEqual(logger.events_path.read_bytes(), original)
        with self.assertRaises(RuntimeError):
            logger.update(200, 1, 1, self.states())

    def test_invalid_states_times_and_parameters_do_not_corrupt_run(self):
        logger = self.logger()
        logger.update(100, 1, 1, self.states())
        with self.assertRaises(ValueError):
            logger.update(99, 1, 1, self.states())
        with self.assertRaises(ValueError):
            logger.update(200, 1, 1, {"head_shake": "yes"})
        with self.assertRaises(ValueError):
            logger.update(200, 1, 1, {"unrecognized": "active"})
        with self.assertRaises(ValueError):
            logger.update(float("nan"), 1, 1, self.states())
        logger.update(200, 1, 1, self.states())
        logger.close()
        self.assertEqual(len(self.read(logger.timeline_path)), 1)
        for interval in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                TimeCsvLogger(self.logs, self.video, interval_seconds=interval)
        with self.assertRaises(ValueError):
            TimeCsvLogger(self.logs, self.video, max_gap_seconds=0)


if __name__ == "__main__":
    unittest.main()
