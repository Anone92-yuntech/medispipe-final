"""Experiment orchestration with source-time CSVs and no native GUI/model use."""

from contextlib import redirect_stderr, redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import experiment
from behavior_csv import TimeCsvLogger
from behavior_engine import BehaviorFrame
from motion_detection import MotionConfig, MotionResult
from posture_detection import PostureResult
from seat_detection import SeatConfig, SeatResult
from target_selection import MappedPose, TrackObservation
from tracking_pipeline import FrameAnalysis, PipelineConfig


def subject_analysis(timestamp, generation=1):
    points = np.full((33, 2), [300, 180], dtype=float)
    points[11:17] = [[250, 240], [350, 240], [220, 320], [380, 320], [270, 370], [330, 370]]
    pose = MappedPose(points, np.ones(33), np.ones(33))
    track = TrackObservation(1, np.array([180, 100, 420, 460]), True, 100, 0, timestamp)
    posture = PostureResult('resting', 'suspected_table_rest', .4, .2, 2.0,
                            20, 150, 100)
    return FrameAnalysis([track], 1, pose, 'locked', 'only_qualified_candidate',
                         {1: {'median_width': 100}}, timestamp,
                         selection_generation=generation, posture=posture)


class FakeCapture:
    def __init__(self, frame_count=2, *, read_error=None, opened=True, fps=10):
        self.frame_count = frame_count
        self.read_error = read_error
        self.opened = opened
        self.fps = fps
        self.read_calls = 0
        self.released = False
        self.seek_calls = []

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return max(0, self.read_calls - 1) * 100
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return 0
        return 0

    def set(self, prop, value):
        self.seek_calls.append((prop, value))
        return True

    def read(self):
        self.read_calls += 1
        if self.read_calls > self.frame_count:
            if self.read_error is not None:
                raise self.read_error
            return False, None
        return True, np.full((540, 640, 3), [80, 90, 100], dtype=np.uint8)

    def release(self):
        self.released = True


class FakePipeline:
    instances = []

    def __init__(self, detector, model, fps, config):
        self.selection_generation = 1
        self.auto_reset_count = 0
        self.pose_recovery_count = 0
        self.processed = []
        self.resets = 0
        self.closed = False
        self.exit_type = None
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        self.exit_type = exc_type

    def process(self, frame, timestamp, frame_index):
        self.processed.append((timestamp, frame_index))
        return subject_analysis(timestamp, self.selection_generation)

    def reset_selection(self):
        self.selection_generation += 1
        self.resets += 1


class FakeDetector:
    provenance = {'name': 'fake-person-detector', 'source': 'unit-test'}
    input_size = 416


class AppConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Path(self.temp.name) / '設定.json'
        self.write_settings({})

    def write_settings(self, content):
        self.settings.write_text(json.dumps(content), encoding='utf-8-sig')

    def parse(self, *arguments):
        return experiment.parse_configuration(['--settings', str(self.settings), *arguments])

    def assert_invalid(self, *arguments):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            self.parse(*arguments)
        self.assertEqual(raised.exception.code, 2)

    def test_default_head_definition_and_target_retention(self):
        args, pipeline, motion, seat = self.parse()
        self.assertEqual(motion.shake_min_turns, 4)
        self.assertEqual(motion.window_seconds, 2.0)
        self.assertTrue(pipeline.retain_target_on_loss)
        self.assertEqual(args.csv_interval, 1.0)
        self.assertEqual(seat.departure_ratio, 1.0)

    def test_settings_then_cli_precedence_and_path_conversion(self):
        self.write_settings({'shake_min_turns': 5, 'wave_min_hz': 3,
                             'seat_hold_seconds': 2, 'output': str(Path(self.temp.name) / '記錄'),
                             'reselect_on_loss': False, 'seat_roi': [100, 120, 400, 500]})
        args, pipeline, motion, seat = self.parse('--shake-min-turns', '6', '--wave-min-hz', '2.5',
                                                  '--reselect-on-loss', '--seat-hold-seconds', '1.2')
        self.assertEqual(motion.shake_min_turns, 6)
        self.assertEqual(motion.wave_min_hz, 2.5)
        self.assertEqual(seat.hold_seconds, 1.2)
        self.assertIsInstance(args.output, Path)
        self.assertEqual(args.seat_roi, [100.0, 120.0, 400.0, 500.0])
        self.assertFalse(pipeline.retain_target_on_loss)

    def test_bad_configuration_is_rejected_before_loading_models(self):
        for args in (('--csv-interval', '0'), ('--csv-interval', '0.0001'),
                     ('--csv-interval', 'nan'), ('--start-seconds', '-1'),
                     ('--max-frames', '0'), ('--sample-stride', '0'),
                     ('--min-samples', '1'), ('--landmark-confidence', '1.1'),
                     ('--wave-min-amplitude', '.5'), ('--shake-min-turns', '1'),
                     ('--seat-hold-seconds', '-1'), ('--seat-departure-ratio', 'inf'),
                     ('--seat-roi', '100', '50', '90', '200')):
            with self.subTest(args=args):
                self.assert_invalid(*args)

    def test_unknown_or_malformed_settings_do_not_silently_apply(self):
        for content in ({'unknown_behavior': 1}, {'headless': 'false'},
                        {'detector_preset': 'invalid'}, {'seat_roi': [0, 1, 2]}, []):
            with self.subTest(content=content):
                self.write_settings(content)
                self.assert_invalid()

    def test_missing_settings_file_reports_parse_error(self):
        self.settings.unlink()
        self.assert_invalid()


class AppGeometryAndPanelTests(unittest.TestCase):
    def tearDown(self):
        experiment._PANEL_CACHE.clear()

    def test_display_roi_maps_original_pixels_including_letterbox_offset(self):
        shape = (1080, 1920, 3)
        width, height, left, top = experiment.preview.display_geometry(shape)
        self.assertGreater(top, 0)
        result = experiment.display_roi_to_source((left, top, width, height), shape)
        np.testing.assert_allclose(result, (0, 0, 1920, 1080))
        result = experiment.display_roi_to_source((left+width/4, top+height/4, width/2, height/2), shape)
        np.testing.assert_allclose(result, (480, 270, 1440, 810))

    def test_sidebar_black_bars_and_empty_roi_are_rejected(self):
        for roi in ((1500, 100, 200, 200), (10, 0, 200, 50), (100, 100, 0, 50),
                    (-1, 200, 100, 200), (100, 1000, 100, 81)):
            with self.subTest(roi=roi):
                self.assertIsNone(experiment.display_roi_to_source(roi, (1080, 1920, 3)))

    def test_fhd_panel_contains_all_six_statuses_and_separate_controls(self):
        analysis = subject_analysis(1000)
        states = dict(zip(experiment.BEHAVIOR_NAMES,
                          ('active', 'normal', 'candidate', 'unknown', 'calibrating', 'active')))
        behavior = BehaviorFrame(states, {}, {}, MotionResult(turn_count=4),
                                 SeatResult('away', 'observed_upper_body_outside_seat',
                                            1.2, 1.5, (200, 0, 400, 540), (300, 240)))
        entries = []
        real_draw = experiment.ImageDraw.Draw

        class RecordingDraw:
            def __init__(self, image):
                self.draw = real_draw(image)

            def text(self, xy, text, **kwargs):
                entries.append((xy, text, kwargs['font'].size))
                return self.draw.text(xy, text, **kwargs)

            def rectangle(self, *args, **kwargs):
                return self.draw.rectangle(*args, **kwargs)

        with patch.object(experiment.ImageDraw, 'Draw', RecordingDraw):
            canvas = experiment.render_preview(np.zeros((540, 640, 3), np.uint8), analysis, behavior,
                                               Path('中文影片.mp4'), 1, 1, 30, 30, 25,
                                               PipelineConfig(), MotionConfig(), 1)
        self.assertEqual(canvas.shape, (1080, 1920, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        positions = []
        for name, label in experiment.BEHAVIOR_NAMES.items():
            expected = label + '：' + experiment.STATUS_NAMES[states[name]]
            matches = [entry for entry in entries if entry[1] == expected]
            self.assertEqual(len(matches), 1, label)
            positions.append(matches[0][0][1])
            self.assertLess(matches[0][0][1] + matches[0][2], 900)
        self.assertTrue(all(after-before >= 24 for before, after in zip(positions, positions[1:])))
        control_entries = [entry for entry in entries if '空白鍵' in entry[1] or '下一部' in entry[1]]
        self.assertEqual(len(control_entries), 2)
        self.assertTrue(all(entry[0][1] >= 916 for entry in control_entries))
        self.assertTrue(np.any(canvas[:, 1500:] != np.array([31, 25, 22])))

    def test_selection_generation_invalidates_cached_panel(self):
        analysis = subject_analysis(100)
        canvas = np.zeros((1080, 1920, 3), np.uint8)
        args = (None, Path('clip.mp4'), 1, 1, 0, 30, 20, PipelineConfig(), MotionConfig(), 1)
        experiment._draw_panel(canvas, analysis, *args)
        key = experiment._PANEL_CACHE['key']
        analysis.selection_generation += 1
        experiment._draw_panel(canvas, analysis, *args)
        self.assertNotEqual(key, experiment._PANEL_CACHE['key'])


class AppPlaybackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'logs'
        self.source = Path(self.temp.name) / '來源影片.mp4'
        self.source.write_bytes(b'not-decoded-by-the-test')
        self.original_source = self.source.read_bytes()
        self.settings = Path(self.temp.name) / 'settings.json'
        self.settings.write_text('{}', encoding='utf-8')
        self.args, self.pipeline_config, self.motion_config, self.seat_config = experiment.parse_configuration(
            ['--settings', str(self.settings), '--output', str(self.output), '--csv-interval', '.05'])
        self.loggers = []
        FakePipeline.instances.clear()

    def logger_factory(self, *args, **kwargs):
        logger = TimeCsvLogger(*args, **kwargs)
        self.loggers.append(logger)
        return logger

    def play(self, capture, keys=None, headless=False):
        self.args.headless = headless
        with patch.object(experiment.cv2, 'VideoCapture', return_value=capture), \
             patch.object(experiment, 'TrackingPipeline', FakePipeline), \
             patch.object(experiment, 'TimeCsvLogger', side_effect=self.logger_factory), \
             patch.object(experiment.cv2, 'imshow'), \
             patch.object(experiment.cv2, 'waitKey', side_effect=keys or [-1, -1]), \
             patch.object(experiment.cv2, 'getWindowProperty', return_value=1), \
             redirect_stdout(StringIO()):
            return experiment.play_video(self.source, FakeDetector(), Path('unused.task'),
                                         self.pipeline_config, self.motion_config,
                                         self.seat_config, self.args)

    def rows(self, path):
        with Path(path).open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))

    def assert_cleanup(self, capture, reason):
        self.assertTrue(capture.released)
        self.assertTrue(FakePipeline.instances[-1].closed)
        self.assertTrue(self.loggers[-1].closed)
        metadata = json.loads(self.loggers[-1].metadata_path.read_text(encoding='utf-8'))
        self.assertTrue(metadata['closed'])
        self.assertEqual(metadata['close_reason'], reason)
        self.assertEqual(self.source.read_bytes(), self.original_source)
        self.assertFalse(list(self.output.glob('*.mp4')))
        self.assertFalse(list(self.output.glob('*.png')))

    def test_pause_manual_reset_next_use_source_time_and_close_csv_events(self):
        capture = FakeCapture(2)
        keep, summary = self.play(capture, [ord(' '), ord('r'), -1, ord(' '), -1, ord('n')])
        self.assertTrue(keep)
        self.assertEqual(summary['frames'], 2)
        self.assertEqual(summary['end_reason'], 'next')
        pipeline = FakePipeline.instances[-1]
        self.assertEqual(pipeline.processed, [(0, 0), (100, 1)])
        self.assertEqual(pipeline.resets, 1)
        events = self.rows(summary['csv_paths']['events'])
        self.assertEqual([row['end_reason'] for row in events], ['manual_reset', 'next'])
        self.assertEqual([row['selection_generation'] for row in events], ['1', '2'])
        timeline = self.rows(summary['csv_paths']['timeline'])
        self.assertEqual([row['sample_time_seconds'] for row in timeline], ['0.000000', '0.100000'])
        self.assert_cleanup(capture, 'next')

    def test_quit_releases_resources_and_returns_stop(self):
        capture = FakeCapture(2)
        keep, summary = self.play(capture, [ord('q')])
        self.assertFalse(keep)
        self.assertEqual(summary['frames'], 1)
        self.assertEqual(summary['display_shape'], [1080, 1920, 3])
        self.assertEqual(self.rows(summary['csv_paths']['events'])[0]['end_reason'], 'quit')
        self.assert_cleanup(capture, 'quit')

    def test_reset_then_quit_reports_cleared_target_and_unknown_states(self):
        capture = FakeCapture(2)
        keep, summary = self.play(capture, [ord('r'), ord('q')])
        self.assertFalse(keep)
        self.assertIsNone(summary['final_target_id'])
        self.assertEqual(summary['final_tracking_state'], 'collecting')
        self.assertEqual(set(summary['final_states'].values()), {'unknown'})
        self.assertEqual(FakePipeline.instances[-1].processed, [(0, 0)])
        self.assertEqual(self.rows(self.loggers[-1].events_path)[0]['end_reason'], 'manual_reset')
        self.assert_cleanup(capture, 'quit')

    def test_roi_selection_while_paused_updates_preview_and_logs_boundary(self):
        capture = FakeCapture(2)
        frame_shape = (540, 640, 3)
        width, height, left, top = experiment.preview.display_geometry(frame_shape)
        roi_display = (left+width/4, top+height/4, width/2, height/2)
        with patch.object(experiment.cv2, 'selectROI', return_value=roi_display) as select, \
             patch.object(experiment.cv2, 'destroyWindow') as destroy:
            keep, summary = self.play(capture, [ord(' '), ord('s'), ord('q')])
        self.assertFalse(keep)
        self.assertEqual(FakePipeline.instances[-1].processed, [(0, 0)])
        self.assertEqual(summary['final_states']['seat_leave'], 'calibrating')
        np.testing.assert_allclose(summary['final_metrics']['seat']['seat_roi'], (160, 135, 480, 405))
        self.assertEqual(self.rows(self.loggers[-1].events_path)[0]['end_reason'], 'seat_roi_changed')
        select.assert_called_once()
        destroy.assert_called_once()
        self.assert_cleanup(capture, 'quit')

    def test_headless_reads_each_frame_and_closes_at_eof(self):
        capture = FakeCapture(2)
        keep, summary = self.play(capture, headless=True)
        self.assertTrue(keep)
        self.assertEqual(summary['frames'], 2)
        self.assertEqual(summary['end_reason'], 'end_video')
        self.assert_cleanup(capture, 'end_video')

    def test_seek_with_zero_backend_clock_preserves_original_source_time(self):
        self.args.start_seconds = 8.0
        self.args.csv_interval = .001

        class ZeroClockCapture(FakeCapture):
            def __init__(self, initial_position):
                super().__init__(2, fps=30)
                self.initial_position = initial_position

            def get(self, prop):
                if prop == cv2.CAP_PROP_POS_MSEC:
                    return 0
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    return self.initial_position
                return super().get(prop)

        for initial_position in (240, 0):
            with self.subTest(reported_position=initial_position):
                capture = ZeroClockCapture(initial_position)
                _, summary = self.play(capture, headless=True)
                self.assertEqual(capture.seek_calls, [(cv2.CAP_PROP_POS_MSEC, 8000.0)])
                self.assertEqual(FakePipeline.instances[-1].processed, [(8000, 240), (8033, 241)])
                timeline = self.rows(summary['csv_paths']['timeline'])
                self.assertEqual([row['sample_time_seconds'] for row in timeline],
                                 ['8.000000', '8.033000'])
                self.assert_cleanup(capture, 'end_video')

    def test_failed_frame_does_not_extend_csv_event_to_unanalyzed_timestamp(self):
        class GapCapture(FakeCapture):
            def get(self, prop):
                if prop == cv2.CAP_PROP_POS_MSEC:
                    return 0 if self.read_calls <= 1 else 10000
                return super().get(prop)

        original_process = FakePipeline.process
        original_engine_update = experiment.BehaviorEngine.update
        original_logger_update = TimeCsvLogger.update

        def fail_pipeline(instance, frame, timestamp, frame_index):
            if timestamp == 10000:
                raise RuntimeError('failed second frame')
            return original_process(instance, frame, timestamp, frame_index)

        def fail_engine(instance, analysis, shape):
            if analysis.timestamp_ms == 10000:
                raise RuntimeError('failed second frame')
            return original_engine_update(instance, analysis, shape)

        def fail_logger(instance, timestamp, *args, **kwargs):
            if timestamp == 10000:
                raise RuntimeError('failed second frame')
            return original_logger_update(instance, timestamp, *args, **kwargs)

        for owner, method, replacement in ((FakePipeline, 'process', fail_pipeline),
                                           (experiment.BehaviorEngine, 'update', fail_engine),
                                           (TimeCsvLogger, 'update', fail_logger)):
            with self.subTest(stage=owner.__name__):
                capture = GapCapture(2)
                with patch.object(owner, method, replacement):
                    with self.assertRaisesRegex(RuntimeError, 'failed second frame'):
                        self.play(capture, headless=True)
                logger = self.loggers[-1]
                timeline = self.rows(logger.timeline_path)
                self.assertEqual(len(timeline), 1)
                events = self.rows(logger.events_path)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]['end_seconds'], '0.000000')
                self.assertEqual(events[0]['duration_seconds'], '0.000000')
                self.assertEqual(events[0]['last_active_seconds'], '0.000000')
                metadata = json.loads(logger.metadata_path.read_text(encoding='utf-8'))
                self.assertEqual(metadata['last_source_seconds'], 0)
                self.assert_cleanup(capture, 'error')

    def test_frame_limit_is_a_recorded_boundary(self):
        self.args.max_frames = 1
        capture = FakeCapture(3)
        _, summary = self.play(capture, headless=True)
        self.assertEqual(summary['frames'], 1)
        self.assert_cleanup(capture, 'max_frames')

    def test_read_error_closes_pipeline_capture_and_active_event(self):
        capture = FakeCapture(1, read_error=OSError('simulated decoding error'))
        with self.assertRaisesRegex(OSError, 'simulated decoding error'):
            self.play(capture, headless=True)
        self.assertEqual(self.rows(self.loggers[-1].events_path)[0]['end_reason'], 'error')
        self.assert_cleanup(capture, 'error')

    def test_keyboard_interrupt_closes_active_event_and_reraises(self):
        capture = FakeCapture(1, read_error=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.play(capture, headless=True)
        self.assertEqual(self.rows(self.loggers[-1].events_path)[0]['end_reason'], 'interrupted')
        self.assert_cleanup(capture, 'interrupted')

    def test_empty_video_has_closed_files_and_no_fabricated_rows(self):
        capture = FakeCapture(0)
        with self.assertRaisesRegex(OSError, 'No decodable'):
            self.play(capture, headless=True)
        self.assertEqual(self.rows(self.loggers[-1].timeline_path), [])
        self.assertEqual(self.rows(self.loggers[-1].events_path), [])
        self.assert_cleanup(capture, 'error')

    def test_unopenable_capture_is_released_without_creating_csv(self):
        capture = FakeCapture(opened=False)
        with self.assertRaisesRegex(OSError, 'Cannot open video'):
            self.play(capture, headless=True)
        self.assertTrue(capture.released)
        self.assertEqual(self.loggers, [])
        self.assertEqual(FakePipeline.instances, [])


class AppMainTests(unittest.TestCase):
    def test_main_launches_fhd_once_iterates_videos_and_always_closes_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / 'settings.json'
            settings.write_text('{}', encoding='utf-8')
            args = ['--settings', str(settings), '--input', folder]
            videos = [Path(folder) / '1.mp4', Path(folder) / '2.mp4']
            with patch.object(experiment.preview, 'find_videos', return_value=videos), \
                 patch.object(experiment, 'ensure_model', return_value=Path('unused.task')), \
                 patch.object(experiment, 'PersonDetector', return_value=FakeDetector()), \
                 patch.object(experiment, 'play_video', side_effect=[(True, {}), (False, {})]) as play, \
                 patch.object(experiment.cv2, 'namedWindow') as named, \
                 patch.object(experiment.cv2, 'resizeWindow') as resize, \
                 patch.object(experiment.cv2, 'destroyAllWindows') as close:
                self.assertEqual(experiment.main(args), 0)
            self.assertEqual(play.call_count, 2)
            self.assertEqual([call.args[-2:] for call in play.call_args_list], [(1, 2), (2, 2)])
            named.assert_called_once()
            resize.assert_called_once_with(experiment.preview.WINDOW_NAME, 1920, 1080)
            close.assert_called_once()

    def test_main_interrupt_is_normal_exit_and_still_closes_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / 'settings.json'
            settings.write_text('{}', encoding='utf-8')
            with patch.object(experiment.preview, 'find_videos', side_effect=KeyboardInterrupt()), \
                 patch.object(experiment.cv2, 'destroyAllWindows') as close:
                result = experiment.main(['--settings', str(settings)])
            self.assertEqual(result, 0)
            close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
