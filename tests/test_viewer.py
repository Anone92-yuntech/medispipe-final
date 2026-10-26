"""FHD geometry and keyboard/playlist behavior without opening a window."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import pose_viewer as viewer
from target_selection import MappedPose
from posture_detection import PostureResult


def analysis(*, pose=None, target_id=None, timestamp_ms=0, state="waiting"):
    return viewer.FrameAnalysis([], target_id, pose, state, "waiting_for_people", {}, timestamp_ms)


class _FakeCapture:
    """Two-frame read-only decoder; callers cannot accidentally seek/write."""

    def __init__(self, frame_count=2):
        self.frame_count = frame_count
        self.position = 0
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        if prop == cv2.CAP_PROP_POS_MSEC:
            return max(0, self.position - 1) * 40.0
        raise AssertionError(f"Unexpected capture property {prop}")

    def read(self):
        self.read_count += 1
        if self.position >= self.frame_count:
            return False, None
        self.position += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class ViewerGeometryTests(unittest.TestCase):
    def test_original_image_markers_and_pose_align_in_landscape_and_portrait(self):
        # Independent expected positions include resize rounding and black bars.
        cases = [
            ((720, 1280), (320, 180), (375, 329), (0, 118, 1500, 962)),
            ((1280, 720), (180, 320), (598, 270), (446, 0, 1054, 1080)),
        ]
        for (height, width), source_xy, expected_xy, bounds in cases:
            with self.subTest(source_shape=(height, width)):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                marker_color = (177, 33, 90)
                cv2.circle(frame, source_xy, 9, marker_color, -1)
                with patch.object(viewer, "_draw_sidebar"):
                    background = viewer.render_fhd(frame, analysis())
                self.assertEqual(background.shape, (1080, 1920, 3))
                x, y = expected_xy
                np.testing.assert_array_equal(background[y, x], marker_color)
                projected = viewer.to_display([source_xy], frame.shape)[0]
                np.testing.assert_allclose(projected, expected_xy, atol=0.01)
                np.testing.assert_allclose(
                    viewer.to_display([[0, 0], [width, height]], frame.shape),
                    np.asarray(bounds).reshape(2, 2), atol=0.01,
                )
                points = np.zeros((33, 2), dtype=float)
                points[11] = source_xy
                confidence = np.zeros(33)
                confidence[11] = 1.0
                pose = MappedPose(points, confidence.copy(), confidence.copy())
                with patch.object(viewer, "_draw_sidebar"):
                    overlay = viewer.render_fhd(frame, analysis(pose=pose, target_id=7))
                # The real render path puts a landmark on the original marker.
                np.testing.assert_array_equal(overlay[y, x], (60, 255, 255))
                self.assertTrue(np.array_equal(overlay[1000, 1800], [0, 0, 0]))

    def test_imshow_receives_the_complete_fhd_canvas(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        with patch.object(viewer, "_draw_sidebar"):
            canvas = viewer.render_fhd(frame, analysis())
        with patch.object(viewer.cv2, "imshow") as show, \
                patch.object(viewer.cv2, "waitKey", return_value=-1), \
                patch.object(viewer.cv2, "getWindowProperty", return_value=1):
            self.assertEqual(viewer.show_frame(canvas, 1), "continue")
        show.assert_called_once()
        self.assertEqual(show.call_args.args[0], viewer.WINDOW_NAME)
        self.assertEqual(show.call_args.args[1].shape, (1080, 1920, 3))


class ViewerControlTests(unittest.TestCase):
    def test_processing_rate_includes_render_and_excludes_user_waits(self):
        clock = SimpleNamespace(now=0.0)
        capture = _FakeCapture()
        original_read = capture.read

        def read():
            clock.now += .01
            return original_read()

        capture.read = read
        pipeline = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = pipeline

        def process(_frame, timestamp, _index):
            clock.now += .02
            return analysis(timestamp_ms=timestamp)

        def render(*_args):
            clock.now += .01
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        def show(*_args):
            clock.now += 5  # UI waiting/pausing must not reduce processing FPS.
            return 'continue'

        pipeline.process.side_effect = process
        args = SimpleNamespace(start_seconds=0, max_frames=2, headless=False)
        with patch.object(viewer.cv2, 'VideoCapture', return_value=capture), \
                patch.object(viewer, 'TrackingPipeline', return_value=context), \
                patch.object(viewer, 'render_fhd', side_effect=render) as renderer, \
                patch.object(viewer, 'show_frame', side_effect=show), \
                patch.object(viewer.time, 'perf_counter', side_effect=lambda: clock.now), \
                patch('builtins.print'):
            _, summary = viewer.play_video(Path('test.mp4'), object(), Path('model'),
                                            viewer.PipelineConfig(), args, 1, 1)
        self.assertAlmostEqual(summary['compute_seconds'], .08)
        self.assertAlmostEqual(summary['processing_fps'], 25)
        self.assertAlmostEqual(renderer.call_args.args[7], 25)
        self.assertGreater(summary['elapsed_seconds'], 10)

    def test_pause_and_reset_refresh_without_advancing_decoder_or_tracking(self):
        capture = _FakeCapture()
        pipeline = MagicMock()
        pipeline.process.side_effect = [
            analysis(target_id=7, state="locked", timestamp_ms=0),
            analysis(target_id=7, state="locked", timestamp_ms=40),
        ]
        context = MagicMock()
        context.__enter__.return_value = pipeline
        shown = []

        def reset_selection():
            # R occurs inside the first frame's pause loop: no new decoder read,
            # detector/tracker process, or timestamp has occurred.
            self.assertEqual(capture.read_count, 1)
            self.assertEqual(pipeline.process.call_count, 1)

        pipeline.reset_selection.side_effect = reset_selection

        def render(frame, result, *_):
            value = 9 if result.target_id is None else 2
            return np.full((1080, 1920, 3), value, dtype=np.uint8)

        args = SimpleNamespace(start_seconds=0, max_frames=None, headless=False)
        keys = [ord(" "), -1, ord("R"), -1, ord(" "), -1, ord("N")]
        with ExitStack() as stack:
            stack.enter_context(patch.object(viewer.cv2, "VideoCapture", return_value=capture))
            stack.enter_context(patch.object(viewer, "TrackingPipeline", return_value=context))
            stack.enter_context(patch.object(viewer, "render_fhd", side_effect=render))
            stack.enter_context(patch.object(viewer.cv2, "imshow", side_effect=lambda _, image: shown.append(image.copy())))
            wait = stack.enter_context(patch.object(viewer.cv2, "waitKey", side_effect=keys))
            stack.enter_context(patch.object(viewer.cv2, "getWindowProperty", return_value=1))
            stack.enter_context(patch("builtins.print"))
            keep_playing, summary = viewer.play_video(
                Path("受試者影片.mp4"), object(), Path("model.task"),
                viewer.PipelineConfig(), args, 1, 2,
            )
        self.assertTrue(keep_playing)
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(capture.read_count, 2)
        self.assertTrue(capture.released)
        self.assertEqual(pipeline.process.call_count, 2)
        self.assertEqual([call.args[1] for call in pipeline.process.call_args_list], [0, 40])
        pipeline.reset_selection.assert_called_once()
        context.__exit__.assert_called_once()
        self.assertTrue(all(image.shape == (1080, 1920, 3) for image in shown))
        # The frame shown immediately after R contains the fresh cleared view.
        self.assertEqual(shown[3][1000, 1400, 0], 9)
        self.assertEqual(shown[0][1000, 1400, 0], 2)
        self.assertEqual([call.args[0] for call in wait.call_args_list][1:5], [30] * 4)

    def test_next_quit_escape_and_window_close(self):
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cases = [
            ([ord("n")], 1, "next"),
            ([ord("N")], 1, "next"),
            ([ord("q")], 1, "quit"),
            ([ord("Q")], 1, "quit"),
            ([27], 1, "quit"),
            ([-1], 0, "quit"),
            ([ord(" "), ord("n")], 1, "next"),
            ([ord(" "), 27], 1, "quit"),
        ]
        for keys, visible, expected in cases:
            with self.subTest(keys=keys, visible=visible), \
                    patch.object(viewer.cv2, "imshow"), \
                    patch.object(viewer.cv2, "waitKey", side_effect=keys), \
                    patch.object(viewer.cv2, "getWindowProperty", return_value=visible):
                self.assertEqual(viewer.show_frame(canvas, 1), expected)

    def test_playlist_creates_fresh_pipeline_and_passes_unicode_path_unchanged(self):
        paths = [Path("C:/videos/first.mp4"), Path("C:/videos/兒童第二部.MP4")]
        captures = [_FakeCapture(frame_count=1), _FakeCapture(frame_count=1)]
        contexts = [MagicMock(), MagicMock()]
        pipelines = [MagicMock(), MagicMock()]
        for index, (context, pipeline) in enumerate(zip(contexts, pipelines)):
            context.__enter__.return_value = pipeline
            pipeline.process.return_value = analysis(target_id=index + 1, state="locked")
        detector = SimpleNamespace(provenance={"name": "mock detector"}, device="cpu", input_size=416)
        with ExitStack() as stack:
            stack.enter_context(patch.object(viewer, "find_videos", return_value=paths))
            stack.enter_context(patch.object(viewer, "ensure_model", return_value=Path("model.task")))
            stack.enter_context(patch.object(viewer, "PersonDetector", return_value=detector))
            open_video = stack.enter_context(patch.object(viewer.cv2, "VideoCapture", side_effect=captures))
            create_pipeline = stack.enter_context(patch.object(viewer, "TrackingPipeline", side_effect=contexts))
            stack.enter_context(patch.object(viewer, "render_fhd", return_value=np.zeros((1080, 1920, 3), dtype=np.uint8)))
            stack.enter_context(patch.object(viewer, "show_frame", side_effect=["next", "quit"]))
            stack.enter_context(patch.object(viewer.cv2, "namedWindow"))
            resize = stack.enter_context(patch.object(viewer.cv2, "resizeWindow"))
            destroy = stack.enter_context(patch.object(viewer.cv2, "destroyAllWindows"))
            stack.enter_context(patch("builtins.print"))
            self.assertEqual(viewer.main(["--input", "unused"]), 0)
        self.assertEqual([call.args[0] for call in open_video.call_args_list], [str(path) for path in paths])
        self.assertEqual(create_pipeline.call_count, 2)
        for capture, pipeline, context in zip(captures, pipelines, contexts):
            self.assertEqual(capture.read_count, 1)
            self.assertTrue(capture.released)
            pipeline.process.assert_called_once()
            self.assertEqual(pipeline.process.call_args.args[1], 0)
            self.assertEqual(pipeline.process.call_args.args[2], 0)
            context.__exit__.assert_called_once()
        resize.assert_called_once_with(viewer.WINDOW_NAME, 1920, 1080)
        destroy.assert_called_once()


class SidebarFreshnessTests(unittest.TestCase):
    def setUp(self):
        viewer.reset_sidebar_cache()

    def tearDown(self):
        viewer.reset_sidebar_cache()

    def test_identity_or_status_changes_refresh_without_waiting_for_numeric_tick(self):
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        images = [np.full((1080, 420, 3), value, np.uint8) for value in (2, 7, 9)]
        with patch.object(viewer, '_render_sidebar', side_effect=images):
            for result, expected in ((analysis(target_id=1, state='locked', timestamp_ms=100), 2),
                                     (analysis(target_id=1, state='lost', timestamp_ms=110), 7),
                                     (analysis(target_id=2, state='locked', timestamp_ms=120), 9)):
                viewer._draw_sidebar(canvas, result, Path('test.mp4'), 1, 1, 1,
                                     30, 25, viewer.PipelineConfig())
                self.assertEqual(canvas[0, 1500, 0], expected)

    def test_reset_discards_cached_panel_with_identical_time_and_state(self):
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = analysis(timestamp_ms=100)
        images = [np.full((1080, 420, 3), value, np.uint8) for value in (2, 9)]
        with patch.object(viewer, '_render_sidebar', side_effect=images):
            viewer._draw_sidebar(canvas, result, Path('test.mp4'), 1, 1, 1,
                                 30, 25, viewer.PipelineConfig())
            viewer.reset_sidebar_cache()
            viewer._draw_sidebar(canvas, result, Path('test.mp4'), 1, 1, 1,
                                 30, 25, viewer.PipelineConfig())
        self.assertEqual(canvas[0, 1500, 0], 9)

    def test_auto_reselection_generation_refreshes_same_id_without_numeric_tick(self):
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        before = analysis(target_id=1, state='locked', timestamp_ms=100)
        after = analysis(target_id=1, state='locked', timestamp_ms=110)
        after.selection_generation = before.selection_generation + 1
        after.auto_reset_count = 1
        after.auto_reset_reason = 'auto_reset_pose_failed'
        images = [np.full((1080, 420, 3), value, np.uint8) for value in (2, 9)]
        with patch.object(viewer, '_render_sidebar', side_effect=images) as renderer:
            for result in (before, after):
                viewer._draw_sidebar(canvas, result, Path('test.mp4'), 1, 1, 1,
                                     30, 25, viewer.PipelineConfig())
        self.assertEqual(renderer.call_count, 2)
        self.assertEqual(canvas[0, 1500, 0], 9)

    def test_posture_and_recovery_changes_refresh_without_numeric_tick(self):
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results = [analysis(target_id=1, state='locked', timestamp_ms=100 + i)
                   for i in range(4)]
        for result, state in zip(results, ('candidate', 'resting', 'unknown', 'unknown')):
            result.posture = PostureResult(state, 'test')
        results[-1].pose_recovery_count = 1
        images = [np.full((1080, 420, 3), value, np.uint8) for value in (2, 5, 7, 9)]
        with patch.object(viewer, '_render_sidebar', side_effect=images) as renderer:
            for result, expected in zip(results, (2, 5, 7, 9)):
                viewer._draw_sidebar(canvas, result, Path('test.mp4'), 1, 1, 1,
                                     30, 25, viewer.PipelineConfig())
                self.assertEqual(canvas[0, 1500, 0], expected)
        self.assertEqual(renderer.call_count, 4)


if __name__ == "__main__":
    unittest.main()
