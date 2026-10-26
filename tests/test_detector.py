"""Detector contract checks; local-model smoke tests never download assets."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from person_detector import (
    DEFAULT_DETECTOR_MODEL, DEFAULT_DETECTOR_PRESET, DETECTOR_PRESETS,
    PersonDetector, ensure_detector_model,
)


class _FakeNet:
    def __init__(self, output):
        self.output = output
        self.blob = None

    def setInput(self, blob):
        self.blob = blob.copy()

    def forward(self):
        return self.output.copy()

    def setPreferableBackend(self, backend):
        pass

    def setPreferableTarget(self, target):
        pass


def _synthetic_detector(raw: np.ndarray, input_size=640) -> PersonDetector:
    """Inject a known head output to verify geometry without model/network access."""
    detector = PersonDetector.__new__(PersonDetector)
    detector.input_size = input_size
    detector.confidence = 0.35
    detector.nms = 0.45
    detector.device = "cpu"
    detector._grid = np.tile(np.array([[2, 3]], dtype=np.float32), (len(raw), 1))
    detector._strides = np.full((len(raw), 1), 8, dtype=np.float32)
    detector.net = _FakeNet(raw[None])
    return detector


def _person_prediction() -> np.ndarray:
    # Known box in model pixels: cx=320, cy=180, width=128, height=80.
    row = np.zeros((1, 85), dtype=np.float32)
    row[0, :4] = [38, 19.5, np.log(16), np.log(10)]
    row[0, 4:6] = [0.9, 0.8]
    return row


class DetectorContractTests(unittest.TestCase):
    def test_decode_original_pixels_and_preserve_bgr_input(self):
        detector = _synthetic_detector(_person_prediction())
        source = np.full((720, 1280, 3), (10, 20, 30), dtype=np.uint8)
        before = source.copy()
        boxes = detector.detect(source)
        # Input is halved with padding only below; original-pixel box is doubled.
        np.testing.assert_allclose(boxes, [[512, 280, 768, 440, 0.72]], atol=0.001)
        self.assertEqual(boxes.dtype, np.float64)
        np.testing.assert_array_equal(source, before)
        self.assertEqual(detector.net.blob.shape, (1, 3, 640, 640))
        self.assertEqual(detector.net.blob.dtype, np.float32)
        np.testing.assert_array_equal(detector.net.blob[0, :, 20, 20], [10, 20, 30])
        np.testing.assert_array_equal(detector.net.blob[0, :, 600, 20], [114, 114, 114])

    def test_only_person_class_is_returned(self):
        row = _person_prediction()
        row[0, 6] = 0.99  # Most likely COCO class is bicycle, not person.
        result = _synthetic_detector(row).detect(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(result.shape, (0, 5))

    def test_416_decode_returns_original_pixels(self):
        # The same model-pixel box at a 416 input has a different scale to FHD.
        detector = _synthetic_detector(_person_prediction(), input_size=416)
        frame = np.zeros((1080, 1920, 3), np.uint8)
        boxes = detector.detect(frame)
        ratio = 416 / 1920
        np.testing.assert_allclose(boxes[0, :4], np.array([256, 140, 384, 220]) / ratio, atol=0.001)
        self.assertEqual(detector.net.blob.shape, (1, 3, 416, 416))

    def test_presets_auto_size_and_raw_head_grid_match(self):
        for preset, metadata in DETECTOR_PRESETS.items():
            with self.subTest(preset=preset):
                fake = _FakeNet(np.empty(0))
                with patch("person_detector.ensure_detector_model", return_value=metadata["path"]), \
                     patch("person_detector._sha256", return_value=metadata["sha256"]), \
                     patch("person_detector.cv2.dnn.readNetFromONNX", return_value=fake):
                    detector = PersonDetector(preset=preset, device="cpu")
                self.assertEqual(detector.input_size, metadata["input_size"])
                self.assertEqual(detector.preset, preset)
                self.assertEqual(len(detector._grid), 3549 if preset != "s" else 8400)
                self.assertEqual(detector.provenance["sha256"], metadata["sha256"])
                self.assertNotIn("path", detector.provenance)

    def test_explicit_original_s_model_keeps_640_without_overriding_default(self):
        metadata = DETECTOR_PRESETS["s"]
        with patch("person_detector.ensure_detector_model", return_value=metadata["path"]), \
             patch("person_detector._sha256", return_value=metadata["sha256"]), \
             patch("person_detector.cv2.dnn.readNetFromONNX", return_value=_FakeNet(np.empty(0))):
            detector = PersonDetector(metadata["path"], "cpu")
        self.assertEqual(detector.preset, "s")
        self.assertEqual(detector.input_size, 640)

    def test_fixed_preset_resolution_mismatch_fails_before_loading(self):
        for preset in DETECTOR_PRESETS:
            wrong_size = 640 if preset != "s" else 416
            with self.subTest(preset=preset), patch("person_detector.ensure_detector_model") as loader:
                with self.assertRaisesRegex(ValueError, "固定輸入"):
                    PersonDetector(preset=preset, input_size=wrong_size)
                loader.assert_not_called()

    def test_custom_model_requires_explicit_resolution(self):
        with patch("person_detector.ensure_detector_model") as loader:
            with self.assertRaisesRegex(ValueError, "input_size"):
                PersonDetector(BASE_DIR / "models" / "custom.onnx")
            loader.assert_not_called()

    def test_duplicate_boxes_are_suppressed(self):
        row = np.repeat(_person_prediction(), 2, axis=0)
        row[1, 4] = 0.8
        result = _synthetic_detector(row).detect(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(result.shape, (1, 5))
        self.assertAlmostEqual(result[0, 4], 0.72, places=5)

    def test_boxes_are_clamped_to_original_image(self):
        row = _person_prediction()
        row[0, :4] = [-2, -3, np.log(16), np.log(10)]  # center at origin.
        result = _synthetic_detector(row).detect(np.zeros((720, 1280, 3), np.uint8))
        np.testing.assert_allclose(result[0, :4], [0, 0, 128, 80], atol=0.001)

    def test_invalid_image_shapes_and_types_are_rejected(self):
        detector = _synthetic_detector(_person_prediction())
        for source in (
            None,
            np.zeros((32, 32), np.uint8),
            np.zeros((0, 32, 3), np.uint8),
            np.zeros((32, 32, 4), np.uint8),
            np.zeros((32, 32, 3), np.float32),
        ):
            with self.subTest(shape=getattr(source, "shape", None)):
                with self.assertRaises(ValueError):
                    detector.detect(source)

    def test_wrong_model_output_has_clear_error(self):
        detector = _synthetic_detector(_person_prediction())
        detector.net.output = np.zeros((1, 1, 6), np.float32)
        with self.assertRaisesRegex(RuntimeError, "YOLOX"):
            detector.detect(np.zeros((32, 32, 3), np.uint8))

    def test_missing_custom_model_never_downloads_default_weights(self):
        missing = BASE_DIR / "models" / f"missing_test_{uuid.uuid4().hex}.onnx"
        with patch("person_detector.urllib.request.urlopen") as downloader:
            with self.assertRaises(FileNotFoundError):
                ensure_detector_model(missing)
        downloader.assert_not_called()

    def test_invalid_configuration_is_rejected_before_model_loading(self):
        for kwargs in ({"confidence": 0}, {"nms": 1.5}, {"input_size": 641}, {"device": "invalid"}, {"preset": "wrong"}):
            with self.subTest(kwargs=kwargs):
                with patch("person_detector.ensure_detector_model") as loader:
                    with self.assertRaises(ValueError):
                        PersonDetector(**kwargs)
                loader.assert_not_called()


@unittest.skipUnless(DEFAULT_DETECTOR_MODEL.is_file(), "local YOLOX model not present; no download in tests")
class LocalModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patching urlopen makes the no-network test contract explicit.
        with patch("person_detector.urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            cls.detector = PersonDetector(device="cpu")

    def test_black_image_has_no_person(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        boxes = self.detector.detect(frame)
        self.assertEqual(boxes.shape, (0, 5))
        self.assertEqual(boxes.dtype, np.float64)

    def test_all_local_official_presets_load_and_black_image_has_no_person(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        for preset, metadata in DETECTOR_PRESETS.items():
            if not metadata["path"].is_file():
                continue
            with self.subTest(preset=preset), patch(
                "person_detector.urllib.request.urlopen", side_effect=AssertionError("network forbidden")
            ):
                detector = PersonDetector(preset=preset, device="cpu")
                self.assertEqual(detector.detect(frame).shape, (0, 5))
                self.assertEqual(detector.input_size, metadata["input_size"])

    @unittest.skipUnless((BASE_DIR / "optional_test_video.mp4").is_file(), "optional source video absent")
    def test_one_source_frame_is_read_only_and_boxes_use_source_pixels(self):
        video = BASE_DIR / "optional_test_video.mp4"
        original_stat = video.stat()
        capture = cv2.VideoCapture(str(video))
        try:
            self.assertTrue(capture.isOpened())
            capture.set(cv2.CAP_PROP_POS_MSEC, 10000)
            success, frame = capture.read()  # Exactly one decoded source frame.
            self.assertTrue(success)
        finally:
            capture.release()
        before = frame.copy()
        boxes = self.detector.detect(frame)
        np.testing.assert_array_equal(frame, before)
        self.assertEqual(boxes.ndim, 2)
        self.assertEqual(boxes.shape[1], 5)
        self.assertGreaterEqual(len(boxes), 1)
        height, width = frame.shape[:2]
        self.assertTrue(np.isfinite(boxes).all())
        self.assertTrue(np.all(boxes[:, :4] >= 0))
        self.assertTrue(np.all(boxes[:, [0, 2]] <= width))
        self.assertTrue(np.all(boxes[:, [1, 3]] <= height))
        # Existing source is FHD: rightmost person edge exceeds model width 640.
        self.assertGreater(float(boxes[:, 2].max()), self.detector.input_size)
        current_stat = video.stat()
        self.assertEqual(current_stat.st_size, original_stat.st_size)
        self.assertEqual(current_stat.st_mtime_ns, original_stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
