"""COCO person detection using official YOLOX ONNX releases and OpenCV DNN.

The returned boxes are measured in ORIGINAL image pixels: [x1, y1, x2, y2, score].
No tracking, age classification, image/video saving, torch, or ONNX Runtime is used.

Preprocessing and output decoding follow the upstream reference implementations:
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md

Custom models must be raw-output YOLOX COCO ONNX exports (80 classes, strides
8/16/32), without --decode_in_inference, with the supplied square input_size.
Nano/Tiny have fixed 416 x 416 inputs; S has a fixed 640 x 640 input.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

import cv2
import numpy as np


MODELS_DIR = Path(__file__).resolve().parent / "models"
_RELEASE_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/"
)
# These are original, unmodified official release assets, pinned after download.
DETECTOR_PRESETS = {
    "nano": {
        "name": "YOLOX-Nano", "filename": "yolox_nano.onnx", "input_size": 416,
        "sha256": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    },
    "tiny": {
        "name": "YOLOX-Tiny", "filename": "yolox_tiny.onnx", "input_size": 416,
        "sha256": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    },
    "s": {
        "name": "YOLOX-S", "filename": "yolox_s.onnx", "input_size": 640,
        "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    },
}
for _preset in DETECTOR_PRESETS.values():
    _size = _preset["input_size"]
    _rows = sum((_size // stride) ** 2 for stride in (8, 16, 32))
    _preset.update({
        "path": MODELS_DIR / _preset["filename"],
        "url": _RELEASE_URL + _preset["filename"],
        "source": _RELEASE_URL + _preset["filename"],
        "release": "0.1.1rc0",
        "dataset": "COCO (person class index 0)",
        "input": f"1x3x{_size}x{_size} BGR float32 0..255; top-left letterbox=114",
        "output": f"1x{_rows}x85 raw YOLOX head, strides 8/16/32",
        "runtime": "OpenCV DNN",
        "upstream_license": "Apache-2.0",
    })
DEFAULT_DETECTOR_PRESET = "nano"
DEFAULT_DETECTOR_MODEL = DETECTOR_PRESETS[DEFAULT_DETECTOR_PRESET]["path"]
DETECTOR_MODEL_URL = DETECTOR_PRESETS[DEFAULT_DETECTOR_PRESET]["url"]
DETECTOR_MODEL_SHA256 = DETECTOR_PRESETS[DEFAULT_DETECTOR_PRESET]["sha256"]
DETECTOR_PROVENANCE = dict(DETECTOR_PRESETS[DEFAULT_DETECTOR_PRESET])
DETECTOR_PROVENANCE.pop("path", None)


def _preset_for_path(path: Path) -> tuple[str, dict] | None:
    return next(
        ((name, item) for name, item in DETECTOR_PRESETS.items() if item["path"] == path),
        None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_detector_model(
    path: Path | None = None, preset: str = DEFAULT_DETECTOR_PRESET,
) -> Path:
    """Download a named official preset atomically, with SHA256 verification.

    An explicit missing custom path is an error; no unrelated weights are silently
    downloaded into it. The downloaded model and temporary file remain in models/.
    """
    if preset not in DETECTOR_PRESETS:
        raise ValueError(f"不支援的偵測器預設：{preset}；可選 {', '.join(DETECTOR_PRESETS)}")
    model_path = DETECTOR_PRESETS[preset]["path"] if path is None else Path(path).expanduser().resolve()
    known = _preset_for_path(model_path)
    metadata = known[1] if known else None
    if model_path.is_file():
        if metadata and _sha256(model_path) != metadata["sha256"]:
            raise RuntimeError(
                f"人物模型 SHA256 不符，請移除損壞的模型後重新執行：{model_path}"
            )
        if model_path.stat().st_size == 0:
            raise RuntimeError(f"人物模型是空檔案：{model_path}")
        return model_path
    if metadata is None:
        raise FileNotFoundError(f"找不到指定的人物偵測模型：{model_path}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=model_path.stem + "_", suffix=".onnx.part", dir=str(MODELS_DIR)
    )
    temporary_path = Path(temporary_name)
    try:
        request = urllib.request.Request(
            metadata["url"], headers={"User-Agent": "MEDIAPIPE-Pose-Viewer/1.0"}
        )
        with os.fdopen(descriptor, "wb") as destination:
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        actual_hash = _sha256(temporary_path)
        if actual_hash != metadata["sha256"]:
            raise RuntimeError(f"下載模型 SHA256 不符：{actual_hash}")
        os.replace(temporary_path, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"人物偵測模型下載失敗；來源：{metadata['url']}\n{exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return model_path


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Single-class NMS following the YOLOX NumPy reference convention."""
    order = np.argsort(scores)[::-1]
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    selected: list[int] = []
    while len(order):
        current = int(order[0])
        selected.append(current)
        others = order[1:]
        if not len(others):
            break
        top_left = np.maximum(boxes[current, :2], boxes[others, :2])
        bottom_right = np.minimum(boxes[current, 2:], boxes[others, 2:])
        intersection_wh = np.maximum(0.0, bottom_right - top_left + 1)
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
        union = areas[current] + areas[others] - intersection
        overlap = intersection / np.maximum(union, 1e-9)
        order = others[overlap <= threshold]
    return selected


class PersonDetector:
    """Pretrained YOLOX COCO person detector; detect() returns an Nx5 float64 copy."""

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "auto",
        confidence: float = 0.35,
        nms: float = 0.45,
        input_size: int | None = None,
        *,
        preset: str = DEFAULT_DETECTOR_PRESET,
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence 必須介於 0（不含）至 1")
        if not 0 < nms <= 1:
            raise ValueError("nms 必須介於 0（不含）至 1")
        if preset not in DETECTOR_PRESETS:
            raise ValueError(f"不支援的偵測器預設：{preset}")
        requested_path = DETECTOR_PRESETS[preset]["path"] if model_path is None else Path(model_path).expanduser().resolve()
        known = _preset_for_path(requested_path)
        if input_size is None:
            if not known:
                raise ValueError("自訂 YOLOX ONNX 必須明確提供 input_size")
            input_size = known[1]["input_size"]
        if not isinstance(input_size, int) or input_size < 32 or input_size % 32:
            raise ValueError("input_size 必須為至少 32 的 32 倍數整數")
        if known and input_size != known[1]["input_size"]:
            raise ValueError(f"{known[1]['name']} 模型的固定輸入為 {known[1]['input_size']}，不可改成 {input_size}")
        requested_device = str(device).lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device 僅支援 auto、cpu、cuda")
        cuda_targets = ()
        if requested_device != "cpu":
            try:
                cuda_targets = cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)
            except cv2.error:
                pass
        cuda_available = cv2.dnn.DNN_TARGET_CUDA in cuda_targets
        if requested_device == "cuda" and not cuda_available:
            raise RuntimeError(
                "目前 OpenCV 未提供可用的 CUDA DNN 後端；請使用 --device cpu 或 auto。"
                "一般 pip OpenCV 套件不含 CUDA DNN。"
            )
        self.device = "cuda" if cuda_available and requested_device != "cpu" else "cpu"
        self.model_path = ensure_detector_model(model_path, preset=preset)
        self.preset = known[0] if known else "custom"
        self.input_size = input_size
        self.confidence = float(confidence)
        self.nms = float(nms)
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        except cv2.error as exc:
            raise RuntimeError(f"OpenCV 無法載入 YOLOX ONNX：{self.model_path}\n{exc}") from exc
        if self.device == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            # A newly loaded OpenCV net defaults to CPU. OpenCV 5's graph engine
            # warns on setPreferableTarget(CPU), so leave that default untouched.
            if int(cv2.__version__.split(".", 1)[0]) < 5:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        grids, strides = [], []
        for stride in (8, 16, 32):
            side = input_size // stride
            xx, yy = np.meshgrid(np.arange(side), np.arange(side))
            grid = np.stack((xx, yy), axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((len(grid), 1), stride))
        self._grid = np.concatenate(grids).astype(np.float32)
        self._strides = np.concatenate(strides).astype(np.float32)
        self.provenance = dict(known[1]) if known else {
            "name": "custom YOLOX COCO ONNX", "source": "user-supplied", "release": "unknown",
            "runtime": "OpenCV DNN", "input_size": input_size,
        }
        # JSON reports must not receive pathlib objects from preset metadata.
        self.provenance.pop("path", None)
        self.provenance.update({
            "model_path": str(self.model_path),
            "sha256": _sha256(self.model_path),
            "opencv_version": cv2.__version__,
            "device": self.device,
        })

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return observed person detections in original pixels; never mutate frame."""
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.shape[0] < 1
            or frame_bgr.shape[1] < 1
            or frame_bgr.dtype != np.uint8
        ):
            raise ValueError("frame_bgr 必須是非空 HxWx3 uint8 BGR 影像")
        height, width = frame_bgr.shape[:2]
        ratio = min(self.input_size / height, self.input_size / width)
        resized_width = max(1, int(width * ratio))
        resized_height = max(1, int(height * ratio))
        resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:resized_height, :resized_width] = resized
        # YOLOX uses BGR 0..255, top-left aligned letterbox, not YOLO RGB /255.
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        try:
            self.net.setInput(blob)
            output = self.net.forward()
        except cv2.error as exc:
            raise RuntimeError(
                f"YOLOX 推論失敗，請確認模型輸入為 {self.input_size}x{self.input_size}、"
                f"且支援 {self.device}：\n{exc}"
            ) from exc
        if output.shape != (1, len(self._grid), 85):
            raise RuntimeError(
                f"不支援的 YOLOX 輸出 {output.shape}；預期 (1, {len(self._grid)}, 85)，"
                "需使用未內建 decode 的 80 類 COCO YOLOX ONNX。"
            )
        predictions = output[0].copy()
        classes = predictions[:, 5:].argmax(axis=1)
        scores = predictions[:, 4] * predictions[:, 5]  # COCO person = class 0.
        keep = (classes == 0) & (scores >= self.confidence) & np.isfinite(scores)
        if not np.any(keep):
            return np.empty((0, 5), dtype=np.float64)
        raw = predictions[keep, :4]
        centers = (raw[:, :2] + self._grid[keep]) * self._strides[keep]
        with np.errstate(over="ignore", invalid="ignore"):
            sizes = np.exp(raw[:, 2:4]) * self._strides[keep]
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / ratio
        valid_scores = scores[keep]
        finite = np.isfinite(boxes).all(axis=1)
        boxes, valid_scores = boxes[finite], valid_scores[finite]
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
        valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes, valid_scores = boxes[valid], valid_scores[valid]
        if not len(boxes):
            return np.empty((0, 5), dtype=np.float64)
        selected = _nms(boxes, valid_scores, self.nms)
        return np.column_stack((boxes[selected], valid_scores[selected])).astype(np.float64, copy=True)
