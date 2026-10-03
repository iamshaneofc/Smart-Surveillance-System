"""F4-F real detector adapter: YOLOX-Tiny behind the Detector abstraction.

Model: YOLOX-Tiny (COCO 80-class), official ONNX artifact pinned by
scripts/weights.lock.json (release tag 0.1.1rc0, sha256
427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7).
Registry status: candidate - UNEVALUATED (no licensed evaluation dataset;
see docs/DETECTOR_SELECTION.md section 8).

Inference contract (empirically validated against the artifact 2026-10-03):
- graph input  'images' [1, 3, 416, 416] float32, pixel range 0..255
  (the /255 normalization is BAKED INTO THE EXPORTED GRAPH - feeding 0..1
  collapses all confidences; verified empirically),
- letterbox: aspect-preserving min-scale resize, pad 114 top/left, BGR->RGB,
- graph output 'output' [1, 3549, 85] raw head - decode and NMS run OUTSIDE
  the graph ((xy + grid) * stride, exp(wh) * stride, strides 8/16/32,
  numpy per-class NMS), boxes un-letterboxed by the resize ratio and
  normalized to [0, 1].

Never downloads weights: a missing artifact raises DetectorError with the
explicit acquisition command. Optional dependency: onnxruntime (pip install
"sentinel[infer]"), imported lazily.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from packages.common.logging import get_logger
from services.camera.types import FramePacket
from services.inference.interfaces import DetectorError
from services.inference.types import BBox, Detection, ModelInfo

log = get_logger(__name__)

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Pinned artifact metadata - MUST match scripts/weights.lock.json
# (tests/test_f4_detector_adapter.py asserts they stay in sync).
YOLOX_SPEC = {
    "model_id": "yolox-tiny",
    "version": "0.1.1rc0",
    "filename": "yolox_tiny.onnx",
    "size_bytes": 20219662,
    "sha256": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
    "license": "Apache-2.0 (repository LICENSE; weights via official release assets)",
    "framework": "onnx",
}

DEFAULT_WEIGHTS_PATH = Path("models") / YOLOX_SPEC["filename"]
DEFAULT_INPUT_SIZE = (416, 416)
DEFAULT_CLASSES: tuple[str, ...] = ("person",)
DEFAULT_CONFIDENCE = 0.3
NMS_IOU = 0.45
CANDIDATE_SCORE_FLOOR = 0.1
MAX_DETECTIONS = 100
STRIDES: tuple[int, ...] = (8, 16, 32)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_weights_path(weights_path: str | Path | None) -> Path:
    """Relative paths resolve against the repository root. No machine paths."""
    if weights_path is None:
        return _REPO_ROOT / DEFAULT_WEIGHTS_PATH
    path = Path(weights_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def letterbox(image, input_size: tuple[int, int]):
    """Aspect-preserving pad-to-size (YOLOX preproc). Returns (canvas, ratio)."""
    import cv2
    import numpy as np

    out_h, out_w = input_size
    canvas = np.full((out_h, out_w, 3), 114.0, dtype=np.float32)
    src_h, src_w = image.shape[:2]
    ratio = min(out_h / src_h, out_w / src_w)
    new_w, new_h = int(src_w * ratio), int(src_h * ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[:new_h, :new_w] = resized
    return canvas, ratio


def decode_and_nms(raw, input_size: tuple[int, int], ratio: float, frame_w: int, frame_h: int) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """Decode raw YOLOX head -> per-class NMS -> (class_id, conf, xyxy normalized to frame).

    Mirrors the official YOLOX demo (release tag 0.1.1rc0): demo_postprocess +
    numpy multiclass NMS. NMS runs on un-letterboxed ORIGINAL-PIXEL boxes (the
    vendor algorithm is pixel-space; normalizing x/y by different frame
    dimensions would distort IoU), then boxes are normalized to [0,1].
    """
    import numpy as np

    preds = raw.copy()
    if preds.ndim == 3:
        preds = preds[0]
    if preds.ndim != 2 or preds.shape[1] != 85:
        raise DetectorError(
            f"unexpected yolox output shape {raw.shape}: expected [1, N, 85]"
        )

    in_h, in_w = input_size
    grids, expanded_strides = [], []
    for stride in STRIDES:
        hs, ws = in_h // stride, in_w // stride
        if hs <= 0 or ws <= 0:
            raise DetectorError(f"input size {input_size} not divisible by stride {stride}")
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], grid.shape[1], 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    preds[:, :2] = (preds[:, :2] + grids) * expanded_strides
    preds[:, 2:4] = np.exp(preds[:, 2:4]) * expanded_strides

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = (cx - w / 2) / ratio
    y1 = (cy - h / 2) / ratio
    x2 = (cx + w / 2) / ratio
    y2 = (cy + h / 2) / ratio
    boxes = np.stack([x1, y1, x2, y2], 1)
    scores = preds[:, 4:5] * preds[:, 5:]

    results: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for class_id in range(scores.shape[1]):
        mask = scores[:, class_id] > CANDIDATE_SCORE_FLOOR
        if not mask.any():
            continue
        cls_scores = scores[mask, class_id]
        cls_boxes = boxes[mask]
        results.extend(_nms(cls_scores, cls_boxes, class_id))

    normalized: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for class_id, confidence, (x1p, y1p, x2p, y2p) in results:
        normalized.append(
            (
                class_id,
                confidence,
                (
                    min(max(x1p / frame_w, 0.0), 1.0),
                    min(max(y1p / frame_h, 0.0), 1.0),
                    min(max(x2p / frame_w, 0.0), 1.0),
                    min(max(y2p / frame_h, 0.0), 1.0),
                ),
            )
        )
    return normalized


def _nms(cls_scores, cls_boxes, class_id: int) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """Single-class greedy NMS (official YOLOX numpy demo algorithm)."""
    import numpy as np

    x1, y1, x2, y2 = cls_boxes[:, 0], cls_boxes[:, 1], cls_boxes[:, 2], cls_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = cls_scores.argsort()[::-1]
    kept: list[tuple[int, float, tuple[float, float, float, float]]] = []
    while order.size > 0:
        i = order[0]
        score = float(cls_scores[i])
        box = tuple(float(v) for v in cls_boxes[i])
        kept.append((class_id, score, box))  # type: ignore[arg-type]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= NMS_IOU)[0] + 1]
    return kept


class YoloXOnnxDetector:
    """YOLOX-Tiny ONNX detector behind the Detector interface."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        classes: Sequence[str] | None = DEFAULT_CLASSES,
        max_detections: int = MAX_DETECTIONS,
        model_id: str = YOLOX_SPEC["model_id"],
        model_version: str = YOLOX_SPEC["version"],
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise DetectorError(
                f"invalid configuration: confidence_threshold {confidence_threshold} outside [0, 1]"
            )
        selected = tuple(classes) if classes is not None else COCO_CLASSES
        if not selected:
            raise DetectorError("invalid configuration: classes must not be empty (use COCO classes)")
        unknown = [c for c in selected if c not in COCO_CLASSES]
        if unknown:
            raise DetectorError(
                f"invalid configuration: unknown class(es) {unknown}; YOLOX COCO taxonomy only"
            )
        if len(input_size) != 2 or any(int(v) <= 0 for v in input_size):
            raise DetectorError(f"invalid configuration: input_size {input_size!r} must be (h, w) > 0")

        self.confidence_threshold = float(confidence_threshold)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.classes = tuple(selected)
        self.max_detections = int(max_detections)
        self.model_id = model_id
        self.model_version = model_version
        self.last_inference_ms: float | None = None
        self.last_detection_count: int | None = None

        self.weights_path = _resolve_weights_path(weights_path)
        if not self.weights_path.is_file():
            raise DetectorError(
                f"yolox weights not found at {self.weights_path.name} "
                f"(expected under models/ relative to the repository root). "
                f"Run explicitly: python scripts/download_models.py --model yolox-tiny "
                f"[pinned sha256 {YOLOX_SPEC['sha256'][:16]}...]"
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectorError(
                "onnxruntime not installed; install with: pip install 'sentinel[infer]'"
            ) from exc

        try:
            self._session = ort.InferenceSession(
                str(self.weights_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise DetectorError(f"failed to load yolox weights {self.weights_path.name}: {exc}") from exc

        model_input = self._session.get_inputs()[0]
        graph_shape = model_input.shape  # e.g. [1, 3, 416, 416]
        if len(graph_shape) == 4 and all(isinstance(v, int) for v in graph_shape[2:4]):
            graph_hw = (graph_shape[2], graph_shape[3])
            if graph_hw != self.input_size:
                raise DetectorError(
                    f"invalid configuration: graph expects input {graph_hw}, requested {self.input_size}"
                )
            self.input_size = graph_hw
        self._input_name = model_input.name
        self._output_name = self._session.get_outputs()[0].name

        self.info = ModelInfo(
            name=model_id,
            version=model_version,
            family="yolox",
            license=YOLOX_SPEC["license"],
            classes=self.classes,
            input_size=self.input_size,
            metrics={
                "status": "not-evaluated",
                "evaluation": "EVALUATION DATASET NOT AVAILABLE",
                "weights_file": self.weights_path.name,
                "weights_sha256": YOLOX_SPEC["sha256"],
                "confidence_threshold": self.confidence_threshold,
                "last_inference_ms": None,
            },
            status="candidate",
        )
        log.info(
            "yolox detector initialized",
            extra={
                "weights": self.weights_path.name,
                "classes": list(self.classes),
                "confidence_threshold": self.confidence_threshold,
                "input_size": list(self.input_size),
            },
        )

    def warmup(self) -> None:
        import numpy as np

        dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
        tensor, _ = self._preprocess(dummy)
        start = time.perf_counter()
        self._session.run([self._output_name], {self._input_name: tensor[None]})
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0

    def _preprocess(self, image):
        """Letterbox + BGR->RGB + float32 0..255 (normalization is in-graph)."""
        import numpy as np

        canvas, ratio = letterbox(image, self.input_size)
        rgb = canvas[:, :, ::-1]
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        return tensor, ratio

    def detect(self, frame: FramePacket) -> list[Detection]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise DetectorError("opencv not installed; install sentinel[video]") from exc

        array = np.frombuffer(frame.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise DetectorError("frame decode failed")
        frame_h, frame_w = image.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            raise DetectorError("frame decode failed: empty frame")

        tensor, ratio = self._preprocess(image)
        start = time.perf_counter()
        try:
            output = self._session.run([self._output_name], {self._input_name: tensor[None]})[0]
        except Exception as exc:
            raise DetectorError(f"yolox inference failed: {exc}") from exc
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0

        try:
            candidates = decode_and_nms(output, self.input_size, ratio, frame_w, frame_h)
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"yolox output post-processing failed: {exc}") from exc

        allowed = set(self.classes)
        candidates.sort(key=lambda c: -c[1])
        detections: list[Detection] = []
        for class_id, confidence, (x1, y1, x2, y2) in candidates:
            if confidence < self.confidence_threshold:
                continue
            class_name = COCO_CLASSES[class_id]
            if class_name not in allowed:
                continue
            bbox = BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1).normalized()
            if bbox.w <= 0.0 or bbox.h <= 0.0:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    timestamp=frame.ts,
                    model_id=self.model_id,
                    model_version=self.model_version,
                )
            )
            if len(detections) >= self.max_detections:
                break

        detections.sort(key=lambda d: (-d.confidence, d.class_id or 0))
        self.last_detection_count = len(detections)
        self.info.metrics["last_inference_ms"] = self.last_inference_ms
        return detections

    def close(self) -> None:
        self._session = None
