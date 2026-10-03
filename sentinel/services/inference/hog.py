"""F1 development detector: OpenCV HOG people detector.

DEVELOPMENT / NON-COMMERCIAL USE ONLY - not an approved production model.
License: Apache-2.0 (bundled OpenCV SVM+HOG pedestrian weights, opencv-python 4.x).
Person class only. Accuracy never evaluated on an evaluation dataset - see
docs/MODEL_REGISTRY.md entry "opencv-hog-people".
"""

from __future__ import annotations

import math

from packages.common.logging import get_logger
from services.camera.types import FramePacket, SourceReadError
from services.inference.interfaces import DetectorError  # noqa: F401  (re-export)
from services.inference.types import BBox, Detection, ModelInfo

log = get_logger(__name__)

HOG_INFO = ModelInfo(
    name="opencv-hog-people",
    version="4.14.0-dev1",
    family="hog-svm",
    license="Apache-2.0 (OpenCV bundled pedestrian detector)",
    classes=("person",),
    input_size=(64, 128),
    metrics={"status": "not-evaluated"},
    status="development",
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def normalize_hog(
    rects,
    weights,
    frame_width: int,
    frame_height: int,
    ts,
    model_id: str,
    model_version: str,
    min_confidence: float = 0.0,
) -> list[Detection]:
    """Convert raw HOG outputs (px xywh + SVM weights) into normalized Detections."""
    detections: list[Detection] = []
    if frame_width <= 0 or frame_height <= 0:
        return detections
    for rect, weight in zip(rects, weights):
        x, y, w, h = (float(v) for v in rect)
        score = float(weight)
        confidence = _sigmoid(score)
        if confidence < min_confidence:
            continue
        bbox = BBox(
            x=x / frame_width,
            y=y / frame_height,
            w=w / frame_width,
            h=h / frame_height,
        ).normalized()
        if bbox.w <= 0.0 or bbox.h <= 0.0:
            continue
        detections.append(
            Detection(
                class_name="person",
                confidence=confidence,
                bbox=bbox,
                timestamp=ts,
                model_id=model_id,
                model_version=model_version,
            )
        )
    return detections


class HogPeopleDetector:
    """OpenCV HOG pedestrian detector behind the Detector interface."""

    def __init__(
        self,
        min_confidence: float = 0.3,
        win_stride: tuple[int, int] = (8, 8),
        padding: tuple[int, int] = (8, 8),
        hit_threshold: float = 0.0,
    ) -> None:
        self.info = HOG_INFO
        self.min_confidence = min_confidence
        self.win_stride = win_stride
        self.padding = padding
        self.hit_threshold = hit_threshold
        self._hog = None

    def warmup(self) -> None:
        import cv2

        if self._hog is None:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog

    def detect(self, frame: FramePacket) -> list[Detection]:
        if self._hog is None:
            self.warmup()
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise DetectorError(
                "opencv not installed; install sentinel[video]"
            ) from exc
        array = np.frombuffer(frame.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise DetectorError("frame decode failed")
        height, width = image.shape[:2]
        try:
            rects, weights = self._hog.detectMultiScale(
                image,
                winStride=self.win_stride,
                padding=self.padding,
                hitThreshold=self.hit_threshold,
            )
        except Exception as exc:
            raise DetectorError(f"hog inference failed: {exc}") from exc
        return normalize_hog(
            rects,
            weights,
            width,
            height,
            frame.ts,
            self.info.name,
            self.info.version,
            min_confidence=self.min_confidence,
        )

    def close(self) -> None:
        self._hog = None
