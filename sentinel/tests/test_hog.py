import math
from datetime import datetime, timezone

import pytest

from services.camera.types import FramePacket
from services.inference.hog import HOG_INFO, DetectorError, HogPeopleDetector, normalize_hog

T0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_hog_info_marks_development_only():
    assert HOG_INFO.name == "opencv-hog-people"
    assert HOG_INFO.license.startswith("Apache-2.0")
    assert HOG_INFO.status == "development"
    assert HOG_INFO.classes == ("person",)
    assert HOG_INFO.metrics.get("status") == "not-evaluated"


def test_registry_exposes_hog_profile():
    from services.inference.interfaces import default_registry

    registry = default_registry()
    assert "hog" in registry.profiles()
    detector = registry.create("hog")
    assert isinstance(detector, HogPeopleDetector)
    assert detector.info.name == "opencv-hog-people"


def test_normalize_hog_maps_pixels_to_normalized_bbox():
    detections = normalize_hog(
        rects=[[100.0, 50.0, 80.0, 160.0]],
        weights=[2.0],
        frame_width=640,
        frame_height=360,
        ts=T0,
        model_id="opencv-hog-people",
        model_version="4.14.0-dev1",
    )
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "person"
    assert det.timestamp == T0
    assert det.model_id == "opencv-hog-people"
    assert det.model_version == "4.14.0-dev1"
    assert det.bbox.x == pytest.approx(100.0 / 640.0)
    assert det.bbox.y == pytest.approx(50.0 / 360.0)
    assert det.bbox.w == pytest.approx(80.0 / 640.0)
    assert det.bbox.h == pytest.approx(160.0 / 360.0)
    assert det.confidence == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))


def test_normalize_hog_filters_low_confidence_and_degenerate_boxes():
    detections = normalize_hog(
        rects=[[10.0, 10.0, 80.0, 160.0], [0.0, 0.0, 0.0, 100.0]],
        weights=[-5.0, 4.0],
        frame_width=640,
        frame_height=360,
        ts=T0,
        model_id="m",
        model_version="v",
        min_confidence=0.3,
    )
    assert detections == []


def test_normalize_hog_rejects_invalid_frame_size():
    assert normalize_hog([[0, 0, 10, 10]], [1.0], 0, 360, T0, "m", "v") == []


def test_detection_to_dict_is_json_friendly():
    detections = normalize_hog([[0, 0, 100, 200]], [1.0], 640, 360, T0, "m", "1.0")
    payload = detections[0].to_dict()
    assert payload["timestamp"] == T0.isoformat()
    assert payload["model_id"] == "m"
    assert payload["bbox"] == {"x": 0.0, "y": 0.0, "w": 100.0 / 640.0, "h": 200.0 / 360.0}


def test_detector_raises_detector_error_on_undecodable_frame():
    detector = HogPeopleDetector()
    packet = FramePacket(camera_id="cam1", frame_id=0, ts=T0, data=b"not-a-jpeg")
    with pytest.raises(DetectorError):
        detector.detect(packet)
    detector.close()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cv2") is None, reason="opencv not installed"
)
def test_detector_on_blank_image_returns_no_people():
    import cv2
    import numpy as np

    detector = HogPeopleDetector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    packet = FramePacket(
        camera_id="cam1", frame_id=0, ts=T0, data=buf.tobytes(), width=640, height=480
    )
    detections = detector.detect(packet)
    assert detections == []
    detector.close()
