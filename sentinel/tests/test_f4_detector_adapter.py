"""F4-H detector adapter tests: YOLOX behind the Detector contract.

Uses a small deterministic ONNX fixture (generated at test time, never
committed) for full-path inference tests, plus the real pinned weights when
present (skipped otherwise). No GPU required.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from services.camera.types import FramePacket
from services.inference.interfaces import DetectorError, default_registry
from services.inference.yolox import (
    COCO_CLASSES,
    DEFAULT_CLASSES,
    YOLOX_SPEC,
    YoloXOnnxDetector,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_WEIGHTS = ROOT / "models" / YOLOX_SPEC["filename"]
LOCK_PATH = ROOT / "scripts" / "weights.lock.json"

IN_H, IN_W = 416, 416
FRAME_W, FRAME_H = 640, 480
S8 = IN_H // 8  # 52


def _frame(width: int = FRAME_W, height: int = FRAME_H, data: bytes | None = None) -> FramePacket:
    if data is None:
        import cv2
        import numpy as np

        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        data = buf.tobytes()
    return FramePacket(
        camera_id="cam-test",
        frame_id=1,
        ts=datetime(2026, 10, 3, 12, 0, 0, tzinfo=timezone.utc),
        data=data,
    )


def _empty_head() -> "object":
    import numpy as np

    return np.zeros((1, 3549, 85), dtype=np.float32)


def _put_box(head, gx: int, gy: int, cx: float, cy: float, w: float, h: float, obj: float, cls_id: int, cls_score: float) -> None:
    import math

    idx = gy * S8 + gx  # stride-8 rows are the first S8*S8 entries
    head[0, idx, 0] = cx / 8.0 - gx
    head[0, idx, 1] = cy / 8.0 - gy
    head[0, idx, 2] = math.log(w / 8.0)
    head[0, idx, 3] = math.log(h / 8.0)
    head[0, idx, 4] = obj
    head[0, idx, 5 + cls_id] = cls_score


def _build_fixture(path: Path, head) -> Path:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    const = numpy_helper.from_array(head, name="const_out")
    zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero")
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, IN_H, IN_W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(head.shape))
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["sum"], keepdims=0),
        helper.make_node("Mul", ["sum", "zero"], ["zeroed"]),
        helper.make_node("Add", ["const_out", "zeroed"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "sentinel-fixture", [inp], [out], initializer=[const, zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@pytest.fixture()
def fixture_weights(tmp_path) -> Path:
    head = _empty_head()
    # person A: conf 0.95 * 0.93 = 0.8835 (kept at default threshold)
    _put_box(head, gx=10, gy=10, cx=84.0, cy=84.0, w=80.0, h=160.0, obj=0.95, cls_id=0, cls_score=0.93)
    # person B: conf 0.35 * 0.60 = 0.21 (above NMS floor, below 0.30 threshold)
    _put_box(head, gx=30, gy=30, cx=244.0, cy=244.0, w=48.0, h=48.0, obj=0.35, cls_id=0, cls_score=0.60)
    # car: conf 0.90 * 0.88 = 0.792 (filtered by default classes=("person",))
    _put_box(head, gx=40, gy=10, cx=324.0, cy=84.0, w=64.0, h=64.0, obj=0.90, cls_id=2, cls_score=0.88)
    return _build_fixture(tmp_path / "fixture.onnx", head)


@pytest.fixture()
def broken_weights(tmp_path) -> Path:
    head = np.zeros((1, 100, 10), dtype=np.float32)
    return _build_fixture(tmp_path / "broken.onnx", head)


# ---------------------------------------------------------------- initialization


def test_registry_exposes_yolox_profile():
    assert "yolox" in default_registry().profiles()


def test_detector_error_is_shared_contract():
    from services.inference import hog

    assert hog.DetectorError is DetectorError


def test_missing_weights_is_deterministic_configuration_error(tmp_path):
    with pytest.raises(DetectorError) as exc:
        YoloXOnnxDetector(weights_path=tmp_path / "nope.onnx")
    message = str(exc.value)
    assert "download_models.py" in message
    assert "not found" in message


def test_invalid_confidence_threshold_rejected(fixture_weights):
    with pytest.raises(DetectorError, match="confidence_threshold"):
        YoloXOnnxDetector(weights_path=fixture_weights, confidence_threshold=1.5)
    with pytest.raises(DetectorError, match="confidence_threshold"):
        YoloXOnnxDetector(weights_path=fixture_weights, confidence_threshold=-0.1)


def test_unknown_class_rejected(fixture_weights):
    with pytest.raises(DetectorError, match="unknown class"):
        YoloXOnnxDetector(weights_path=fixture_weights, classes=["spaceship"])


def test_empty_class_list_rejected(fixture_weights):
    with pytest.raises(DetectorError, match="classes"):
        YoloXOnnxDetector(weights_path=fixture_weights, classes=[])


def test_input_size_mismatch_with_graph_rejected(fixture_weights):
    with pytest.raises(DetectorError, match="graph expects input"):
        YoloXOnnxDetector(weights_path=fixture_weights, input_size=(640, 640))


def test_missing_onnxruntime_reported_clearly(fixture_weights, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(DetectorError, match="onnxruntime not installed"):
        YoloXOnnxDetector(weights_path=fixture_weights)


def test_corrupted_weights_reported_clearly(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file")
    with pytest.raises(DetectorError, match="failed to load"):
        YoloXOnnxDetector(weights_path=bad)


def test_spec_matches_pinned_lock():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    artifact = next(a for a in lock["artifacts"] if a["id"] == YOLOX_SPEC["model_id"])
    for key in ("filename", "version", "sha256", "url", "size_bytes"):
        assert artifact[key] == YOLOX_SPEC[key], f"{key} out of sync with weights.lock.json"


# ---------------------------------------------------------------- inference path


def test_valid_inference_fixture_normalized_boxes(fixture_weights):
    det = YoloXOnnxDetector(weights_path=fixture_weights)
    detections = det.detect(_frame())
    assert len(detections) == 1
    person = detections[0]
    assert person.class_name == "person"
    assert person.class_id == 0
    assert person.confidence == pytest.approx(0.8835, abs=1e-3)
    # hand-computed normalization: ratio = 640/416 -> 0.65 input scale
    assert person.bbox.x == pytest.approx(0.1058, abs=0.01)
    assert person.bbox.y == pytest.approx(0.0128, abs=0.01)
    assert person.bbox.w == pytest.approx(0.1923, abs=0.01)
    assert person.bbox.h == pytest.approx(0.5128, abs=0.01)
    for d in detections:
        assert 0.0 <= d.bbox.x <= 1.0 and 0.0 <= d.bbox.y <= 1.0
        assert 0.0 < d.bbox.w <= 1.0 and 0.0 < d.bbox.h <= 1.0
        assert 0.0 <= d.bbox.x + d.bbox.w <= 1.0001
        assert 0.0 <= d.bbox.y + d.bbox.h <= 1.0001


def test_confidence_threshold_filters(fixture_weights):
    lenient = YoloXOnnxDetector(weights_path=fixture_weights, confidence_threshold=0.15, classes=("person",))
    detections = lenient.detect(_frame())
    confs = sorted(d.confidence for d in detections)
    assert confs == pytest.approx([0.21, 0.8835], abs=1e-3)
    strict = YoloXOnnxDetector(weights_path=fixture_weights, confidence_threshold=0.3, classes=("person",))
    assert len(strict.detect(_frame())) == 1


def test_class_mapping_filter(fixture_weights):
    det = YoloXOnnxDetector(weights_path=fixture_weights, confidence_threshold=0.15, classes=("person", "car"))
    names = {d.class_name for d in det.detect(_frame())}
    assert names == {"person", "car"}
    assert "car" in COCO_CLASSES and COCO_CLASSES[2] == "car"


def test_model_provenance_on_detections_and_info(fixture_weights):
    det = YoloXOnnxDetector(weights_path=fixture_weights)
    detections = det.detect(_frame())
    assert detections
    for d in detections:
        assert d.model_id == "yolox-tiny"
        assert d.model_version == "0.1.1rc0"
    assert det.info.name == "yolox-tiny"
    assert det.info.status == "candidate"
    assert det.info.license.startswith("Apache-2.0")
    assert det.info.metrics["status"] == "not-evaluated"
    assert det.info.metrics["evaluation"] == "EVALUATION DATASET NOT AVAILABLE"
    assert det.info.metrics["weights_sha256"] == YOLOX_SPEC["sha256"]


def test_deterministic_output_shape(fixture_weights):
    det = YoloXOnnxDetector(weights_path=fixture_weights)
    first = [(d.class_id, d.class_name, d.confidence, d.bbox.to_dict()) for d in det.detect(_frame())]
    second = [(d.class_id, d.class_name, d.confidence, d.bbox.to_dict()) for d in det.detect(_frame())]
    assert first == second


def test_inference_timing_recorded(fixture_weights):
    det = YoloXOnnxDetector(weights_path=fixture_weights)
    det.detect(_frame())
    assert det.last_inference_ms is not None and det.last_inference_ms > 0
    assert det.info.metrics["last_inference_ms"] == det.last_inference_ms
    assert det.last_detection_count == 1


def test_corrupted_frame_raises_detector_error(fixture_weights):
    det = YoloXOnnxDetector(weights_path=fixture_weights)
    with pytest.raises(DetectorError, match="frame decode failed"):
        det.detect(_frame(data=b"definitely-not-a-jpeg"))


def test_invalid_detection_output_raises(fixture_weights, broken_weights):
    det = YoloXOnnxDetector(weights_path=fixture_weights)
    from services.inference.yolox import decode_and_nms

    import numpy as np

    with pytest.raises(DetectorError, match="unexpected yolox output shape"):
        decode_and_nms(np.zeros((1, 100, 10), dtype=np.float32), (416, 416), 1.0, 640, 480)
    with pytest.raises(DetectorError, match="unexpected yolox output shape"):
        decode_and_nms(np.zeros((3549, 84), dtype=np.float32), (416, 416), 1.0, 640, 480)


def test_warmup_and_close_safe(fixture_weights):
    det = YoloXOnnxDetector(weights_path=fixture_weights)
    det.warmup()
    assert det.last_inference_ms is not None
    det.close()
    assert det._session is None


def test_default_class_filter_is_person():
    assert DEFAULT_CLASSES == ("person",)


# ------------------------------------------------------- real pinned weights


@pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="real yolox weights not acquired (scripts/download_models.py)")
def test_real_weights_full_path():
    det = YoloXOnnxDetector()
    det.warmup()
    detections = det.detect(_frame())
    assert isinstance(detections, list)
    for d in detections:
        assert d.class_name in COCO_CLASSES
        assert 0.0 <= d.bbox.x + d.bbox.w <= 1.0001
        assert 0.0 <= d.bbox.y + d.bbox.h <= 1.0001
        assert d.confidence >= det.confidence_threshold
        assert d.model_version == "0.1.1rc0"


@pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="real yolox weights not acquired (scripts/download_models.py)")
def test_real_weights_registry_profile_builds():
    det = default_registry().create("yolox")
    assert det.info.name == "yolox-tiny"
    assert det.info.status == "candidate"
    det.close()


# ------------------------------------------------------- weight management


def test_download_script_check_mode_passes_when_present():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "download_models.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if not REAL_WEIGHTS.is_file():
        assert result.returncode == 1
        assert "MISSING" in result.stdout
    else:
        assert result.returncode == 0
        assert "sha256 verified" in result.stdout


def test_weights_are_gitignored():
    import subprocess

    if not REAL_WEIGHTS.is_file():
        pytest.skip("weights not present")
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(REAL_WEIGHTS)],
        cwd=str(ROOT),
    )
    assert result.returncode == 0, "model weights must be gitignored"
