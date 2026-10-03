"""F4-I/J/K/L tests: benchmark runner, evaluation mechanics smoke,
threshold-analysis infrastructure and error-analysis records.

All numbers produced here are MECHANICS on synthetic fixtures, confined to
temporary files - they are never accuracy claims (F4 critical rule).
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation.runner import UNAVAILABLE_STATUS, run_evaluation
from evaluation.threshold_sweep import NOT_EVALUATED, sweep_thresholds
from services.inference.types import BBox, Detection, ModelInfo

ROOT = Path(__file__).resolve().parents[1]
REAL_WEIGHTS = ROOT / "models" / "yolox_tiny.onnx"

from tests.test_f4_detector_adapter import (  # noqa: E402
    _build_fixture,
    _empty_head,
    _put_box,
)


class ScriptedDetector:
    info = ModelInfo(
        name="scripted",
        version="9.9",
        family="test",
        license="apache-2.0",
        classes=("person",),
        input_size=(416, 416),
    )

    def __init__(self, boxes=(), confidence_threshold=None):
        self._boxes = list(boxes)
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        return [
            Detection(
                class_name=cls,
                confidence=conf,
                bbox=BBox(*bbox),
                timestamp=frame.ts,
                model_id=self.info.name,
                model_version=self.info.version,
            )
            for cls, bbox, conf in self._boxes
        ]

    def warmup(self):
        return None

    def close(self):
        return None


def _jpg_bytes():
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _dataset(tmp_path, boxes):
    """One-entry labeled dataset. boxes: (class, (x,y,w,h))."""
    media = tmp_path / "media"
    ann = tmp_path / "annotations"
    media.mkdir()
    ann.mkdir()
    (media / "e1.jpg").write_bytes(_jpg_bytes())
    (ann / "e1.json").write_text(
        json.dumps(
            {
                "entry_id": "e1",
                "frames": [
                    {"frame_index": 0, "boxes": [{"class_name": c, "bbox": list(b)} for c, b in boxes]}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "f4-fixture",
                "dataset_id": "f4-fixture",
                "version": "1",
                "license": "CC0-1.0",
                "source": "synthetic mechanics fixture",
                "class_mapping": {"person": "person"},
                "entries": [
                    {
                        "entry_id": "e1",
                        "media": "media/e1.jpg",
                        "media_type": "image",
                        "annotation": "annotations/e1.json",
                        "camera_id": "cam-fixture",
                        "capture_context": "synthetic",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


# ------------------------------------------------------------------ F4-I


def test_benchmark_runner_works(tmp_path):
    out = tmp_path / "bench.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_pipeline.py"),
            "--source", "synthetic",
            "--frames", "5",
            "--profile", "stub",
            "--json", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "DEVELOPMENT ENVIRONMENT BENCHMARK" in result.stdout
    assert "NOT an accuracy benchmark" in result.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["label"] == "DEVELOPMENT ENVIRONMENT"
    assert payload["frames"] == 5
    assert payload["detector_latency_ms"]["p50"] is not None
    assert payload["warmup_policy"]["warmup_calls"] == 1
    assert payload["model"]["batch_size"] == 1
    assert payload["model"]["input_resolution"]
    assert payload["software"]["python"]
    assert payload["throughput_fps"] > 0


@pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="yolox weights not acquired")
def test_benchmark_runner_with_yolox(tmp_path):
    out = tmp_path / "bench-yolox.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_pipeline.py"),
            "--source", "synthetic",
            "--frames", "8",
            "--profile", "yolox",
            "--json", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["model"]["id"] == "yolox-tiny"
    assert payload["model"]["version"] == "0.1.1rc0"
    assert payload["model"]["status"] == "candidate"
    assert payload["model"]["confidence_threshold"] == 0.3
    assert payload["software"]["onnxruntime"]
    assert payload["detector_errors"] == 0
    assert payload["model_versions"]["detector"] == "yolox-tiny:0.1.1rc0"


# ------------------------------------------------------------------ F4-J


def test_evaluation_runner_reports_unavailable_without_dataset(tmp_path):
    report = run_evaluation(tmp_path / "absent.json", ScriptedDetector())
    assert report["status"] == UNAVAILABLE_STATUS
    assert report["metrics"] is None


def test_evaluation_mechanics_smoke_with_yolox_fixture(tmp_path):
    """F4-D item 5: synthetic/test fixtures verify MECHANICS only."""
    head = _empty_head()
    _put_box(head, gx=10, gy=10, cx=84.0, cy=84.0, w=80.0, h=160.0, obj=0.95, cls_id=0, cls_score=0.93)
    weights = _build_fixture(tmp_path / "fixture.onnx", head)

    from services.inference.yolox import YoloXOnnxDetector

    manifest = _dataset(
        tmp_path,
        [("person", (0.1058, 0.0128, 0.1923, 0.5128))],  # matches fixture box A
    )
    detector = YoloXOnnxDetector(weights_path=weights, confidence_threshold=0.3)
    report_path = tmp_path / "reports" / "smoke.json"
    report = run_evaluation(manifest, detector, output_path=report_path)

    assert report["status"] == "completed"
    assert report["model"]["name"] == "yolox-tiny"
    assert report["model"]["version"] == "0.1.1rc0"
    metrics = report["metrics"]["object_detection"]
    assert metrics["tp"] == 1  # mechanics: fixture GT matches fixture prediction
    assert metrics["fp"] == 0 and metrics["fn"] == 0
    assert report["latency_ms"]["p50"] is not None
    # report stays in tmp - no evaluation artifacts committed
    assert str(report_path).startswith(str(tmp_path))


# ------------------------------------------------------------------ F4-K


def test_threshold_sweep_unavailable_dataset(tmp_path):
    result = sweep_thresholds(tmp_path / "absent.json", ScriptedDetector())
    assert result["status"] == UNAVAILABLE_STATUS
    assert result["rows"] == []
    assert result["operational_threshold"] == NOT_EVALUATED
    assert "NOT EVALUATED" in result["notes"]


def test_threshold_sweep_produces_rows_but_never_selects(tmp_path):
    manifest = _dataset(tmp_path, [("person", (0.4, 0.3, 0.2, 0.4))])
    detector = ScriptedDetector(boxes=[("person", (0.4, 0.3, 0.2, 0.4), 0.55)])
    out = tmp_path / "sweep.json"
    result = sweep_thresholds(
        manifest, detector, thresholds=(0.3, 0.5, 0.6), output_path=out
    )
    assert result["status"] == "completed"
    assert [r["threshold"] for r in result["rows"]] == [0.3, 0.5, 0.6]
    # conf 0.55: passes 0.3/0.5, fails 0.6 -> observed metric change, no selection
    assert result["rows"][0]["f1"] == pytest.approx(1.0)
    assert result["rows"][2]["f1"] == pytest.approx(0.0)
    assert result["operational_threshold"] == NOT_EVALUATED
    assert result["best_f1_observation"]["threshold"] in (0.3, 0.5)
    assert json.loads(out.read_text(encoding="utf-8"))["operational_threshold"] == NOT_EVALUATED


def test_threshold_sweep_warns_when_detector_floor_caps_range(tmp_path):
    manifest = _dataset(tmp_path, [("person", (0.4, 0.3, 0.2, 0.4))])
    detector = ScriptedDetector(confidence_threshold=0.3)
    result = sweep_thresholds(manifest, detector, thresholds=(0.2, 0.4))
    assert "warning" in result
    assert "floor" in result["warning"]


# ------------------------------------------------------------------ F4-L


def test_error_examples_include_iou_and_model_version(tmp_path):
    manifest = _dataset(tmp_path, [("person", (0.4, 0.3, 0.2, 0.4))])
    detector = ScriptedDetector(boxes=[("person", (0.0, 0.0, 0.1, 0.1), 0.7)])
    report = run_evaluation(manifest, detector)
    examples = report["error_examples"]
    assert {e["kind"] for e in examples} == {"false_negative", "false_positive"}
    for e in examples:
        assert e["model_version"] == "9.9"
        assert e["iou"] is not None and 0.0 <= e["iou"] <= 1.0
        assert "expected_class" in e and "predicted_class" in e
    fn = next(e for e in examples if e["kind"] == "false_negative")
    assert fn["expected_class"] == "person" and fn["predicted_class"] is None
    fp = next(e for e in examples if e["kind"] == "false_positive")
    assert fp["predicted_class"] == "person" and fp["confidence"] == 0.7


def test_error_examples_capped_at_ten(tmp_path):
    boxes = [("person", (0.4, 0.3, 0.2, 0.4))]
    manifest = _dataset(tmp_path, boxes)
    # every prediction misses -> many FP/FN records; cap must hold
    detector = ScriptedDetector(boxes=[(f"person", (0.9, 0.9, 0.05, 0.05), 0.9)] * 30)
    report = run_evaluation(manifest, detector)
    assert len(report["error_examples"]) <= 10
