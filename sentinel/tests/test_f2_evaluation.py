import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation.annotations import AnnotationFile, Box, FrameAnnotation
from evaluation.metrics import DetectionMetrics, frame_matches, iou
from evaluation.runner import UNAVAILABLE_STATUS, run_evaluation
from services.inference.types import BBox, Detection, ModelInfo

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


class StubDetector:
    """Deterministic detector: boxes keyed by camera_id (one camera per entry)."""

    info = ModelInfo(
        name="stub-eval",
        version="1.0",
        family="test",
        license="apache-2.0",
        classes=("person",),
    )

    def __init__(self, script=None):
        self.script = script or {}
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [
            Detection(
                class_name=cls,
                confidence=conf,
                bbox=BBox(*bbox),
                timestamp=frame.ts,
                model_id=self.info.name,
                model_version=self.info.version,
            )
            for cls, bbox, conf in self.script.get(frame.camera_id, [])
        ]

    def warmup(self):
        return None

    def close(self):
        return None


def _jpg_bytes():
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _build_dataset(tmp_path, entries):
    """Create media/annotations/manifest. entries: (entry_id, camera_id, boxes)."""
    media_dir = tmp_path / "media"
    ann_dir = tmp_path / "annotations"
    media_dir.mkdir()
    ann_dir.mkdir()

    manifest_entries = []
    for entry_id, camera_id, boxes in entries:
        (media_dir / f"{entry_id}.jpg").write_bytes(_jpg_bytes())
        annotation = AnnotationFile(
            entry_id=entry_id,
            frames=[
                FrameAnnotation(
                    frame_index=0,
                    boxes=[Box(class_name=c, bbox=list(b)) for c, b in boxes],
                )
            ],
        )
        (ann_dir / f"{entry_id}.json").write_text(
            annotation.model_dump_json(), encoding="utf-8"
        )
        manifest_entries.append(
            {
                "entry_id": entry_id,
                "media": f"media/{entry_id}.jpg",
                "media_type": "image",
                "annotation": f"annotations/{entry_id}.json",
                "camera_id": camera_id,
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"name": "unit-eval", "version": "1", "license": "CC0-1.0",
                    "source": "synthetic test data", "entries": manifest_entries}),
        encoding="utf-8",
    )
    return manifest_path


def test_iou_known_values():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert iou([0, 0, 1, 1], [2, 2, 1, 1]) == 0.0
    assert iou([0, 0, 2, 2], [1, 0, 2, 2]) == pytest.approx(1 / 3)


def test_frame_matches_greedy_by_confidence():
    gt = [Box.xywh("person", 0.1, 0.1, 0.2, 0.2)]
    preds = [
        Box(class_name="person", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.5),
        Box(class_name="person", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.9),
    ]
    matches = frame_matches(gt, preds, iou_threshold=0.5)
    assert len(matches) == 1
    assert matches[0][1].confidence == 0.9


def test_metrics_precision_recall_f1_and_map():
    metrics = DetectionMetrics(iou_threshold=0.5)
    metrics.add_frame(
        [Box.xywh("person", 0.1, 0.1, 0.2, 0.2)],
        [Box(class_name="person", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.9)],
    )
    metrics.add_frame(
        [Box.xywh("person", 0.5, 0.5, 0.2, 0.2)],
        [Box(class_name="person", bbox=[0.9, 0.9, 0.1, 0.1], confidence=0.8)],
    )
    precision, recall, f1 = metrics.precision_recall()
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)
    summary = metrics.summary()
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["map"] == pytest.approx(0.5)
    assert "person" in summary["per_class"]


def test_runner_missing_manifest(tmp_path):
    detector = StubDetector()
    report = run_evaluation(tmp_path / "nope.json", detector, output_path=tmp_path / "r.json")
    assert report["status"] == UNAVAILABLE_STATUS
    assert report["metrics"] is None
    assert UNAVAILABLE_STATUS in report["notes"]
    written = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert written["status"] == UNAVAILABLE_STATUS


def test_runner_empty_manifest(tmp_path):
    manifest = tmp_path / "empty.json"
    manifest.write_text(json.dumps({"name": "empty", "entries": []}), encoding="utf-8")
    report = run_evaluation(manifest, StubDetector())
    assert report["status"] == UNAVAILABLE_STATUS
    assert report["metrics"] is None


def test_runner_missing_media_reports_unavailable(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "ghost",
                "entries": [
                    {"entry_id": "g1", "media": "media/g1.jpg", "media_type": "image"}
                ],
            }
        ),
        encoding="utf-8",
    )
    report = run_evaluation(manifest, StubDetector())
    assert report["status"] == UNAVAILABLE_STATUS
    assert report["dataset"]["entries_skipped_missing_media"] == 1
    assert report["metrics"] is None


def test_runner_computes_metrics_on_labeled_media(tmp_path):
    manifest_path = _build_dataset(
        tmp_path,
        [
            ("e1", "cam-e1", [("person", (0.4, 0.3, 0.2, 0.4))]),
            ("e2", "cam-e2", [("person", (0.1, 0.1, 0.3, 0.3))]),
        ],
    )
    detector = StubDetector(
        script={"cam-e1": [("person", (0.4, 0.3, 0.2, 0.4), 0.9)]}
    )
    report_path = tmp_path / "reports" / "report.json"
    report = run_evaluation(
        manifest_path, detector, output_path=report_path, confidence_threshold=0.3
    )

    assert report["status"] == "completed"
    assert report["dataset"]["entries_evaluated"] == 2
    assert report["dataset"]["license"] == "CC0-1.0"
    assert report["model"]["name"] == "stub-eval"
    assert report["model"]["license"] == "apache-2.0"

    metrics = report["metrics"]["object_detection"]
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert report["metrics"]["event_detection"] is None

    assert report["latency_ms"] is not None
    assert report["latency_ms"]["count"] == 2
    assert report["latency_ms"]["p50"] is not None

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["status"] == "completed"
    assert written["config"]["iou_threshold"] == 0.5


def test_runner_records_error_examples(tmp_path):
    manifest_path = _build_dataset(
        tmp_path, [("e1", "cam-e1", [("person", (0.4, 0.3, 0.2, 0.4))])]
    )
    detector = StubDetector(
        script={"cam-e1": [("person", (0.0, 0.0, 0.1, 0.1), 0.7)]}
    )
    report = run_evaluation(manifest_path, detector)
    kinds = {e["kind"] for e in report["error_examples"]}
    assert kinds == {"false_negative", "false_positive"}
    assert report["error_examples"][0]["entry_id"] == "e1"


def test_example_template_is_schema_valid():
    from evaluation.manifest import load_manifest

    template = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "manifests"
        / "example.template.json"
    )
    manifest = load_manifest(template)
    assert manifest.name == "example-manifest"
    assert len(manifest.entries) == 2
    assert manifest.entries[0].media_type == "image"


def test_template_manifest_run_reports_unavailable(tmp_path):
    template = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "manifests"
        / "example.template.json"
    )
    report = run_evaluation(template, StubDetector(), output_path=tmp_path / "t.json")
    assert report["status"] == UNAVAILABLE_STATUS
    assert report["metrics"] is None
    assert report["dataset"]["entries_skipped_missing_media"] == 2
