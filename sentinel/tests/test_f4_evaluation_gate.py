"""F4-D/E evaluation gate tests: honest data state + deterministic validation."""

import json
from pathlib import Path

import pytest

from evaluation.validate_manifest import main, validate_manifest

DOCS = Path(__file__).resolve().parents[1]
TEMPLATE = DOCS / "evaluation" / "manifests" / "example.template.json"


def _write_annotation(path: Path, entry_id: str, boxes: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {"entry_id": entry_id, "frames": [{"frame_index": 0, "boxes": boxes}]}
        ),
        encoding="utf-8",
    )


def _write_manifest(tmp_path: Path, entries: list[dict], filename: str = "manifest.json", **overrides) -> Path:
    data = {
        "name": "unit-eval",
        "dataset_id": "sentinel-unit-eval",
        "version": "1.0",
        "license": "CC-BY-4.0",
        "source": "synthetic fixture built by tests",
        "class_mapping": {"person": "person", "car": "car"},
        "entries": entries,
    }
    data.update(overrides)
    path = tmp_path / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _valid_entry(tmp_path: Path) -> dict:
    (tmp_path / "media").mkdir(exist_ok=True)
    (tmp_path / "media" / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0fixture")
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(exist_ok=True)
    _write_annotation(
        ann_dir / "e1.json",
        "e1",
        [{"class_name": "person", "bbox": [0.1, 0.1, 0.2, 0.4]}],
    )
    return {
        "entry_id": "e1",
        "media": "media/a.jpg",
        "media_type": "image",
        "annotation": "annotations/e1.json",
        "source_resolution": "640x480",
        "capture_context": "synthetic",
    }


def test_valid_manifest_passes(tmp_path):
    path = _write_manifest(tmp_path, [_valid_entry(tmp_path)])
    report = validate_manifest(path)
    assert report.ok, report.render()
    assert report.errors == 0
    assert report.entries_total == 1
    assert report.entries_with_errors == 0


def test_missing_metadata_detected(tmp_path):
    path = _write_manifest(
        tmp_path,
        [_valid_entry(tmp_path)],
        dataset_id="",
        license="",
        source="",
    )
    report = validate_manifest(path)
    assert not report.ok
    assert sum(1 for i in report.issues if i.code == "missing_metadata" and i.severity == "error") == 3


def test_placeholder_metadata_warns(tmp_path):
    path = _write_manifest(tmp_path, [_valid_entry(tmp_path)], license="REPLACE: license")
    report = validate_manifest(path)
    assert report.ok  # placeholder = warning, not error
    assert any(i.code == "missing_metadata" and i.severity == "warning" for i in report.issues)


def test_duplicate_entry_ids_detected(tmp_path):
    entry = _valid_entry(tmp_path)
    path = _write_manifest(tmp_path, [entry, dict(entry)])
    report = validate_manifest(path)
    assert any(i.code == "duplicate_identifiers" for i in report.issues)
    assert not report.ok


def test_unsupported_format_detected(tmp_path):
    entry = _valid_entry(tmp_path)
    entry["media"] = "media/a.txt"
    path = _write_manifest(tmp_path, [entry])
    report = validate_manifest(path)
    assert any(i.code == "unsupported_format" and i.severity == "error" for i in report.issues)


def test_missing_files_detected(tmp_path):
    entry = _valid_entry(tmp_path)
    entry["media"] = "media/missing.jpg"
    path = _write_manifest(tmp_path, [entry])
    report = validate_manifest(path)
    assert any(i.code == "missing_files" and i.severity == "error" for i in report.issues)
    assert report.entries_with_errors == 1


def test_malformed_annotation_detected(tmp_path):
    entry = _valid_entry(tmp_path)
    ann = tmp_path / "annotations" / "e1.json"
    ann.write_text("{not json", encoding="utf-8")
    path = _write_manifest(tmp_path, [entry])
    report = validate_manifest(path)
    assert any(i.code == "malformed_annotations" for i in report.issues)


def test_unknown_class_detected(tmp_path):
    entry = _valid_entry(tmp_path)
    _write_annotation(
        tmp_path / "annotations" / "e1.json",
        "e1",
        [{"class_name": "spaceship", "bbox": [0.1, 0.1, 0.2, 0.4]}],
    )
    path = _write_manifest(tmp_path, [entry])
    report = validate_manifest(path)
    assert any(i.code == "unknown_classes" and i.severity == "error" for i in report.issues)


@pytest.mark.parametrize(
    "bbox",
    (
        [0.5, 0.5, 0.6, 0.2],  # exceeds frame
        [0.1, 0.1, 0.0, 0.4],  # zero width
        [-0.2, 0.1, 0.2, 0.4],  # negative origin
        [0.1, 0.1, 0.9, 0.9],  # exactly at edge is fine -> 0.1+0.9=1.0
    ),
)
def test_bounding_box_validation(tmp_path, bbox):
    entry = _valid_entry(tmp_path)
    _write_annotation(
        tmp_path / "annotations" / "e1.json",
        "e1",
        [{"class_name": "person", "bbox": bbox}],
    )
    path = _write_manifest(tmp_path, [entry])
    report = validate_manifest(path)
    has_box_error = any(i.code == "invalid_bounding_boxes" for i in report.issues)
    should_fail = bbox != [0.1, 0.1, 0.9, 0.9]
    assert has_box_error == should_fail, report.render()


def test_context_metadata_warnings(tmp_path):
    entry = _valid_entry(tmp_path)
    entry.pop("source_resolution")
    entry.pop("capture_context")
    path = _write_manifest(tmp_path, [entry])
    report = validate_manifest(path)
    assert any(i.code == "missing_metadata" and i.severity == "warning" and "capture_context" in i.detail for i in report.issues)


def test_template_manifest_structurally_loads():
    from evaluation.manifest import load_manifest

    manifest = load_manifest(TEMPLATE)
    assert manifest.dataset_id
    assert "REPLACE" in manifest.dataset_id
    assert manifest.class_mapping
    entry = manifest.entries[0]
    assert entry.source_resolution
    assert "REPLACE" in entry.capture_context


def test_template_validation_reports_missing_files_not_silently():
    report = validate_manifest(TEMPLATE)
    assert not report.ok
    codes = {i.code for i in report.issues}
    assert "missing_files" in codes


def test_cli_exit_codes(tmp_path):
    good = _write_manifest(tmp_path, [_valid_entry(tmp_path)], filename="good.json")
    bad = _write_manifest(tmp_path, [], filename="bad.json", dataset_id="")
    assert main([str(good), "--quiet"]) == 0
    assert main([str(bad), "--quiet"]) == 1


def test_cli_writes_json_report(tmp_path):
    good = _write_manifest(tmp_path, [_valid_entry(tmp_path)])
    out = tmp_path / "report.json"
    assert main([str(good), "--json", str(out), "--quiet"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["errors"] == 0
    assert "manifest_readable" in payload["checks"]


def test_acquisition_spec_documents_honest_gate():
    text = (DOCS / "datasets" / "README.md").read_text(encoding="utf-8")
    assert "EVALUATION DATASET NOT AVAILABLE" in text
    assert "BLOCKED" in text
    assert "validate_manifest" in text
    assert "no valid, licensed evaluation dataset" in text
