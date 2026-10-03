"""F2 doc-contract tests: MODEL_REGISTRY.md and DETECTOR_SELECTION.md stay honest.

Guards the acceptance criteria: closed status vocabulary, required registry
fields, HOG registered as development/not-evaluated, and the detector research
gate explicitly selecting no winner.
"""

from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

STATUS_VALUES = ("candidate", "development", "evaluated", "approved", "deprecated")
LEGACY_BACKTICKED = ("`blocked`", "`research-only`", "`development-only`")

REQUIRED_FIELDS = (
    "name",
    "source",
    "framework",
    "task",
    "intended-use",
    "license",
    "version",
    "commercial-use status",
    "redistribution restrictions",
    "training-data status",
    "metrics",
    "status",
    "notes",
)


def _read(name: str) -> str:
    path = DOCS / name
    assert path.is_file(), f"missing doc: docs/{name}"
    return path.read_text(encoding="utf-8")


def _line_after(text: str, prefix: str) -> str:
    """Return the first non-empty line after the first line starting with prefix."""
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith(prefix))
    for ln in lines[idx + 1 :]:
        if ln.strip():
            return ln.strip()
    raise AssertionError(f"no content after line starting with {prefix!r}")


def test_model_registry_status_vocabulary_is_closed():
    text = _read("MODEL_REGISTRY.md")
    vocab_line = _line_after(text, "Status values")
    assert vocab_line == "`candidate | development | evaluated | approved | deprecated`"
    for legacy in LEGACY_BACKTICKED:
        assert legacy not in text, f"legacy status {legacy} must not appear as a value"
    for value in STATUS_VALUES:
        assert f"`{value}`" in text


def test_model_registry_required_fields():
    text = _read("MODEL_REGISTRY.md")
    field_line = _line_after(text, "Required fields")
    for field in REQUIRED_FIELDS:
        assert field in field_line, f"missing required field '{field}'"
    for column in ("Framework", "Task", "Intended use"):
        assert f"| {column} |" in text, f"missing column {column}"


def test_model_registry_hog_is_development_and_not_evaluated():
    text = _read("MODEL_REGISTRY.md")
    hog_row = next(ln for ln in text.splitlines() if "`opencv-hog-people`" in ln)
    assert "`development`" in hog_row
    assert "Apache-2.0" in hog_row
    assert "not evaluated" in hog_row
    assert "`approved`" not in hog_row
    assert "Nothing in this registry has been evaluated yet." in text
    assert "No model is approved for production." in text


def test_model_registry_rows_use_only_valid_statuses():
    text = _read("MODEL_REGISTRY.md")
    table_rows = [ln for ln in text.splitlines() if ln.startswith("| ") and ln.count("|") > 5]
    counted = {value: 0 for value in STATUS_VALUES}
    for row in table_rows:
        for value in STATUS_VALUES:
            counted[value] += row.count(f"`{value}`")
        for legacy in LEGACY_BACKTICKED:
            assert legacy not in row, f"row uses legacy status {legacy}: {row[:80]}"
    assert counted["candidate"] >= 2
    assert counted["development"] >= 1
    assert counted["approved"] == 0, "no row may be approved in this phase"


def test_detector_selection_gate_is_honest():
    text = _read("DETECTOR_SELECTION.md")
    assert "No winner is selected" in text
    assert "No model is approved" in text
    assert "NOT SENTINEL measurements" in text
    # F2 statement preserved; F4 records the single real, checksum-pinned
    # acquisition and insists it is NOT an evaluation.
    assert "no weights were downloaded or run" in text
    assert "still **unevaluated**" in text
    assert "NO accuracy evaluation" in text
    for family in ("RT-DETRv2", "RF-DETR", "YOLOX"):
        assert family in text
    assert "AGPL" in text, "AGPL exclusions must be documented"
    assert "Sources" in text, "research must cite sources"
    assert "Evaluation plan" in text
    assert "FIRST EVALUATION CANDIDATE" in text


def test_detector_selection_cross_references_registry():
    selection = _read("DETECTOR_SELECTION.md")
    registry = _read("MODEL_REGISTRY.md")
    assert "MODEL_REGISTRY.md" in selection
    assert "DETECTOR_SELECTION.md" in registry
