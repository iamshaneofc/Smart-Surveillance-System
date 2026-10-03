"""Deterministic evaluation-manifest validation (F4-E).

Detects: missing metadata, unsupported formats, duplicate identifiers,
missing files, malformed annotations, unknown classes and invalid bounding
boxes. Produces a clear machine/human readable report. Never silently
discards invalid data - every issue is listed with its location.

Usage:
    python evaluation/validate_manifest.py evaluation/manifests/<name>.json
        [--json report-out.json] [--quiet]

Exit codes: 0 = no errors (warnings allowed), 1 = errors found, 2 = usage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ANNOTATION_EXTENSIONS = {".json"}
PLACEHOLDER_MARKERS = ("REPLACE", "TODO", "CHANGE_ME")

CHECKS = (
    "manifest_readable",
    "missing_metadata",
    "duplicate_identifiers",
    "unsupported_format",
    "missing_files",
    "malformed_annotations",
    "unknown_classes",
    "invalid_bounding_boxes",
)


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    detail: str
    entry_id: str | None = None
    frame_index: int | None = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "detail": self.detail,
            "entry_id": self.entry_id,
            "frame_index": self.frame_index,
        }

    def render(self) -> str:
        where = []
        if self.entry_id:
            where.append(f"entry={self.entry_id}")
        if self.frame_index is not None:
            where.append(f"frame={self.frame_index}")
        loc = f" [{' '.join(where)}]" if where else ""
        return f"[{self.severity}] {self.code}{loc}: {self.detail}"


@dataclass
class ValidationReport:
    manifest: str
    ok: bool = True
    entries_total: int = 0
    entries_with_errors: int = 0
    issues: list[Issue] = field(default_factory=list)
    checks: tuple[str, ...] = CHECKS

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def finalize(self) -> "ValidationReport":
        self.ok = self.errors == 0
        return self

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "entries_total": self.entries_total,
            "entries_with_errors": self.entries_with_errors,
            "checks": list(self.checks),
            "issues": [i.to_dict() for i in self.issues],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def render(self) -> str:
        lines = [
            f"manifest: {self.manifest}",
            f"result: {'OK' if self.ok else 'FAILED'} "
            f"({self.errors} error(s), {self.warnings} warning(s), "
            f"{self.entries_total} entries, {self.entries_with_errors} entries with errors)",
            "checks: " + ", ".join(self.checks),
        ]
        for issue in self.issues:
            lines.append("  " + issue.render())
        return "\n".join(lines)


def _is_placeholder(value: str) -> bool:
    return any(marker in value for marker in PLACEHOLDER_MARKERS)


def _box_issues(class_name: str, bbox: list[float], confidence: float | None) -> list[str]:
    problems: list[str] = []
    if len(bbox) != 4:
        return [f"bbox must have 4 values, got {len(bbox)}"]
    x, y, w, h = bbox
    if not all(math.isfinite(v) for v in bbox):
        return [f"bbox contains non-finite values: {bbox}"]
    if w <= 0 or h <= 0:
        problems.append(f"non-positive size w={w} h={h}")
    if x < 0 or y < 0:
        problems.append(f"negative origin x={x} y={y}")
    if x > 1.0 or y > 1.0:
        problems.append(f"origin outside normalized range x={x} y={y}")
    if x + w > 1.0001 or y + h > 1.0001:
        problems.append(f"box exceeds normalized frame x+w={round(x + w, 4)} y+h={round(y + h, 4)}")
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        problems.append(f"confidence outside [0,1]: {confidence}")
    if not class_name:
        problems.append("empty class_name")
    return problems


def validate_manifest(manifest_path: str | Path) -> ValidationReport:
    """Validate a manifest and everything it references. Deterministic order."""
    from evaluation.annotations import load_annotation
    from evaluation.manifest import (
        load_manifest,
        resolve_annotation_path,
        resolve_media_path,
    )

    path = Path(manifest_path)
    report = ValidationReport(manifest=str(path))

    if not path.is_file():
        report.add(Issue("error", "missing_files", f"manifest file not found: {path}"))
        return report.finalize()

    try:
        manifest = load_manifest(path)
    except Exception as exc:
        report.add(Issue("error", "manifest_readable", f"manifest unreadable: {exc}"))
        return report.finalize()

    report.entries_total = len(manifest.entries)

    # missing_metadata (dataset-level)
    for field_name in ("dataset_id", "license", "source"):
        value = getattr(manifest, field_name)
        if not value or not value.strip():
            report.add(Issue("error", "missing_metadata", f"dataset-level '{field_name}' is empty"))
        elif _is_placeholder(value):
            report.add(Issue("warning", "missing_metadata", f"dataset-level '{field_name}' still contains a placeholder: {value!r}"))
    if not manifest.version:
        report.add(Issue("error", "missing_metadata", "dataset-level 'version' is empty"))
    if not manifest.class_mapping:
        report.add(Issue("warning", "unknown_classes", "no class_mapping declared - annotation class names are used as-is and cannot be checked against a vocabulary"))

    # duplicate_identifiers
    seen: set[str] = set()
    for entry in manifest.entries:
        if entry.entry_id in seen:
            report.add(Issue("error", "duplicate_identifiers", f"duplicate entry_id '{entry.entry_id}'", entry_id=entry.entry_id))
        seen.add(entry.entry_id)

    entry_error_ids: set[str] = set()

    for entry in manifest.entries:
        entry_errors_before = report.errors

        # unsupported_format
        media_ext = Path(entry.media).suffix.lower()
        expected = IMAGE_EXTENSIONS if entry.media_type == "image" else VIDEO_EXTENSIONS
        if media_ext not in expected:
            report.add(Issue("error", "unsupported_format", f"media extension '{media_ext}' is not a supported {entry.media_type} format for {entry.media}", entry_id=entry.entry_id))
        if entry.annotation:
            ann_ext = Path(entry.annotation).suffix.lower()
            if ann_ext not in ANNOTATION_EXTENSIONS:
                report.add(Issue("error", "unsupported_format", f"annotation extension '{ann_ext}' is not supported: {entry.annotation}", entry_id=entry.entry_id))

        # missing_files
        media_path = resolve_media_path(entry, path)
        if not media_path.exists():
            report.add(Issue("error", "missing_files", f"media not found: {media_path}", entry_id=entry.entry_id))
        annotation_path = resolve_annotation_path(entry, path)
        if annotation_path is None or not Path(annotation_path).exists():
            report.add(Issue("error", "missing_files", "annotation not found: " + (str(annotation_path) if annotation_path else f"(default path for {entry.entry_id})"), entry_id=entry.entry_id))

        # context metadata (optional - warn when neither is available)
        if not entry.source_resolution and not entry.capture_context:
            report.add(Issue("warning", "missing_metadata", "no source_resolution and no capture_context recorded for this entry", entry_id=entry.entry_id))

        # malformed_annotations + unknown_classes + invalid_bounding_boxes
        if annotation_path is not None and Path(annotation_path).exists():
            try:
                annotation = load_annotation(annotation_path)
            except Exception as exc:
                report.add(Issue("error", "malformed_annotations", f"annotation unreadable: {exc}", entry_id=entry.entry_id))
            else:
                if annotation.entry_id != entry.entry_id:
                    report.add(Issue("warning", "malformed_annotations", f"annotation entry_id '{annotation.entry_id}' does not match manifest entry_id '{entry.entry_id}'", entry_id=entry.entry_id))
                for frame in annotation.frames:
                    if frame.frame_index < 0:
                        report.add(Issue("error", "malformed_annotations", f"negative frame_index {frame.frame_index}", entry_id=entry.entry_id, frame_index=frame.frame_index))
                    for box in frame.boxes:
                        if manifest.class_mapping and box.class_name not in manifest.class_mapping:
                            report.add(Issue("error", "unknown_classes", f"class '{box.class_name}' is not a key of class_mapping", entry_id=entry.entry_id, frame_index=frame.frame_index))
                        for problem in _box_issues(box.class_name, box.bbox, box.confidence):
                            report.add(Issue("error", "invalid_bounding_boxes", f"{problem} (class '{box.class_name}')", entry_id=entry.entry_id, frame_index=frame.frame_index))

        if report.errors > entry_errors_before:
            entry_error_ids.add(entry.entry_id)

    report.entries_with_errors = len(entry_error_ids)
    return report.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an evaluation manifest (F4-E).")
    parser.add_argument("manifest", help="path to manifest JSON")
    parser.add_argument("--json", dest="json_out", help="write machine-readable report to this path")
    parser.add_argument("--quiet", action="store_true", help="print only the result line")
    args = parser.parse_args(argv)

    report = validate_manifest(args.manifest)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if args.quiet:
        print(f"{'OK' if report.ok else 'FAILED'}: {report.errors} error(s), {report.warnings} warning(s)")
    else:
        print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
