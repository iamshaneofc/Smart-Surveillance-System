from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ManifestEntry(BaseModel):
    """One labeled media item. Paths are absolute or relative to the manifest."""

    entry_id: str = Field(min_length=1)
    media: str
    media_type: Literal["image", "video"] = "image"
    annotation: str | None = None
    camera_id: str = "unknown"
    scene: str = ""
    split: Literal["dev", "val", "test"] = "val"
    source_resolution: str | None = None  # e.g. "1920x1080" where known
    capture_context: str = ""  # e.g. "indoor, consented license plate footage"


class EvaluationManifest(BaseModel):
    """F4-E dataset/manifest contract.

    Required provenance: dataset_id, version, source (provenance), license.
    class_mapping maps dataset class names to model class names; empty means
    annotation class names are used as-is (validator then warns).
    """

    name: str
    dataset_id: str = ""
    version: str = "1"
    license: str = ""
    source: str = ""
    class_mapping: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    created_at: datetime | None = None
    entries: list[ManifestEntry] = Field(default_factory=list)


def load_manifest(path: str | Path) -> EvaluationManifest:
    import json

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvaluationManifest.model_validate(data)


def resolve_media_path(entry: ManifestEntry, manifest_path: Path) -> Path:
    candidate = Path(entry.media)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate


def resolve_annotation_path(entry: ManifestEntry, manifest_path: Path) -> Path | None:
    if entry.annotation:
        candidate = Path(entry.annotation)
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        return candidate
    default = manifest_path.parent.parent / "annotations" / f"{entry.entry_id}.json"
    return default if default.exists() else None
