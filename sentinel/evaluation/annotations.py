from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Box(BaseModel):
    class_name: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float | None = None

    @classmethod
    def xywh(cls, class_name: str, x: float, y: float, w: float, h: float) -> "Box":
        return cls(class_name=class_name, bbox=[x, y, w, h])


class FrameAnnotation(BaseModel):
    frame_index: int = 0
    boxes: list[Box] = Field(default_factory=list)


class AnnotationFile(BaseModel):
    entry_id: str
    frames: list[FrameAnnotation] = Field(default_factory=list)


def load_annotation(path: str | Path) -> AnnotationFile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AnnotationFile.model_validate(data)
