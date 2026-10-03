from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class BBox:
    x: float
    y: float
    w: float
    h: float

    def normalized(self) -> "BBox":
        return BBox(
            x=min(max(self.x, 0.0), 1.0),
            y=min(max(self.y, 0.0), 1.0),
            w=min(max(self.w, 0.0), 1.0),
            h=min(max(self.h, 0.0), 1.0),
        )

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x + self.w, other.x + other.w)
        iy2 = min(self.y + self.h, other.y + other.h)
        iw = max(ix2 - ix1, 0.0)
        ih = max(iy2 - iy1, 0.0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int | None = None
    class_name: str = ""
    confidence: float = 0.0
    bbox: BBox = field(default_factory=lambda: BBox(0.0, 0.0, 0.0, 0.0))
    track_id: int | None = None
    timestamp: datetime | None = None
    model_id: str | None = None
    model_version: str | None = None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "track_id": self.track_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_id": self.model_id,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    family: str
    license: str
    classes: tuple[str, ...]
    input_size: tuple[int, int] = (640, 640)
    metrics: dict = field(default_factory=dict)
    status: str = "candidate"

    @property
    def label(self) -> str:
        return f"{self.name}:{self.version}"
