from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    SNAPSHOT = "snapshot"
    CLIP = "clip"


class EvidenceRef(BaseModel):
    evidence_id: str
    type: EvidenceType
    uri: str


class EvidenceItem(BaseModel):
    evidence_id: str
    event_id: str
    camera_id: str
    type: EvidenceType
    uri: str
    sha256: str | None = None
    size_bytes: int = 0
    content_type: str | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    captured_at: datetime
    expires_at: datetime | None = None
    storage_backend: str = "local"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
