from dataclasses import dataclass, field
from datetime import datetime

from packages.schemas.common import Severity


@dataclass
class AlertMessage:
    alert_id: str
    event_id: str
    camera_id: str
    event_type: str
    severity: Severity
    title: str
    body: str
    created_at: datetime
    summary: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class AlertResult:
    channel: str
    status: str  # sent | skipped | failed
    reason: str = ""
    alert_id: str | None = None
