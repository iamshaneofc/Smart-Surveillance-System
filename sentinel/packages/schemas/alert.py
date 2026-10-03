from datetime import datetime

from pydantic import BaseModel


class AlertOut(BaseModel):
    id: str
    event_id: str
    channel: str
    status: str = "pending"
    target: str | None = None
    attempts: int = 0
    error: str | None = None
    sent_at: datetime | None = None
    created_at: datetime
