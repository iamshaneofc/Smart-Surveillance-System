from typing import Any

from sqlalchemy.orm import Session

from packages.db import models
from packages.schemas.auth import Principal


def record_audit(
    session: Session,
    actor: Principal | str | None,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict[str, Any] | None = None,
) -> None:
    if isinstance(actor, str) or actor is None:
        actor_name = actor or "system"
    else:
        actor_name = actor.user
    session.add(
        models.AuditLog(
            actor=actor_name,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
        )
    )
