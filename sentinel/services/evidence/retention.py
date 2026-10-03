from dataclasses import dataclass, field
from datetime import datetime, timedelta

from packages.common.logging import get_logger
from packages.common.textutil import redact_secrets
from packages.common.timeutil import utcnow
from packages.db import models
from packages.db.audit import record_audit
from packages.schemas.common import Severity
from packages.schemas.evidence import EvidenceItem
from services.evidence.store import EvidenceStore

log = get_logger(__name__)

ACTIVE_EVENT_STATUSES = ("new", "acknowledged")


def resolve_expiry(
    severity: Severity | str,
    days_by_severity: dict[str, int],
    at: datetime,
) -> datetime | None:
    key = severity.value if isinstance(severity, Severity) else str(severity).lower()
    days = days_by_severity.get(key)
    if days is None or days <= 0:
        return None
    return at + timedelta(days=days)


def is_expired(item: EvidenceItem, now: datetime) -> bool:
    if item.expires_at is None:
        return False
    return now >= item.expires_at


@dataclass
class RetentionReport:
    candidates: int = 0
    deleted: int = 0
    missing_files: int = 0
    failed: int = 0
    skipped_active: int = 0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "candidates": self.candidates,
            "deleted": self.deleted,
            "missing_files": self.missing_files,
            "failed": self.failed,
            "skipped_active": self.skipped_active,
            "dry_run": self.dry_run,
            "errors": list(self.errors),
        }


class RetentionService:
    """Deletes expired evidence: storage object first, then the metadata row.

    Storage rows are only removed after the blob is gone (deleted or confirmed
    missing), so the database never points at files that no longer exist.
    Failures keep the row so the next sweep can retry. Every real deletion and
    failure is written to the audit log.
    """

    def __init__(
        self,
        store: EvidenceStore,
        allow_active_event_deletion: bool = False,
        actor: str = "system:retention",
    ) -> None:
        self._store = store
        self._allow_active = allow_active_event_deletion
        self._actor = actor

    def sweep(self, session, now: datetime | None = None, dry_run: bool = False) -> RetentionReport:
        now = now or utcnow()
        report = RetentionReport(dry_run=dry_run)

        from sqlalchemy import select

        rows = (
            session.execute(
                select(models.Evidence).where(
                    models.Evidence.expires_at.is_not(None),
                    models.Evidence.expires_at <= now,
                )
            )
            .scalars()
            .all()
        )

        for row in rows:
            report.candidates += 1
            event = session.get(models.Event, row.event_id)
            if (
                event is not None
                and event.status in ACTIVE_EVENT_STATUSES
                and not self._allow_active
            ):
                report.skipped_active += 1
                log.info(
                    "retention skipped evidence tied to active event",
                    extra={"evidence_id": row.id, "event_id": row.event_id},
                )
                continue

            if dry_run:
                continue

            file_missing = not self._store.exists(row.uri)
            if not file_missing:
                try:
                    self._store.delete(row.uri)
                except Exception as exc:
                    reason = redact_secrets(f"{type(exc).__name__}: {exc}")
                    report.failed += 1
                    report.errors.append(reason)
                    record_audit(
                        session,
                        self._actor,
                        action="evidence.retention_failed",
                        resource_type="evidence",
                        resource_id=row.id,
                        details={
                            "event_id": row.event_id,
                            "uri": row.uri,
                            "error": reason,
                        },
                    )
                    log.warning(
                        "retention storage deletion failed",
                        extra={
                            "evidence_id": row.id,
                            "event_id": row.event_id,
                            "error": reason,
                        },
                    )
                    continue

            session.delete(row)
            report.deleted += 1
            if file_missing:
                report.missing_files += 1
            record_audit(
                session,
                self._actor,
                action="evidence.retention_delete",
                resource_type="evidence",
                resource_id=row.id,
                details={
                    "event_id": row.event_id,
                    "uri": row.uri,
                    "sha256": row.sha256,
                    "expires_at": row.expires_at.isoformat() if row.expires_at else None,
                    "reason": "file_missing" if file_missing else "expired",
                },
            )
            log.info(
                "retention deleted evidence",
                extra={
                    "evidence_id": row.id,
                    "event_id": row.event_id,
                    "reason": "file_missing" if file_missing else "expired",
                },
            )

        session.flush()
        return report


def run_retention_sweep(
    session,
    store: EvidenceStore,
    allow_active_event_deletion: bool = False,
    now: datetime | None = None,
    dry_run: bool = False,
) -> RetentionReport:
    service = RetentionService(
        store, allow_active_event_deletion=allow_active_event_deletion
    )
    return service.sweep(session, now=now, dry_run=dry_run)
