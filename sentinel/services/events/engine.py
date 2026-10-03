from datetime import datetime, timedelta

from packages.common.ids import new_id
from packages.common.logging import get_logger
from packages.schemas.common import Severity
from packages.schemas.event import (
    ALLOWED_TRANSITIONS,
    Event,
    EventDraft,
    EventStatus,
)
from packages.schemas.rule import RuleDefinition
from services.rules.base import RuleMatch

log = get_logger(__name__)

_SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]

_SEVERITY_ALIASES = {
    "low": Severity.LOW,
    "medium": Severity.MEDIUM,
    "high": Severity.HIGH,
    "critical": Severity.CRITICAL,
}


def _severity(value: Severity | str) -> Severity:
    if isinstance(value, Severity):
        return value
    return _SEVERITY_ALIASES.get(str(value).lower(), Severity.MEDIUM)


def _bump_severity(severity: Severity) -> Severity:
    idx = _SEVERITY_ORDER.index(_severity(severity))
    return _SEVERITY_ORDER[min(idx + 1, len(_SEVERITY_ORDER) - 1)]


def dedup_key_for(camera_id: str, event_type: str, zone_id: str | None, track_ids: list[int]) -> str:
    tracks = ",".join(str(t) for t in sorted(track_ids))
    return f"{camera_id}|{event_type}|{zone_id or '-'}|{tracks}"


def _build_summary(match: RuleMatch, confirmed_ts: datetime) -> str:
    tracks = ", ".join(f"#{t}" for t in match.track_ids) or "#?"
    entered = match.metadata.get("entered_zone_at")
    if entered and match.zone_name:
        try:
            entered_dt = datetime.fromisoformat(str(entered))
        except ValueError:
            entered_dt = None
        if entered_dt is not None:
            dwell = max((confirmed_ts - entered_dt).total_seconds(), 0.0)
            return (
                f"Track {tracks} entered {match.zone_name} "
                f"and remained inside for {dwell:.1f} seconds."
            )
    custom = match.metadata.get("summary")
    if custom:
        return str(custom)
    if match.zone_name:
        return f"Track {tracks} entered {match.zone_name} ({match.event_type})."
    return f"Track {tracks}: {match.event_type}."


class EventEngine:
    def __init__(
        self,
        rules: dict[str, RuleDefinition] | None = None,
        pending_ttl_seconds: float = 30.0,
        default_cooldown_seconds: float = 60.0,
        default_confirm_seconds: float = 2.0,
        escalation_repeats: int = 3,
    ) -> None:
        self.rules = rules or {}
        self.pending_ttl = pending_ttl_seconds
        self.default_cooldown = default_cooldown_seconds
        self.default_confirm = default_confirm_seconds
        self.escalation_repeats = escalation_repeats
        self._pending: dict[str, tuple[datetime, datetime, RuleMatch, int]] = {}
        self._open: dict[str, Event] = {}
        self._closed_at: dict[str, datetime] = {}
        self._recurrences: dict[str, int] = {}

    def _rule_params(self, rule_key: str | None) -> tuple[float, float]:
        definition = self.rules.get(rule_key or "")
        if definition is None:
            return self.default_confirm, self.default_cooldown
        return definition.confirm_seconds, definition.cooldown_seconds

    def confirm(
        self,
        match: RuleMatch,
        model_versions: dict[str, str] | None = None,
    ) -> Event | None:
        confirm_seconds, cooldown_seconds = self._rule_params(match.rule_key)
        key = dedup_key_for(match.camera_id, match.event_type, match.zone_id, match.track_ids)

        if key in self._open:
            return None
        closed_at = self._closed_at.get(key)
        if closed_at is not None:
            gap = match.ts - closed_at
            if gap < timedelta(seconds=cooldown_seconds):
                return None
            if gap > timedelta(seconds=max(cooldown_seconds * 10, 3600)):
                self._recurrences.pop(key, None)

        if confirm_seconds > 0:
            entry = self._pending.get(key)
            if entry is None:
                self._pending[key] = (match.ts, match.ts, match, 1)
                self._prune(match.ts)
                return None
            first_ts, last_ts, best, evaluations = entry
            if match.ts - last_ts > timedelta(seconds=self.pending_ttl):
                first_ts, last_ts, best, evaluations = match.ts, match.ts, match, 1
            elif match.confidence >= (best.confidence or 0):
                best = match
            evaluations += 1
            if match.ts - first_ts < timedelta(seconds=confirm_seconds):
                self._pending[key] = (first_ts, match.ts, best, evaluations)
                return None
            self._pending.pop(key, None)
            match = best
            confirmed_ts = last_ts
        else:
            self._pending.pop(key, None)
            confirmed_ts = match.ts

        severity = _severity(match.severity)
        self._recurrences[key] = self._recurrences.get(key, 0) + 1
        if self._recurrences[key] >= self.escalation_repeats:
            severity = _bump_severity(severity)

        now = datetime.now(tz=match.ts.tzinfo)
        event = Event(
            event_id=new_id("evt"),
            camera_id=match.camera_id,
            timestamp=confirmed_ts,
            event_type=match.event_type,
            severity=severity,
            status=EventStatus.NEW,
            confidence=match.confidence,
            summary=_build_summary(match, confirmed_ts),
            track_ids=list(match.track_ids),
            zone_id=match.zone_id,
            zone_name=match.zone_name,
            rule_id=match.rule_key,
            rule_name=match.rule_name,
            conditions=list(match.conditions),
            model_versions=model_versions or {},
            metadata=dict(match.metadata),
            created_at=now,
            updated_at=now,
        )
        self._open[key] = event
        log.info(
            "event created",
            extra={
                "event_id": event.event_id,
                "event_type": event.event_type,
                "camera_id": event.camera_id,
                "severity": event.severity.value,
            },
        )
        return event

    def cancel_pending(
        self, camera_id: str, event_type: str, zone_id: str | None, track_ids: list[int]
    ) -> bool:
        """Drop a pending confirmation when its triggering condition stops holding."""
        key = dedup_key_for(camera_id, event_type, zone_id, track_ids)
        return self._pending.pop(key, None) is not None

    def pending_keys(self) -> set[str]:
        return set(self._pending)

    def cancel_pending_key(self, key: str) -> bool:
        return self._pending.pop(key, None) is not None

    def open_events(self) -> list[Event]:
        return list(self._open.values())

    def _prune(self, now: datetime) -> None:
        stale = [
            key
            for key, (first, last, _, _) in self._pending.items()
            if now - last > timedelta(seconds=self.pending_ttl)
        ]
        for key in stale:
            self._pending.pop(key, None)

    def transition(
        self,
        event: Event,
        new_status: EventStatus,
        actor: str | None = None,
        at: datetime | None = None,
    ) -> Event:
        allowed = ALLOWED_TRANSITIONS[event.status]
        if new_status not in allowed:
            raise ValueError(f"invalid event transition {event.status.value} -> {new_status.value}")
        now = at or datetime.now(tz=event.updated_at.tzinfo)
        event.status = new_status
        event.updated_at = now
        if new_status == EventStatus.ACKNOWLEDGED:
            event.acknowledged_by = actor
            event.acknowledged_at = now
        if new_status in (EventStatus.RESOLVED, EventStatus.DISMISSED):
            event.resolved_at = now
            key = dedup_key_for(event.camera_id, event.event_type, event.zone_id, event.track_ids)
            self._open.pop(key, None)
            self._closed_at[key] = now
        log.info(
            "event transition",
            extra={"event_id": event.event_id, "status": new_status.value, "actor": actor},
        )
        return event

    def active_events(self) -> list[Event]:
        return list(self._open.values())
