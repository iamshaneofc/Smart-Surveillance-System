from collections import OrderedDict
from datetime import datetime, timedelta

from packages.common.ids import new_id
from packages.common.logging import get_logger
from packages.common.textutil import redact_secrets
from packages.schemas.event import Event
from services.alerts.notifiers import Notifier
from services.alerts.types import AlertMessage, AlertResult

log = get_logger(__name__)

DISPATCH_MEMORY = 10_000


class AlertRouter:
    def __init__(
        self,
        notifiers: dict[str, Notifier] | None = None,
        default_channels: list[str] | None = None,
        cooldown_seconds: float = 120.0,
        rate_cap_per_minute: int = 30,
        escalation_after_seconds: float = 300.0,
        escalation_channel: str | None = None,
    ) -> None:
        self.notifiers = notifiers or {}
        self.default_channels = default_channels or list(self.notifiers)
        self.cooldown_seconds = cooldown_seconds
        self.rate_cap_per_minute = rate_cap_per_minute
        self.escalation_after_seconds = escalation_after_seconds
        self.escalation_channel = escalation_channel
        self._last_sent: dict[tuple[str, str, str], datetime] = {}
        self._camera_counts: dict[str, list[datetime]] = {}
        self._dispatched: OrderedDict[str, None] = OrderedDict()

    def _already_dispatched(self, event_id: str) -> bool:
        return event_id in self._dispatched

    def _mark_dispatched(self, event_id: str) -> None:
        self._dispatched[event_id] = None
        while len(self._dispatched) > DISPATCH_MEMORY:
            self._dispatched.popitem(last=False)

    def _within_cooldown(self, event: Event, channel: str) -> bool:
        key = (event.camera_id, event.event_type, channel)
        last = self._last_sent.get(key)
        if last is None:
            return False
        return (event.timestamp - last) < timedelta(seconds=self.cooldown_seconds)

    def _within_rate_cap(self, event: Event) -> bool:
        window_start = event.timestamp - timedelta(minutes=1)
        hits = [t for t in self._camera_counts.get(event.camera_id, []) if t >= window_start]
        return len(hits) < self.rate_cap_per_minute

    def dispatch(self, event: Event, channels: list[str] | None = None) -> list[AlertResult]:
        results: list[AlertResult] = []
        targets = channels if channels is not None else self.default_channels
        if self._already_dispatched(event.event_id):
            return [
                AlertResult(channel=channel, status="skipped", reason="duplicate")
                for channel in targets
            ]
        for channel in targets:
            notifier = self.notifiers.get(channel)
            if notifier is None:
                results.append(AlertResult(channel=channel, status="skipped", reason="no notifier"))
                continue
            if not self._within_rate_cap(event):
                results.append(
                    AlertResult(channel=channel, status="skipped", reason="rate_cap")
                )
                continue
            if self._within_cooldown(event, channel):
                results.append(AlertResult(channel=channel, status="skipped", reason="cooldown"))
                continue
            message = self._build_message(event, channel)
            try:
                notifier.send(message)
            except Exception as exc:
                log.exception("alert send failed", extra={"channel": channel})
                results.append(
                    AlertResult(
                        channel=channel, status="failed", reason=redact_secrets(str(exc))
                    )
                )
                continue
            self._last_sent[(event.camera_id, event.event_type, channel)] = event.timestamp
            self._camera_counts.setdefault(event.camera_id, []).append(event.timestamp)
            results.append(
                AlertResult(channel=channel, status="sent", alert_id=message.alert_id)
            )
        self._mark_dispatched(event.event_id)
        return results

    def escalation_candidates(self, events: list[Event], now: datetime) -> list[AlertResult]:
        if self.escalation_channel is None:
            return []
        results: list[AlertResult] = []
        for event in events:
            if event.acknowledged_at is not None:
                continue
            if event.resolved_at is not None:
                continue
            age = (now - event.timestamp).total_seconds()
            if age < self.escalation_after_seconds:
                continue
            notifier = self.notifiers.get(self.escalation_channel)
            if notifier is None:
                continue
            message = self._build_message(event, self.escalation_channel)
            message.title = f"[ESCALATED] {message.title}"
            try:
                notifier.send(message)
                results.append(
                    AlertResult(
                        channel=self.escalation_channel,
                        status="sent",
                        alert_id=message.alert_id,
                    )
                )
            except Exception as exc:
                results.append(
                    AlertResult(channel=self.escalation_channel, status="failed", reason=str(exc))
                )
        return results

    def _build_message(self, event: Event, channel: str) -> AlertMessage:
        severity = event.severity.value if hasattr(event.severity, "value") else str(event.severity)
        zone = f" zone={event.zone_name}" if event.zone_name else ""
        confidence = f" conf={event.confidence:.2f}" if event.confidence is not None else ""
        return AlertMessage(
            alert_id=new_id("alr"),
            event_id=event.event_id,
            camera_id=event.camera_id,
            event_type=event.event_type,
            severity=event.severity,
            title=f"{severity.upper()} {event.event_type} @ {event.camera_id}",
            body=f"{event.event_type}{zone} at {event.timestamp.isoformat()}{confidence}",
            created_at=event.timestamp,
            summary=event.summary or "",
            evidence_ids=list(event.evidence_ids),
            metadata={
                "rule_id": event.rule_id,
                "track_ids": list(event.track_ids),
                "channel": channel,
            },
        )
