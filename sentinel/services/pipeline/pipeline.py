from __future__ import annotations

from datetime import datetime

from packages.common.logging import get_logger
from packages.common.textutil import redact_secrets
from packages.schemas.common import HealthState
from packages.schemas.event import Event, EventStatus
from services.camera.types import FramePacket
from services.events.engine import EventEngine, dedup_key_for
from services.inference.interfaces import Detector
from services.rules.base import RuleSet
from services.rules.context import EvaluationContext, SpatioTemporalState, ZoneContext

log = get_logger(__name__)


class SurveillancePipeline:
    """Camera frame -> detection -> tracking -> zone rules -> events -> evidence.

    Persistence-agnostic: results flow through the EventEngine, the
    EvidenceService (its own on_finalize callback) and an optional
    event_sink, so every stage stays independently testable.
    """

    def __init__(
        self,
        camera_id: str,
        detector: Detector,
        tracker,
        rule_set: RuleSet,
        engine: EventEngine,
        spatial: SpatioTemporalState,
        zones: list[ZoneContext],
        evidence_service,
        camera_timezone: str = "UTC",
        rule_version: str = "",
        event_sink=None,
        buffer=None,
        bus=None,
        on_ai_status=None,
        alert_dispatcher=None,
    ) -> None:
        self.camera_id = camera_id
        self.detector = detector
        self.tracker = tracker
        self.rule_set = rule_set
        self.engine = engine
        self.spatial = spatial
        self.zones = zones
        self.evidence = evidence_service
        self.camera_timezone = camera_timezone
        self.rule_version = rule_version
        self.event_sink = event_sink
        self.buffer = buffer
        self.bus = bus
        self.on_ai_status = on_ai_status
        self.alert_dispatcher = alert_dispatcher
        self.frames_processed = 0
        self.detector_errors = 0
        self.tracker_errors = 0
        self.events_created = 0
        self.model_versions = {"detector": detector.info.label}

    def _ai(self, status: HealthState) -> None:
        if self.on_ai_status is not None:
            try:
                self.on_ai_status(status)
            except Exception:
                log.exception("ai status callback failed")

    def process_frame(self, packet: FramePacket) -> list[Event]:
        events: list[Event] = []
        if self.buffer is not None:
            self.buffer.append(packet)
        try:
            self.evidence.on_frame(packet)
        except Exception:
            log.exception(
                "evidence frame update failed",
                extra={"camera_id": self.camera_id, "frame_id": packet.frame_id},
            )

        try:
            detections = self.detector.detect(packet)
            self._ai(HealthState.HEALTHY)
        except Exception as exc:
            self.detector_errors += 1
            self._ai(HealthState.DEGRADED)
            log.warning(
                "detector error",
                extra={
                    "camera_id": self.camera_id,
                    "error": redact_secrets(exc),
                    "detector": self.detector.info.name,
                },
            )
            detections = []

        try:
            tracks = self.tracker.update(detections, packet.ts)
        except Exception as exc:
            self.tracker_errors += 1
            self._ai(HealthState.DEGRADED)
            log.warning(
                "tracker error",
                extra={"camera_id": self.camera_id, "error": redact_secrets(exc)},
            )
            return events

        enriched = self.spatial.enrich(tracks, self.zones)
        self.spatial.update(enriched, self.zones, packet.ts)
        ctx = EvaluationContext(
            camera_id=self.camera_id,
            camera_timezone=self.camera_timezone,
            ts=packet.ts,
            zones=self.zones,
            tracks=enriched,
            transitions=self.spatial.transitions,
            spatial=self.spatial,
        )
        matches = self.rule_set.evaluate_all(ctx)
        matched_keys: set[str] = set()
        for match in matches:
            matched_keys.add(
                dedup_key_for(match.camera_id, match.event_type, match.zone_id, match.track_ids)
            )
            match.metadata.setdefault("rule_version", self.rule_version)

        self._close_stale_events(matched_keys, packet.ts)
        self._cancel_stale_pending(matched_keys)

        for match in matches:
            event = self.engine.confirm(match, model_versions=dict(self.model_versions))
            if event is None:
                continue
            self.events_created += 1
            if self.event_sink is not None:
                try:
                    self.event_sink(event)
                except Exception:
                    log.exception(
                        "event persistence failed",
                        extra={"event_id": event.event_id},
                    )
            try:
                self.evidence.begin(event)
            except Exception:
                log.exception(
                    "evidence capture failed",
                    extra={"event_id": event.event_id},
                )
            if self.bus is not None:
                try:
                    self.bus.publish(
                        "events.created",
                        {
                            "event_id": event.event_id,
                            "camera_id": event.camera_id,
                            "event_type": event.event_type,
                            "severity": event.severity.value,
                        },
                    )
                except Exception:
                    log.exception("event bus publish failed")
            if self.alert_dispatcher is not None:
                try:
                    self.alert_dispatcher.submit(event)
                except Exception:
                    log.exception(
                        "alert submit failed",
                        extra={"event_id": event.event_id},
                    )
            events.append(event)
        self.frames_processed += 1
        return events

    def _close_stale_events(self, matched_keys: set[str], now: datetime) -> None:
        prefix = f"{self.camera_id}|"
        for event in self.engine.open_events():
            key = dedup_key_for(
                event.camera_id, event.event_type, event.zone_id, event.track_ids
            )
            if not key.startswith(prefix) or key in matched_keys:
                continue
            still_active = False
            if event.zone_id:
                still_active = any(
                    self.spatial.zone_since(track_id, event.zone_id) is not None
                    for track_id in event.track_ids
                )
            if still_active:
                continue
            try:
                self.engine.transition(event, EventStatus.RESOLVED, actor="system", at=now)
            except ValueError:
                log.warning(
                    "event auto-resolve skipped",
                    extra={"event_id": event.event_id},
                )
                continue
            log.info(
                "event auto-resolved (condition cleared)",
                extra={"event_id": event.event_id, "camera_id": self.camera_id},
            )
            if self.event_sink is not None:
                try:
                    self.event_sink(event)
                except Exception:
                    log.exception(
                        "event status persistence failed",
                        extra={"event_id": event.event_id},
                    )

    def _cancel_stale_pending(self, matched_keys: set[str]) -> None:
        prefix = f"{self.camera_id}|"
        for key in list(self.engine.pending_keys()):
            if key.startswith(prefix) and key not in matched_keys:
                self.engine.cancel_pending_key(key)
