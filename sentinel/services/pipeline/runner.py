from __future__ import annotations

import threading

from packages.common.logging import get_logger
from packages.common.timeutil import utcnow
from packages.config import Settings
from packages.schemas.rule import RulePack
from services.camera.manager import CameraWorker, WorkerConfig
from services.camera.sources import source_factory_for
from services.events.engine import EventEngine
from services.events.repository import SqlEventRepository
from services.evidence.buffer import RollingFrameBuffer
from services.evidence.service import EvidenceService
from services.evidence.store import LocalDiskEvidenceStore
from services.inference.interfaces import default_registry
from services.pipeline.pipeline import SurveillancePipeline
from services.rules.base import RuleSet
from services.rules.context import SpatioTemporalState, ZoneContext
from services.rules.pack import load_pack
from services.tracking.iou import IOUTracker, NullTracker

log = get_logger(__name__)

TRACKERS = {"iou": IOUTracker, "null": NullTracker}


def build_tracker(name: str):
    tracker_cls = TRACKERS.get(name)
    if tracker_cls is None:
        raise ValueError(f"unknown tracker '{name}' (available: {', '.join(sorted(TRACKERS))})")
    return tracker_cls()


class PipelineRunner:
    """Owns one camera worker + pipeline, wiring persistence, health and lifecycle."""

    def __init__(self, settings: Settings, bus=None) -> None:
        self.settings = settings
        self.pipeline_settings = settings.pipeline
        self.bus = bus
        p = self.pipeline_settings

        self.pack: RulePack = load_pack(
            _resolve_pack_path(p.rule_pack)
        )
        self.rule_version = self.pack.version
        self.rule_set = RuleSet(list(self.pack.rules))
        self.engine = EventEngine(
            rules={d.rule_id: d for d in self.pack.rules},
        )
        self.zones = [
            ZoneContext(
                id=z.id,
                name=z.name,
                zone_type=z.zone_type,
                polygon=[tuple(pt) for pt in z.polygon],
                enabled=z.enabled,
                anchor=z.anchor,
            )
            for z in p.zones
        ]
        self.detector = default_registry().create(p.detector_profile, **p.detector_options)
        self.tracker = build_tracker(p.tracker)
        self.spatial = SpatioTemporalState()

        pre = settings.evidence.pre_seconds
        buffer_seconds = max(pre + 5.0, 30.0)
        self.buffer = RollingFrameBuffer(p.camera_id, max_seconds=buffer_seconds)
        self.store = LocalDiskEvidenceStore(settings.evidence.root)
        clip_writer = None
        try:
            from services.evidence.service import OpenCvClipWriter

            clip_writer = OpenCvClipWriter()
        except Exception:  # pragma: no cover - optional video writer
            clip_writer = None
        self.evidence = EvidenceService(
            store=self.store,
            buffer=self.buffer,
            clip_writer=clip_writer,
            pre_seconds=pre,
            post_seconds=settings.evidence.post_seconds,
            snapshot_count=settings.evidence.snapshot_count,
            days_by_severity=settings.evidence.retention_days_by_severity,
            on_finalize=self._persist_evidence,
        )
        self.alert_dispatcher = self._build_alert_dispatcher(settings)
        self.pipeline = SurveillancePipeline(
            camera_id=p.camera_id,
            detector=self.detector,
            tracker=self.tracker,
            rule_set=self.rule_set,
            engine=self.engine,
            spatial=self.spatial,
            zones=self.zones,
            evidence_service=self.evidence,
            camera_timezone=p.timezone,
            rule_version=self.rule_version,
            event_sink=self._persist_event,
            buffer=self.buffer,
            bus=bus,
            alert_dispatcher=self.alert_dispatcher,
        )
        self.worker = CameraWorker(
            WorkerConfig(
                camera_id=p.camera_id,
                detection_fps=p.detection_fps,
                reconnect_initial_seconds=settings.camera_reconnect_initial_seconds,
                reconnect_max_seconds=settings.camera_reconnect_max_seconds,
            ),
            source_factory=source_factory_for(p.camera_id, p.source_type, p.stream_url),
            on_frame=self._on_frame,
        )
        self.pipeline.on_ai_status = self.worker.report_ai_status
        self._thread: threading.Thread | None = None
        self._last_health_flush = 0.0

    # -- persistence -----------------------------------------------------

    def _build_alert_dispatcher(self, settings: Settings):
        from services.alerts.dispatcher import AlertDispatcher
        from services.alerts.notifiers import DatabaseNotifier, WebhookNotifier
        from services.alerts.router import AlertRouter

        a = settings.alerts
        if not a.enabled:
            return None
        notifiers = {}
        for channel in a.channels:
            if channel == "in_app":
                notifiers[channel] = DatabaseNotifier()
            elif channel == "webhook":
                notifiers[channel] = WebhookNotifier(
                    url=a.webhook_url,
                    timeout=a.webhook_timeout_seconds,
                    max_retries=a.webhook_max_retries,
                    backoff_seconds=a.webhook_backoff_seconds,
                )
            else:
                raise ValueError(
                    f"unknown alert channel '{channel}' (available: in_app, webhook)"
                )
        if not notifiers:
            raise ValueError("alerts.enabled is true but alerts.channels is empty")
        router = AlertRouter(
            notifiers=notifiers,
            default_channels=list(notifiers),
            cooldown_seconds=a.cooldown_seconds,
            rate_cap_per_minute=a.rate_cap_per_minute,
        )
        log.info(
            "alerts configured",
            extra={"channels": list(notifiers), "queue_max": a.queue_max},
        )
        return AlertDispatcher(router, queue_size=a.queue_max, bus=self.bus)

    def _persist_event(self, event) -> None:
        from packages.db import base as db_base

        with db_base.session_scope() as session:
            SqlEventRepository(session).save(event)

    def _persist_evidence(self, event, items) -> None:
        from packages.db import base as db_base
        from packages.db import models

        if not items:
            return
        with db_base.session_scope() as session:
            repo = SqlEventRepository(session)
            row = session.get(models.Event, event.event_id)
            if row is None:
                repo.save(event)
                row = session.get(models.Event, event.event_id)
            existing = list(row.evidence_ids or [])
            for item in items:
                if item.evidence_id in existing:
                    continue
                session.add(
                    models.Evidence(
                        id=item.evidence_id,
                        event_id=item.event_id,
                        camera_id=item.camera_id,
                        type=item.type.value,
                        uri=item.uri,
                        sha256=item.sha256,
                        size_bytes=item.size_bytes,
                        content_type=item.content_type,
                        width=item.width,
                        height=item.height,
                        duration_ms=item.duration_ms,
                        captured_at=item.captured_at,
                        expires_at=item.expires_at,
                        storage_backend=item.storage_backend,
                        metadata_=dict(item.metadata),
                    )
                )
                existing.append(item.evidence_id)
            event.evidence_ids = existing
            repo.save(event)

    # -- camera / health -------------------------------------------------

    def ensure_camera_row(self) -> None:
        from sqlalchemy import select

        from packages.db import base as db_base
        from packages.db import models

        p = self.pipeline_settings
        with db_base.session_scope() as session:
            row = session.execute(
                select(models.Camera).where(models.Camera.camera_id == p.camera_id)
            ).scalar_one_or_none()
            if row is None:
                row = models.Camera(
                    camera_id=p.camera_id,
                    name=p.camera_name,
                    location=p.camera_location,
                    source_type=p.source_type,
                    stream_url=p.stream_url or f"{p.source_type}://configured",
                    detection_fps=p.detection_fps,
                    timezone=p.timezone,
                    model_profile=p.detector_profile,
                    rule_profile=p.rule_pack,
                )
                session.add(row)
                log.info("camera row created", extra={"camera_id": p.camera_id})

    def flush_health(self) -> None:
        from packages.db import base as db_base
        from packages.db import models

        snap = self.worker.snapshot()
        try:
            with db_base.session_scope() as session:
                session.add(
                    models.CameraHealth(
                        camera_id=snap.camera_id,
                        state=snap.state.value,
                        health=snap.health.value,
                        ai_status=snap.ai_status.value,
                        fps=snap.fps,
                        frame_drops=snap.frame_drops,
                        latency_ms=snap.latency_ms,
                        reconnect_count=snap.reconnect_count,
                        frames_processed=snap.frames_processed,
                        last_frame_at=snap.last_frame_at,
                        error=snap.error,
                        details=dict(snap.details),
                        ts=utcnow(),
                    )
                )
        except Exception:
            log.exception("camera health flush failed", extra={"camera_id": snap.camera_id})
        if self.bus is not None:
            try:
                self.bus.publish(
                    "camera.health",
                    {
                        "camera_id": snap.camera_id,
                        "state": snap.state.value,
                        "health": snap.health.value,
                        "ai_status": snap.ai_status.value,
                        "fps": snap.fps,
                        "error": snap.error,
                        "ts": utcnow().isoformat(),
                    },
                )
            except Exception:
                log.exception("camera.health publish failed")

    def _on_frame(self, packet) -> None:
        self.pipeline.process_frame(packet)
        now = self.worker._monotonic()
        if now - self._last_health_flush >= self.pipeline_settings.health_flush_seconds:
            self._last_health_flush = now
            self.flush_health()

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self.detector.warmup()
        self.ensure_camera_row()
        if self.alert_dispatcher is not None:
            self.alert_dispatcher.start()
        self._thread = threading.Thread(
            target=self.worker.run, name=f"camera-{self.pipeline_settings.camera_id}", daemon=True
        )
        self._thread.start()
        log.info(
            "pipeline started",
            extra={
                "camera_id": self.pipeline_settings.camera_id,
                "source_type": self.pipeline_settings.source_type,
                "detector": self.detector.info.name,
                "tracker": self.pipeline_settings.tracker,
                "rule_pack": self.pipeline_settings.rule_pack,
            },
        )

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the camera worker thread exits. True when it is finished."""
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def stop(self, join_timeout: float = 10.0) -> None:
        self.worker.stop()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        if self.alert_dispatcher is not None:
            self.alert_dispatcher.stop()
        try:
            self.evidence.finalize(self.pipeline_settings.camera_id)
        except Exception:
            log.exception("evidence finalize on shutdown failed")
        self.flush_health()
        try:
            self.detector.close()
        except Exception:
            log.exception("detector close failed")
        log.info(
            "pipeline stopped",
            extra={
                "camera_id": self.pipeline_settings.camera_id,
                "frames_processed": self.pipeline.frames_processed,
                "events_created": self.pipeline.events_created,
                "detector_errors": self.pipeline.detector_errors,
            },
        )


def _resolve_pack_path(rule_pack: str):
    from pathlib import Path

    candidate = Path(rule_pack)
    if candidate.exists():
        return candidate
    packs_dir = Path(__file__).resolve().parents[2] / "rules" / "packs"
    return packs_dir / rule_pack
