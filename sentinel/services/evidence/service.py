from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from packages.common.ids import new_id
from packages.common.logging import get_logger
from packages.common.timeutil import utcnow
from packages.schemas.common import Severity
from packages.schemas.evidence import EvidenceItem, EvidenceType
from packages.schemas.event import Event
from services.camera.types import FramePacket
from services.evidence.buffer import RollingFrameBuffer
from services.evidence.retention import resolve_expiry
from services.evidence.store import EvidenceStore

log = get_logger(__name__)


class ClipWriter:
    def write(self, frames: list[FramePacket]) -> tuple[bytes, str, str] | None:
        raise NotImplementedError


class OpenCvClipWriter(ClipWriter):
    def write(self, frames: list[FramePacket]) -> tuple[bytes, str, str] | None:
        if len(frames) < 2:
            return None
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        import tempfile
        from pathlib import Path

        deltas = []
        for a, b in zip(frames, frames[1:]):
            deltas.append((b.ts - a.ts).total_seconds())
        avg = sum(deltas) / len(deltas) if deltas else 0.0
        fps = 1.0 / avg if avg > 0 else 10.0
        fps = min(max(fps, 1.0), 60.0)

        first = cv2.imdecode(np.frombuffer(frames[0].data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return None
        height, width = first.shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        writer = None
        try:
            writer = cv2.VideoWriter(
                str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            for packet in frames:
                frame = cv2.imdecode(np.frombuffer(packet.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
            data = tmp_path.read_bytes()
        finally:
            if writer is not None:
                writer.release()
            tmp_path.unlink(missing_ok=True)
        if not data:
            return None
        return data, "video/mp4", "mp4"


@dataclass
class EvidenceCaptureSession:
    event: Event
    camera_id: str
    pre_frames: list[FramePacket] = field(default_factory=list)
    post_frames: list[FramePacket] = field(default_factory=list)
    deadline: datetime | None = None

    @property
    def complete(self) -> bool:
        return self.deadline is None


class EvidenceService:
    def __init__(
        self,
        store: EvidenceStore,
        buffer: RollingFrameBuffer | None = None,
        clip_writer: ClipWriter | None = None,
        pre_seconds: float = 10.0,
        post_seconds: float = 10.0,
        snapshot_count: int = 3,
        days_by_severity: dict[str, int] | None = None,
        clock=utcnow,
        on_finalize: Callable[[Event, list[EvidenceItem]], None] | None = None,
    ) -> None:
        self.store = store
        self.buffer = buffer
        self.clip_writer = clip_writer
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.snapshot_count = snapshot_count
        self.days_by_severity = days_by_severity or {
            "critical": 90,
            "high": 30,
            "medium": 14,
            "low": 3,
        }
        self.clock = clock
        self.on_finalize = on_finalize
        self._sessions: dict[str, EvidenceCaptureSession] = {}

    def begin(self, event: Event) -> EvidenceCaptureSession:
        now = event.timestamp
        pre: list[FramePacket] = []
        if self.buffer is not None and self.buffer.camera_id == event.camera_id:
            pre = self.buffer.recent(self.pre_seconds, now)
        if self.post_seconds <= 0:
            session = EvidenceCaptureSession(event=event, camera_id=event.camera_id, pre_frames=pre)
            session.deadline = None
            self._sessions[event.camera_id] = session
            return session
        session = EvidenceCaptureSession(
            event=event,
            camera_id=event.camera_id,
            pre_frames=pre,
            deadline=now + timedelta(seconds=self.post_seconds),
        )
        self._sessions[event.camera_id] = session
        return session

    def on_frame(self, packet: FramePacket) -> bool:
        session = self._sessions.get(packet.camera_id)
        if session is None or session.deadline is None:
            return False
        if packet.ts <= session.deadline:
            session.post_frames.append(packet)
            return True
        self.finalize(packet.camera_id)
        return False

    def finalize(self, camera_id: str) -> list[EvidenceItem]:
        session = self._sessions.pop(camera_id, None)
        if session is None:
            return []
        frames = sorted(session.pre_frames + session.post_frames, key=lambda f: f.ts)
        event = session.event
        items: list[EvidenceItem] = []
        expires = resolve_expiry(event.severity, self.days_by_severity, event.timestamp)

        if self.clip_writer is not None and frames:
            written = self.clip_writer.write(frames)
            if written is not None:
                data, content_type, ext = written
                blob = self.store.save(
                    event.camera_id,
                    event.event_id,
                    f"clip.{ext}",
                    data,
                    content_type,
                )
                duration_ms = int(
                    max((frames[-1].ts - frames[0].ts).total_seconds(), 0.0) * 1000
                )
                items.append(
                    EvidenceItem(
                        evidence_id=new_id("evd"),
                        event_id=event.event_id,
                        camera_id=event.camera_id,
                        type=EvidenceType.CLIP,
                        uri=blob.uri,
                        sha256=blob.sha256,
                        size_bytes=blob.size_bytes,
                        content_type=content_type,
                        duration_ms=duration_ms,
                        captured_at=event.timestamp,
                        created_at=event.timestamp,
                        expires_at=expires,
                        storage_backend=self.store.backend,
                        metadata=self._metadata(event),
                    )
                )

        snapshots = self._pick_snapshots(frames)
        for index, packet in enumerate(snapshots):
            blob = self.store.save(
                event.camera_id,
                event.event_id,
                f"snapshot_{index:02d}.jpg",
                packet.data,
                "image/jpeg",
            )
            items.append(
                EvidenceItem(
                    evidence_id=new_id("evd"),
                    event_id=event.event_id,
                    camera_id=event.camera_id,
                    type=EvidenceType.SNAPSHOT,
                    uri=blob.uri,
                    sha256=blob.sha256,
                    size_bytes=blob.size_bytes,
                    content_type="image/jpeg",
                    width=packet.width or None,
                    height=packet.height or None,
                    captured_at=packet.ts,
                    created_at=packet.ts,
                    expires_at=expires,
                    storage_backend=self.store.backend,
                    metadata=self._metadata(event),
                )
            )
        log.info(
            "evidence finalized",
            extra={
                "event_id": event.event_id,
                "camera_id": camera_id,
                "items": len(items),
                "frames": len(frames),
            },
        )
        for item in items:
            log.info(
                "evidence created",
                extra={
                    "evidence_id": item.evidence_id,
                    "event_id": item.event_id,
                    "type": item.type.value,
                    "sha256": item.sha256,
                },
            )
        if self.on_finalize is not None:
            try:
                self.on_finalize(event, items)
            except Exception:
                log.exception(
                    "evidence finalize callback failed",
                    extra={"event_id": event.event_id},
                )
        return items

    def _pick_snapshots(self, frames: list[FramePacket]) -> list[FramePacket]:
        if not frames:
            return []
        count = min(self.snapshot_count, len(frames))
        if count == 1:
            return [frames[len(frames) // 2]]
        indices = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
        seen: set[int] = set()
        picked: list[FramePacket] = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                picked.append(frames[idx])
        return picked

    def _metadata(self, event: Event) -> dict:
        return {
            "event_type": event.event_type,
            "severity": event.severity.value
            if isinstance(event.severity, Severity)
            else str(event.severity),
            "zone_id": event.zone_id,
            "zone_name": event.zone_name,
            "track_ids": list(event.track_ids),
            "rule_id": event.rule_id,
            "rule_name": event.rule_name,
            "rule_version": event.metadata.get("rule_version"),
            "confidence": event.confidence,
            "model_versions": dict(event.model_versions),
        }
