import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from packages.common.logging import get_logger
from packages.common.textutil import redact_secrets
from packages.common.timeutil import ensure_utc, utcnow
from packages.schemas.camera import CameraHealthSnapshot, CameraState
from packages.schemas.common import HealthState
from services.camera.interfaces import FpsGate, SourceFactory
from services.camera.types import FramePacket, SourceReadError, SourceUnavailableError

log = get_logger(__name__)

OnFrame = Callable[[FramePacket], None]


@dataclass
class WorkerConfig:
    camera_id: str
    detection_fps: float = 5.0
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    max_consecutive_failures: int = 5


class CameraWorker:
    def __init__(
        self,
        config: WorkerConfig,
        source_factory: SourceFactory,
        on_frame: OnFrame | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._source_factory = source_factory
        self._on_frame = on_frame
        self._monotonic = monotonic
        self._gate = FpsGate(config.detection_fps) if config.detection_fps > 0 else None
        self._source = None
        self._state = CameraState.IDLE
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._frames = 0
        self._drops = 0
        self._gated = 0
        self._reconnects = 0
        self._consecutive_failures = 0
        self._error: str | None = None
        self._last_frame_at = None
        self._last_seen_frame_id: int | None = None
        self._ema_interval: float | None = None
        self._last_frame_mono: float | None = None
        self._latency_ms: float | None = None
        self._ai_status = HealthState.OFFLINE

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    def report_ai_status(self, status: HealthState) -> None:
        with self._lock:
            self._ai_status = status

    def _set_state(self, state: CameraState) -> None:
        with self._lock:
            if state != self._state:
                log.info(
                    "camera state change",
                    extra={"camera_id": self.config.camera_id, "from": self._state.value, "to": state.value},
                )
                self._state = state

    def _backoff(self) -> float:
        exp = min(self._reconnects, 16)
        return min(
            self.config.reconnect_initial_seconds * (2**exp),
            self.config.reconnect_max_seconds,
        )

    def _handle_failure(self, exc: Exception) -> bool:
        while True:
            with self._lock:
                self._consecutive_failures += 1
                self._error = redact_secrets(exc)
                self._reconnects += 1
                failures = self._consecutive_failures
                attempt = self._reconnects
            log.warning(
                "camera disconnected",
                extra={
                    "camera_id": self.config.camera_id,
                    "error": self._error,
                    "attempt": attempt,
                },
            )
            if failures >= self.config.max_consecutive_failures:
                self._set_state(CameraState.OFFLINE)
            else:
                self._set_state(CameraState.RECONNECTING)
            if self._stop.wait(self._backoff()):
                return False
            try:
                if self._source is not None:
                    try:
                        self._source.close()
                    except Exception:
                        pass
                self._source = self._source_factory()
                self._source.open()
            except Exception as reopen_exc:
                exc = reopen_exc
                continue
            with self._lock:
                self._consecutive_failures = 0
                self._error = None
            log.info(
                "camera reconnected",
                extra={"camera_id": self.config.camera_id, "reconnects": attempt},
            )
            self._set_state(CameraState.STREAMING)
            return True

    def _on_new_frame(self, packet: FramePacket) -> None:
        now_mono = self._monotonic()
        with self._lock:
            if self._last_seen_frame_id is not None and packet.frame_id > self._last_seen_frame_id + 1:
                self._drops += packet.frame_id - (self._last_seen_frame_id + 1)
            self._last_seen_frame_id = packet.frame_id
            if self._last_frame_mono is not None:
                interval = max(now_mono - self._last_frame_mono, 1e-6)
                self._ema_interval = (
                    interval if self._ema_interval is None else 0.8 * self._ema_interval + 0.2 * interval
                )
            self._last_frame_mono = now_mono
            self._frames += 1
            self._last_frame_at = packet.ts
            self._latency_ms = max((utcnow() - ensure_utc(packet.ts)).total_seconds(), 0.0) * 1000.0
        if self._on_frame is not None:
            self._on_frame(packet)

    def run(self, max_frames: int | None = None) -> None:
        self._set_state(CameraState.CONNECTING)
        try:
            self._source = self._source_factory()
            self._source.open()
            self._set_state(CameraState.STREAMING)
            log.info("camera connected", extra={"camera_id": self.config.camera_id})
        except Exception as exc:
            if not self._handle_failure(exc):
                return
        while not self._stop.is_set():
            if max_frames is not None and self._frames >= max_frames:
                break
            source = self._source
            if source is None:
                break
            try:
                packet = source.read()
            except (SourceReadError, SourceUnavailableError) as exc:
                if not self._handle_failure(exc):
                    break
                continue
            except Exception as exc:
                if not self._handle_failure(SourceReadError(redact_secrets(exc))):
                    break
                continue
            if packet is None:
                if source.is_exhausted:
                    break
                continue
            if self._gate is not None and not self._gate.allow(packet.ts.timestamp()):
                with self._lock:
                    self._gated += 1
                    if self._last_seen_frame_id is None or packet.frame_id > self._last_seen_frame_id:
                        self._last_seen_frame_id = packet.frame_id
                continue
            self._on_new_frame(packet)
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        source = self._source
        if source is not None:
            try:
                source.close()
            except Exception:
                log.exception("camera source close failed", extra={"camera_id": self.config.camera_id})
        self._set_state(CameraState.STOPPED)

    @property
    def frames_processed(self) -> int:
        return self._frames

    @property
    def frame_drops(self) -> int:
        return self._drops

    @property
    def reconnect_count(self) -> int:
        return self._reconnects

    def snapshot(self) -> CameraHealthSnapshot:
        with self._lock:
            state = self._state
            fps = 1.0 / self._ema_interval if self._ema_interval else 0.0
            health = {
                CameraState.STREAMING: HealthState.HEALTHY,
                CameraState.CONNECTING: HealthState.DEGRADED,
                CameraState.RECONNECTING: HealthState.DEGRADED,
                CameraState.OFFLINE: HealthState.ERROR if self._error else HealthState.OFFLINE,
                CameraState.STOPPED: HealthState.OFFLINE,
                CameraState.IDLE: HealthState.OFFLINE,
            }[state]
            return CameraHealthSnapshot(
                camera_id=self.config.camera_id,
                state=state,
                health=health,
                fps=round(fps, 2),
                frame_drops=self._drops,
                latency_ms=round(self._latency_ms, 1) if self._latency_ms is not None else None,
                reconnect_count=self._reconnects,
                frames_processed=self._frames,
                last_frame_at=self._last_frame_at,
                error=self._error,
                ai_status=self._ai_status,
                ts=utcnow(),
                details={"detection_fps": self.config.detection_fps, "fps_gated": self._gated},
            )
