import threading
from datetime import timedelta

from packages.schemas.camera import CameraState
from packages.schemas.common import HealthState
from services.camera.interfaces import FpsGate
from services.camera.manager import CameraWorker, WorkerConfig
from services.camera.sources import SyntheticSource
from services.camera.types import SourceReadError, SourceUnavailableError


def test_worker_reads_frames_and_stops():
    seen = []
    worker = CameraWorker(
        WorkerConfig(camera_id="cam1", detection_fps=0.0, reconnect_initial_seconds=0.0, reconnect_max_seconds=0.0),
        source_factory=lambda: SyntheticSource("cam1", max_frames=10),
        on_frame=seen.append,
    )
    worker.run()
    assert len(seen) == 10
    assert seen[0].frame_id == 0
    assert worker.state == CameraState.STOPPED
    snap = worker.snapshot()
    assert snap.frames_processed == 10
    assert snap.health == HealthState.OFFLINE
    assert snap.camera_id == "cam1"


def test_worker_reconnects_after_read_failure():
    state = {"opens": 0}

    def factory():
        state["opens"] += 1
        if state["opens"] == 1:
            return SyntheticSource("cam1", fail_after=3)
        return SyntheticSource("cam1", max_frames=100)

    worker = CameraWorker(
        WorkerConfig(
            camera_id="cam1",
            detection_fps=0.0,
            reconnect_initial_seconds=0.0,
            reconnect_max_seconds=0.0,
            max_consecutive_failures=5,
        ),
        source_factory=factory,
    )
    worker.run(max_frames=5)
    assert worker.reconnect_count == 1
    assert worker.frames_processed == 5
    assert worker.state == CameraState.STOPPED


def test_worker_goes_offline_when_source_cannot_open():
    def factory():
        raise SourceReadError("no route to camera")

    worker = CameraWorker(
        WorkerConfig(
            camera_id="cam1",
            detection_fps=0.0,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.01,
            max_consecutive_failures=2,
        ),
        source_factory=factory,
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    for _ in range(200):
        if worker.state == CameraState.OFFLINE:
            break
        threading.Event().wait(0.01)
    assert worker.state == CameraState.OFFLINE
    snap = worker.snapshot()
    assert snap.health == HealthState.ERROR
    assert snap.error
    worker.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_worker_counts_frame_gaps_as_drops():
    seen = []

    def factory():
        return GapSource("cam1")

    worker = CameraWorker(
        WorkerConfig(camera_id="cam1", detection_fps=0.0, reconnect_initial_seconds=0.0, reconnect_max_seconds=0.0),
        source_factory=factory,
        on_frame=seen.append,
    )
    worker.run()
    assert worker.frame_drops == 2
    assert worker.frames_processed == 4


def test_worker_reconnects_after_connection_failure():
    state = {"opens": 0}

    def factory():
        state["opens"] += 1
        if state["opens"] == 1:
            raise SourceUnavailableError("connection refused")
        return SyntheticSource("cam1", max_frames=100)

    worker = CameraWorker(
        WorkerConfig(
            camera_id="cam1",
            detection_fps=0.0,
            reconnect_initial_seconds=0.0,
            reconnect_max_seconds=0.0,
            max_consecutive_failures=3,
        ),
        source_factory=factory,
    )
    worker.run(max_frames=3)
    assert state["opens"] == 2
    assert worker.reconnect_count == 1
    assert worker.frames_processed == 3


def test_worker_fps_gating_drops_extra_frames():
    from datetime import datetime, timezone

    from services.camera.types import FramePacket

    t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    class ScriptedSource:
        camera_id = "cam1"

        def __init__(self):
            self._i = 0

        @property
        def is_exhausted(self):
            return self._i >= 10

        def open(self):
            pass

        def read(self):
            if self._i >= 10:
                return None
            packet = FramePacket(
                camera_id="cam1",
                frame_id=self._i,
                ts=t0 + timedelta(seconds=self._i * 0.1),
                data=b"x",
            )
            self._i += 1
            return packet

        def close(self):
            pass

    seen = []
    worker = CameraWorker(
        WorkerConfig(
            camera_id="cam1",
            detection_fps=2.0,
            reconnect_initial_seconds=0.0,
            reconnect_max_seconds=0.0,
        ),
        source_factory=ScriptedSource,
        on_frame=seen.append,
    )
    worker.run()
    # gate at 2 fps => every 0.5s: frames at t=0.0 and t=0.5 only
    assert [p.frame_id for p in seen] == [0, 5]
    snap = worker.snapshot()
    assert snap.details["fps_gated"] == 8
    assert snap.frames_processed == 2
    assert worker.frame_drops == 0


def test_redact_secrets_strips_url_credentials():
    from packages.common.textutil import redact_secrets

    msg = redact_secrets("cannot open rtsp://admin:secret123@10.0.0.1/stream")
    assert "secret123" not in msg
    assert "***" in msg
    keyed = redact_secrets("header X-API-Key: 'abc123def'")
    assert "abc123def" not in keyed


class GapSource:
    def __init__(self, camera_id):
        self._camera_id = camera_id
        self._seq = [0, 1, 4, 5]
        self._i = 0

    @property
    def camera_id(self):
        return self._camera_id

    @property
    def is_exhausted(self):
        return self._i >= len(self._seq)

    def open(self):
        pass

    def read(self):
        from packages.common.timeutil import utcnow
        from services.camera.types import FramePacket

        if self._i >= len(self._seq):
            return None
        packet = FramePacket(
            camera_id=self._camera_id,
            frame_id=self._seq[self._i],
            ts=utcnow(),
            data=b"x",
        )
        self._i += 1
        return packet

    def close(self):
        pass


def test_fps_gate():
    gate = FpsGate(2.0)
    assert gate.allow(0.0) is True
    assert gate.allow(0.1) is False
    assert gate.allow(0.5) is True
    assert gate.allow(0.6) is False
    assert gate.allow(1.1) is True

