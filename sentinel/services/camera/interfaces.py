from typing import Protocol

from services.camera.types import FramePacket


class CameraSource(Protocol):
    @property
    def camera_id(self) -> str: ...

    @property
    def is_exhausted(self) -> bool: ...

    def open(self) -> None: ...

    def read(self) -> FramePacket | None: ...

    def close(self) -> None: ...


class SourceFactory(Protocol):
    def __call__(self) -> CameraSource: ...


class FpsGate:
    def __init__(self, fps: float) -> None:
        self.min_interval = 1.0 / fps if fps > 0 else 0.0
        self._last: float | None = None

    def allow(self, ts_epoch: float) -> bool:
        if self._last is None or (ts_epoch - self._last) >= self.min_interval:
            self._last = ts_epoch
            return True
        return False
