from collections import deque

from services.camera.types import FramePacket


class RollingFrameBuffer:
    def __init__(self, camera_id: str, max_seconds: float = 30.0, max_frames: int = 600) -> None:
        self.camera_id = camera_id
        self.max_seconds = max_seconds
        self.max_frames = max_frames
        self._frames: deque[FramePacket] = deque()

    def append(self, packet: FramePacket) -> None:
        self._frames.append(packet)
        self._prune(packet)

    def _prune(self, newest: FramePacket) -> None:
        from datetime import timedelta

        cutoff = newest.ts - timedelta(seconds=self.max_seconds)
        while self._frames and (
            len(self._frames) > self.max_frames or self._frames[0].ts < cutoff
        ):
            self._frames.popleft()

    def frames_since(self, since_ts) -> list[FramePacket]:
        return [f for f in self._frames if f.ts >= since_ts]

    def recent(self, seconds: float, now) -> list[FramePacket]:
        from datetime import timedelta

        return self.frames_since(now - timedelta(seconds=seconds))

    def latest(self) -> FramePacket | None:
        """Newest frame, or None. Safe to call from another thread.

        Only atomic deque operations are used (len + index), so a concurrent
        append/popleft from the pipeline thread cannot raise.
        """
        return self._frames[-1] if self._frames else None

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()
