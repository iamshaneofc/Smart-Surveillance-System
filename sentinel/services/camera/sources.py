import time
from datetime import timedelta
from pathlib import Path

from packages.common.logging import get_logger
from packages.common.textutil import redact_secrets
from packages.common.timeutil import utcnow
from services.camera.types import FramePacket, SourceReadError, SourceUnavailableError

log = get_logger(__name__)


class SyntheticSource:
    def __init__(
        self,
        camera_id: str,
        fps: float = 10.0,
        width: int = 640,
        height: int = 360,
        frame_bytes: bytes = b"\x00\x01",
        max_frames: int | None = None,
        fail_after: int | None = None,
        pace: bool = False,
    ) -> None:
        self._camera_id = camera_id
        self.fps = fps
        self.width = width
        self.height = height
        self._payload = frame_bytes
        self._max_frames = max_frames
        self._fail_after = fail_after
        self._pace = pace
        self._frame_id = 0
        self._opened = False
        self._closed = False

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def is_exhausted(self) -> bool:
        if self._max_frames is not None and self._frame_id >= self._max_frames:
            return True
        return self._closed

    def open(self) -> None:
        self._opened = True
        self._closed = False
        self._frame_id = 0

    def read(self) -> FramePacket | None:
        if not self._opened or self._closed:
            raise SourceUnavailableError("source not open")
        if self._max_frames is not None and self._frame_id >= self._max_frames:
            return None
        if self._fail_after is not None and self._frame_id >= self._fail_after:
            raise SourceReadError("synthetic failure injected")
        if self._pace and self.fps > 0:
            time.sleep(1.0 / self.fps)
        packet = FramePacket(
            camera_id=self._camera_id,
            frame_id=self._frame_id,
            ts=utcnow(),
            data=self._payload,
            width=self.width,
            height=self.height,
        )
        self._frame_id += 1
        return packet

    def close(self) -> None:
        self._closed = True


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SourceUnavailableError(
            "opencv-python(-headless) is not installed; install sentinel[video]"
        ) from exc
    return cv2


class FileSource:
    """Local video file source with deterministic video-timeline timestamps.

    Frame timestamps are derived from the video fps anchored at open() time
    (ts = open_time + frame_index / fps), so the pipeline sees a stable
    timeline regardless of read speed.
    """

    def __init__(self, camera_id: str, path: str) -> None:
        self._camera_id = camera_id
        self._path = path
        self._cv2 = None
        self._cap = None
        self._exhausted = False
        self._fps = 30.0
        self._base_ts = None

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    def open(self) -> None:
        self._cv2 = _require_cv2()
        if not Path(self._path).exists():
            raise SourceUnavailableError(f"video file not found: {self._path}")
        self._cap = self._cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise SourceUnavailableError(f"cannot open file: {self._path}")
        fps = float(self._cap.get(self._cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if fps > 0 else 30.0
        self._base_ts = utcnow()
        self._exhausted = False

    def read(self) -> FramePacket | None:
        if self._cap is None or self._base_ts is None:
            raise SourceUnavailableError("source not open")
        ok, frame = self._cap.read()
        if not ok:
            self._exhausted = True
            return None
        ok_enc, buf = self._cv2.imencode(".jpg", frame)
        if not ok_enc:
            raise SourceReadError("frame encode failed")
        h, w = frame.shape[:2]
        index = int(self._cap.get(self._cv2.CAP_PROP_POS_FRAMES)) - 1
        ts = self._base_ts + timedelta(seconds=index / self._fps)
        return FramePacket(
            camera_id=self._camera_id,
            frame_id=max(index, 0),
            ts=ts,
            data=buf.tobytes(),
            width=w,
            height=h,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class WebcamSource:
    def __init__(self, camera_id: str, device_index: int = 0) -> None:
        self._camera_id = camera_id
        self._device_index = device_index
        self._cv2 = None
        self._cap = None

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def is_exhausted(self) -> bool:
        return False

    def open(self) -> None:
        self._cv2 = _require_cv2()
        self._cap = self._cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise SourceUnavailableError(f"cannot open webcam {self._device_index}")

    def read(self) -> FramePacket | None:
        if self._cap is None:
            raise SourceUnavailableError("source not open")
        ok, frame = self._cap.read()
        if not ok:
            raise SourceReadError("webcam read failed")
        ok_enc, buf = self._cv2.imencode(".jpg", frame)
        if not ok_enc:
            raise SourceReadError("frame encode failed")
        h, w = frame.shape[:2]
        return FramePacket(
            camera_id=self._camera_id,
            frame_id=int(self._cap.get(self._cv2.CAP_PROP_POS_FRAMES)),
            ts=utcnow(),
            data=buf.tobytes(),
            width=w,
            height=h,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RTSPSource:
    """RTSP source (no ONVIF discovery in F1). Credentials never appear in logs."""

    def __init__(self, camera_id: str, url: str, open_timeout_seconds: float = 10.0) -> None:
        self._camera_id = camera_id
        self._url = url
        self._open_timeout_seconds = open_timeout_seconds
        self._cv2 = None
        self._cap = None

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def is_exhausted(self) -> bool:
        return False

    def open(self) -> None:
        self._cv2 = _require_cv2()
        timeout_ms = int(self._open_timeout_seconds * 1000)
        try:
            self._cap = self._cv2.VideoCapture(
                self._url,
                self._cv2.CAP_FFMPEG,
                [self._cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms],
            )
        except Exception:
            self._cap = self._cv2.VideoCapture(self._url, self._cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise SourceUnavailableError(
                redact_secrets(f"cannot open stream for camera {self._camera_id}: {self._url}")
            )

    def read(self) -> FramePacket | None:
        if self._cap is None:
            raise SourceUnavailableError("source not open")
        ok, frame = self._cap.read()
        if not ok:
            raise SourceReadError(
                redact_secrets(f"stream read failed for camera {self._camera_id}")
            )
        ok_enc, buf = self._cv2.imencode(".jpg", frame)
        if not ok_enc:
            raise SourceReadError("frame encode failed")
        h, w = frame.shape[:2]
        return FramePacket(
            camera_id=self._camera_id,
            frame_id=int(self._cap.get(self._cv2.CAP_PROP_POS_FRAMES)),
            ts=utcnow(),
            data=buf.tobytes(),
            width=w,
            height=h,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def source_factory_for(camera_id: str, source_type: str, stream_url: str, device_index: int = 0):
    if source_type == "synthetic":
        return lambda: SyntheticSource(camera_id)
    if source_type == "file":
        return lambda: FileSource(camera_id, stream_url)
    if source_type == "webcam":
        return lambda: WebcamSource(camera_id, device_index)
    if source_type == "rtsp":
        return lambda: RTSPSource(camera_id, stream_url)
    raise SourceUnavailableError(f"unsupported source type: {source_type}")
