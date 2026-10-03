from services.camera.types import FramePacket
from services.inference.types import BBox, Detection, ModelInfo

_BASE_INFO = ModelInfo(
    name="stub-detector",
    version="0.0.0",
    family="none",
    license="internal",
    classes=(),
    status="development",
)


class NullDetector:
    def __init__(self) -> None:
        self.info = ModelInfo(
            name="null-detector",
            version="0.0.0",
            family="none",
            license="internal",
            classes=(),
        )

    def detect(self, frame: FramePacket) -> list[Detection]:
        return []

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


class StubDetector:
    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.info = _BASE_INFO
        self._detections = detections or [
            Detection(class_name="person", confidence=0.9, bbox=BBox(0.4, 0.3, 0.2, 0.4))
        ]

    def detect(self, frame: FramePacket) -> list[Detection]:
        return list(self._detections)

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None
