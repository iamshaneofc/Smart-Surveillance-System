from typing import Protocol

from services.camera.types import FramePacket
from services.inference.types import Detection, ModelInfo


class DetectorError(Exception):
    """Raised when a detector fails (config, weights, inference); the pipeline must survive it."""


class Detector(Protocol):
    @property
    def info(self) -> ModelInfo: ...

    def detect(self, frame: FramePacket) -> list[Detection]: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...


class DetectorFactory(Protocol):
    def __call__(self, **kwargs) -> Detector: ...


class DetectorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, DetectorFactory] = {}

    def register(self, profile: str, factory: DetectorFactory) -> None:
        self._factories[profile] = factory

    def create(self, profile: str, **kwargs) -> Detector:
        if profile not in self._factories:
            known = ", ".join(sorted(self._factories)) or "none"
            raise KeyError(f"unknown detector profile '{profile}' (registered: {known})")
        return self._factories[profile](**kwargs)

    def profiles(self) -> list[str]:
        return sorted(self._factories)


def default_registry() -> DetectorRegistry:
    from services.inference.stub import NullDetector, StubDetector

    registry = DetectorRegistry()
    registry.register("null", lambda **_: NullDetector())
    registry.register("stub", lambda **_: StubDetector())

    def _hog(**kwargs):
        from services.inference.hog import HogPeopleDetector

        return HogPeopleDetector(**kwargs)

    registry.register("hog", _hog)

    def _yolox(**kwargs):
        from services.inference.yolox import YoloXOnnxDetector

        return YoloXOnnxDetector(**kwargs)

    registry.register("yolox", _yolox)
    return registry
