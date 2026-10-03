from services.inference.interfaces import Detector, DetectorFactory, DetectorRegistry, default_registry
from services.inference.stub import NullDetector, StubDetector
from services.inference.types import BBox, Detection, ModelInfo

__all__ = [
    "Detector",
    "DetectorFactory",
    "DetectorRegistry",
    "default_registry",
    "NullDetector",
    "StubDetector",
    "BBox",
    "Detection",
    "ModelInfo",
]
