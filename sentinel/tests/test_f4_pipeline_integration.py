"""F4-M/N: YOLOX detector profile through the pipeline - integration and
failure/degraded-mode behavior.

Uses a deterministic tiny ONNX fixture (never committed, no weights needed)
so the tests exercise the real YoloXOnnxDetector -> tracker -> zone rule ->
event path, with assertions on provenance and failure isolation only -
fixture detections are mechanics, never accuracy claims.
"""

from pathlib import Path

import pytest

from packages.schemas.common import HealthState
from services.camera.types import FramePacket
from services.inference.interfaces import DetectorError, default_registry
from services.inference.yolox import YoloXOnnxDetector
from tests.test_f4_detector_adapter import _build_fixture, _empty_head, _put_box
from tests.test_pipeline import CAMERA, STEP, T0, build

# Frame geometry the fixture box math below assumes (ratio = 416/640 = 0.65).
FRAME_W, FRAME_H = 640, 480
RATIO = 416 / FRAME_W

# IoU-compatible zone geometry (from test_pipeline): track boxes OUT/IN keep
# IoU 0.5 so one track survives the position change.
OUT_BOX = (0.22, 0.4, 0.12, 0.3)   # center x 0.28 -> outside the zone
IN_BOX = (0.26, 0.4, 0.12, 0.3)    # center x 0.32 -> inside the zone
CONFIRM_FRAMES = 15  # pending at frame 5, confirm_seconds 2.0 @ 0.2s step


def _jpg640() -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


JPG640 = _jpg640()


def frame640(i: int, data: bytes | None = None) -> FramePacket:
    return FramePacket(
        camera_id=CAMERA,
        frame_id=i,
        ts=T0 + STEP * i,
        data=JPG640 if data is None else data,
        width=FRAME_W,
        height=FRAME_H,
    )


def _fixture_at(tmp_path, name: str, norm_box) -> Path:
    """Tiny ONNX graph emitting one person box at a fixed normalized position."""
    x, y, w, h = norm_box
    cx = (x + w / 2) * FRAME_W * RATIO
    cy = (y + h / 2) * FRAME_H * RATIO
    bw = w * FRAME_W * RATIO
    bh = h * FRAME_H * RATIO
    head = _empty_head()
    _put_box(
        head,
        gx=int(cx // 8),
        gy=int(cy // 8),
        cx=cx,
        cy=cy,
        w=bw,
        h=bh,
        obj=0.95,
        cls_id=0,
        cls_score=0.93,
    )
    return _build_fixture(Path(tmp_path) / name, head)


def _detector_at(tmp_path, name: str, norm_box) -> YoloXOnnxDetector:
    return YoloXOnnxDetector(weights_path=_fixture_at(tmp_path, name, norm_box))


# ------------------------------------------------------------------ F4-M


def test_registry_yolox_profile_buildable_with_explicit_weights(tmp_path):
    det = default_registry().create(
        "yolox", weights_path=str(_fixture_at(tmp_path, "reg.onnx", IN_BOX))
    )
    try:
        assert det.info.name == "yolox-tiny"
        assert det.info.status == "candidate"
        assert det.info.label == "yolox-tiny:0.1.1rc0"
    finally:
        det.close()


def test_pipeline_runner_initializes_with_yolox_profile(settings, tmp_path):
    from packages.config.settings import PipelineSettings
    from services.pipeline.runner import PipelineRunner

    settings.pipeline = PipelineSettings(
        camera_id="cam-yolox",
        source_type="synthetic",
        detector_profile="yolox",
        detector_options={"weights_path": str(_fixture_at(tmp_path, "runner.onnx", IN_BOX))},
        rule_pack="factory.yaml",
    )
    settings.evidence.root = str(tmp_path / "evidence")

    runner = PipelineRunner(settings)
    try:
        assert runner.detector.info.name == "yolox-tiny"
        assert runner.pipeline.detector is runner.detector
        assert runner.pipeline.model_versions["detector"] == "yolox-tiny:0.1.1rc0"
    finally:
        runner.detector.close()


def test_yolox_adapter_drives_zone_event_with_provenance(tmp_path):
    det_out = _detector_at(tmp_path, "out.onnx", OUT_BOX)
    det_in = _detector_at(tmp_path, "in.onnx", IN_BOX)
    ai: list[HealthState] = []

    pipeline, engine, store, evidence = build(
        lambda i: None, detector=det_out, on_ai_status=ai.append
    )
    events = []
    for i in range(CONFIRM_FRAMES + 6):
        if i >= 5:
            pipeline.detector = det_in  # person walks into the zone
        events.extend(pipeline.process_frame(frame640(i)))

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "restricted_zone_intrusion"
    assert event.model_versions["detector"] == "yolox-tiny:0.1.1rc0"
    assert pipeline.model_versions == {"detector": "yolox-tiny:0.1.1rc0"}
    assert engine.open_events() == [event]
    # tracking, rules and confirmation all ran on real adapter detections
    assert pipeline.detector_errors == 0
    assert pipeline.tracker_errors == 0
    assert pipeline.frames_processed == CONFIRM_FRAMES + 6
    assert ai and set(ai) == {HealthState.HEALTHY}


# ------------------------------------------------------------------ F4-N


def test_missing_weights_pipeline_startup_fails_clearly(settings):
    from packages.config.settings import PipelineSettings
    from services.pipeline.runner import PipelineRunner

    settings.pipeline = PipelineSettings(
        camera_id="cam-missing",
        source_type="synthetic",
        detector_profile="yolox",
        detector_options={"weights_path": "models/__missing_for_test__.onnx"},
    )
    with pytest.raises(DetectorError) as exc:
        PipelineRunner(settings)
    message = str(exc.value)
    assert "not found" in message
    assert "download_models.py" in message


def test_detector_failure_isolated_and_recovers(tmp_path):
    """Corrupt frames -> DetectorError counted + DEGRADED, then HEALTHY again.

    Failure is never recorded as 'no detections': detector_errors increments
    and ai_status degrades while the frame loop keeps running.
    """
    det_out = _detector_at(tmp_path, "out2.onnx", OUT_BOX)
    det_in = _detector_at(tmp_path, "in2.onnx", IN_BOX)
    ai: list[HealthState] = []
    total = 25

    pipeline, engine, store, evidence = build(
        lambda i: None, detector=det_out, on_ai_status=ai.append
    )
    events = []
    for i in range(total):
        if i >= 10:
            pipeline.detector = det_in
        data = b"definitely-not-a-jpeg" if i in (3, 4, 5) else None
        events.extend(pipeline.process_frame(frame640(i, data=data)))

    # failure isolated: counted, degraded, frames still processed
    assert pipeline.detector_errors == 3
    assert pipeline.frames_processed == total
    assert len(ai) == total
    assert set(ai[:3]) == {HealthState.HEALTHY}      # before failure
    assert set(ai[3:6]) == {HealthState.DEGRADED}    # failure window
    assert set(ai[6:]) == {HealthState.HEALTHY}      # recovery
    # rules still fired after recovery, with yolox provenance
    assert len(events) == 1
    assert events[0].model_versions["detector"] == "yolox-tiny:0.1.1rc0"
