"""End-to-end F1 vertical slice: PipelineRunner -> persistence -> API retrieval."""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

T0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
STEP = timedelta(seconds=0.2)
CAMERA = "cam-e2e"


def _jpg():
    import numpy as np

    import cv2

    ok, buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_vertical_slice_frame_to_api(settings, client, tmp_path):
    import cv2  # noqa: F401
    from packages.config.settings import PipelineSettings, ZoneSpec
    from services.camera.types import FramePacket
    from services.pipeline.runner import PipelineRunner

    settings.pipeline = PipelineSettings(
        camera_id=CAMERA,
        camera_name="E2E camera",
        source_type="synthetic",
        detector_profile="stub",
        tracker="iou",
        rule_pack="factory.yaml",
        detection_fps=5.0,
        zones=[
            ZoneSpec(
                id="restricted",
                name="Restricted Area",
                zone_type="restricted",
                polygon=[(0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)],
            )
        ],
    )
    settings.evidence.root = str(tmp_path / "evidence")

    runner = PipelineRunner(settings)
    runner.ensure_camera_row()

    jpg = _jpg()
    emitted = []
    for i in range(40):
        packet = FramePacket(
            camera_id=CAMERA,
            frame_id=i,
            ts=T0 + STEP * i,
            data=jpg,
            width=64,
            height=48,
        )
        emitted.extend(runner.pipeline.process_frame(packet))
    assert len(emitted) == 1
    event = emitted[0]
    runner.evidence.finalize(CAMERA)
    runner.flush_health()

    # camera visible without leaking any stream configuration
    cameras = client.get("/api/v1/cameras").json()
    assert cameras["total"] == 1
    cam = cameras["items"][0]
    assert cam["camera_id"] == CAMERA
    assert cam["source_type"] == "synthetic"
    assert "stream_url" not in cam

    # health snapshot flushed by the runner
    health = client.get(f"/api/v1/cameras/{CAMERA}/health").json()
    assert health
    assert health[0]["ai_status"] == "healthy"
    assert health[0]["details"]["fps_gated"] == 0

    # event retrievable with explainability + summary + rule version
    listing = client.get("/api/v1/events", params={"camera_id": CAMERA}).json()
    assert listing["total"] == 1
    item = listing["items"][0]
    assert item["event_id"] == event.event_id
    assert item["status"] == "new"
    assert item["severity"] == "high"
    assert item["event_type"] == "restricted_zone_intrusion"
    assert item["rule_id"] == "restricted-zone-entry"
    assert item["zone_name"] == "Restricted Area"
    assert "Restricted Area" in item["summary"]
    assert item["metadata"]["rule_version"] == "0.1"
    assert item["conditions"]
    assert item["model_versions"]["detector"].startswith("stub-detector")
    assert item["evidence_ids"]

    detail = client.get(f"/api/v1/events/{event.event_id}").json()
    assert detail["summary"] == item["summary"]
    assert detail["track_ids"] == item["track_ids"]

    # evidence rows linked to the event, hashes verifiable against disk
    evidence_page = client.get("/api/v1/evidence", params={"event_id": event.event_id}).json()
    assert evidence_page["total"] >= 1
    root = Path(settings.evidence.root)
    for evd in evidence_page["items"]:
        assert evd["event_id"] == event.event_id
        assert evd["sha256"]
        assert evd["metadata"]["rule_version"] == "0.1"
        on_disk = (root / evd["uri"]).read_bytes()
        assert hashlib.sha256(on_disk).hexdigest() == evd["sha256"]
        single = client.get(f"/api/v1/evidence/{evd['evidence_id']}")
        assert single.status_code == 200
        assert single.json()["sha256"] == evd["sha256"]

    # operator can acknowledge the event
    ack = client.post(
        f"/api/v1/events/{event.event_id}/status", json={"status": "acknowledged"}
    )
    assert ack.status_code == 200
    assert ack.json()["status"] == "acknowledged"
    assert ack.json()["acknowledged_by"] == "dev-admin"


def test_runner_start_stop_with_synthetic_source(settings, tmp_path):
    from packages.config.settings import PipelineSettings, ZoneSpec
    from services.pipeline.runner import PipelineRunner

    settings.pipeline = PipelineSettings(
        camera_id="cam-live",
        source_type="synthetic",
        detector_profile="null",
        tracker="null",
        rule_pack="factory.yaml",
        detection_fps=5.0,
        zones=[
            ZoneSpec(
                id="restricted",
                name="Restricted Area",
                polygon=[(0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)],
            )
        ],
    )
    settings.evidence.root = str(tmp_path / "evidence")

    runner = PipelineRunner(settings)
    runner.start()
    import time

    time.sleep(0.5)
    snap = runner.worker.snapshot()
    runner.stop(join_timeout=5.0)
    assert runner.worker.state.value == "stopped"
    assert snap.camera_id == "cam-live"

