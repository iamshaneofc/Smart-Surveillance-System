from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.camera.sources import FileSource, source_factory_for
from services.camera.types import SourceUnavailableError


def test_missing_file_raises_source_unavailable(tmp_path):
    source = FileSource("cam1", str(tmp_path / "nope.mp4"))
    with pytest.raises(SourceUnavailableError):
        source.open()


def test_unknown_source_type_raises():
    with pytest.raises(SourceUnavailableError):
        source_factory_for("cam1", "carrier-pigeon", "")


def test_file_source_missing_path_via_factory(tmp_path):
    factory = source_factory_for("cam1", "file", str(tmp_path / "absent.avi"))
    source = factory()
    with pytest.raises(SourceUnavailableError):
        source.open()


def test_file_source_deterministic_video_timeline(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("video writer unavailable on this platform")
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    assert path.exists()

    source = FileSource("cam1", str(path))
    source.open()
    first = source.read()
    base = first.ts
    assert first.frame_id == 0
    assert first.width == 64
    assert first.height == 48
    assert source.fps == pytest.approx(20.0, abs=0.5)

    expected = None
    packets = [first]
    for i in range(1, 10):
        packet = source.read()
        assert packet is not None
        assert packet.frame_id == i
        packets.append(packet)
    for i, packet in enumerate(packets):
        expected = base.timestamp() + i / source.fps
        assert packet.ts.timestamp() == pytest.approx(expected, abs=1e-6)

    assert source.read() is None
    assert source.is_exhausted
    source.close()


def test_file_source_read_before_open_raises(tmp_path):
    source = FileSource("cam1", "whatever.mp4")
    with pytest.raises(SourceUnavailableError):
        source.read()
