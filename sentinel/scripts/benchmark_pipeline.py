"""DEVELOPMENT ENVIRONMENT BENCHMARK - latency/throughput of the F1 pipeline.

NOT an accuracy benchmark: no mAP, precision/recall or FP/hour claims are made
here or anywhere in F1 (no evaluation dataset has been run - see
docs/MODEL_REGISTRY.md). Numbers describe THIS machine only.

Usage:
  python scripts/benchmark_pipeline.py --source path/to/video.mp4 --profile hog
  python scripts/benchmark_pipeline.py --source synthetic --frames 200 --profile hog
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.schemas.common import Severity  # noqa: E402
from packages.schemas.rule import RuleDefinition, RuleType  # noqa: E402
from services.camera.sources import FileSource  # noqa: E402
from services.camera.types import FramePacket  # noqa: E402
from services.events.engine import EventEngine  # noqa: E402
from services.inference.interfaces import default_registry  # noqa: E402
from services.inference.types import Detection  # noqa: E402
from services.pipeline.pipeline import SurveillancePipeline  # noqa: E402
from services.rules.base import RuleSet  # noqa: E402
from services.rules.context import SpatioTemporalState, ZoneContext  # noqa: E402
from services.tracking.iou import IOUTracker  # noqa: E402

LEFT_ZONE = [(0.0, 0.0), (0.25, 0.0), (0.25, 1.0), (0.0, 1.0)]
CENTER_ZONE = [(0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)]


class TimingDetector:
    """Wraps a detector and records per-call latency."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.info = inner.info
        self.durations: list[float] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def detect(self, frame: FramePacket) -> list[Detection]:
        start = time.perf_counter()
        try:
            return self._inner.detect(frame)
        finally:
            self.durations.append((time.perf_counter() - start) * 1000.0)

    def warmup(self) -> None:
        self._inner.warmup()

    def close(self) -> None:
        self._inner.close()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((p / 100.0) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def summarize(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: no samples"
    return (
        f"{label}: n={len(values)} p50={percentile(values, 50):.1f}ms "
        f"p95={percentile(values, 95):.1f}ms p99={percentile(values, 99):.1f}ms "
        f"max={max(values):.1f}ms mean={statistics.fmean(values):.1f}ms"
    )


def synthetic_frames(count: int, camera_id: str):
    base = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    try:
        import cv2
        import numpy as np

        ok, buf = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
        payload = buf.tobytes() if ok else b"\x00"
    except Exception:
        payload = b"\x00"
    for i in range(count):
        yield FramePacket(
            camera_id=camera_id,
            frame_id=i,
            ts=base + timedelta(seconds=i / 30.0),
            data=payload,
            width=640,
            height=480,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="video file path or 'synthetic'")
    parser.add_argument("--profile", default="hog", help="detector profile (hog|stub|null|yolox)")
    parser.add_argument("--frames", type=int, default=300, help="frame cap (synthetic/all sources)")
    parser.add_argument("--zone", choices=["left", "center"], default="left")
    parser.add_argument("--camera-id", default="bench")
    parser.add_argument("--warmup", type=int, default=1, help="warm-up iterations before timing (policy recorded in report)")
    parser.add_argument("--json", dest="json_out", help="also write the benchmark report as JSON to this path")
    args = parser.parse_args()

    zone_polygon = LEFT_ZONE if args.zone == "left" else CENTER_ZONE
    zone = ZoneContext(
        id="restricted",
        name="Benchmark zone",
        zone_type="restricted",
        polygon=zone_polygon,
    )
    definition = RuleDefinition(
        rule_id="restricted-zone-entry",
        name="Restricted zone intrusion",
        rule_type=RuleType.RESTRICTED_ZONE_INTRUSION,
        event_type="restricted_zone_intrusion",
        severity=Severity.HIGH,
        zone_ids=["restricted"],
        params={"classes": ["person"]},
        min_confidence=0.5,
        confirm_seconds=2.0,
        cooldown_seconds=30.0,
    )

    detector = TimingDetector(default_registry().create(args.profile))
    for _ in range(max(0, args.warmup)):
        detector.warmup()
    pipeline = SurveillancePipeline(
        camera_id=args.camera_id,
        detector=detector,
        tracker=IOUTracker(),
        rule_set=RuleSet([definition]),
        engine=EventEngine(rules={definition.rule_id: definition}),
        spatial=SpatioTemporalState(),
        zones=[zone],
        evidence_service=_NullEvidence(),
        rule_version="bench",
    )

    if args.source == "synthetic":
        frames = synthetic_frames(args.frames, args.camera_id)
        source_label = f"synthetic x{args.frames}"
    else:
        source = FileSource(args.camera_id, args.source)
        source.open()
        source_label = args.source
        frames = (packet for packet in iter(source.read, None))

    frame_durations: list[float] = []
    count = 0
    wall_start = time.perf_counter()
    for packet in frames:
        start = time.perf_counter()
        pipeline.process_frame(packet)
        frame_durations.append((time.perf_counter() - start) * 1000.0)
        count += 1
        if count >= args.frames:
            break
    wall = time.perf_counter() - wall_start
    detector.close()

    if args.source != "synthetic":
        source.close()

    print("=" * 72)
    print("DEVELOPMENT ENVIRONMENT BENCHMARK - latency/throughput only.")
    print("NOT an accuracy benchmark: no mAP / precision / recall / FP-hour claims.")
    print("=" * 72)
    print(f"host            : {platform.platform()} cpus={platform.machine()} cores={os.cpu_count()}")
    print(f"python          : {platform.python_version()}")
    try:
        import cv2

        print(f"opencv          : {cv2.__version__}")
    except Exception:
        print("opencv          : not installed")
    try:
        import onnxruntime

        print(f"onnxruntime     : {onnxruntime.__version__}")
    except Exception:
        pass
    print(f"source          : {source_label}")
    print(f"detector        : {detector.info.name}:{detector.info.version} ({args.profile})")
    print(f"detector license: {detector.info.license}")
    print(f"model status    : {detector.info.status} / metrics={detector.info.metrics.get('status')}")
    print(f"input resolution: {list(detector.info.input_size)} batch size=1")
    print(f"conf threshold  : {getattr(detector, 'confidence_threshold', getattr(detector, 'min_confidence', 'n/a'))}")
    print(f"warm-up policy  : {args.warmup} warmup() call(s) before timing")
    print(f"tracker         : IOUTracker (DEVELOPMENT TRACKER)")
    print(f"zone            : {args.zone} {zone_polygon}")
    print(f"frames          : {count}")
    print(f"wall time       : {wall:.2f}s -> {count / wall if wall else 0.0:.1f} fps process rate")
    print(summarize("detector        ", detector.durations))
    print(summarize("frame end-to-end", frame_durations))
    print(f"events created  : {pipeline.events_created}")
    print(f"detector errors : {pipeline.detector_errors}")
    print(f"tracker errors  : {pipeline.tracker_errors}")
    print(f"model_versions  : {pipeline.model_versions}")

    if args.json_out:
        import json

        report = {
            "label": "DEVELOPMENT ENVIRONMENT",
            "kind": "latency/throughput only - NOT an accuracy benchmark",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "gpu": "none used (CPUExecutionProvider)",
            },
            "software": {
                "python": platform.python_version(),
                "opencv": _try_version("cv2"),
                "onnxruntime": _try_version("onnxruntime"),
            },
            "model": {
                "id": detector.info.name,
                "version": detector.info.version,
                "license": detector.info.license,
                "status": detector.info.status,
                "metrics_status": detector.info.metrics.get("status"),
                "input_resolution": list(detector.info.input_size),
                "batch_size": 1,
                "confidence_threshold": getattr(detector, "confidence_threshold", getattr(detector, "min_confidence", None)),
            },
            "warmup_policy": {"warmup_calls": args.warmup},
            "source": source_label,
            "frames": count,
            "wall_seconds": round(wall, 3),
            "throughput_fps": round(count / wall, 3) if wall else None,
            "detector_latency_ms": _stats(detector.durations),
            "frame_latency_ms": _stats(frame_durations),
            "events_created": pipeline.events_created,
            "detector_errors": pipeline.detector_errors,
            "tracker_errors": pipeline.tracker_errors,
            "model_versions": dict(pipeline.model_versions),
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json report     : {out}")
    return 0


def _try_version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:
        return None


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


class _NullEvidence:
    def on_frame(self, packet) -> bool:
        return False

    def begin(self, event) -> None:
        return None

    def finalize(self, camera_id):
        return []


if __name__ == "__main__":
    raise SystemExit(main())
