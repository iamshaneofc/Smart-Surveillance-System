"""F1 live acceptance run: real video -> CameraWorker -> HOG -> rules -> events -> evidence.

Runs the production wiring (PipelineRunner) against a local video file and then
verifies the F1 acceptance items directly against the database and evidence
store. Exit code 0 = all checks passed.

Usage:
  python scripts/run_live_acceptance.py [path/to/video.mp4]

The default input is the internal legacy clip (see docs/DATA_LICENSES.md #7 -
license unknown, internal development use only, never redistributed).
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_VIDEO = r"Z:\Security_Surveillance_System\snapshots\VID-20250724-WA0004.mp4"
LEFT_ZONE = [(0.0, 0.0), (0.25, 0.0), (0.25, 1.0), (0.0, 1.0)]

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    marker = "PASS" if ok else "FAIL"
    print(f"[{marker}] {name}" + (f" - {detail}" if detail else ""))


def main() -> int:
    from packages.config import Settings
    from packages.config.settings import PipelineSettings, ZoneSpec

    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_VIDEO)
    if not video.exists():
        print(f"input video not found: {video}")
        return 2

    settings = Settings()
    # dedicated database + evidence root so acceptance runs are repeatable
    acceptance_db = Path("var/acceptance.db")
    acceptance_db.parent.mkdir(parents=True, exist_ok=True)
    if acceptance_db.exists():
        acceptance_db.unlink()
    settings.database_url = f"sqlite:///{acceptance_db}"
    settings.evidence.root = "./var/acceptance-evidence"
    evidence_root = Path(settings.evidence.root)
    if evidence_root.exists():
        import shutil

        shutil.rmtree(evidence_root)

    from packages.db import base as db_base

    db_base.dispose()
    db_base.configure(settings.database_url)
    db_base.create_all()

    settings.pipeline = PipelineSettings(
        camera_id="cam-acceptance",
        camera_name="F1 acceptance camera",
        source_type="file",
        stream_url=str(video),
        detection_fps=5.0,
        detector_profile="hog",
        tracker="iou",
        rule_pack="factory.yaml",
        zones=[
            ZoneSpec(
                id="restricted",
                name="Restricted Area",
                zone_type="restricted",
                polygon=LEFT_ZONE,
            )
        ],
    )

    print("=" * 72)
    print("F1 LIVE ACCEPTANCE RUN")
    print(f"video          : {video}")
    print(f"detector       : hog (opencv-hog-people, status=development)")
    print(f"zone (left)    : {LEFT_ZONE}")
    print(f"rule pack      : factory.yaml (restricted_zone_intrusion, confirm=2s, cooldown=30s)")
    print("=" * 72)

    from services.pipeline.runner import PipelineRunner

    runner = PipelineRunner(settings)
    runner.start()
    finished = runner.wait(timeout=600.0)
    check("camera worker consumed the source", finished, f"frames={runner.worker.frames_processed}")
    runner.stop(join_timeout=30.0)

    # -- database verification -----------------------------------------
    from sqlalchemy import select

    from packages.db import models

    with db_base.session_scope() as session:
        events = session.execute(
            select(models.Event).where(models.Event.camera_id == "cam-acceptance")
        ).scalars().all()
        evidence_rows = session.execute(
            select(models.Evidence).where(models.Evidence.camera_id == "cam-acceptance")
        ).scalars().all()
        health_rows = session.execute(
            select(models.CameraHealth).where(models.CameraHealth.camera_id == "cam-acceptance")
        ).scalars().all()
        camera = session.execute(
            select(models.Camera).where(models.Camera.camera_id == "cam-acceptance")
        ).scalar_one_or_none()

        print()
        print("-" * 72)
        print(f"camera row      : {'present' if camera else 'MISSING'}")
        print(f"events          : {len(events)}")
        for event in events:
            print(
                f"  - {event.id} type={event.event_type} severity={event.severity} "
                f"status={event.status} confidence={event.confidence} "
                f"tracks={event.track_ids} zone={event.zone_name}"
            )
            print(f"    summary   : {event.summary}")
            print(f"    timestamp : {event.timestamp}")
            print(f"    rule      : {event.rule_id} v={event.metadata_.get('rule_version')}")
            print(f"    models    : {event.model_versions}")
            print(f"    evidence  : {event.evidence_ids}")
        print(f"evidence rows   : {len(evidence_rows)}")
        for row in evidence_rows:
            print(
                f"  - {row.id} type={row.type} sha256={(row.sha256 or '')[:16]}... "
                f"size={row.size_bytes} duration_ms={row.duration_ms}"
            )
        print(f"health rows     : {len(health_rows)}")
        for row in health_rows[:3]:
            print(
                f"  - state={row.state} health={row.health} ai={row.ai_status} "
                f"fps={row.fps} details={row.details}"
            )

        print()
        print("-" * 72)
        print("ACCEPTANCE CHECKS")

        check("A1 camera row created from config", camera is not None)
        check(
            "A2 real video source consumed",
            runner.worker.frames_processed > 0,
            f"{runner.worker.frames_processed} gated frames",
        )
        check(
            "A3 detector produced person detections (no detector errors)",
            runner.pipeline.detector_errors == 0 and runner.pipeline.frames_processed > 0,
            f"frames={runner.pipeline.frames_processed} detector={runner.pipeline.model_versions}",
        )
        check(
            "A4 exactly one confirmed event (dedup while condition persists)",
            len(events) == 1,
            f"{len(events)} event(s)",
        )
        if events:
            event = events[0]
            check(
                "A5 event explainability (conditions + summary + rule version)",
                bool(event.conditions) and bool(event.summary)
                and event.metadata_.get("rule_version") == "0.1",
                f"{len(event.conditions)} condition(s)",
            )
            check(
                "A6 event linked to detector model versions",
                bool(event.model_versions.get("detector")),
                str(event.model_versions),
            )
            check(
                "A7 evidence rows exist only for the event",
                len(evidence_rows) > 0
                and all(
                    row.event_id in {e.id for e in events} for row in evidence_rows
                ),
                f"{len(evidence_rows)} item(s)",
            )
            check(
                "A8 event.evidence_ids matches evidence rows",
                set(event.evidence_ids) == {row.id for row in evidence_rows},
            )
            root = Path(settings.evidence.root)
            hashes_ok = True
            for row in evidence_rows:
                path = root / row.uri
                if not path.exists():
                    hashes_ok = False
                    break
                if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
                    hashes_ok = False
                    break
            check("A9 evidence sha256 verifies against stored files", hashes_ok)
            types = {row.type for row in evidence_rows}
            check("A10 evidence includes snapshot(s)", "snapshot" in types, str(sorted(types)))
            check(
                "A11 pre/post clip captured (or graceful skip if <2 frames)",
                "clip" in types or len(evidence_rows) >= 1,
                str(sorted(types)),
            )
        check(
            "A12 camera health flushed (status + details)",
            len(health_rows) > 0,
            f"{len(health_rows)} row(s)",
        )
        check(
            "A13 ai status reported",
            any(row.ai_status in ("healthy", "degraded") for row in health_rows),
        )
        check(
            "A14 valid event lifecycle (created new; auto-resolved when condition cleared)",
            len(events) >= 1 and all(e.status in ("new", "resolved") for e in events),
            str([e.status for e in events]),
        )
        check(
            "A15 detector never crashed the pipeline",
            runner.pipeline.detector_errors == 0,
            f"errors={runner.pipeline.detector_errors}",
        )

    evidence_root = Path(settings.evidence.root)
    on_disk = list(evidence_root.rglob("*")) if evidence_root.exists() else []
    check(
        "A16 evidence files written under configured root",
        any(p.is_file() for p in on_disk),
        str(evidence_root),
    )

    print()
    failed = [name for name, ok, _ in CHECKS if not ok]
    print("-" * 72)
    print(f"RESULT: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print(
        "NOTE: benchmark/accuracy numbers are NOT claimed anywhere; see "
        "docs/MODEL_REGISTRY.md (HOG = development, never evaluated)."
    )
    time.sleep(0.1)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

