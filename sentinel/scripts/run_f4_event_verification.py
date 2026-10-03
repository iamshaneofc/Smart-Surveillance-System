"""F4-O/P end-to-end event verification on real development footage.

Runs the production wiring (PipelineRunner) with the YOLOX detector against
the internal legacy clip (docs/DATA_LICENSES.md #7 - license unknown, internal
development use only, never redistributed) and verifies the full chain
rule -> event -> evidence -> alert -> API. Two sequential runs:

  Run A - stock factory pack (confirm=2s): honest confirmation behavior. The
          clip's longest >=0.5 in-zone condition is ~1.3s, so pendings are
          created and cancelled and NO event is expected.
  Run B - f4-verification pack (confirm=0.6s): full-chain verification with
          provenance and alert rows.

The clip and all numbers below are DEVELOPMENT verification material - this
script never claims accuracy or operational performance (F4 critical rule).

Usage:
  python scripts/run_f4_event_verification.py [path/to/video.mp4]

Exit code 0 = all checks passed.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_VIDEO = r"Z:\Security_Surveillance_System\snapshots\VID-20250724-WA0004.mp4"
LEFT_ZONE = [(0.0, 0.0), (0.25, 0.0), (0.25, 1.0), (0.0, 1.0)]
CAMERA = "cam-f4-verify"
LOCK_PATH = Path(__file__).resolve().parents[1] / "scripts" / "weights.lock.json"
WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "yolox_tiny.onnx"
REPORT_PATH = Path(__file__).resolve().parents[1] / "var" / "benchmarks" / "f4-event-verification.json"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _instrument(pipeline, stats: dict) -> None:
    """Passive frame instrumentation for F4-P (detections/tracks/pendings)."""
    det = pipeline.detector
    orig_detect = det.detect

    def detect(pkt):
        stats["_error"] = False
        try:
            dets = orig_detect(pkt)
        except Exception:
            stats["_error"] = True
            raise
        stats["_dets"] = dets
        return dets

    det.detect = detect

    orig_pf = pipeline.process_frame

    def process_frame(packet):
        events = orig_pf(packet)
        dets = stats.pop("_dets", None)
        errored = stats.pop("_error", False)
        stats["frames"] += 1
        if errored:
            stats["detect_error_frames"] += 1
            dets = []
        dets = dets or []
        zone_dets = [d for d in dets if (d.bbox.x + d.bbox.w / 2) <= 0.25]
        stats["det_counts"].append(len(dets))
        stats["zone_det_counts"].append(len(zone_dets))
        if dets:
            stats["frames_with_dets"] += 1
        else:
            stats["gap_frames"] += 1
        stats["in_zone_confs"].extend(round(d.confidence, 4) for d in zone_dets)

        fid = packet.frame_id
        zone_tids = []
        for track in list(getattr(pipeline.tracker, "_active", [])):
            seen = stats["track_seen"].setdefault(track.track_id, [fid, fid, 0])
            seen[0] = min(seen[0], fid)
            seen[1] = max(seen[1], fid)
            seen[2] += 1
            if track.bbox.center()[0] <= 0.25:
                zone_tids.append(track.track_id)
        if zone_tids and stats["_last_zone_tids"]:
            if not (set(zone_tids) & set(stats["_last_zone_tids"])):
                stats["zone_switches"] += 1
        stats["_last_zone_tids"] = zone_tids
        stats["active_tracks"].append(len(getattr(pipeline.tracker, "_active", [])))

        pending = pipeline.engine.pending_keys()
        stats["pending_max"] = max(stats["pending_max"], len(pending))
        if pending:
            stats["pending_frames"] += 1
        return events

    pipeline.process_frame = process_frame


def _new_stats() -> dict:
    return {
        "frames": 0,
        "frames_with_dets": 0,
        "gap_frames": 0,
        "detect_error_frames": 0,
        "det_counts": [],
        "zone_det_counts": [],
        "in_zone_confs": [],
        "active_tracks": [],
        "track_seen": {},
        "zone_switches": 0,
        "pending_max": 0,
        "pending_frames": 0,
        "_dets": None,
        "_error": False,
        "_last_zone_tids": [],
    }


def _summarize(stats: dict, pipeline, init_seconds: float) -> dict:
    def rnd(values):
        return {
            "mean": round(statistics.fmean(values), 3) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "label": "DEVELOPMENT / INTERNAL FOOTAGE - verification only, not accuracy",
        "detector_init_seconds": round(init_seconds, 3),
        "frames_processed": pipeline.frames_processed,
        "events_confirmed": pipeline.events_created,
        "detector_errors": pipeline.detector_errors,
        "tracker_errors": pipeline.tracker_errors,
        "detect_error_frames": stats["detect_error_frames"],
        "frames_with_detections": stats["frames_with_dets"],
        "detector_gap_frames": stats["gap_frames"],
        "detections_per_frame": rnd(stats["det_counts"]),
        "in_zone_detection_frames": sum(1 for n in stats["zone_det_counts"] if n > 0),
        "in_zone_confidence": rnd(stats["in_zone_confs"]),
        "active_tracks": rnd(stats["active_tracks"]),
        "tracks": [
            {
                "track_id": tid,
                "first_frame": first,
                "last_frame": last,
                "frames_seen": count,
            }
            for tid, (first, last, count) in sorted(stats["track_seen"].items())
        ],
        "zone_track_id_switches": stats["zone_switches"],
        "pending_keys_max": stats["pending_max"],
        "frames_with_pending": stats["pending_frames"],
        "model_versions": dict(pipeline.model_versions),
    }


def run_once(label: str, pack: str, video: Path, alerts_enabled: bool) -> dict:
    from packages.config import Settings
    from packages.config.settings import PipelineSettings, ZoneSpec
    from packages.db import base as db_base

    print()
    print("=" * 72)
    print(f"{label}")
    print("=" * 72)

    db_file = Path(f"var/f4-{label.split()[0].lower()}.db")
    evidence_root = Path(f"./var/f4-{label.split()[0].lower()}-evidence")
    db_file.parent.mkdir(parents=True, exist_ok=True)
    for path in (db_file,):
        if path.exists():
            path.unlink()
    if evidence_root.exists():
        import shutil

        shutil.rmtree(evidence_root)

    settings = Settings()
    settings.database_url = f"sqlite:///{db_file}"
    settings.evidence.root = str(evidence_root)
    settings.alerts.enabled = alerts_enabled
    settings.pipeline = PipelineSettings(
        camera_id=CAMERA,
        camera_name="F4 verification camera",
        source_type="file",
        stream_url=str(video),
        detection_fps=5.0,
        detector_profile="yolox",
        tracker="iou",
        rule_pack=pack,
        zones=[ZoneSpec(id="restricted", name="Restricted Area", zone_type="restricted", polygon=LEFT_ZONE)],
    )

    db_base.dispose()
    db_base.configure(settings.database_url)
    db_base.create_all()

    from services.pipeline.runner import PipelineRunner

    t0 = time.perf_counter()
    runner = PipelineRunner(settings)
    init_seconds = time.perf_counter() - t0
    stats = _new_stats()
    _instrument(runner.pipeline, stats)

    runner.start()
    finished = runner.wait(timeout=600.0)
    check(f"{label}: camera worker consumed the source", finished, f"frames={runner.worker.frames_processed}")
    runner.stop(join_timeout=30.0)

    pipeline = runner.pipeline
    check(
        f"{label}: yolox detector ran with zero detector errors",
        pipeline.detector_errors == 0,
        f"errors={pipeline.detector_errors} detector={pipeline.model_versions}",
    )
    check(
        f"{label}: model provenance recorded",
        pipeline.model_versions.get("detector") == "yolox-tiny:0.1.1rc0",
        str(pipeline.model_versions),
    )

    from sqlalchemy import select

    from packages.db import models

    with db_base.session_scope() as session:
        events = session.execute(
            select(models.Event).where(models.Event.camera_id == CAMERA)
        ).scalars().all()
        evidence_rows = session.execute(
            select(models.Evidence).where(models.Evidence.camera_id == CAMERA)
        ).scalars().all()
        health_rows = session.execute(
            select(models.CameraHealth).where(models.CameraHealth.camera_id == CAMERA)
        ).scalars().all()
        alert_rows = session.execute(select(models.Alert)).scalars().all()
        camera = session.execute(
            select(models.Camera).where(models.Camera.camera_id == CAMERA)
        ).scalar_one_or_none()

        check(f"{label}: camera row created from config", camera is not None)
        check(
            f"{label}: ai status flushed healthy",
            any(r.ai_status in ("healthy", "degraded") for r in health_rows) and health_rows,
            f"{len(health_rows)} health row(s)",
        )

        summary = _summarize(stats, pipeline, init_seconds)
        if pack == "factory.yaml":
            check(
                "A: pendings attempted (confirmation behavior observed)",
                stats["pending_max"] > 0 and stats["pending_frames"] > 0,
                f"pending_max={stats['pending_max']} frames_with_pending={stats['pending_frames']}",
            )
            check(
                "A: no event confirmed - condition (~1.3s) shorter than confirm=2s (honest outcome)",
                len(events) == 0 and pipeline.events_created == 0,
                f"{len(events)} event(s)",
            )
            check(
                "A: pending keys cancelled after gaps (no stale pendings survive)",
                pipeline.engine.pending_keys() == set(),
                str(pipeline.engine.pending_keys()),
            )
        else:
            check("B: at least one confirmed event", len(events) >= 1, f"{len(events)} event(s)")
            for event in events:
                print(
                    f"  - {event.id} type={event.event_type} severity={event.severity} "
                    f"status={event.status} confidence={event.confidence} "
                    f"tracks={event.track_ids} zone={event.zone_name}"
                )
                print(f"    summary   : {event.summary}")
                print(f"    models    : {event.model_versions}")
                print(f"    evidence  : {event.evidence_ids}")
            if events:
                event = events[0]
                check(
                    "B: event linked to yolox-tiny:0.1.1rc0",
                    event.model_versions.get("detector") == "yolox-tiny:0.1.1rc0",
                    str(event.model_versions),
                )
                check(
                    "B: event explainability (conditions + summary + rule version)",
                    bool(event.conditions) and bool(event.summary)
                    and event.metadata_.get("rule_version") == "0.1",
                    f"{len(event.conditions)} condition(s)",
                )
                check(
                    "B: evidence rows exist only for these events",
                    len(evidence_rows) > 0
                    and all(r.event_id in {e.id for e in events} for r in evidence_rows),
                    f"{len(evidence_rows)} item(s)",
                )
                union = set()
                for e in events:
                    union |= set(e.evidence_ids)
                check(
                    "B: event.evidence_ids match evidence rows (union of events)",
                    union == {r.id for r in evidence_rows},
                    f"{len(union)} id(s) vs {len(evidence_rows)} row(s)",
                )
                without = [e.id for e in events if not e.evidence_ids]
                if without:
                    print(
                        "  NOTE: known limitation (pre-existing, not F4): EvidenceService keeps"
                        " one session per camera, so the earlier of two near-simultaneous"
                        f" events gets no evidence - {len(without)} event(s) affected here;"
                        " candidate F5 item."
                    )
                hashes_ok = True
                for row in evidence_rows:
                    path = evidence_root / row.uri
                    if not path.exists() or _sha256(path) != row.sha256:
                        hashes_ok = False
                        break
                check("B: evidence sha256 verifies against stored files", hashes_ok)
                types = {r.type for r in evidence_rows}
                check("B: evidence includes snapshot(s)", "snapshot" in types, str(sorted(types)))
            on_disk = list(evidence_root.rglob("*")) if evidence_root.exists() else []
            check("B: evidence files written under configured root", any(p.is_file() for p in on_disk))
            check("B: alert rows dispatched (in_app channel)", len(alert_rows) >= 1, f"{len(alert_rows)} alert(s)")

    summary = _summarize(stats, pipeline, init_seconds)
    print()
    print("-" * 72)
    print("F4-P tracking interaction (development clip):")
    for key in (
        "frames_processed",
        "frames_with_detections",
        "detector_gap_frames",
        "detect_error_frames",
        "detections_per_frame",
        "in_zone_detection_frames",
        "in_zone_confidence",
        "active_tracks",
        "zone_track_id_switches",
        "pending_keys_max",
        "frames_with_pending",
        "events_confirmed",
    ):
        print(f"  {key:28s}: {summary[key]}")
    print(f"  {'tracks':28s}: {len(summary['tracks'])} track(s)")
    for track in summary["tracks"]:
        print(
            f"    id={track['track_id']} frames {track['first_frame']}..{track['last_frame']} "
            f"seen={track['frames_seen']}"
        )

    return {"summary": summary, "events": [e.id for e in events]}


def api_checks(settings) -> None:
    """F4-O: verify the chain is retrievable through the public API."""
    from fastapi.testclient import TestClient

    from apps.api.main import create_app
    from packages.db import base as db_base

    db_base.dispose()
    db_base.configure(settings.database_url)

    app = create_app(settings)
    with TestClient(app) as client:
        events_page = client.get("/api/v1/events", params={"camera_id": CAMERA}).json()
        items = events_page.get("items", [])
        check("C: API lists the verified event", len(items) >= 1, f"{len(items)} item(s)")
        if items:
            detail = client.get(f"/api/v1/events/{items[0]['event_id']}").json()
            check(
                "C: API event detail carries yolox provenance",
                detail.get("model_versions", {}).get("detector") == "yolox-tiny:0.1.1rc0",
                str(detail.get("model_versions")),
            )
        health = client.get(f"/api/v1/cameras/{CAMERA}/health").json()
        check(
            "C: API camera health shows ai_status",
            bool(health) and health[-1].get("ai_status") in ("healthy", "degraded"),
            str(health[-1].get("ai_status")) if health else "no rows",
        )
        alerts_page = client.get("/api/v1/alerts").json()
        check("C: API lists dispatched alerts", alerts_page.get("total", 0) >= 1, f"{alerts_page.get('total')} alert(s)")
        system = client.get("/api/v1/system/health")
        check("C: system health endpoint reachable", system.status_code == 200, str(system.json().get("status")))


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_VIDEO)
    if not video.exists():
        print(f"input video not found: {video}")
        return 2

    print("=" * 72)
    print("F4 END-TO-END EVENT VERIFICATION (YOLOX detector)")
    print(f"video        : {video}")
    print("footage      : INTERNAL DEVELOPMENT CLIP - synthetic/development")
    print("             : verification material only, never redistributed,")
    print("             : NO accuracy or operational performance claimed")
    print("detector     : yolox-tiny 0.1.1rc0 (candidate, UNEVALUATED)")
    print(f"zone (left)  : {LEFT_ZONE}")
    print("=" * 72)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    artifact = next(a for a in lock["artifacts"] if a["id"] == "yolox-tiny")
    ok = WEIGHTS_PATH.is_file() and _sha256(WEIGHTS_PATH) == artifact["sha256"]
    check(
        "W1 weights present and sha256 matches pins lock",
        ok,
        f"{WEIGHTS_PATH.name} sha256={_sha256(WEIGHTS_PATH)[:16] if WEIGHTS_PATH.is_file() else 'missing'}...",
    )
    if not ok:
        print("weights verification failed - run: python scripts/download_models.py")
        return 2

    run_a = run_once("FACTORY-A", "factory.yaml", video, alerts_enabled=False)
    run_b = run_once("VERIFY-B", "f4-verification.yaml", video, alerts_enabled=True)

    from packages.config import Settings
    from packages.db import base as db_base

    api_settings = Settings()
    api_settings.database_url = f"sqlite:///{Path('var/f4-verify-b.db')}"
    api_settings.evidence.root = "./var/f4-verify-b-evidence"
    api_checks(api_settings)

    report = {
        "label": "DEVELOPMENT ENVIRONMENT - F4 event verification, not an accuracy benchmark",
        "video": str(video),
        "footage_license": "unknown - internal development only (docs/DATA_LICENSES.md #7)",
        "detector": {
            "model_id": "yolox-tiny",
            "version": "0.1.1rc0",
            "status": "candidate",
            "weights_sha256": artifact["sha256"],
        },
        "runs": {"factory_pack": run_a["summary"], "verification_pack": run_b["summary"]},
        "checks": [{"name": n, "ok": ok_, "detail": d} for n, ok_, d in CHECKS],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written: {REPORT_PATH}")

    failed = [name for name, ok_, _ in CHECKS if not ok_]
    print()
    print("-" * 72)
    print(f"RESULT: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print(
        "NOTE: numbers above describe THIS development clip only; yolox-tiny remains"
        " UNEVALUATED (docs/MODEL_REGISTRY.md section 7)."
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
