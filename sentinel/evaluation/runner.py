from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from packages.common.logging import get_logger

log = get_logger(__name__)

UNAVAILABLE_STATUS = "EVALUATION DATASET NOT AVAILABLE"


class LatencyStats(BaseModel):
    count: int = 0
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    mean: float | None = None


class EvaluationReport(BaseModel):
    status: str
    generated_at: datetime
    model: dict = Field(default_factory=dict)
    dataset: dict = Field(default_factory=dict)
    environment: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    metrics: dict | None = None
    latency_ms: LatencyStats | None = None
    error_examples: list[dict] = Field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return round(ordered[index], 3)


def _latency_stats(samples: list[float]) -> LatencyStats | None:
    if not samples:
        return None
    return LatencyStats(
        count=len(samples),
        p50=_percentile(samples, 50),
        p95=_percentile(samples, 95),
        p99=_percentile(samples, 99),
        mean=round(sum(samples) / len(samples), 3),
    )


def _environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": "cpu",
    }
    try:
        import cv2

        env["opencv"] = cv2.__version__
    except Exception:
        env["opencv"] = None
    return env


def _model_info(detector) -> dict:
    info = getattr(detector, "info", None)
    if info is None:
        return {"name": type(detector).__name__}
    return {
        "name": info.name,
        "version": getattr(info, "version", ""),
        "family": getattr(info, "family", ""),
        "license": getattr(info, "license", ""),
        "classes": list(getattr(info, "classes", ()) or ()),
    }


def _unavailable_report(manifest_path: Path | None, detector, config: dict, reason: str) -> EvaluationReport:
    return EvaluationReport(
        status=UNAVAILABLE_STATUS,
        generated_at=datetime.now(timezone.utc),
        model=_model_info(detector),
        dataset={
            "manifest": str(manifest_path) if manifest_path else None,
            "entries_total": 0,
            "entries_evaluated": 0,
            "entries_skipped_missing_media": 0,
            "entries_skipped_missing_annotation": 0,
            "license": "",
        },
        environment=_environment(),
        config=config,
        metrics=None,
        latency_ms=None,
        notes=f"{UNAVAILABLE_STATUS}: {reason}",
    )


def _iter_video_frames(path: Path, wanted: set[int]):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path.name}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                ok_enc, buf = cv2.imencode(".jpg", frame)
                yield index, buf.tobytes() if ok_enc else None
            index += 1
            if wanted and index > max(wanted):
                break
    finally:
        cap.release()


def run_evaluation(
    manifest_path: str | Path,
    detector,
    output_path: str | Path | None = None,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    max_entries: int | None = None,
) -> dict:
    """Run a detector over a labeled manifest and write a JSON report.

    Never fabricates metrics: when the manifest or its media is missing the
    report status is "EVALUATION DATASET NOT AVAILABLE" and metrics are null.
    """
    from evaluation.annotations import Box, load_annotation
    from evaluation.manifest import (
        load_manifest,
        resolve_annotation_path,
        resolve_media_path,
    )
    from evaluation.metrics import DetectionMetrics
    from services.camera.types import FramePacket

    config = {
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "max_entries": max_entries,
    }
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        report = _unavailable_report(
            None, detector, config, f"manifest not found: {manifest_path.name}"
        )
        _write_report(report, output_path, "unavailable")
        log.warning("evaluation skipped", extra={"reason": "manifest missing"})
        return report.model_dump(mode="json")

    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        report = _unavailable_report(
            manifest_path, detector, config, f"manifest unreadable: {exc}"
        )
        _write_report(report, output_path, "unavailable")
        return report.model_dump(mode="json")

    if not manifest.entries:
        report = _unavailable_report(manifest_path, detector, config, "manifest has no entries")
        _write_report(report, output_path, "unavailable")
        log.warning("evaluation skipped", extra={"reason": "empty manifest"})
        return report.model_dump(mode="json")

    warmup = getattr(detector, "warmup", None)
    if callable(warmup):
        try:
            warmup()
        except Exception:
            log.exception("detector warmup failed")

    metrics = DetectionMetrics(iou_threshold=iou_threshold)
    latencies: list[float] = []
    error_examples: list[dict] = []
    evaluated = 0
    skipped_missing_media = 0
    skipped_missing_annotation = 0

    entries = manifest.entries[: max_entries or len(manifest.entries)]
    for entry in entries:
        media_path = resolve_media_path(entry, manifest_path)
        if not media_path.exists():
            skipped_missing_media += 1
            continue
        annotation_path = resolve_annotation_path(entry, manifest_path)
        if annotation_path is None or not Path(annotation_path).exists():
            skipped_missing_annotation += 1
            continue
        annotation = load_annotation(annotation_path)
        wanted = {f.frame_index for f in annotation.frames}
        frames_by_index = {f.frame_index: f for f in annotation.frames}

        try:
            if entry.media_type == "image":
                data = media_path.read_bytes()
                frames = [(index, data) for index in sorted(wanted)]
            else:
                frames = list(_iter_video_frames(media_path, wanted))
        except Exception:
            log.exception(
                "evaluation media unreadable",
                extra={"entry_id": entry.entry_id, "media": media_path.name},
            )
            skipped_missing_media += 1
            continue

        for index, frame_data in frames:
            if frame_data is None:
                continue
            packet = FramePacket(
                camera_id=entry.camera_id,
                frame_id=index,
                ts=datetime.now(timezone.utc),
                data=frame_data,
            )
            import time as _time

            start = _time.perf_counter()
            try:
                detections = detector.detect(packet)
            except Exception:
                log.exception(
                    "detector failed during evaluation",
                    extra={"entry_id": entry.entry_id, "frame_index": index},
                )
                detections = []
            latencies.append((_time.perf_counter() - start) * 1000.0)

            gt_boxes = [
                Box(class_name=b.class_name, bbox=list(b.bbox))
                for b in frames_by_index[index].boxes
            ]
            predictions = [
                Box(
                    class_name=d.class_name,
                    bbox=list(
                        (
                            d.bbox.x,
                            d.bbox.y,
                            d.bbox.w,
                            d.bbox.h,
                        )
                    ),
                    confidence=d.confidence,
                )
                for d in detections
                if d.confidence >= confidence_threshold
            ]
            _collect_errors(
                error_examples,
                entry.entry_id,
                index,
                gt_boxes,
                predictions,
                model_version=_model_version(detector),
            )
            metrics.add_frame(gt_boxes, predictions)

        evaluated += 1

    if evaluated == 0:
        report = _unavailable_report(
            manifest_path,
            detector,
            config,
            "no usable labeled media ("
            f"{skipped_missing_media} missing media, "
            f"{skipped_missing_annotation} missing annotations)",
        )
        report.dataset.update(
            {
                "name": manifest.name,
                "version": manifest.version,
                "license": manifest.license,
                "source": manifest.source,
                "entries_total": len(entries),
                "entries_skipped_missing_media": skipped_missing_media,
                "entries_skipped_missing_annotation": skipped_missing_annotation,
            }
        )
        _write_report(report, output_path, "unavailable")
        log.warning(
            "evaluation dataset not available",
            extra={
                "manifest": manifest.name,
                "missing_media": skipped_missing_media,
                "missing_annotations": skipped_missing_annotation,
            },
        )
        return report.model_dump(mode="json")

    summary = metrics.summary()
    report = EvaluationReport(
        status="completed",
        generated_at=datetime.now(timezone.utc),
        model=_model_info(detector),
        dataset={
            "name": manifest.name,
            "version": manifest.version,
            "license": manifest.license,
            "source": manifest.source,
            "entries_total": len(entries),
            "entries_evaluated": evaluated,
            "entries_skipped_missing_media": skipped_missing_media,
            "entries_skipped_missing_annotation": skipped_missing_annotation,
        },
        environment=_environment(),
        config=config,
        metrics={"object_detection": summary, "event_detection": None},
        latency_ms=_latency_stats(latencies),
        error_examples=error_examples[:10],
        notes="object-level metrics on annotated frames; event-level metrics require labeled events (not available)",
    )
    _write_report(report, output_path, manifest.name)
    log.info(
        "evaluation completed",
        extra={
            "manifest": manifest.name,
            "entries_evaluated": evaluated,
            "precision": summary["precision"],
            "recall": summary["recall"],
        },
    )
    return report.model_dump(mode="json")


def _model_version(detector) -> str | None:
    info = getattr(detector, "info", None)
    if info is None:
        return None
    return str(getattr(info, "version", None) or "") or None


def _collect_errors(
    examples: list[dict],
    entry_id: str,
    frame_index: int,
    gt_boxes,
    predictions,
    model_version: str | None = None,
) -> None:
    """F4-L error-analysis records: capped at 10, metadata only (no imagery)."""
    from evaluation.metrics import frame_matches, iou

    if len(examples) >= 10:
        return
    matched = frame_matches(list(gt_boxes), list(predictions))
    matched_gt = {id(gt) for gt, _, _ in matched}
    matched_pred = {id(pred) for _, pred, _ in matched}

    def _best_iou(box, pool) -> float:
        candidates = [iou(box.bbox, other.bbox) for other in pool if other.class_name == box.class_name]
        return round(max(candidates), 4) if candidates else 0.0

    for gt in gt_boxes:
        if id(gt) not in matched_gt and len(examples) < 10:
            examples.append(
                {
                    "entry_id": entry_id,
                    "frame_index": frame_index,
                    "kind": "false_negative",
                    "expected_class": gt.class_name,
                    "predicted_class": None,
                    "class_name": gt.class_name,
                    "bbox": [round(v, 4) for v in gt.bbox],
                    "confidence": None,
                    "iou": _best_iou(gt, predictions),
                    "model_version": model_version,
                }
            )
    for pred in predictions:
        if id(pred) not in matched_pred and len(examples) < 10:
            examples.append(
                {
                    "entry_id": entry_id,
                    "frame_index": frame_index,
                    "kind": "false_positive",
                    "expected_class": None,
                    "predicted_class": pred.class_name,
                    "class_name": pred.class_name,
                    "bbox": [round(v, 4) for v in pred.bbox],
                    "confidence": pred.confidence,
                    "iou": _best_iou(pred, gt_boxes),
                    "model_version": model_version,
                }
            )


def _write_report(report: EvaluationReport, output_path: str | Path | None, name: str) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    if path.is_dir() or str(path).endswith(("/", "\\")):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = path / f"{name}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
