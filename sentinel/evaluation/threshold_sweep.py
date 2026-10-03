"""F4-K threshold analysis infrastructure.

Evaluates multiple confidence thresholds against labeled ground truth and
produces precision/recall/F1 tables. This is INFRASTRUCTURE ONLY: with no
licensed evaluation dataset in the repository (F4-D gate) the sweep reports
status "EVALUATION DATASET NOT AVAILABLE" and the operational threshold
remains "NOT EVALUATED". Selecting a production threshold requires real
ground truth plus documented reasoning - never a fixture run.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evaluation.runner import UNAVAILABLE_STATUS, run_evaluation

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
NOT_EVALUATED = "NOT EVALUATED"


def sweep_thresholds(
    manifest_path,
    detector,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    iou_threshold: float = 0.5,
    output_path=None,
) -> dict:
    """Run the evaluation once per threshold; returns a table + honesty markers."""
    detector_floor = getattr(detector, "confidence_threshold", getattr(detector, "min_confidence", None))

    rows: list[dict] = []
    for threshold in thresholds:
        report = run_evaluation(
            manifest_path,
            detector,
            output_path=None,
            iou_threshold=iou_threshold,
            confidence_threshold=threshold,
        )
        if report["status"] != UNAVAILABLE_STATUS:
            metrics = report["metrics"]["object_detection"]
            rows.append(
                {
                    "threshold": threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
            )
            continue
        result = {
            "status": UNAVAILABLE_STATUS,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reason": report["notes"],
            "iou_threshold": iou_threshold,
            "thresholds": list(thresholds),
            "rows": [],
            "best_f1_observation": None,
            "detector_confidence_floor": detector_floor,
            "operational_threshold": NOT_EVALUATED,
            "notes": "threshold selection remains NOT EVALUATED until a licensed evaluation dataset exists (docs/datasets/README.md)",
        }
        return _write(result, output_path)

    best = max(rows, key=lambda r: r["f1"])
    result = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reason": "",
        "iou_threshold": iou_threshold,
        "thresholds": list(thresholds),
        "rows": rows,
        "best_f1_observation": best,
        "detector_confidence_floor": detector_floor,
        # Deliberately NOT chosen here: operational selection requires a
        # licensed dataset + documented reasoning in docs/DETECTOR_SELECTION.md.
        "operational_threshold": NOT_EVALUATED,
        "notes": "fixture/computed rows are mechanics output only; operational threshold NOT EVALUATED",
    }
    if detector_floor is not None and detector_floor > min(thresholds):
        result["warning"] = (
            f"detector confidence floor {detector_floor} caps thresholds below it; "
            "recreate the detector with a lower floor for a valid sweep"
        )
    return _write(result, output_path)


def _write(result: dict, output_path) -> dict:
    if output_path is not None:
        import json
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
