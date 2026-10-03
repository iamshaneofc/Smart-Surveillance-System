"""CLI for threshold analysis (F4-K).

Usage:
  python evaluation/runners/run_threshold_sweep.py --manifest evaluation/manifests/<name>.json --detector yolox

Without a licensed evaluation dataset the output status is
"EVALUATION DATASET NOT AVAILABLE" and the operational threshold is
"NOT EVALUATED" - never a fabricated choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Confidence-threshold sweep over ground truth")
    parser.add_argument("--manifest", required=True, help="path to manifest JSON")
    parser.add_argument("--detector", default="hog", help="detector profile from the registry")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold")
    parser.add_argument(
        "--thresholds",
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="comma-separated confidence thresholds",
    )
    parser.add_argument("--output", default=None, help="optional JSON report path")
    args = parser.parse_args()

    from evaluation.threshold_sweep import sweep_thresholds
    from services.inference.interfaces import default_registry

    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    detector = default_registry().create(args.detector)
    try:
        result = sweep_thresholds(
            args.manifest,
            detector,
            thresholds=thresholds,
            iou_threshold=args.iou,
            output_path=args.output,
        )
    finally:
        detector.close()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "EVALUATION DATASET NOT AVAILABLE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
