"""CLI for running detector evaluations (F2-H).

Usage:
  python evaluation/runners/run_evaluation.py --manifest evaluation/manifests/example.template.json
  python evaluation/runners/run_evaluation.py --manifest path/to/manifest.json --detector hog --iou 0.5

The manifest references media that lives OUTSIDE the repository (see
evaluation/README.md); without it the run reports
"EVALUATION DATASET NOT AVAILABLE" instead of fabricated metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a detector evaluation")
    parser.add_argument("--manifest", required=True, help="path to manifest JSON")
    parser.add_argument("--detector", default="hog", help="detector profile from the registry")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold")
    parser.add_argument("--confidence", type=float, default=0.3, help="min confidence")
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument(
        "--output",
        default="evaluation/reports",
        help="report file or directory (default: evaluation/reports/)",
    )
    args = parser.parse_args()

    from evaluation.runner import UNAVAILABLE_STATUS, run_evaluation
    from services.inference.interfaces import default_registry

    detector = default_registry().create(args.detector)
    report = run_evaluation(
        args.manifest,
        detector,
        output_path=args.output,
        iou_threshold=args.iou,
        confidence_threshold=args.confidence,
        max_entries=args.max_entries,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] != UNAVAILABLE_STATUS else 3


if __name__ == "__main__":
    raise SystemExit(main())
