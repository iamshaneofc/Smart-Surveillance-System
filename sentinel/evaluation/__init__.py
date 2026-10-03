"""Evaluation framework for detector benchmarks (F2-H).

Runs a Detector against a manifest of labeled media, computes object-detection
metrics and writes a machine-readable JSON report. Missing data yields an
explicit "EVALUATION DATASET NOT AVAILABLE" report - never fabricated metrics.
"""

from evaluation.metrics import DetectionMetrics, frame_matches, iou
from evaluation.runner import EvaluationReport, run_evaluation

__all__ = [
    "DetectionMetrics",
    "EvaluationReport",
    "frame_matches",
    "iou",
    "run_evaluation",
]
