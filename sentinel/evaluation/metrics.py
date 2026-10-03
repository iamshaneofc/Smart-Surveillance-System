from __future__ import annotations

from collections import defaultdict

from evaluation.annotations import Box

Box4 = list[float]


def iou(a: Box4, b: Box4) -> float:
    """Intersection-over-union of two [x, y, w, h] boxes (any consistent units)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def frame_matches(
    ground_truth: list[Box],
    predictions: list[Box],
    iou_threshold: float = 0.5,
) -> list[tuple[Box, Box, float]]:
    """Greedy confidence-descending matching. Returns (gt, pred, iou) TP pairs."""
    remaining = list(ground_truth)
    pairs: list[tuple[Box, Box, float]] = []
    ranked = sorted(predictions, key=lambda p: p.confidence or 0.0, reverse=True)
    for pred in ranked:
        best: Box | None = None
        best_iou = 0.0
        for gt in remaining:
            if gt.class_name != pred.class_name:
                continue
            score = iou(gt.bbox, pred.bbox)
            if score >= iou_threshold and score > best_iou:
                best = gt
                best_iou = score
        if best is not None:
            pairs.append((best, pred, best_iou))
            remaining.remove(best)
    return pairs


class DetectionMetrics:
    """Accumulates TP/FP/FN and confidence-ranked outcomes for P/R/F1 + mAP."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._gt_counts: dict[str, int] = defaultdict(int)
        self._predictions: dict[str, list[tuple[float, bool]]] = defaultdict(list)

    @property
    def ground_truth_total(self) -> int:
        return sum(self._gt_counts.values())

    @property
    def prediction_total(self) -> int:
        return sum(len(v) for v in self._predictions.values())

    @property
    def tp_total(self) -> int:
        return sum(1 for preds in self._predictions.values() for _, tp in preds if tp)

    @property
    def fp_total(self) -> int:
        return self.prediction_total - self.tp_total

    @property
    def fn_total(self) -> int:
        return self.ground_truth_total - self.tp_total

    def add_frame(self, ground_truth: list[Box], predictions: list[Box]) -> None:
        remaining = list(ground_truth)
        for gt in ground_truth:
            self._gt_counts[gt.class_name] += 1

        ranked = sorted(predictions, key=lambda p: p.confidence or 0.0, reverse=True)
        for pred in ranked:
            best_gt: Box | None = None
            best_iou = 0.0
            for gt in remaining:
                if gt.class_name != pred.class_name:
                    continue
                score = iou(gt.bbox, pred.bbox)
                if score >= self.iou_threshold and score > best_iou:
                    best_gt = gt
                    best_iou = score
            confidence = pred.confidence if pred.confidence is not None else 1.0
            if best_gt is not None:
                self._predictions[pred.class_name].append((confidence, True))
                remaining.remove(best_gt)
            else:
                self._predictions[pred.class_name].append((confidence, False))

    def precision_recall(self) -> tuple[float, float, float]:
        tp, fp, fn = self.tp_total, self.fp_total, self.fn_total
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    def average_precision(self, class_name: str) -> float | None:
        preds = sorted(self._predictions.get(class_name, []), key=lambda x: x[0], reverse=True)
        gt = self._gt_counts.get(class_name, 0)
        if gt == 0:
            return None
        tp_cum = 0
        fp_cum = 0
        precisions: list[float] = []
        recalls: list[float] = []
        for _, is_tp in preds:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / gt)
        if not precisions:
            return 0.0
        ap = 0.0
        prev_recall = 0.0
        for precision, recall in zip(precisions, recalls):
            if recall > prev_recall:
                ap += precision * (recall - prev_recall)
                prev_recall = recall
        return ap

    def summary(self) -> dict:
        precision, recall, f1 = self.precision_recall()
        per_class: dict[str, dict] = {}
        aps: list[float] = []
        for class_name, gt_count in sorted(self._gt_counts.items()):
            ap = self.average_precision(class_name)
            per_class[class_name] = {
                "ground_truth": gt_count,
                "predictions": len(self._predictions.get(class_name, [])),
                "ap": ap,
            }
            if ap is not None:
                aps.append(ap)
        return {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "map": round(sum(aps) / len(aps), 6) if aps else None,
            "tp": self.tp_total,
            "fp": self.fp_total,
            "fn": self.fn_total,
            "ground_truth_total": self.ground_truth_total,
            "prediction_total": self.prediction_total,
            "per_class": per_class,
        }
