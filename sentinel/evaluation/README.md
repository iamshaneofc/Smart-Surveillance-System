# SENTINEL evaluation framework

Runs a registered detector over a manifest of labeled media, computes
object-detection metrics (precision / recall / F1 / mAP, latency percentiles)
and writes a machine-readable JSON report to `reports/`.

## What this is not

- No dataset is committed to this repository. Media, annotations and manifests
  with real data live **outside** the repo (see `docs/DATA_LICENSES.md`).
- No metrics are ever fabricated. When the manifest, media or annotations are
  missing, the report status is `EVALUATION DATASET NOT AVAILABLE`, `metrics`
  is `null` and the notes say exactly why.
- Event-level metrics require labeled event datasets; until one exists the
  report carries `"event_detection": null`.

## Layout

```
evaluation/
  manifests/     manifest JSON files (example.template.json is a template)
  annotations/   per-entry frame annotations (see annotations/README.md)
  reports/       generated JSON reports (gitignored)
  runners/       CLI entry points
  manifest.py    manifest schema + loaders (dataset_id, class_mapping, context)
  annotations.py annotation schema + loaders
  metrics.py     IoU matching, precision/recall/F1, mAP
  runner.py      run_evaluation() - the whole flow
  validate_manifest.py  deterministic manifest validation (F4-E)
```

## Running an evaluation

0. **Validate the manifest first** (F4-E deterministic gate):

```bash
python evaluation/validate_manifest.py evaluation/manifests/<name>.json \
    --json evaluation/reports/<name>-validation.json
```

Exit `0` = no errors (warnings allowed), `1` = errors. Checks: manifest
readable, missing metadata, duplicate entry ids, unsupported formats, missing
media/annotation files, malformed annotations, unknown classes (against
`class_mapping`), invalid bounding boxes (normalized `[0,1]`, positive size,
finite). Every issue is reported with its entry/frame location — nothing is
silently discarded. See `../datasets/README.md` for the data acquisition
specification (F4-D): **no licensed evaluation dataset currently exists in
this repository, so evaluation metrics remain `N/A`.**

1. Store labeled media anywhere outside the repository, e.g.
   `D:\datasets\sentinel-eval\images\*.jpg` with matching annotation files.
2. Copy `manifests/example.template.json` to `manifests/<name>.json` and
   point `media` / `annotation` at those paths (absolute paths are fine).
3. Run:

```bash
python evaluation/runners/run_evaluation.py \
    --manifest evaluation/manifests/<name>.json \
    --detector hog --iou 0.5 --confidence 0.3 \
    --output evaluation/reports/
```

Exit codes: `0` report written (completed), `3` dataset not available.

The same flow is available in code:

```python
from evaluation import run_evaluation
from services.inference.interfaces import default_registry

detector = default_registry().create("hog")
report = run_evaluation("evaluation/manifests/<name>.json", detector,
                        output_path="evaluation/reports/")
```

## Report structure

```json
{
  "status": "completed | EVALUATION DATASET NOT AVAILABLE",
  "generated_at": "...",
  "model": {"name": "...", "version": "...", "license": "..."},
  "dataset": {"name": "...", "license": "...", "entries_evaluated": 0},
  "environment": {"python": "...", "opencv": "...", "hardware": "cpu"},
  "config": {"iou_threshold": 0.5, "confidence_threshold": 0.3},
  "metrics": {"object_detection": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "map": 0.0},
              "event_detection": null},
  "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0},
  "error_examples": [{"kind": "false_positive | false_negative", "...": "..."}],
  "notes": "..."
}
```

Metrics are object-level on annotated frames (IoU >= threshold matches).
Latency is measured around `detector.detect()` on the evaluation machine and
is only comparable between runs on the same hardware.
