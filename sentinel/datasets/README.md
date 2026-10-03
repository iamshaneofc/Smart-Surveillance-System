# Datasets — evaluation data acquisition specification (F4-D)

**Current status: EVALUATION DATASET NOT AVAILABLE.**
The repository contains **no valid, licensed evaluation dataset** (inspected
2026-10-03). No evaluation metrics exist; all evaluation metrics are `N/A`.
This file is the data acquisition specification for changing that state
honestly.

## 1. Why nothing is here today

| Candidate source | Verdict | Evidence |
|---|---|---|
| Legacy `Violence_system/Violence_Dataset/` | **BLOCKED** | no license file, undocumented provenance (`docs/DATA_LICENSES.md` #2) |
| Legacy weapon/action datasets | **BLOCKED** | never in repo, unknown license (`docs/DATA_LICENSES.md` #4/#5) |
| Legacy `snapshots/` images | **BLOCKED for evaluation** | mixed third-party content, internal regression use only (#6) |
| Demo clip `VID-20250724-WA0004.mp4` | **NOT evaluation data** | no license; development/acceptance footage only (#7) |
| `suspicious_motion/sus/dataset/` | not a detection dataset | classification images; partial CC BY 4.0 (#3) |
| Arbitrary internet/Google images | **prohibited** | unknown per-image licenses |
| COCO val2017 / other public detection sets | not present; acquisition requires an explicit licensing decision | would be a *substitute dataset* decision, not made unilaterally |

Per the F4 rules: no blocked dataset is reused, no unrelated dataset is
downloaded quietly, and no metric is ever fabricated.

## 2. What a qualifying dataset must provide

1. **Known provenance** — provider, URL/citation, export date, version.
2. **Usable license** — written license permitting the intended use *and*
   model training/evaluation; recorded in `docs/DATA_LICENSES.md` with status
   `verified` before any use.
3. **Clear classes** — a declared class vocabulary that maps to detector
   classes via `class_mapping`.
4. **Annotations/ground truth** — per-frame boxes in the repository's
   normalized `[x, y, w, h]` format (see `evaluation/annotations/README.md`),
   one JSON per media entry.
5. **Reproducible split/manifest** — media-level splits (no frame-level
   leakage between `dev`/`val`/`test`), referenced by an
   `evaluation/manifests/<name>.json` manifest.
6. **Enough context** — source resolution, capture context (indoor/outdoor,
   consent/licence basis, conditions) recorded per entry.

## 3. How to acquire (checklist)

1. Shortlist a source and verify its license text (not a website claim).
2. Record it in `docs/DATA_LICENSES.md` with full provenance fields; do not
   proceed without a usable license.
3. Store media + annotations **outside the repository** (media is never
   committed; only paths go in the manifest).
4. Copy `evaluation/manifests/example.template.json`, fill every `REPLACE`
   field (`dataset_id`, `license`, `source`, `class_mapping`, per-entry
   `capture_context`).
5. Validate deterministically:
   `python evaluation/validate_manifest.py evaluation/manifests/<name>.json`
   (fix every error; warnings must be resolved or explicitly accepted).
6. Run the evaluation:
   `python evaluation/runners/run_evaluation.py --manifest evaluation/manifests/<name>.json --detector yolox`
7. Record results + threshold analysis in `docs/DETECTOR_SELECTION.md`
   section 4 order of operations, then (and only then) consider a status
   change in `docs/MODEL_REGISTRY.md`.

## 4. Registration rules (unchanged from F2)

Only **licensed, documented** datasets belong here. Register every dataset in
`docs/DATA_LICENSES.md` first (source, license, commercial use,
redistribution) and include `LICENSE` + `ATTRIBUTION` files inside each
dataset folder. Legacy datasets with unknown/absent licenses are **blocked**
and must not be used for commercial training or evaluation.
