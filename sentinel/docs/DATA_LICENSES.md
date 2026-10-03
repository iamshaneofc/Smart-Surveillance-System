# SENTINEL — Data License Registry

**Status: FOUNDATION PHASE. No dataset here is cleared for a commercial release unless explicitly marked.**

Rule: **never ship a model trained on an undocumented dataset.** Training data provenance blocks commercial use exactly like a code license does.

## 1. Datasets present in the legacy repository

| # | Dataset | Location (legacy) | Size / classes | Source | License evidence | Commercial use | Disposition |
|---|---|---|---|---|---|---|---|
| 1 | Face-mask images | `Mask-Face-Detector/dataset/` | 1915 `with_mask` + 1918 `without_mask` flat JPEGs, no train/val split | "Kaggle" (legacy `README.md` only) | **None** — no LICENSE file, no provenance beyond one word | **BLOCKED** | Research only. Not usable to train any commercial model. Find a licensed alternative (e.g. explicitly licensed mask/PPE sets) before retraining |
| 2 | Violence videos | `Violence_system/Violence_Dataset/` | 1000 videos (500 `Violence` / 500 `NonViolence`), 571 MB | **Undocumented** | **None** | **BLOCKED** | Research only. Any future violence/fight model must be trained on licensed corpora (e.g. officially licensed XD-Violence/UCF-Crime-style research data under compliant terms, or self-collected consented footage) |
| 3 | Suspicious-action images | `suspicious_motion/sus/dataset/` | 4325 JPEGs: climbing 617, crawling 511, Falling 1137, Sitting 400, Standing 396, Walking 1264 | Roboflow export | **CC BY 4.0** — documented in `sus/dataset/crawling/README.roboflow.txt` (Roboflow export 2023-09-24) | **USABLE with attribution** — only the `crawling` class folder carries the license README; other class folders must be verified as same-export before commercial training | Keep as the one usable research dataset. Add attribution file. Mark `verified` after checking full export provenance |
| 4 | Weapon datasets | **Not in repository** | unknown | Colab `/content/guns-3`, `/content/weapones-3` | **None** — never existed in repo | **BLOCKED** (nothing to license) | Cannot reproduce or verify. Any weapon model must be retrained on a documented licensed dataset |
| 5 | YOLO-violence frames | **Not in repository** | extracted frames | Colab `/content/data.yaml` | **None** | **BLOCKED** | Same as #4 |
| 6 | Legacy evidence/snapshot images | legacy `snapshots/`, `weapon_detector/snapshots/` | demo screenshots, including screenshots of Google Images results and a YouTube video | Mixed third-party | Not redistributable | **INTERNAL ONLY** | Migrate to `evaluation/golden_set/` for regression use; do not publish, ship, or include in any distributed dataset. Label each file with origin notes |
| 7 | Demo surveillance clip (F1 acceptance) | legacy `snapshots/VID-20250724-WA0004.mp4` | 864x480, 219 frames @30fps, indoor person movement | Unknown — pre-existing file in the legacy repository (name suggests a WhatsApp export) | **None** — no license file or provenance | **BLOCKED for redistribution**; usable for **internal development/acceptance runs only** | Used to demonstrate the F1 pipeline locally (HOG detects persons on real footage) and again in F4 for chain verification with yolox (`scripts/run_f4_event_verification.py`, explicitly labeled development footage). Never shipped, never used to train a model, never published. Replace with consented/licensed footage before any demo recording leaves the machine |

## 2. Model–data coupling

A model inherits the strictest status of its training data (see `MODEL_REGISTRY.md`):

- `mask_detector.h5` ← dataset #1 → blocked.
- `modelnew.h5` ← dataset #2 → blocked.
- `human_action_model.pth` ← dataset #3 → provisional (attribution + verification).
- All `*.pt` YOLO weights ← datasets #4/#5 → blocked (plus Ultralytics AGPL on the weights themselves).
- `yolox-tiny` (F4) ← **COCO** pretraining (documented by YOLOX) → **REVIEW REQUIRED**: COCO annotations are CC BY 4.0, but individual images carry their own licenses; no Megvii statement restricts the weights, yet no separate weight license file exists either (`DETECTOR_SELECTION.md` section 8).

## 2b. Model weight artifacts (acquired in phase F4)

| # | Artifact | Location | Size / sha256 | Source | License evidence | Commercial use | Disposition |
|---|---|---|---|---|---|---|---|
| W1 | `yolox_tiny.onnx` | `sentinel/models/` (**gitignored, never committed**) | 20,219,662 B; sha256 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` | Official YOLOX release asset `Megvii-BaseDetection/YOLOX` tag `0.1.1rc0` (`.../releases/download/0.1.1rc0/yolox_tiny.onnx`), downloaded 2026-10-03 via `scripts/download_models.py` (checksum-enforced) | Repository `LICENSE` = Apache-2.0; **no separate weight license published**; training data = COCO (see coupling note) | **REVIEW REQUIRED** — software channel is permissive; weight-license silence + COCO posture need a human sign-off before any commercial claim | Acquisition-approved for **evaluation/development only** (status `candidate`); never re-hosted; missing weight = deterministic configuration error, never a silent download |

**F4-D evaluation-data decision:** no evaluation dataset was acquired in F4.
Nothing on the blocked list (rows 1–5) and no unlicensed public set (including
COCO itself) was downloaded as an evaluation substitute — an unauthorized
substitution would poison the gate. Evaluation status is therefore
`EVALUATION DATASET NOT AVAILABLE`, and no accuracy metric exists for any
model. `datasets/README.md` holds the acquisition specification for when a
licensed source is chosen; any new dataset must pass section 3 first.

## 3. Requirements for new datasets

Before adding any dataset to SENTINEL:

1. Record: name, source URL/provider, license identifier (SPDX-style), version/export date, class inventory, intended use, redistribution constraints.
2. Store the license text/attribution in `datasets/<name>/LICENSE` + `ATTRIBUTION`.
3. Confirm commercial use + model-training permission (some "research only" licenses forbid it).
4. Confirm consent/privacy posture for any footage containing identifiable people (prefer synthetic, consented, or properly licensed corpora; no covert collection).
5. Add the row to this file with status `verified` before it may feed an `approved` model.

## 4. Rules of the road

- No undocumented dataset in a commercial release — zero exceptions.
- Attribution files are generated from this registry for shipped products.
- Legacy blocked datasets stay on disk for reproducibility of research results only; they are excluded from any training pipeline.
