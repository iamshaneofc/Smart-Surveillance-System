# SENTINEL — Model Registry

**Status: FOUNDATION PHASE. No model is approved for production. No accuracy claims.**

Every model used by SENTINEL must be registered here (human doc) and, later, as `Model`/`ModelVersion` rows (see `packages/db/models.py`). A model may only be selected by a runtime **model profile** after it appears in this registry with a completed license review.

## 1. Registration requirements

Required fields for every model:

`name, source, framework, task, intended-use, license, version, commercial-use status, redistribution restrictions, training-data status, metrics, status, notes`

Status values (exact vocabulary; enforced by `tests/test_f2_docs_contract.py`):

`candidate | development | evaluated | approved | deprecated`

- `candidate` — identified and license-screened in principle; **not yet evaluated**.
- `development` — used for development/testing only; never approved for production camera profiles.
- `evaluated` — evaluation results exist for this exact version (video-level split, precision/recall/F1/mAP, FP/hour, latency p50/p95/p99, calibration) but license sign-off or approval is still outstanding. A row with status `evaluated` must have metrics; a row without metrics must not be `evaluated`.
- `approved` — evaluation results exist **and** license sign-off is complete; may be bound to a model profile.
- `deprecated` — must not be used in new code, profiles, or training; retained for provenance.

Legacy note: the pre-F2 statuses (research-only, blocked) were migrated to `deprecated` (or `candidate` where screening was never completed); the original reasons are preserved per row.

Promotion rule: `candidate → evaluated` requires evaluation-framework results (see `sentinel/evaluation/` and `docs/DETECTOR_SELECTION.md` section 4). `evaluated → approved` additionally requires license sign-off. **Nothing in this registry has been evaluated yet.**

Field definitions used below:

- **Framework** — runtime/framework the artifact requires (e.g. Keras 3, PyTorch, OpenCV).
- **Task** — detection / classification / segmentation / action recognition (mirrors `Model.task`, default `detection`).
- **Intended use** — what SENTINEL would use it for if approved; "none" means no intended use.

## 2. Existing research models (from legacy repository `Z:\Security_Surveillance_System`)

| # | Model | Location (legacy) | Framework | Task | Intended use | Architecture | Metrics | License risk | Training data | Status | Disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `mask_detector.h5` | `Mask-Face-Detector/` | Keras 3 / TensorFlow | classification (attribute) | none (PPE attribute detection out of F2 scope) | MobileNetV2, 2-class mask (224²) | **N/A** (never evaluated) | Unknown — trained on Kaggle mask dataset with **no license file** | 1915+1918 images, Kaggle, license unknown | `deprecated` (data license unknown, no metrics, known train/inference preprocessing mismatch bug) | **Do not migrate.** Re-train later on a licensed dataset through a new adapter if mask/PPE attribute detection is productized |
| 2 | `mask_detector.model` | `Mask-Face-Detector/` | Keras 2.2.4-tf | classification (attribute) | none | legacy copy of #1 | N/A | Same as #1 | Same as #1 | `deprecated` (orphan, unreferenced) | **Never use.** Archive/delete candidate |
| 3 | `res10_300x300_ssd_iter_140000.caffemodel` | `Mask-Face-Detector/Face_detection/` | OpenCV DNN (Caffe) | detection (face) | provisional research only — detection never recognition | SSD (Caffe) | N/A | OpenCV sample weights — redistribution terms need review | Unknown (academic sample) | `candidate` pending license review | Provisional research use; target replacement: SCRFD/YOLO-face behind `Detector` interface. **No face recognition ever** — detection only |
| 4 | `modelnew.h5` + `ModelWeights.weights.h5` | `Violence_system/` | Keras 3 / TensorFlow | classification (frame) | none (violence out of F2 scope) | MobileNetV2 binary violence (128²) | 0.9122 accuracy **invalid** — frame-level split leakage; ~255 ms/frame | Training on undocumented `Violence_Dataset` (1000 videos, no license) | Undocumented source/license | `deprecated` (leaked metric + unlicensed data) | **Do not migrate as production model.** Replaced by future temporal analyzer (VideoMAE-family clip model) trained on licensed data |
| 5 | `best.pt` (weapon) | `weapon_detector/snapshots/` | PyTorch / Ultralytics | detection (weapon) | none (weapons detection out of F2 scope) | YOLOv8s FP16, 6-class weapons (`Grenade, Gun, Knife, Pistol, handgun, rifle`), imgsz 800 | **N/A** — checkpoint stripped (no mAP), no eval artifacts | **Ultralytics AGPL-3.0** (fine-tuned weights) + training data not in repo (Colab `/content/guns-3`) with unknown license; class taxonomy inconsistent (`Gun` vs `handgun`) | Not present; provenance unknown | `deprecated` (AGPL + data + no metrics) + documented false positives (e.g. "Grenade 0.67" on a head; fires on screenshots at conf 0.29) | **Not the commercial detector.** Either (a) Ultralytics Enterprise License + full re-eval, or (b) preferred: retrain permissive architecture (RT-DETRv2 / RF-DETR / YOLOX) on a licensed weapon dataset |
| 6 | `best3.pt` | `weapon_detector/snapshots/` | PyTorch / Ultralytics | detection (weapon) | none | YOLOv8n, 1-class `guns`, 250 ep, ultralytics 8.2.58 (2024-07-16, AGPL header present) | N/A | AGPL-3.0 | Colab `/content/weapones-3`, absent | `deprecated` (orphan, unreferenced) | **Never use.** Archive/delete candidate |
| 7 | `Yolo_nano_weights.pt` | `Violence_system/fight_updated/` | PyTorch / Ultralytics | detection (action proxy) | none | YOLOv8n, 2-class (`non_violence, violence`), imgsz 640, ultralytics 8.2.90 (2024-09-09, AGPL header present) | N/A | AGPL-3.0 | Colab `/content/data.yaml`, absent | `deprecated` (AGPL + absent data + broken consumer script points at deleted path) | Research reference only; replacement path = temporal analyzer, not frame/box violence hacks |
| 8 | `human_action_model.pth` | `suspicious_motion/` | PyTorch | action recognition | candidate for future temporal analyzer evaluation | ResNet18 state_dict, 6-class actions (climb/crawl/fall/sit/stand/walk) | N/A — no metadata saved | Trained on `sus/dataset` (Roboflow, **CC BY 4.0** for the documented `crawling` export — verify remaining classes) | Present in repo (4325 images) | `candidate` pending verification + evaluation | Promising: data present + permissive license (attribution required). Candidate for evaluation framework first (video-level split is impossible — image data; needs clip-based protocol) |

**Known-FP evidence images** (legacy `snapshots/`, `weapon_detector/snapshots/`): screenshot detections, head-detected-as-grenade, full-frame "knife" box. These are **golden regression assets**, not models — migrate to `evaluation/golden_set/` (internal use only; some images derive from third-party content — do not redistribute).

## 3. F1 development detector (added in phase F1)

| # | Model | Profile | Framework | Task | Intended use | Source | Version | License | Training data | Metrics | Status | Purpose |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | `opencv-hog-people` | `hog` | OpenCV (`opencv-python` 4.14.0, HOG + linear SVM) | detection (person) | development/testing only — vertical-slice demonstrator | OpenCV bundled HOG + linear-SVM pedestrian weights, `cv2.HOGDescriptor_getDefaultPeopleDetector()` | `4.14.0-dev1` | **Apache-2.0** (weights ship with OpenCV); commercial use permitted | OpenCV default pedestrian SVM (historically trained on the INRIA person dataset; weights redistributed with OpenCV, not re-hosted by SENTINEL) | **N/A — not evaluated.** Never run against an evaluation dataset; synthetic frames do not trigger HOG | `development` | F1 vertical-slice demonstrator: person detection feeding the restricted-zone intrusion rule. **NOT an approved production model** — must not be selected for a production camera profile |

Notes:

- Registered here before any code referenced it (section 5 rule 1). It is **not** written to the DB `model`/`model_version` tables as an approved model.
- `opencv-python` is pinned `<5` because OpenCV 5 removed `HOGDescriptor`.
- Detector metadata is attached to every event: `Event.model_versions["detector"] = "opencv-hog-people:4.14.0-dev1"`.
- Code constant: `services/inference/hog.py` `HOG_INFO.status == "development"`; `metrics["status"] == "not-evaluated"`.
- Replacement path: permissive families from `docs/DETECTOR_SELECTION.md`, evaluated per its section 4 before any status change.
- The IOU tracker is not a model but is documented as **DEVELOPMENT TRACKER** in `SENTINEL_ARCHITECTURE.md` — non-production.

## 4. Target detector lineup (commercial path)

Permissive-license families to evaluate first (all support the `Detector` adapter interface); full research record in `docs/DETECTOR_SELECTION.md`:

| Family | License (weights) | Why |
|---|---|---|
| RT-DETRv2 / RT-DETR | Apache-2.0 | transformer detector, strong accuracy, no AGPL exposure |
| RF-DETR | Apache-2.0 for nano–large (XL/2XL are PML 1.0 — excluded) | modern distillation-trained real-time DETR, edge-capable |
| YOLOX | Apache-2.0 | mature, decoupled head, well understood deploy story (ONNX/OpenVINO/TensorRT) |
| Ultralytics YOLO (v8/11/26) | **AGPL-3.0** or paid Enterprise License | excluded without explicit license sign-off |

Selection criteria (evaluation phase): mAP + FP/hour on golden set, latency p95 on target hardware, calibration (ECE), GPU memory, license status. **F2 phase: no model was downloaded and none was claimed superior. F4 has since acquired the YOLOX-Tiny candidate (section 7) — it is still unevaluated and not claimed superior.**

## 5. Runtime model profiles

`Camera.model_profile` selects a named profile resolved through `DetectorRegistry` (see `services/inference`). Profile config (future): detector adapter, weights URI + sha256, input size, confidence threshold per class, taxonomy mapping, temporal analyzers enabled.

## 6. Registry policy

1. New model → `candidate` entry here with all required fields (section 1) before any code references it.
2. Evaluate with `evaluation/` harness → record metrics; only then may status become `evaluated`.
3. License review → record in `DATA_LICENSES.md` / this file.
4. `approved` → only then bind to a model profile.
5. Deployments (`Deployment` table) record which `ModelVersion` ran where and when — version always attached to events (`Event.model_versions`).
6. Status vocabulary is closed: only `candidate`, `development`, `evaluated`, `approved`, `deprecated`. DB mirror: `ModelVersion.status` (default `candidate`); commercial-use status is tracked separately in `Model.commercial_status`.

## 7. F4 first evaluation candidate (added in phase F4)

| # | Model | Profile | Framework | Task | Intended use | Source | Version | License | Training data | Metrics | Status | Purpose |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | `yolox-tiny` (artifact `models/yolox_tiny.onnx`, gitignored) | `yolox` (opt-in; never a default) | ONNX (`onnxruntime` 1.30.0, MIT) + numpy decode/NMS | detection (COCO 80-class; SENTINEL binds person/vehicle classes) | first evaluation candidate for camera profiles **after** evaluation passes | Official YOLOX release asset: `github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx` | artifact from release tag `0.1.1rc0`; **sha256 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`**, 20,219,662 bytes (enforced by `scripts/download_models.py`) | Apache-2.0 (repository `LICENSE`); weights ship as official release assets of that repository with **no separate weight license published** → commercial sign-off REVIEW REQUIRED | COCO (YOLOX pretraining; annotations CC BY 4.0, images individually licensed) → REVIEW REQUIRED | **N/A — never evaluated; no licensed evaluation dataset exists (F4-D gate)** | `candidate` | First loop of the F4 trustworthy-evaluation process: acquire → verify → adapter → (evaluation pending licensed data). Full gate in `docs/DETECTOR_SELECTION.md` sections 7–9 |

Required fields (section 1) for row 10:

- **name** `yolox-tiny`; **source** official Megvii YOLOX release `0.1.1rc0` (URL + sha256 pinned); **framework** ONNX/onnxruntime; **task** detection.
- **intended-use** — evaluation candidate for person/vehicle detection on SENTINEL cameras; not bound to any camera profile by default.
- **license** — Apache-2.0 code/weights channel; **commercial-use status** REVIEW REQUIRED (no separate weight license; COCO training-data posture).
- **redistribution restrictions** — artifact is not re-hosted or committed; gitignored; fetched only by the explicit checksum-pinned developer command.
- **training-data status** — COCO documented; see `DATA_LICENSES.md`.
- **metrics** — N/A (evaluation dataset not available). **status** `candidate`.
- **notes** — inference contract, benchmark procedure, and evaluation procedure: `docs/DETECTOR_SELECTION.md` sections 7–12 and `docs/OPERATIONS.md` section 8 (F4).

Notes:

- Registered **before** any code referenced it (section 6 rule 1).
- Weights are **never committed to Git** (`.gitignore` blocks `models/*.onnx`); the repository contains only `models/README.md` with provenance.
- `yolox` may only be selected explicitly (`SENTINEL_PIPELINE__DETECTOR_PROFILE=yolox` or camera `model_profile`); the shipped default stays `hog`.
- Promotion beyond `candidate` requires the evaluation plan in `docs/DETECTOR_SELECTION.md` section 4 — which currently cannot run: **no licensed evaluation dataset exists in the repository (F4-D)**.

F4 execution record (verification of mechanics — **not** evaluation, status unchanged):

- Adapter + weight management + 56 new tests green; failure isolation (degraded → recovery, clear startup config error) proven by `tests/test_f4_pipeline_integration.py`.
- Latency benchmark on the internal development clip: detector p50 34.9 / p95 48.8 / p99 74.6 ms, 24.8 fps process rate, 0 detector errors, CPU-only (`DETECTOR_SELECTION.md` section 12, `var/benchmarks/yolox-devclip.json`). Latency/throughput only — no accuracy implied.
- End-to-end chain verified 28/28 on development footage: rule → event → evidence (sha256-verified) → alert → API, events carrying `yolox-tiny:0.1.1rc0` (`scripts/run_f4_event_verification.py`, report `var/benchmarks/f4-event-verification.json`).
- **Nothing promoted**: `yolox-tiny` remains `candidate`, `opencv-hog-people` remains `development`, no row is `approved`, no accuracy metric was created.
