# SENTINEL — Detector Selection (Research Gate)

**Status (F4): FIRST EVALUATION CANDIDATE selected — provisional: YOLOX-Tiny ONNX (registry status `candidate`). Weights acquired through the documented checksum-pinned process (section 8). Still NO accuracy evaluation, NO threshold selection, NO approval, NO "best detector" claim. Sections 1–6 are the preserved F2 research record; sections 7–9 record the F4-A/B/C gate; sections 10–12 record the F4-D..P execution (evaluation data gate, adapter/weight/tests/pipeline integration, benchmarks and end-to-end verification — latency and mechanics only).**

This document records the F2 research on three permissive-license detector families
(RT-DETRv2, RF-DETR, YOLOX) to define the shortlist and the evaluation plan for the
future detector-selection gate. All numbers below are **vendor/repository-reported
claims copied from public sources — they are NOT SENTINEL measurements** and must not
be cited as performance facts. Nothing here changes any status in
`docs/MODEL_REGISTRY.md`; every candidate stays `candidate` until it passes the
evaluation plan in section 4.

## 1. Scope and non-goals

In scope: license review, framework/runtime facts, model variants, hardware support,
export/deployment paths, project maturity — for the three families named above.

Explicitly out of scope: picking a winner, downloading weights, running benchmarks,
any accuracy/latency claim for SENTINEL's cameras, and any change to the active
detector (the active profile remains `hog` = `opencv-hog-people`,
development, see `docs/MODEL_REGISTRY.md` section 3).

## 2. Family research

### 2.1 RT-DETRv2 (and its RT-DETR predecessor)

- **Source:** official implementation `github.com/lyuwenyu/RT-DETR` (papers:
  "DETRs Beat YOLOs on Real-time Object Detection" and "RT-DETRv2: Improved
  Baseline with Bag-of-Freebies for Real-Time Detection Transformer",
  arXiv:2407.17140; RT-DETR accepted at CVPR 2024).
- **License:** Apache-2.0 (repository `LICENSE` file; mirrors/forks confirm
  Apache-2.0).
- **Framework:** PyTorch and Paddle implementations; also available in
  Hugging Face Transformers (`RTDetrV2ForObjectDetection`) and in
  `ultralytics` (which is AGPL — using RT-DETR *through* ultralytics would pull
  in AGPL code, so any integration must use the official repo or HF, not
  ultralytics).
- **Variants:** RT-DETRv2-S/M/L/X (ResNet18/34/50/101 backbones; v2 also has an
  R50 "M*" variant). RT-DETR adds HGNetv2-L/X and RegNet/DLA34 variants.
- **Reported benchmarks (vendor claims, unverified):** e.g. RT-DETRv2-S 47.9 AP
  at 217 FPS (T4, TensorRT FP16, COCO val, 640 input) per the official README
  table; RT-DETR-R18 46.5 AP. COCO-trained weights cover 80 classes.
- **Export/deployment:** ONNX export expected; v2 introduced a discrete sampling
  operator specifically to remove the `grid_sample` deployment constraint that
  complicated earlier DETR ONNX exports. Hugging Face export path also exists.
- **CPU/GPU:** practical for GPU; CPU inference is possible but untested here —
  must be measured on target hardware during evaluation. TensorRT path is
  NVIDIA-GPU-only.
- **Maturity:** active upstream (2024–2025 releases incl. RT-DETRv4 note in
  forks); widely mirrored; research-grade code (training + inference), so a
  pinned commit + weight hash would be required for reproducibility.
- **Risks:** no vendor SLA; training code is research-grade; must avoid the
  AGPL `ultralytics` path; ONNX export quality for v2 must be verified
  empirically.

### 2.2 RF-DETR

- **Source:** `github.com/roboflow/rf-detr` (Roboflow; paper on Hugging Face
  arXiv:2511.09554, "RF-DETR: Neural Architecture Search for Real-Time
  Detection Transformers"; docs at `rfdetr.roboflow.com`).
- **License:** **Apache-2.0 for code and core model weights (Nano → Large)**.
  Important caveat: XL and 2XL detection variants are `rfdetr_plus` components
  licensed **PML 1.0**, and Roboflow's platform-hosted "Platform Model
  Licensed" weights are for non-commercial research — those variants/weights
  must NOT be used under the Apache assumption. Only Apache-designated
  Nano–Large weights qualify for SENTINEL's commercial path.
- **Framework:** Python package `rfdetr` (pip-installable; Python ≥3.10 per
  recent docs), PyTorch backend; also usable via Roboflow `inference` and via
  Hugging Face Transformers.
- **Variants:** nano (~10M), small (~25M), base (29M), medium (~75M), large
  (129M) params — all COCO-pretrained (80 classes), open access.
- **Reported benchmarks (vendor claims, unverified):** Roboflow states it is
  "the first real-time model to exceed 60 AP on COCO" (base size) and SOTA on
  their RF100-VL domain-transfer benchmark. Treat as marketing claims until
  reproduced.
- **Export/deployment:** backends documented by Roboflow `inference`: `torch`
  (CPU and multiple CUDA versions), `onnx` (CPU/CUDA), TensorRT (`trt10`);
  Jetson edge deployment documented. ONNX export is a first-class path.
- **CPU/GPU:** explicitly positioned for edge/CPU usage (nano/base sizes);
  latency claims must still be measured on SENTINEL's target hardware.
- **Maturity:** active commercial open-source project (Roboflow backing),
  frequent releases, good docs, Hugging Face integration (PR #36895).
- **Risks:** Apache/PML split across variants (license boundary must be
  documented in `DATA_LICENSES.md` before use); dependency footprint of the
  `rfdetr` package must be reviewed against the repo's minimal-dependency
  rule; upstream is a company — long-term maintenance risk is moderate.

### 2.3 YOLOX

- **Source:** `github.com/Megvii-BaseDetection/YOLOX` (Megvii; arXiv YOLOX
  paper; docs at yolox.readthedocs.io).
- **License:** **Apache-2.0** (repository `LICENSE`, copyright 2021-2022
  Megvii Inc.). A separate MegEngine implementation (`MegEngine/YOLOX`) is also
  Apache-2.0.
- **Framework:** PyTorch (primary) + MegEngine variant.
- **Variants:** YOLOX-Tiny/Nano/S/M/L/X (and custom-N/S). Anchor-free,
  decoupled head. COCO-pretrained weights distributed from the README table.
- **Reported benchmarks (vendor claims, unverified):** e.g. YOLOX-S 40.5 mAP
  at 640 (COCO val), Tiny 32.2 at 416 — per official README tables.
- **Export/deployment:** ONNX, TensorRT, ncnn, OpenVINO documented — the most
  diverse deployment story of the three (OpenVINO matters for Intel-CPU/NPU
  targets).
- **CPU/GPU:** ONNX/OpenVINO/ncnn paths make CPU deployment well-trodden;
  TensorRT for NVIDIA GPUs. Must be measured on target hardware.
- **Maturity:** established and widely used (~10.6k GitHub stars), but
  upstream updates are slower than RF-DETR's release cadence (last notable
  feature updates 2022–2023); still maintained and stable for inference.
- **Risks:** older architecture than the DETR-based options (lower expected
  ceiling on accuracy per compute — a hypothesis to test, not a fact);
  pretraining provenance of some converted third-party weights must be
  checked (use official Megvii weights only).

### 2.4 Comparison summary

| Family | License (weights) | Variants | Export | Commercial path | Maturity |
|---|---|---|---|---|---|
| RT-DETRv2 | Apache-2.0 | S/M/L/X (+RT-DETR backbone zoo) | ONNX (improved in v2), TensorRT, HF | yes | active research repo |
| RF-DETR | Apache-2.0 (Nano–Large; XL/2XL = PML 1.0) | nano→large | torch/ONNX/TensorRT/Jetson | yes, license split must be tracked | active, well documented |
| YOLOX | Apache-2.0 | Tiny→X | ONNX, TensorRT, ncnn, OpenVINO | yes | stable, slower cadence |

All three satisfy the license gate (no AGPL) — unlike Ultralytics YOLO
(AGPL-3.0 / paid Enterprise), which remains excluded without explicit
sign-off (`docs/MODEL_REGISTRY.md` section 4).

## 3. Shortlist

Evaluation shortlist, in no particular order:

1. **RF-DETR base/small** (Apache-designated weights only)
2. **RT-DETRv2-S** (official repo or Hugging Face path — never ultralytics)
3. **YOLOX-S/Tiny**

Contingent/excluded:

- Ultralytics YOLO family: excluded by default (AGPL-3.0).
- RF-DETR XL/2XL and Roboflow "Platform Model Licensed" weights: excluded
  (PML 1.0 / non-commercial).
- Legacy `best.pt`/`best3.pt`/`Yolo_nano_weights.pt`: already `deprecated`
  (AGPL + unlicensed training data) — not part of this shortlist.

## 4. Evaluation plan (gate for `candidate → evaluated`)

No shortlist member advances until **all** of the following are executed with the
F2 evaluation framework (`sentinel/evaluation/`):

1. **Manifest + data:** curated evaluation dataset with video-level splits
   (no frame-level leakage), per-entry annotations, dataset license recorded in
   `docs/DATA_LICENSES.md`. Missing media/annotations → the runner already
   reports `EVALUATION DATASET NOT AVAILABLE` with null metrics — never
   fabricate numbers.
2. **Golden set:** include the legacy known-FP images (screenshot detections,
   head-as-grenade, full-frame knife) as regression examples.
3. **Metrics per model/version:** precision, recall, F1, mAP (IoU 0.5 and
   0.5:0.95), TP/FP/FN, per-class AP, FP/hour on surveillance-like footage,
   error examples (top FN/FP).
4. **Latency:** p50/p95/p99 + mean on (a) CPU of the target host and (b) target
   GPU if one exists — measured by `evaluation/runner.py` `LatencyStats`,
   batch size 1, per-frame.
5. **Calibration:** Expected Calibration Error at operating thresholds;
   threshold sensitivity sweep per rule type (intrusion needs high precision,
   may trade recall).
6. **Environment reproducibility:** report records python/platform/opencv/
   hardware; weights pinned by sha256; model version recorded per event via
   `Event.model_versions`.
7. **License sign-off:** full license text (code + weights) attached to the
   `candidate` entry and `DATA_LICENSES.md` before any `evaluated` status.
8. **Decision:** results documented in this file and `MODEL_REGISTRY.md`;
   `evaluated → approved` only after license sign-off (per registry policy).

## 5. Explicit statements

- No winner is selected. No model is approved.
- F2 phase: no weights were downloaded or run. (F4 later acquired the
  YOLOX-Tiny ONNX artifact under section 8 — still **unevaluated**.)
- All benchmark numbers in section 2 are vendor-reported and unverified.
- The active/default detector remains `opencv-hog-people` (status
  `development`). The YOLOX profile added in F4 is opt-in only and is not the
  default for any camera.
- This document does not authorize any change to `Camera.model_profile`
  defaults (`packages/config/settings.py` `detector_profile = "hog"`).

## 6. Sources (accessed 2026-10-03)

- RT-DETR official repo: https://github.com/lyuwenyu/RT-DETR (README model
  tables; Apache-2.0 `LICENSE`)
- RT-DETRv2 technical report: https://arxiv.org/abs/2407.17140
- RT-DETRv2 in Hugging Face:
  https://huggingface.co/docs/transformers/model_doc/rt_detr_v2
- RF-DETR repo: https://github.com/roboflow/rf-detr (Apache 2.0 core /
  PML 1.0 plus split)
- RF-DETR docs: https://rfdetr.roboflow.com/ (install, quickstart, benchmarks
  claims)
- RF-DETR model IDs/backends:
  https://inference-models.roboflow.com/models/rfdetr-object-detection
- RF-DETR base on HF (Apache-2.0 model card):
  https://huggingface.co/Roboflow/rf-detr-base
- YOLOX repo: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0
  `LICENSE`; README benchmark table)
- YOLOX docs: https://yolox.readthedocs.io/

## 7. F4-A — Detector contract (as implemented)

The pipeline is decoupled from any detector vendor. Everything above/below the
detector sees only this contract; vendor code lives exclusively inside one
adapter module.

### 7.1 Interface

`services/inference/interfaces.py`:

- `Detector` Protocol — `info -> ModelInfo`, `detect(frame: FramePacket) -> list[Detection]`, `warmup()`, `close()`.
- `DetectorRegistry` — `register(profile, factory)`, `create(profile, **kwargs)`, `profiles()`; unknown profile raises `KeyError` listing registered names.
- `default_registry()` — lazy-imports and registers `null`, `stub`, `hog` (and `yolox` from F4-F). Lazy import keeps optional runtime deps out of process startup.

`services/inference/types.py`:

- `BBox` — normalized `x, y, w, h` in `[0,1]` (top-left origin), with `iou()`, `center()`.
- `Detection` — **`class_id`, `class_name`, `confidence`, `bbox`, `track_id`, `timestamp`, `model_id`, `model_version`** (the normalized provenance-bearing representation required by F4-A).
- `ModelInfo` — `name, version, family, license, classes, input_size, metrics, status`; `label = name:version`.

### 7.2 Wiring

- `packages/config/settings.py` `PipelineSettings.detector_profile` (default `hog`) + `detector_options: dict` (env `SENTINEL_PIPELINE__DETECTOR_PROFILE` / `SENTINEL_PIPELINE__DETECTOR_OPTIONS`).
- `services/pipeline/runner.py` builds the detector through `default_registry().create(profile, **detector_options)` — no detector type is imported by the pipeline itself.
- Events carry provenance via `Event.model_versions["detector"] = "<name>:<version>"` (and tracker version), preserving the F1 rule.

### 7.3 Failure isolation (already enforced)

- `services/pipeline/pipeline.py` catches detector exceptions per frame: increments `detector_errors`, sets `ai_status = degraded`, keeps the camera worker alive — a detector failure is never silently reported as "no detections".
- `runner.py` health snapshot exposes `detector_errors` so camera health can distinguish *camera healthy + detector degraded* from *camera offline*.

### 7.4 Replaceability rules

1. No vendor SDK import outside the adapter module (`services/inference/<vendor>.py`); optional deps imported lazily inside the adapter.
2. Detectors communicate only through `Detection`/`ModelInfo`; downstream (tracker, zones, rules, events) never branches on model identity.
3. Swapping detectors is a `detector_profile`/`detector_options` change — no pipeline code change.

## 8. F4-B — License / provenance gate (YOLOX-Tiny ONNX artifact)

Verified 2026-10-03 before integration (F4-B checklist):

| # | Gate item | Evidence |
|---|---|---|
| 1 | Software license | YOLOX repository `LICENSE` = **Apache License 2.0** (copyright Megvii Inc.), verified at tag `0.1.1rc0` |
| 2 | Model-weight license | Weights are distributed **only as official GitHub release assets of the Apache-2.0 repository**; YOLOX publishes **no separate weight license file** and no use restrictions. Industry practice treats them as Apache-2.0, but absent an explicit weight statement this is recorded as **REVIEW REQUIRED** for commercial sign-off (F4-U) |
| 3 | Framework/dependency licenses | `onnxruntime` 1.30.0 **MIT**; `numpy` BSD-3-Clause; `opencv-python`/`opencv-headless` Apache-2.0; `onnx` (test/dev only) Apache-2.0; Python 3.12 PSF. No AGPL in the inference path |
| 4 | Source / provenance | Exact artifact: `yolox_tiny.onnx`, **20,219,662 bytes**, **sha256 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`**, from `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx` (release tag `0.1.1rc0`, assets published by the YOLOX maintainers). Acquired 2026-10-03; GitHub-reported size matches the local file exactly |
| 5 | Supported usage | Official model zoo entry for YOLOX-Tiny (COCO-pretrained, 80 classes, 416 input); official ONNX export + ONNXRuntime demo documented by the project |
| 6 | Redistribution | Apache-2.0 permits redistribution with license notice. SENTINEL does **not** re-host or re-commit the weight — it is gitignored and fetched only by the explicit developer command (`scripts/download_models.py`, checksum-pinned) |
| 7 | Commercial-use constraints | None documented by Megvii for the weights. **Training data = COCO** (per YOLOX paper/README): COCO annotations are CC BY 4.0; individual images carry their own licenses — commercial posture recorded **REVIEW REQUIRED** (F4-U), per the rule that model license ≠ training-data license |

Empirical inference contract (validated 2026-10-03 against the artifact itself —
the released ONNX differs from the tagged demo script, and the artifact wins):

- Input `images`: fixed `[1, 3, 416, 416]` float32; **pixel range 0–255** (the `/255` normalization is baked into the exported graph — feeding the demo's 0–1 input collapses all scores; verified empirically).
- Letterbox: aspect-preserving `min`-scale resize, pad value 114, pad at top/left, BGR→RGB channel swap (matches YOLOX `preproc`).
- Output `output`: raw head `[1, 3549, 85]` (`3549 = 52² + 26² + 13²`, `85 = 4 box + 1 obj + 80 class`); **decode happens outside the graph** (`(xy + grid) * stride`, `exp wh * stride`, strides 8/16/32), then numpy NMS (IoU 0.45, score floor 0.1) and a final configurable confidence threshold.
- Boxes are un-letterboxed by dividing by the resize ratio (top/left padding → exact inverse), then normalized to `[0,1]` for `BBox`.
- Smoke-validated on YOLOX's own `assets/dog.jpg` (bicycle 0.88, dog 0.53) and `ultralytics/assets bus.jpg` (bus 0.92, three persons 0.84–0.88) — these images prove *mechanics only* and are **not** an evaluation dataset.

## 9. F4-C — Candidate selection (FIRST EVALUATION CANDIDATE)

Selection criteria, scored from documented engineering evidence (F2 research in
section 2 + F4-B gates). This is **not** a claim of being best — it selects the
first candidate for the evaluation loop.

| Criterion | RT-DETRv2-S | RF-DETR base | **YOLOX-Tiny (selected)** |
|---|---|---|---|
| Software license | Apache-2.0 | Apache-2.0 (XL/2XL PML — split) | Apache-2.0 |
| Weight license posture | Apache-2.0 repo, no separate weight file | Apache-2.0 declared for nano–large | Apache-2.0 release assets; no separate weight file → **REVIEW REQUIRED** (same class of uncertainty as RT-DETRv2) |
| Detector task support | detection (80 cls) | detection (80 cls) | detection (80 cls) |
| Inference framework on CPU | torch/HF stack | torch package (`rfdetr`) | **onnxruntime (MIT), numpy post-process** |
| CPU feasibility | plausible but unmeasured; torch required | positioned for edge; torch required | **~5 M params, 416 input, 20 MB ONNX — measured here (section 12 benchmarks)** |
| GPU feasibility | TensorRT path | TensorRT/Jetson documented | TensorRT/OpenVINO documented upstream |
| ONNX/export options | improved v2 export, still research-tooling | first-class export | **official prebuilt ONNX released by maintainers — no export step needed** |
| Deployment complexity | medium (research repo + export) | medium (pip package, heavier dep tree) | **low (one ONNX file + onnxruntime)** |
| Model size | S ≈ 20 M params | base 29 M | Tiny ≈ 5 M params / 20.2 MB ONNX |
| Available checkpoints | official zoo | official HF/repo zoo | official release assets (`0.1.1rc0`) |
| Reproducibility | pinned commit + weight hash required | versioned package | **pinned URL + tag + sha256 (recorded, enforced by download script)** |
| Integration complexity | higher (torch or HF runtime) | higher (`rfdetr` dep tree vs repo's minimal-deps rule) | **lowest — lazy optional dep, numpy decode, no training code** |

Decision: **YOLOX-Tiny = FIRST EVALUATION CANDIDATE** (provisional until
evaluation, per section 4). RT-DETRv2 and RF-DETR remain shortlist members for
later comparison; nothing is discarded.

Why Tiny rather than YOLOX-S: Tiny's released 416 ONNX makes the full
acquire→verify→integrate loop reproducible on CPU-only hardware; S-class
weights would require a torch export step in the loop, which is orthogonal risk
for the first evaluation cycle. This is an engineering-ordering decision, not
an accuracy judgment.

## 10. F4-D/E — Evaluation data gate & manifest contract

**F4-D decision: evaluation status = N/A ("EVALUATION DATASET NOT AVAILABLE").**
No dataset in the repository or available for download in this phase satisfies
the licensing gate of section 4. In particular **COCO itself was not
downloaded**: substituting it for a purpose-built evaluation set would be an
unauthorized dataset substitution (the evaluation gate forbids it). Consequences,
all enforced by tests (`tests/test_f4_evaluation_gate.py`, 18 tests):

- `evaluation/run_evaluation.py` on a missing/absent manifest returns
  `status = EVALUATION DATASET NOT AVAILABLE` with **null metrics** — never
  fabricated numbers. The F2 honesty behavior is preserved, not weakened.
- No mAP / precision / recall number exists anywhere for yolox-tiny or HOG.
  The registry row status stays `candidate` / `development` respectively.
- `datasets/README.md` was rewritten as the data-acquisition specification:
  qualifying-dataset requirements, blocked sources, acquisition checklist,
  registration rules. Acquisition itself is deferred (needs a human-approved
  source selection).

**F4-E manifest contract** (so evaluation can run honestly the day data exists):

- `evaluation/manifest.py` extended with `dataset_id`, `class_mapping`,
  per-entry `source_resolution` and `capture_context`; the template
  `evaluation/manifests/example.template.json` carries explicit `REPLACE` fields
  (a template can never be mistaken for real data).
- `evaluation/validate_manifest.py` — 8 checks (`manifest_readable`,
  `missing_metadata`, `duplicate_identifiers`, `unsupported_format`,
  `missing_files`, `malformed_annotations`, `unknown_classes`,
  `invalid_bounding_boxes`) with `Issue`/`ValidationReport` output, `--json` /
  `--quiet` modes, exit 0/1. **Nothing is silently discarded**: every rejected
  entry is reported. The evaluation runner refuses to start on an invalid
  manifest.

## 11. F4-F..N — Adapter, weight management, tests, pipeline integration

- **Adapter** (`services/inference/yolox.py`): `YoloXOnnxDetector` behind the
  section 7 contract; `YOLOX_SPEC` pins filename/version/sha256/url and is
  asserted in sync with `scripts/weights.lock.json` by tests. The empirical
  input/output contract of section 8 is implemented exactly (0–255 input,
  letterbox, external decode + pixel-space NMS). Registered as profile `yolox`
  in `default_registry()`; the shipped default stays `hog`.
- **Weight management**: `scripts/download_models.py` (stdlib only, size +
  sha256 enforced, `--check` mode), `scripts/weights.lock.json`,
  `models/README.md` provenance table, `.gitignore` excludes `models/*.onnx`
  (weights are never committed). A missing artifact raises `DetectorError`
  with the exact acquisition command — there is **no automatic download** at
  runtime.
- **Tests** (56 new, all green): manifest/evaluation gate 18
  (`test_f4_evaluation_gate.py`), adapter 24 (`test_f4_detector_adapter.py`,
  deterministic ONNX fixture generated at test time + real-weights tests that
  skip when absent), benchmark/evaluation mechanics 9
  (`test_f4_benchmark_eval.py`), pipeline integration/failure 5
  (`test_f4_pipeline_integration.py`).
- **Failure isolation proven** (F4-N): three corrupt frames mid-run →
  `detector_errors = 3`, `ai_status` `healthy → degraded → healthy`, frame loop
  never stops, and the condition is **never** recorded as "no detections".
  Missing weights at startup → `PipelineRunner` construction fails with the
  acquisition command (the API logs `pipeline failed to start - API continues
  without it` and serves the rest of the system). Invalid configuration
  (bad confidence, unknown class, input-size mismatch, corrupt file, missing
  onnxruntime) → deterministic `DetectorError`, never a silent fallback.
- **Provenance preserved** (F4-M): events created under the yolox profile carry
  `model_versions["detector"] = "yolox-tiny:0.1.1rc0"` from detection through
  tracking, zone rules, confirmation and persistence (asserted both in tests
  and in the live run of section 12).

## 12. F4-I/K/L/O/P — Benchmarks & verification record (development only)

All numbers below characterize **this machine, this development clip, latency
only**. They are **not** accuracy results — no mAP/precision/recall exists
(section 10) — and they are not vendor benchmark numbers.

**Environment.** Windows 11 (10.0.26200), Intel Core i5-10300H (4C/8T),
16 GB RAM, CPU-only inference (`CPUExecutionProvider`), Python 3.12.4,
onnxruntime 1.30.0, opencv 4.14.0, numpy (venv `sentinel/.venv`).

**Latency / throughput** (`scripts/benchmark_pipeline.py --profile yolox
--json`, 219-frame internal clip, batch 1, conf 0.3, 1 warmup call;
`var/benchmarks/yolox-devclip.json`):

| Metric | Value |
|---|---|
| detector p50 / p95 / p99 | **34.9 ms / 48.8 ms / 74.6 ms** (mean 37.4, max 139.1) |
| frame end-to-end p50 / p95 / p99 | 35.1 / 49.0 / 74.8 ms |
| process rate | 24.8 fps (8.85 s wall for 219 frames) |
| detector / tracker errors | 0 / 0 |

Run-to-run variance observed: a first cold-cache run showed p50 ≈ 59 ms
(14.4 fps). The JSON report records hardware, software versions, warmup
policy, batch size, input resolution, threshold and status (`candidate /
not-evaluated`) so no number can be read as an accuracy claim.

**Resources.** Detector init (ONNX session load) 0.07–0.33 s; process peak
working set 42 MB → **129 MB** after warmup + 30 detections; weights 19.3 MB
on disk (gitignored).

**End-to-end event verification** (`scripts/run_f4_event_verification.py`,
**28/28 checks passed**; report `var/benchmarks/f4-event-verification.json`).
Two runs over the internal clip (license row 7 of `DATA_LICENSES.md` —
development footage, never redistributed, labeled synthetic/development):

1. **Stock factory pack** (confirm 2 s): pendings were created (17 frames
   with a pending key) and cancelled — **0 events confirmed**, because the
   clip's longest ≥0.5 in-zone condition is ~1.3 s. This is the honest
   confirmation behavior of the shipped configuration, recorded as a pass.
2. **`f4-verification.yaml` pack** (confirm 0.6 s, documented development-only
   pack): **2 confirmed `restricted_zone_intrusion` events**, both carrying
   `yolox-tiny:0.1.1rc0`; explainability (4 conditions, summary, rule version)
   intact; 4 evidence rows (snapshot + clip) with sha256 verified against
   files on disk; 1 in-app alert dispatched; camera health flushed
   `ai_status=healthy`; everything retrievable through the public API
   (`/events`, `/events/{id}`, `/cameras/{id}/health`, `/alerts`,
   `/system/health`).

**Tracking interaction (F4-P), 32 gated frames (5 fps) per run:** 19 frames
with detections, 13 detector-gap frames (real footage gaps — not errors,
`detect_error_frames = 0`), 1.34 detections/frame (max 5), 18 in-zone
detection frames, in-zone confidence mean 0.53 / max 0.83, 8 tracks with
lifetimes recorded (longest 7–217), **0 in-zone track-ID switches**, pending
window observed cancelling on confidence gaps. Known limitation surfaced (pre-
existing, not introduced by F4): `EvidenceService` keeps **one session per
camera**, so of two near-simultaneous events only the later one receives
evidence — recorded as a candidate F5 item, not a regression.

**Registry status after all of this: still `candidate`.** Nothing in this
section promotes the model; promotion requires section 4 evaluation plus
license sign-off.

