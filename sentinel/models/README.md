# Models directory

Weights are **never stored or committed to Git**. `.gitignore` blocks
`models/*.onnx`, `models/*.pth`, `models/*.pt`, `models/*.h5` and nested
weight files; only this README and provenance metadata are tracked.

Before any file lands here it must be registered in `docs/MODEL_REGISTRY.md`
with license + commercial-use status. Runtime selection goes through
`DetectorRegistry` model profiles (`services/inference`).

## Registered artifacts (F4)

| File | Model | Version / source | sha256 | License | Status |
|---|---|---|---|---|---|
| `yolox_tiny.onnx` | YOLOX-Tiny (80-class COCO, 416×416 input) | official release asset, `Megvii-BaseDetection/YOLOX` tag `0.1.1rc0`, `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx` | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` (20,219,662 bytes) | Apache-2.0 repository LICENSE; no separate weight license published → commercial REVIEW REQUIRED (`docs/DATA_LICENSES.md` W1) | `candidate` — unevaluated |

## Acquisition (explicit developer command only)

```bash
python scripts/download_models.py          # download + verify pinned artifacts
python scripts/download_models.py --check  # verify only, never downloads
```

- The application **never** downloads weights at startup; a missing weight
  file produces a clear configuration error from the detector adapter.
- The script verifies size + sha256 against `scripts/weights.lock.json` and
  fails if the file is present but corrupt.
- Nothing here may be re-hosted or redistributed by SENTINEL; acquisition is
  for internal evaluation/development use pending license sign-off.
