#!/usr/bin/env python
"""Deterministic model-weight acquisition (F4-G).

Downloads pinned artifacts from scripts/weights.lock.json into models/,
verifying size and sha256 before anything lands in place. This is an EXPLICIT
developer command: the application itself never downloads weights, and a
missing weight file is reported as a configuration error by the adapter.

Usage:
    python scripts/download_models.py            # download + verify all
    python scripts/download_models.py --check    # verify only (no network)
    python scripts/download_models.py --model yolox-tiny

Exit codes: 0 = success/verified, 1 = failure, 2 = usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "scripts" / "weights.lock.json"
DEFAULT_DEST = ROOT / "models"
CHUNK = 1 << 20


def load_lock() -> list[dict]:
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    artifacts = data.get("artifacts")
    if not artifacts:
        raise SystemExit("weights lock contains no artifacts")
    return artifacts


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, artifact: dict) -> None:
    actual_size = path.stat().st_size
    if actual_size != artifact["size_bytes"]:
        raise SystemExit(
            f"{path.name}: size mismatch (expected {artifact['size_bytes']}, got {actual_size})"
        )
    actual_sha = sha256_of(path)
    if actual_sha != artifact["sha256"]:
        raise SystemExit(
            f"{path.name}: sha256 mismatch (expected {artifact['sha256']}, got {actual_sha})"
        )


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, tempfile.NamedTemporaryFile(
        delete=False, dir=dest.parent, prefix=".download-", suffix=".part"
    ) as tmp:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch pinned model weights (never implicit).")
    parser.add_argument("--check", action="store_true", help="verify existing files only; no network")
    parser.add_argument("--model", help="only operate on this artifact id (e.g. yolox-tiny)")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="destination directory")
    args = parser.parse_args(argv)

    try:
        artifacts = load_lock()
    except Exception as exc:
        print(f"error: cannot read {LOCK_PATH}: {exc}", file=sys.stderr)
        return 1

    if args.model:
        artifacts = [a for a in artifacts if a["id"] == args.model]
        if not artifacts:
            print(f"error: no artifact id '{args.model}' in {LOCK_PATH}", file=sys.stderr)
            return 2

    dest = Path(args.dest)
    failures = 0

    for artifact in artifacts:
        target = dest / artifact["filename"]
        if args.check:
            if not target.exists():
                print(f"MISSING {target} (run: python scripts/download_models.py --model {artifact['id']})")
                failures += 1
                continue
            try:
                verify(target, artifact)
            except SystemExit as exc:
                print(f"FAIL {exc}")
                failures += 1
                continue
            print(f"OK {target.name} (sha256 verified, {artifact['size_bytes']} bytes)")
            continue

        if target.exists():
            try:
                verify(target, artifact)
                print(f"OK {target.name} already present and verified")
                continue
            except SystemExit:
                print(f"existing {target.name} failed verification - re-downloading")
                target.unlink()

        dest.mkdir(parents=True, exist_ok=True)
        try:
            download(artifact["url"], target)
            verify(target, artifact)
        except SystemExit as exc:
            print(f"FAIL {exc}", file=sys.stderr)
            target.unlink(missing_ok=True)
            failures += 1
            continue
        except Exception as exc:
            print(f"FAIL download {artifact['url']}: {exc}", file=sys.stderr)
            target.unlink(missing_ok=True)
            failures += 1
            continue
        print(f"OK {target.name} downloaded + sha256 verified ({artifact['size_bytes']} bytes)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
