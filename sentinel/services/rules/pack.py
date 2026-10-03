from pathlib import Path
from typing import Any

import yaml

from packages.schemas.rule import RulePack

PACKS_DIR = Path(__file__).resolve().parents[2] / "rules" / "packs"


def load_pack_dict(data: dict[str, Any]) -> RulePack:
    return RulePack.model_validate(data)


def load_pack(path: str | Path) -> RulePack:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"rule pack {path} must contain a YAML mapping")
    return load_pack_dict(data)


def available_packs(directory: Path | None = None) -> list[Path]:
    directory = directory or PACKS_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("*.yaml"))
