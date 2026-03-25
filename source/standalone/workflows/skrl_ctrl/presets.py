"""Load eval / finetune preset files (JSON or YAML) and coefficient helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_preset_file(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Preset not found: {p}")
    suf = p.suffix.lower()
    with open(p, encoding="utf-8") as f:
        if suf in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as e:
                raise ImportError("Install PyYAML to use .yaml/.yml presets, or use .json") from e
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Preset root must be a mapping, got {type(data)}")
    return data


def load_coeff_json(path: str | Path) -> list[list[float]]:
    """Load coefficient rows for ``predefined_task_coeff`` (shorter rows are zero-padded to degree 10)."""
    with open(Path(path), encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"Empty coefficient file: {path}")
    if isinstance(data[0], (int, float)):
        return [[float(x) for x in data]]
    return [[float(x) for x in row] for row in data]


def pool_indices(B: int, pool_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, pool_size, size=B, dtype=np.int64)


def merge_preset(preset: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(preset)
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    return out
