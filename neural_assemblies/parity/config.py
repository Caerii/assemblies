"""Load per-paper YAML config registries."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import config_dir


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=32)
def load_paper_config(paper_key: str) -> dict:
    """Load ``parity/configs/{paper_key}.yaml`` (e.g. ``colt2022``, ``pnas2020``)."""
    path = config_dir() / f"{paper_key}.yaml"
    return _load_yaml(path)


def load_regime(paper_key: str, regime: str) -> dict[str, Any]:
    """Load a named parameter regime from a paper config."""
    cfg = load_paper_config(paper_key)
    regimes = cfg.get("param_regimes") or cfg.get("regimes") or {}
    if regime in regimes:
        return dict(regimes[regime])
    if regime in cfg:
        block = cfg[regime]
        return dict(block) if isinstance(block, dict) else {"value": block}
    raise KeyError(f"regime {regime!r} not in {paper_key}.yaml")


def brain_defaults(regime: str = "ci_parity") -> dict[str, Any]:
    """Global Brain defaults: try pnas2020 ci_parity, else fallbacks."""
    try:
        return load_regime("pnas2020", regime)
    except (FileNotFoundError, KeyError):
        return {
            "n": 5000, "k": 80, "p": 0.05, "beta": 0.1,
            "rounds": 10, "seed": 42, "engine": "numpy_sparse",
        }
