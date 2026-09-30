"""Resolve repository and literature parity directories."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    env = os.environ.get("ASSEMBLIES_REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repo root (set ASSEMBLIES_REPO_ROOT)")


@lru_cache(maxsize=1)
def parity_root() -> Path:
    env = os.environ.get("ASSEMBLIES_PARITY_ROOT")
    if env:
        return Path(env).resolve()
    return repo_root() / "research" / "literature" / "parity"


def config_dir() -> Path:
    return parity_root() / "configs"


def golden_dir() -> Path:
    return parity_root() / "golden"


def registry_path() -> Path:
    return parity_root() / "registry.json"
