"""Paper-scoped reproduction entrypoints."""

from __future__ import annotations

from typing import Any

from .config import load_paper_config, load_regime
from .runner import run_protocol, verify_protocol


def coin2024(protocol_id: str = "coin2024_demo", **kwargs: Any):
    return run_protocol(protocol_id, **kwargs)


def colt2022(protocol_id: str = "colt2022_mnist", **kwargs: Any):
    return run_protocol(protocol_id, **kwargs)


def nemo2025(protocol_id: str = "nemo2025_scaffold", **kwargs: Any):
    return run_protocol(protocol_id, **kwargs)


def pnas2020(protocol_id: str = "pnas2020_scaling", **kwargs: Any):
    return run_protocol(protocol_id, **kwargs)


def direct2026(**kwargs: Any):
    return run_protocol("direct2026_pearl", **kwargs)


def hoff2026(**kwargs: Any):
    return run_protocol("hoff2026_size_dist", **kwargs)


__all__ = [
    "coin2024",
    "colt2022",
    "nemo2025",
    "pnas2020",
    "direct2026",
    "hoff2026",
    "load_paper_config",
    "load_regime",
    "verify_protocol",
]
