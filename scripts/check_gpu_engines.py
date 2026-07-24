#!/usr/bin/env python3
"""Report GPU / engine availability for assembly calculus.

Usage::

    uv sync --extra gpu
    uv run python scripts/check_gpu_engines.py
"""

from __future__ import annotations

import sys


def main() -> int:
    print("=== Assembly calculus GPU / engine check ===\n")

    try:
        import torch

        print(f"torch:           {torch.__version__}")
        print(f"cuda available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device:     {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"cuda capability: {props.major}.{props.minor}")
            print(f"device memory:   {props.total_memory / (1024**3):.1f} GiB")
    except ImportError:
        print("torch:           NOT INSTALLED")
        print("  Fix: uv sync --extra gpu")

    print()
    from neural_assemblies.core.engine import list_engines
    from neural_assemblies.core.backend import detect_best_engine, detect_fastest_engine
    from neural_assemblies.assembly_calculus.emergent.training.perf import (
        resolve_engine,
        force_gpu_enabled,
    )

    engines = list_engines()
    print(f"registered engines: {engines}")
    print(f"detect_fastest:     {detect_fastest_engine()}")
    for n in (10_000, 100_000, 1_000_000):
        print(f"detect_best(n={n:,}): {detect_best_engine(n)}")
    print(f"resolve_engine(auto, n=10k):  {resolve_engine('auto', n_hint=10_000)}")
    print(f"resolve_engine(auto, n=1M):   {resolve_engine('auto', n_hint=1_000_000)}")
    if force_gpu_enabled():
        print("ASSEMBLIES_FORCE_GPU=1 is set")

    if "torch_sparse" not in engines:
        print("\n[!] torch_sparse not registered — install CUDA torch: uv sync --extra gpu")
        return 1

    print("\n[ok] torch_sparse is available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
