#!/usr/bin/env python3
"""Sweep perf benchmark — backbone, fork, delta cell breakdown.

Usage:
    EMERGENT_SWEEP_MODE=1 EMERGENT_FAST_TRAINING=1 uv run python research/experiments/benchmark_sweep_perf.py
"""

from __future__ import annotations

import statistics
import sys
import time

from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import (
    build_parser_backbone,
    fork_parser,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import run_delta_cell


def _timed(label: str, fn, *, repeats: int = 1) -> float:
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    sec = statistics.mean(times)
    print(f"  {label:28} {sec:6.3f}s" + (f"  (n={repeats})" if repeats > 1 else ""))
    return result


def main() -> None:
    n, k, depth = 300, 8, "TWO_WORD"
    if len(sys.argv) >= 3:
        n, k = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) >= 4:
        depth = sys.argv[3]

    print("=" * 56)
    print(f"SWEEP PERF  n={n} k={k} depth={depth}")
    print("=" * 56)

    bb = _timed(
        "build_parser_backbone",
        lambda: build_parser_backbone(depth, seed=42, n=n, k=k),
    )
    print(f"    train={bb.train_seconds:.2f}s cal={bb.calibration_seconds:.2f}s")

    _timed("fork_parser", lambda: fork_parser(bb), repeats=5)

    _timed(
        "run_delta_cell (balanced)",
        lambda: run_delta_cell(bb, pattern="balanced", wobbly=False),
        repeats=3,
    )

    print("=" * 56)


if __name__ == "__main__":
    main()
