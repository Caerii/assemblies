#!/usr/bin/env python3
"""Benchmark serial vs threaded vs process-parallel delta cells."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("EMERGENT_SWEEP_MODE", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

BbKey = Tuple[str, int, int, int]  # depth, seed, n, k
CellKey = Tuple[str, bool]  # pattern, wobbly

# Process-worker backbone cache (one per worker process).
_WORKER_BACKBONES: Dict[BbKey, object] = {}


def _worker_init() -> None:
    os.environ["EMERGENT_FAST_TRAINING"] = "1"
    os.environ["EMERGENT_SWEEP_MODE"] = "1"
    os.environ["TRAIN_PROGRESS"] = "0"


def _worker_run_cell(task: Tuple[BbKey, str, bool]) -> float:
    """Build backbone on first use in this process, run one delta cell."""
    from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import (
        build_parser_backbone,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
        run_delta_cell,
    )

    bb_key, pattern, wobbly = task
    depth, seed, n, k = bb_key
    if bb_key not in _WORKER_BACKBONES:
        _WORKER_BACKBONES[bb_key] = build_parser_backbone(
            depth, seed=seed, n=n, k=k, fast_training=True, calibrate=True,
        )
    t0 = time.perf_counter()
    run_delta_cell(_WORKER_BACKBONES[bb_key], pattern=pattern, wobbly=wobbly)
    return time.perf_counter() - t0


def _thread_run_cell(bb, pattern: str, wobbly: bool) -> float:
    from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
        run_delta_cell,
    )

    t0 = time.perf_counter()
    run_delta_cell(bb, pattern=pattern, wobbly=wobbly)
    return time.perf_counter() - t0


def explore_grid() -> Tuple[List[BbKey], List[Tuple[BbKey, str, bool]]]:
    """Explore preset: 3 seeds × 6 cells."""
    seeds = [42, 43, 44]
    patterns = ["balanced", "subject_only", "object_only"]
    wobbly_flags = [False, True]
    n, k = 300, 8
    depth = "TWO_WORD"
    backbones = [(depth, s, n, k) for s in seeds]
    tasks: List[Tuple[BbKey, str, bool]] = []
    for bb in backbones:
        for pattern in patterns:
            for wobbly in wobbly_flags:
                tasks.append((bb, pattern, wobbly))
    return backbones, tasks


def run_serial(tasks: List[Tuple[BbKey, str, bool]]) -> float:
    from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import (
        build_parser_backbone,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
        run_delta_cell,
    )

    cache: Dict[BbKey, object] = {}
    t0 = time.perf_counter()
    for bb_key, pattern, wobbly in tasks:
        if bb_key not in cache:
            depth, seed, n, k = bb_key
            cache[bb_key] = build_parser_backbone(
                depth, seed=seed, n=n, k=k, fast_training=True, calibrate=True,
            )
        run_delta_cell(cache[bb_key], pattern=pattern, wobbly=wobbly)
    return time.perf_counter() - t0


def run_threaded(bb_cache: Dict, tasks: List, workers: int) -> float:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(
            ex.map(
                lambda t: _thread_run_cell(bb_cache[t[0]], t[1], t[2]),
                tasks,
            ),
        )
    return time.perf_counter() - t0


def run_process(tasks: List, workers: int) -> float:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        list(ex.map(_worker_run_cell, tasks))
    return time.perf_counter() - t0


def run_seed_partitioned(backbones: List[BbKey], tasks: List, workers: int) -> float:
    """One process per seed: build backbone once, run all cells serially in worker."""
    from collections import defaultdict

    from research.experiments.sweep_parallel import run_backbone_batch

    cells_by_bb: Dict[BbKey, List[CellKey]] = defaultdict(list)
    for bb_key, pattern, wobbly in tasks:
        cells_by_bb[bb_key].append((pattern, wobbly))

    batches = [
        (depth, seed, n, k, cells_by_bb[bb_key], "small")
        for bb_key in backbones
        for depth, seed, n, k in [bb_key]
    ]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(workers, len(batches)),
        initializer=_worker_init,
    ) as ex:
        list(ex.map(run_backbone_batch, batches))
    return time.perf_counter() - t0


def main() -> None:
    import multiprocessing as mp

    mp.freeze_support()
    cpus = os.cpu_count() or 1
    backbones, tasks = explore_grid()
    print(f"CPUs (logical): {cpus}")
    print(f"Grid: {len(backbones)} backbones × {len(tasks) // len(backbones)} cells = {len(tasks)} tasks")
    print()

    serial_s = run_serial(tasks)
    print(f"serial (backbone cache):     {serial_s:6.1f}s")

    # Threaded with shared backbone (upper bound if GIL released — it isn't)
    from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import (
        build_parser_backbone,
    )

    bb_cache: Dict = {}
    for bb_key in backbones:
        depth, seed, n, k = bb_key
        bb_cache[bb_key] = build_parser_backbone(
            depth, seed=seed, n=n, k=k, fast_training=True, calibrate=True,
        )
    for w in [2, 4, min(8, cpus)]:
        wall = run_threaded(bb_cache, tasks, w)
        print(f"threads={w} (shared bb):      {wall:6.1f}s  ({serial_s / wall:.2f}x)")

    print()
    for w in [1, 2, 3, 4, min(6, cpus)]:
        wall = run_process(tasks, w)
        print(f"processes={w} (per-cell):       {wall:6.1f}s  ({serial_s / wall:.2f}x)")

    print()
    for w in [1, 2, 3, 4]:
        wall = run_seed_partitioned(backbones, tasks, w)
        print(f"processes={w} (per-seed):       {wall:6.1f}s  ({serial_s / wall:.2f}x)")


if __name__ == "__main__":
    main()
