"""Picklable process-pool workers for sweep_dynamics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# depth, seed, n, k
BbKey = Tuple[str, int, int, int]
CellSpec = Tuple[str, bool]  # pattern, wobbly
BatchTask = Tuple[str, int, int, int, List[CellSpec], str, Optional[str]]  # + holdout, cache_dir


def _worker_log(message: str) -> None:
    print(f"[worker] {message}", flush=True)


def _ensure_env() -> None:
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("EMERGENT_SWEEP_MODE", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")


def run_backbone_batch(task: BatchTask) -> List[Dict[str, object]]:
    """Build one backbone in-process, run all delta cells for it."""
    _ensure_env()
    depth, seed, n, k, cells, holdout, cache_dir_s = task
    cache_dir = Path(cache_dir_s) if cache_dir_s else None

    from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import (
        backbone_cache_path,
        build_parser_backbone,
        load_backbone_cache,
        save_backbone_cache,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        default_holdout_set,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
        run_delta_cell,
    )

    holdout_frozen = frozenset(default_holdout_set())
    _worker_log(
        f"backbone start depth={depth} seed={seed} n={n} k={k} "
        f"cells={len(cells)}",
    )
    bb = None
    if cache_dir is not None:
        cp_path = backbone_cache_path(
            cache_dir, depth, seed=seed, n=n, k=k, holdout_words=holdout_frozen,
        )
        bb = load_backbone_cache(cp_path)
    if bb is None:
        bb = build_parser_backbone(
            depth, seed=seed, n=n, k=k, fast_training=True, calibrate=True,
            show_progress=False,
        )
        if cache_dir is not None:
            save_backbone_cache(bb, cp_path)
    _worker_log(
        f"backbone done train={bb.train_seconds:.1f}s "
        f"cal={bb.calibration_seconds:.1f}s",
    )
    rows: List[Dict[str, object]] = []
    for i, (pattern, wobbly) in enumerate(cells, start=1):
        _worker_log(
            f"cell {i}/{len(cells)} pattern={pattern} wobbly={wobbly}",
        )
        delta = run_delta_cell(
            bb, pattern=pattern, wobbly=wobbly, holdout=holdout,
            log_steps=False,
        )
        rows.append({
            "seed": seed,
            "depth": depth,
            "holdout_pattern": pattern,
            "wobbly_bootstrap": wobbly,
            **delta,
        })
    return rows
