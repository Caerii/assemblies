#!/usr/bin/env python3
"""Resumable dynamics sweep — seeds × depth × wobbly × holdout exposure pattern.

Backbone + fork mode (default): train and calibrate once per (seed, depth), then
fork for each (pattern, wobbly) cell — only exposure + mining differ.

Usage::

    EMERGENT_SWEEP_MODE=1 EMERGENT_FAST_TRAINING=1 \\
        uv run python research/experiments/sweep_dynamics.py \\
        --preset explore --output research/results/sweeps/explore.csv

Use ``--quiet`` to silence curriculum progress (``TRAIN_PROGRESS=0``) and step logs.

Progress streams to **stdout** when ``EMERGENT_SWEEP_MODE=1`` (curriculum phases,
backbone train, per-cell fork/mine steps).

Append/resume: re-run with same --output skips completed rows (seed+depth+pattern+wobbly).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("EMERGENT_SWEEP_MODE", "1")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")


def _force_line_buffered() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except (OSError, ValueError):
                pass

from neural_assemblies.assembly_calculus.emergent.evaluation.checkpoint import (
    ParserCheckpoint,
    backbone_cache_path,
    build_parser_backbone,
    load_backbone_cache,
    save_backbone_cache,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
    default_holdout_set,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
    DEFAULT_K,
    DEFAULT_N,
    delta_mining_sentences,
    ingest_holdout_pattern,
    reset_sweep_log,
    run_delta_cell,
    sweep_log,
)

HOLDOUT = "small"
PROBE_SENT = ["the", "bird", "finds", HOLDOUT]
FIELDNAMES = [
    "seed",
    "depth",
    "holdout_pattern",
    "wobbly_bootstrap",
    "train_s",
    "calibration_s",
    "fork_s",
    "n400_cohens_d",
    "p600_cohens_d",
    "p600_ready",
    "holdout_acc",
    "holdout_small_cat",
    "wobble_rate",
    "wobbly_episodes",
    "cache_hit",
]

PRESETS = {
    "explore": {
        "seeds": [42, 43, 44],
        "depths": ["TWO_WORD"],
        "patterns": ["balanced", "subject_only", "object_only"],
        "n": 300,
        "k": 8,
    },
    "standard": {
        "seeds": [42, 43, 44],
        "depths": ["TWO_WORD", "SENTENCES"],
        "patterns": ["balanced", "subject_only", "object_only"],
        "n": 300,
        "k": 8,
    },
    "publish": {
        "seeds": [42, 43, 44, 45, 46],
        "depths": ["SENTENCES"],
        "patterns": ["balanced", "subject_only", "object_only"],
        "n": DEFAULT_N,
        "k": DEFAULT_K,
    },
    "gpu": {
        "seeds": [42],
        "depths": ["TWO_WORD"],
        "patterns": ["balanced"],
        "n": 1_000_000,
        "k": 100,
    },
}


def _row_key(row: Dict[str, str]) -> tuple:
    return (
        row["seed"],
        row["depth"],
        row["holdout_pattern"],
        row["wobbly_bootstrap"],
    )


def _load_completed(path: Path) -> Set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {_row_key(r) for r in reader}


def _cell_post_mine(
    parser,
    *,
    pattern: str,
    wobbly: bool,
    mined_probes: list,
    mem,
    holdout: Set[str],
) -> Dict[str, object]:
    """Shared post-mining steps for full-retrain and delta cells."""
    from neural_assemblies.assembly_calculus.emergent.acquisition import (
        classify_word_bootstrapped,
        infer_holdout_categories,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.erp import (
        assess_erp_readiness,
    )

    if wobbly and mem.episodes:
        from neural_assemblies.assembly_calculus.emergent.acquisition.wobbly.replay import (
            replay_wobbly_episodes,
        )
        replay_wobbly_episodes(parser, mem)

    infer_holdout_categories(parser, holdout)
    cat, _ = classify_word_bootstrapped(parser, HOLDOUT)
    readiness = assess_erp_readiness(parser)
    wobble = (
        sum(1 for p in mined_probes if p.wobbly) / len(mined_probes)
        if mined_probes else 0.0
    )
    expected = {"small": "ADJ", "bird": "NOUN", "finds": "VERB"}
    correct = sum(
        1 for w in holdout
        if w in expected
        and classify_word_bootstrapped(parser, w)[0] == expected[w]
    )
    total = sum(1 for w in holdout if w in expected)
    return {
        "holdout_acc": round(correct / max(1, total), 4),
        "holdout_small_cat": cat,
        "p600_ready": readiness.p600_ready,
        "wobble_rate": round(wobble, 4),
        "wobbly_episodes": len(mem.episodes),
    }


def run_cell_full_retrain(
    *,
    seed: int,
    depth: str,
    pattern: str,
    wobbly: bool,
    n: int,
    k: int,
) -> Dict[str, object]:
    """Legacy path: full train + calibrate per cell (parity / regression)."""
    from neural_assemblies.assembly_calculus.emergent.acquisition import (
        mine_wobbly_episodes,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.erp import (
        ensure_parser_erp_calibration,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        train_parser_to_depth,
    )

    holdout = default_holdout_set()
    t0 = time.perf_counter()
    parser = train_parser_to_depth(
        depth, n=n, k=k, seed=seed, holdout_words=holdout, fast_training=True,
    )
    train_s = time.perf_counter() - t0

    t_cal = time.perf_counter()
    cal_report = ensure_parser_erp_calibration(parser, fast=True)
    calibration_s = time.perf_counter() - t_cal

    ingest_holdout_pattern(parser, pattern, holdout=HOLDOUT)

    mined_probes: list = []
    mem = mine_wobbly_episodes(
        parser,
        delta_mining_sentences(HOLDOUT, probe_sent=PROBE_SENT),
        target_words={HOLDOUT},
        probe_depth="mining",
        collected_probes=mined_probes,
    )
    post = _cell_post_mine(
        parser,
        pattern=pattern,
        wobbly=wobbly,
        mined_probes=mined_probes,
        mem=mem,
        holdout=holdout,
    )

    return {
        "seed": seed,
        "depth": depth,
        "holdout_pattern": pattern,
        "wobbly_bootstrap": wobbly,
        "train_s": round(train_s, 3),
        "calibration_s": round(calibration_s, 3),
        "fork_s": 0.0,
        "n400_cohens_d": round(cal_report.separation.get("n400_cohens_d", 0.0), 4),
        "p600_cohens_d": round(cal_report.separation.get("p600_cohens_d", 0.0), 4),
        "cache_hit": False,
        **post,
    }


def _load_or_build_backbone(
    depth: str,
    seed: int,
    n: int,
    k: int,
    *,
    show_progress: bool,
    backbone_cache_dir: Optional[Path],
) -> ParserCheckpoint:
    holdout = frozenset(default_holdout_set())
    if backbone_cache_dir is not None:
        cp_path = backbone_cache_path(
            backbone_cache_dir,
            depth,
            seed=seed,
            n=n,
            k=k,
            holdout_words=holdout,
        )
        cached = load_backbone_cache(cp_path)
        if cached is not None:
            sweep_log(
                f"backbone cache hit seed={seed} depth={depth} "
                f"n={n} k={k} ({cp_path.name})",
            )
            return cached

    sweep_log(
        f"backbone train start seed={seed} depth={depth} n={n:,} k={k}",
    )
    backbone = build_parser_backbone(
        depth, seed=seed, n=n, k=k,
        fast_training=True, calibrate=True,
        show_progress=show_progress,
    )
    if backbone_cache_dir is not None:
        cp_path = backbone_cache_path(
            backbone_cache_dir,
            depth,
            seed=seed,
            n=n,
            k=k,
            holdout_words=holdout,
        )
        save_backbone_cache(backbone, cp_path)
        sweep_log(f"backbone cached ({cp_path.name})")
    sweep_log(
        f"backbone train done {backbone.train_seconds:.1f}s "
        f"cal={backbone.calibration_seconds:.1f}s "
        f"engine={getattr(backbone.parser, 'engine_name', '?')}",
    )
    return backbone


def run_sweep(
    *,
    seeds: Iterable[int],
    depths: Iterable[str],
    patterns: Iterable[str],
    wobbly_flags: Iterable[bool],
    output: Path,
    n: int = DEFAULT_N,
    k: int = DEFAULT_K,
    full_retrain: bool = False,
    workers: int = 1,
    show_progress: bool = True,
    backbone_cache_dir: Optional[Path] = None,
) -> None:
    reset_sweep_log()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(output)
    write_header = not output.exists()

    seed_list = list(seeds)
    depth_list = list(depths)
    pattern_list = list(patterns)
    wobbly_list = list(wobbly_flags)
    total_cells = (
        len(seed_list) * len(depth_list)
        * len(pattern_list) * len(wobbly_list)
    )

    pending_serial: list = []
    pending_parallel: list = []

    for seed in seed_list:
        for depth in depth_list:
            cells: List[tuple] = []
            for pattern in pattern_list:
                for wobbly in wobbly_list:
                    key = (str(seed), depth, pattern, str(wobbly))
                    if key in completed:
                        continue
                    cells.append((pattern, wobbly))
            if not cells:
                continue
            if full_retrain:
                for pattern, wobbly in cells:
                    pending_serial.append((seed, depth, pattern, wobbly))
            else:
                pending_parallel.append((depth, seed, n, k, cells))

    rows_to_write: List[Dict[str, object]] = []
    total_new_cells = (
        len(pending_serial)
        + sum(len(cells) for _, _, _, _, cells in pending_parallel)
    )
    cell_num = 0
    sweep_log(
        f"grid: {total_new_cells} new cells "
        f"({len(completed)} rows already in {output.name})",
    )

    for seed, depth, pattern, wobbly in pending_serial:
        rows_to_write.append(
            run_cell_full_retrain(
                seed=seed, depth=depth, pattern=pattern,
                wobbly=wobbly, n=n, k=k,
            ),
        )

    if pending_parallel:
        cache_dir_s = (
            str(backbone_cache_dir) if backbone_cache_dir is not None else None
        )
        batches = [
            (depth, seed, bk_n, bk_k, cells, HOLDOUT, cache_dir_s)
            for depth, seed, bk_n, bk_k, cells in pending_parallel
        ]
        if workers > 1 and len(batches) > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from research.experiments.sweep_parallel import run_backbone_batch

            mp.freeze_support()
            print(
                f"  parallel mode: {len(batches)} backbone batches, "
                f"workers={min(workers, len(batches))}",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=min(workers, len(batches)),
            ) as ex:
                for batch_rows in ex.map(run_backbone_batch, batches):
                    rows_to_write.extend(batch_rows)
        else:
            backbones: Dict[tuple, ParserCheckpoint] = {}
            for depth, seed, bk_n, bk_k, cells in pending_parallel:
                backbone_key = (seed, depth, bk_n, bk_k)
                if backbone_key in backbones:
                    backbone = backbones[backbone_key]
                    sweep_log(
                        f"reuse backbone seed={seed} depth={depth} "
                        f"({len(cells)} cells)",
                    )
                else:
                    t_bb = time.perf_counter()
                    backbone = _load_or_build_backbone(
                        depth, seed, bk_n, bk_k,
                        show_progress=show_progress,
                        backbone_cache_dir=backbone_cache_dir,
                    )
                    backbones[backbone_key] = backbone
                    sweep_log(
                        f"backbone wall {time.perf_counter() - t_bb:.1f}s "
                        f"seed={seed} depth={depth}",
                    )
                for pattern, wobbly in cells:
                    cell_num += 1
                    sweep_log(
                        f"cell {cell_num}/{total_new_cells} "
                        f"seed={seed} depth={depth} pattern={pattern} "
                        f"wobbly={wobbly}",
                    )
                    delta = run_delta_cell(
                        backbone,
                        pattern=pattern,
                        wobbly=wobbly,
                        holdout=HOLDOUT,
                        probe_sent=PROBE_SENT,
                        log_steps=show_progress,
                    )
                    rows_to_write.append({
                        "seed": seed,
                        "depth": depth,
                        "holdout_pattern": pattern,
                        "wobbly_bootstrap": wobbly,
                        **delta,
                    })

    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows_to_write:
            writer.writerow({k: row[k] for k in FIELDNAMES})
            f.flush()
            print(
                f"  seed={row['seed']} depth={row['depth']} "
                f"pattern={row['holdout_pattern']} wobbly={row['wobbly_bootstrap']} "
                f"train={row['train_s']}s fork={row.get('fork_s', 0)}s "
                f"acc={row['holdout_acc']}",
                flush=True,
            )

    skipped = total_cells - len(rows_to_write)
    print(f"\nSweep done: {len(rows_to_write)} new rows, {skipped} skipped", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Emergent parser dynamics sweep")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--depths", nargs="+", default=None)
    p.add_argument("--patterns", nargs="+", default=None)
    p.add_argument("--wobbly", dest="wobbly_flags", action="store_true", default=None)
    p.add_argument("--no-wobbly", dest="wobbly_flags", action="store_false")
    p.set_defaults(wobbly_flags=None)
    p.add_argument("-n", type=int, default=None)
    p.add_argument("-k", type=int, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/sweeps/dynamics.csv"),
    )
    p.add_argument(
        "--full-retrain",
        action="store_true",
        help="Retrain from scratch per cell (slow; parity baseline)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process pool size for parallel backbone batches (default 1=serial). "
        "Use 3-4 on multi-core CPUs; not threads.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Silence curriculum progress and per-step sweep logs",
    )
    p.add_argument(
        "--backbone-cache-dir",
        type=Path,
        default=None,
        help="Directory to load/save backbone checkpoints (skip retrain on resume)",
    )
    args = p.parse_args()

    _force_line_buffered()

    if args.quiet:
        os.environ["TRAIN_PROGRESS"] = "0"
        os.environ["SWEEP_LOG"] = "0"
    else:
        os.environ.setdefault("TRAIN_PROGRESS", "1")
        os.environ.setdefault("SWEEP_LOG", "1")

    cfg = PRESETS.get(args.preset or "", {})
    seeds = args.seeds or cfg.get("seeds", [42, 43, 44])
    depths = args.depths or cfg.get("depths", ["TWO_WORD", "SENTENCES"])
    patterns = args.patterns or cfg.get("patterns", ["balanced", "subject_only", "object_only"])
    n = args.n if args.n is not None else cfg.get("n", DEFAULT_N)
    k = args.k if args.k is not None else cfg.get("k", DEFAULT_K)

    wobbly_flags: List[bool]
    if args.wobbly_flags is None:
        wobbly_flags = [False, True]
    else:
        wobbly_flags = [args.wobbly_flags]

    mode = "full-retrain" if args.full_retrain else "backbone+fork"
    par = f", workers={args.workers}" if args.workers > 1 and not args.full_retrain else ""
    from neural_assemblies.assembly_calculus.emergent.training.perf import (
        resolve_engine,
        warn_engine_scale_mismatch,
    )

    engine = resolve_engine("auto", n_hint=n)
    warn_engine_scale_mismatch(engine, n, where="sweep_dynamics")
    print(
        f"Dynamics sweep ({mode}{par}): {len(seeds)} seeds × {len(depths)} depths × "
        f"{len(patterns)} patterns × {len(wobbly_flags)} wobbly "
        f"= {len(seeds)*len(depths)*len(patterns)*len(wobbly_flags)} cells "
        f"(n={n:,}, k={k}, engine={engine})",
        flush=True,
    )
    show_progress = not args.quiet and args.workers == 1
    if args.workers > 1 and not args.quiet:
        print(
            "  note: per-stage progress disabled with --workers > 1 "
            "(use --workers 1 for live curriculum logs)",
            flush=True,
        )
    t0 = time.perf_counter()
    run_sweep(
        seeds=seeds,
        depths=depths,
        patterns=patterns,
        wobbly_flags=wobbly_flags,
        output=args.output,
        n=n,
        k=k,
        full_retrain=args.full_retrain,
        workers=max(1, args.workers),
        show_progress=show_progress,
        backbone_cache_dir=args.backbone_cache_dir,
    )
    print(f"Total wall time: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
