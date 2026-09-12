"""Reproducible operational throughput benchmark for projection.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-operational-benchmark

This is a performance diagnostic, not scientific evidence.  It reports one
timing per seed and summary quantiles so a single fast or slow process cannot
look like a stable performance claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from neural_assemblies.assembly_calculus import project
from neural_assemblies.core.brain import Brain


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90 = statistics.quantiles(ordered, n=10, method="inclusive")[8]
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": p90,
        "max": ordered[-1],
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _materialization_storage(engine: str) -> str:
    """Return a representation the selected backend actually implements."""
    return "csr" if engine == "torch_sparse" else "dense"


def benchmark(*, engine: str, sizes: list[int], k: int, rounds: int,
              seeds: list[int], materialize: bool = True,
              warmup: bool = True) -> dict:
    if not seeds or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("throughput benchmark requires at least three unique seeds")
    if any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError("sizes must be positive integers")
    if type(k) is not int or k <= 0 or type(rounds) is not int or rounds <= 0:
        raise ValueError("k and rounds must be positive integers")
    if type(warmup) is not bool:
        raise ValueError("warmup must be a boolean")
    storage = _materialization_storage(engine)
    if warmup:
        warm = Brain(p=0.05, seed=0, engine=engine, save_winners=True)
        warm.add_stimulus("stimulus", k)
        warm.add_area("target", max(k, 40), k, beta=0.1)
        if materialize:
            warm.materialize_area("target", storage=storage)
        project(warm, "stimulus", "target", rounds=1, recurrent=True)
    cells = []
    resolved_model = None
    for n in sizes:
        rows = []
        for seed in seeds:
            brain = Brain(p=0.05, seed=seed, engine=engine, save_winners=True)
            current_model = brain.model_semantics.to_dict()
            if resolved_model is None:
                resolved_model = current_model
            elif current_model != resolved_model:
                raise RuntimeError("benchmark cells resolved different model semantics")
            brain.add_stimulus("stimulus", k)
            brain.add_area("target", n, k, beta=0.1)
            if materialize:
                brain.materialize_area("target", storage=storage)
            started = time.perf_counter()
            project(brain, "stimulus", "target", rounds=rounds, recurrent=True)
            elapsed = time.perf_counter() - started
            rows.append({
                "seed": seed,
                "seconds": elapsed,
                "rounds_per_second": rounds / elapsed if elapsed else float("inf"),
            })
        rates = [row["rounds_per_second"] for row in rows]
        cells.append({"n": n, "k": k, "rounds": rounds,
                      "per_seed": rows, "rounds_per_second": _quantiles(rates)})
    return {
        "benchmark": "projection-throughput-v1",
        "status": "diagnostic",
        "engine": engine,
        "storage": "materialized" if materialize else "sampled",
        "warmup": warmup,
        "model_semantics": resolved_model,
        "seeds": seeds,
        "cells": cells,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "git_commit": _git_commit(),
                    "recorded_utc": datetime.now(timezone.utc).isoformat()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("numpy_exact", "numpy_sparse"), default="numpy_sparse")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 2_000, 5_000])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--sampled", action="store_true",
                        help="opt into lazy sampled storage; avoid for recurrent science")
    parser.add_argument("--no-warmup", action="store_true",
                        help="include first-use initialization in the timing")
    parser.add_argument("--output", type=Path, help="new JSON path; existing files are rejected")
    args = parser.parse_args(argv)
    result = benchmark(engine=args.engine, sizes=args.sizes, k=args.k,
                       rounds=args.rounds, seeds=args.seeds,
                       materialize=not args.sampled, warmup=not args.no_warmup)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
