"""Reproducible operational throughput benchmark for projection.

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
import time

from neural_assemblies.assembly_calculus import project
from neural_assemblies.core.brain import Brain


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
    }


def benchmark(*, engine: str, sizes: list[int], k: int, rounds: int,
              seeds: list[int]) -> dict:
    if not seeds or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("throughput benchmark requires at least three unique seeds")
    if any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError("sizes must be positive integers")
    if type(k) is not int or k <= 0 or type(rounds) is not int or rounds <= 0:
        raise ValueError("k and rounds must be positive integers")
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
        "model_semantics": resolved_model,
        "seeds": seeds,
        "cells": cells,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "recorded_utc": datetime.now(timezone.utc).isoformat()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("numpy_exact", "numpy_sparse"), default="numpy_sparse")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 2_000, 5_000])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, help="new JSON path; existing files are rejected")
    args = parser.parse_args(argv)
    result = benchmark(engine=args.engine, sizes=args.sizes, k=args.k,
                       rounds=args.rounds, seeds=args.seeds)
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
