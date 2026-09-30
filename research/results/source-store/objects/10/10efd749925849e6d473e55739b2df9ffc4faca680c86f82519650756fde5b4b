#!/usr/bin/env python3
"""Compare throughput and ERP quality across n/k scales.

Usage:
    EMERGENT_FAST_TRAINING=1 EMERGENT_ERP_FAST=1 TRAIN_PROGRESS=0 \\
        uv run python research/experiments/baseline_scale.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.evaluation.erp import (
    calibrate_erp_thresholds,
    run_incremental_erp_probes,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
    default_holdout_set,
    train_parser_to_depth,
)
from neural_assemblies.assembly_calculus.ops import project
from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import NOUN_CORE

SCALES = [
    (3000, 30),
    (1500, 20),
    (1000, 15),
    (750, 12),
    (500, 10),
    (300, 8),
]


def bench_scale(n: int, k: int, seed: int = 42) -> dict:
    holdout = default_holdout_set()
    row: dict = {"n": n, "k": k, "ok": True, "error": ""}

    try:
        t0 = time.perf_counter()
        p = EmergentParser(n=n, k=k, seed=seed, fast_training=True)
        p.register_word("dog")
        for _ in range(50):
            project(p.brain, p.stim_map["dog"], NOUN_CORE, rounds=4)
        row["micro_50proj_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        parser = train_parser_to_depth(
            "TWO_WORD", n=n, k=k, seed=seed, holdout_words=holdout,
        )
        row["two_word_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        parser = train_parser_to_depth(
            "SENTENCES", n=n, k=k, seed=seed, holdout_words=holdout,
        )
        row["sentences_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        report = calibrate_erp_thresholds(parser, fast=True)
        row["calibration_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        _, probes = run_incremental_erp_probes(
            parser, ["the", "bird", "finds", "small"], probe_depth="mining",
        )
        row["probe_s"] = round(time.perf_counter() - t0, 3)

        row["p600_ready"] = report.readiness.p600_ready
        row["p600_d"] = round(report.separation.get("p600_cohens_d", 0.0), 3)
        row["wobbly"] = any(p.wobbly for p in probes)
        row["total_s"] = round(
            row["sentences_s"] + row["calibration_s"] + row["probe_s"], 3,
        )
    except Exception as exc:
        row["ok"] = False
        row["error"] = str(exc)[:80]

    return row


def main() -> None:
    print("=" * 72)
    print("SCALE BENCHMARK (TWO_WORD + SENTENCES + fast calibration + probe)")
    print("=" * 72)
    print(f"{'n':>6} {'k':>4} {'50proj':>8} {'TWO_WORD':>9} {'SENTENCES':>10} "
          f"{'cal':>6} {'probe':>6} {'total':>7} {'p600_r':>7} {'p600_d':>7} {'ok':>4}")
    print("-" * 72)

    baseline_total = None
    for n, k in SCALES:
        row = bench_scale(n, k)
        if not row["ok"]:
            print(f"{n:>6} {k:>4}  FAILED: {row['error']}")
            continue
        if baseline_total is None:
            baseline_total = row["total_s"]
        speedup = baseline_total / row["total_s"] if row["total_s"] else 0
        print(
            f"{n:>6} {k:>4} "
            f"{row['micro_50proj_s']:>7.2f}s "
            f"{row['two_word_s']:>8.2f}s "
            f"{row['sentences_s']:>9.2f}s "
            f"{row['calibration_s']:>5.2f}s "
            f"{row['probe_s']:>5.2f}s "
            f"{row['total_s']:>6.2f}s "
            f"{str(row['p600_ready']):>7} "
            f"{row['p600_d']:>7.2f} "
            f"{'Y' if row['p600_ready'] else 'N':>4}  "
            f"({speedup:.1f}x vs n=3000)",
        )

    print("=" * 72)
    print("total = SENTENCES train + fast calibration + one mining probe sentence")
    print("Use smallest n/k where p600_ready=Y and p600_d > 0.3 for sweeps.")


if __name__ == "__main__":
    main()
