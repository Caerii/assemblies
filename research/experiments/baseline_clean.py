#!/usr/bin/env python3
"""Clean baseline benchmark — run after reboot with minimal background load.

Usage:
    TRAIN_PROGRESS=0 EMERGENT_FAST_TRAINING=1 uv run python research/experiments/baseline_clean.py
"""

from __future__ import annotations

import platform
import statistics
import sys
import time

from neural_assemblies.assembly_calculus.emergent import (
    EmergentParser,
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.acquisition.wobbly.mining import (
    mine_wobbly_episodes,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import NOUN_CORE
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp import (
    calibrate_erp_thresholds,
    ensure_parser_erp_calibration,
    run_incremental_erp_probes,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
    train_parser_to_depth,
)
from neural_assemblies.assembly_calculus.ops import project
from neural_assemblies.core.engine import list_engines

N, K = 3000, 30


def main() -> None:
    print("=" * 62)
    print("CLEAN BASELINE BENCHMARK")
    print("=" * 62)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor:  {platform.processor()}")
    print(f"Engines:    {list_engines()}")
    print(f"Config:     n={N}, k={K}, EMERGENT_FAST_TRAINING=1, TRAIN_PROGRESS=0")
    print()

    # Micro: raw projection throughput
    p_micro = EmergentParser(n=N, k=K, seed=0, fast_training=True)
    p_micro.register_word("dog")
    t0 = time.perf_counter()
    for _ in range(100):
        project(p_micro.brain, p_micro.stim_map["dog"], NOUN_CORE, rounds=4)
    micro = time.perf_counter() - t0
    print(f"100x project(dog->NOUN_CORE, r=4)     {micro:7.2f}s  ({micro / 100 * 1000:.1f} ms/proj)")
    print()

    # Single training pass phase breakdown
    print("--- Single training pass (medium vocab) ---")
    sents = create_training_sentences()
    raw = [s.words for s in sents]
    p = EmergentParser(
        n=N, k=K, seed=42, fast_training=True,
        vocabulary=build_vocabulary_preset("medium"),
    )
    phases: list[tuple[str, float]] = []
    t_total = time.perf_counter()

    t = time.perf_counter()
    for s in raw:
        p.ingest_raw_sentence(s)
    phases.append(("ingest", time.perf_counter() - t))

    t = time.perf_counter()
    p.train_lexicon(skip_known=False)
    phases.append(("lexicon", time.perf_counter() - t))

    for name, fn in [
        ("roles", lambda: p.train_roles(sents)),
        ("phrases", lambda: p.train_phrases(sents)),
        ("next_token", lambda: p.train_next_token(sents)),
    ]:
        t = time.perf_counter()
        fn()
        phases.append((name, time.perf_counter() - t))

    train_total = time.perf_counter() - t_total
    for name, sec in phases:
        print(f"  {name:12} {sec:6.2f}s ({100 * sec / train_total:5.1f}%)")
    print(f"  {'TOTAL':12} {train_total:6.2f}s")
    print(
        f"  rounds={p.rounds} infer={p.inference_rounds} "
        f"bridge={p.bridge_rounds} engine={p.brain._engine.__class__.__name__}"
    )
    print()

    # Curriculum depth (3-run mean)
    print("--- Curriculum depth (3-run mean, seeds 42-44) ---")
    for depth in ("TWO_WORD", "SENTENCES"):
        times = []
        for seed in (42, 43, 44):
            t0 = time.perf_counter()
            train_parser_to_depth(
                depth, n=N, k=K, seed=seed, holdout_words={"small"},
            )
            times.append(time.perf_counter() - t0)
        print(
            f"  train_parser_to_depth({depth:10})  "
            f"mean={statistics.mean(times):6.2f}s  "
            f"min={min(times):6.2f}s  max={max(times):6.2f}s"
        )
    print()

    # ERP / wobbly path
    print("--- ERP / wobbly path ---")
    p_erp = train_parser_to_depth(
        "SENTENCES", n=N, k=K, seed=42, holdout_words={"small"},
    )
    t0 = time.perf_counter()
    calibrate_erp_thresholds(p_erp, fast=False)
    print(f"calibrate_erp_thresholds (full)            {time.perf_counter() - t0:7.2f}s")

    p_fast = train_parser_to_depth(
        "SENTENCES", n=N, k=K, seed=99, holdout_words={"small"},
    )
    t0 = time.perf_counter()
    calibrate_erp_thresholds(p_fast, fast=True)
    print(f"calibrate_erp_thresholds (fast)            {time.perf_counter() - t0:7.2f}s")

    p_cached = train_parser_to_depth(
        "SENTENCES", n=N, k=K, seed=88, holdout_words={"small"},
    )
    ensure_parser_erp_calibration(p_cached, fast=True)
    t0 = time.perf_counter()
    ensure_parser_erp_calibration(p_cached, fast=True)
    print(f"ensure_parser_erp_calibration (cached)       {time.perf_counter() - t0:7.2f}s")

    probe_sents = [
        ["the", "small", "bird"],
        ["the", "bird", "runs"],
        ["the", "bird", "finds", "small"],
    ]
    ensure_parser_erp_calibration(p_erp)
    t0 = time.perf_counter()
    for s in probe_sents:
        run_incremental_erp_probes(p_erp, s, probe_depth="calibration")
    erp_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    for s in probe_sents:
        run_incremental_erp_probes(p_erp, s, probe_depth="mining")
    erp_mine = time.perf_counter() - t0
    print(f"ERP probes x3 (calibration depth)          {erp_full:7.2f}s  ({erp_full / len(probe_sents):.2f}s/sent)")
    print(f"ERP probes x3 (mining depth)               {erp_mine:7.2f}s  ({erp_mine / len(probe_sents):.2f}s/sent)")

    t0 = time.perf_counter()
    mine_wobbly_episodes(p_erp, probe_sents, target_words={"small"}, probe_depth="mining")
    print(f"mine_wobbly_episodes (3 sents)             {time.perf_counter() - t0:7.2f}s")

    print()
    print("=" * 62)
    print("Baseline complete.")


if __name__ == "__main__":
    main()
