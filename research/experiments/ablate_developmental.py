#!/usr/bin/env python3
"""Ablate developmental training components (science discovery).

Usage::

    python research/experiments/ablate_developmental.py --seed 42
    python research/experiments/ablate_developmental.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["EMERGENT_DEV_CURRICULUM"] = "1"


ABLATIONS = {
    "full": {},
    "no_babble": {"babble": False},
    "no_fuzzy": {"fuzzy_early_words": False},
    "no_adaptive": {"adaptive": False},
    "no_gates": {"gate_enforcement": False},
    "no_babble_fuzzy": {"babble": False, "fuzzy_early_words": False},
    "no_sleep": {"inter_stage_sleep_enabled": False},
}


def run_ablation(
    name: str,
    seed: int,
    n: int,
    k: int,
    holdout: set,
    **kwargs,
) -> Dict[str, object]:
    from neural_assemblies.assembly_calculus.emergent import (
        EmergentParser,
        build_vocabulary_preset,
    )
    from neural_assemblies.assembly_calculus.emergent.acquisition import (
        run_developmental_acquisition,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.composition_battery import (
        evaluate_composition_battery,
    )

    vocab = build_vocabulary_preset("core")
    parser = EmergentParser(
        n=n, k=k, seed=seed, vocabulary=vocab, fast_training=True,
    )
    t0 = time.perf_counter()
    report = run_developmental_acquisition(
        parser,
        max_stage="SENTENCES",
        holdout_words=holdout,
        seed=seed,
        **kwargs,
    )
    train_s = time.perf_counter() - t0
    battery = evaluate_composition_battery(parser, seed=seed)
    return {
        "ablation": name,
        "seed": seed,
        "train_seconds": round(train_s, 3),
        "blocked_at_stage": report.blocked_at_stage,
        "remedial_sentences": sum(
            float(r.metrics.get("remedial_sentences", 0.0))
            for r in report.reflections
        ),
        **{k: battery[k] for k in (
            "science_score", "novel_easy", "novel_strain",
            "systematicity", "bridge_oov_top5", "bridge_direct_top5",
            "bridge_seen_top5", "holdout_bootstrap",
        )},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Developmental ablation study.")
    parser.add_argument("-n", type=int, default=3000)
    parser.add_argument("-k", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/dev_runs/ablate_developmental.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("research/results/dev_runs/ablate_developmental.csv"),
    )
    args = parser.parse_args()

    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        default_holdout_set,
    )

    holdout = default_holdout_set()
    rows: List[Dict[str, object]] = []

    for seed in args.seeds:
        for name, flags in ABLATIONS.items():
            print(f"\n=== {name} seed={seed} ===", flush=True)
            row = run_ablation(
                name, seed, args.n, args.k, holdout, **flags,
            )
            rows.append(row)
            print(
                f"  science={row['science_score']:.3f} "
                f"strain={row['novel_strain']:.1%} "
                f"bridge_oov={row['bridge_oov_top5']:.1%} "
                f"train={row['train_seconds']}s "
                f"remedial={row['remedial_sentences']:.0f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {args.output} and {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
