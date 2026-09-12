#!/usr/bin/env python3
"""Multi-seed comparison of training paths on the composition battery.

Usage::

    python research/experiments/compare_training_paths.py
    python research/experiments/compare_training_paths.py --seeds 42 43 44 45 46
    python research/experiments/compare_training_paths.py --paths developmental curriculum_only
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

from research.json_documents import write_new_document

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["EMERGENT_DEV_CURRICULUM"] = "1"

BATTERY_KEYS = (
    "science_score",
    "novel_easy",
    "novel_strain",
    "systematicity",
    "strain_gap",
    "holdout_bootstrap",
    "bridge_oov_top5",
    "bridge_direct_top5",
    "bridge_seen_top5",
    "dialogue",
    "composite",
    "train_seconds",
)


def _run_developmental(n: int, k: int, seed: int, holdout: set) -> Dict[str, object]:
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
        gate_enforcement=True,
    )
    train_s = time.perf_counter() - t0
    battery = evaluate_composition_battery(parser, seed=seed)
    row = {"path": "developmental", "seed": seed, "train_seconds": round(train_s, 3)}
    row.update(battery)
    row["blocked_at_stage"] = report.blocked_at_stage
    row["remedial_sentences"] = sum(
        float(r.metrics.get("remedial_sentences", 0.0))
        for r in report.reflections
    )
    return row


def _run_curriculum_only(n: int, k: int, seed: int, holdout: set) -> Dict[str, object]:
    from neural_assemblies.assembly_calculus.emergent import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.evaluation.composition_battery import (
        evaluate_composition_battery,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        train_parser_to_depth,
    )

    t0 = time.perf_counter()
    parser = train_parser_to_depth(
        "SENTENCES",
        n=n,
        k=k,
        seed=seed,
        holdout_words=holdout,
        vocabulary=build_vocabulary_preset("core"),
        fast_training=True,
    )
    train_s = time.perf_counter() - t0
    battery = evaluate_composition_battery(parser, seed=seed)
    row = {"path": "curriculum_only", "seed": seed, "train_seconds": round(train_s, 3)}
    row.update(battery)
    row["remedial_sentences"] = 0.0
    return row


def _run_chat_bootstrap(n: int, k: int, seed: int, holdout: set) -> Dict[str, object]:
    from neural_assemblies.assembly_calculus.emergent import build_vocabulary_preset
    from neural_assemblies.assembly_calculus.emergent.evaluation.composition_battery import (
        evaluate_composition_battery,
    )
    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        train_parser_to_depth,
    )

    t0 = time.perf_counter()
    parser = train_parser_to_depth(
        "DIALOGUE_FAST",
        n=n,
        k=k,
        seed=seed,
        holdout_words=holdout,
        vocabulary=build_vocabulary_preset("medium"),
        fast_training=True,
    )
    train_s = time.perf_counter() - t0
    battery = evaluate_composition_battery(parser, seed=seed)
    row = {"path": "chat_bootstrap", "seed": seed, "train_seconds": round(train_s, 3)}
    row.update(battery)
    row["remedial_sentences"] = 0.0
    return row


def _summarize_by_path(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    paths = sorted({str(r["path"]) for r in rows})
    summary: List[Dict[str, object]] = []
    for path in paths:
        path_rows = [r for r in rows if r["path"] == path]
        entry: Dict[str, object] = {"path": path, "n_seeds": len(path_rows)}
        for key in BATTERY_KEYS:
            if key == "train_seconds":
                vals = [float(r[key]) for r in path_rows]
            else:
                vals = [float(r.get(key, 0.0)) for r in path_rows]  # type: ignore[arg-type]
            if not vals:
                continue
            entry[f"{key}_mean"] = statistics.mean(vals)
            if len(vals) > 1:
                entry[f"{key}_std"] = statistics.stdev(vals)
        summary.append(entry)
    summary.sort(key=lambda r: float(r.get("science_score_mean", 0.0)), reverse=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-seed training-path comparison on composition battery.",
    )
    parser.add_argument("-n", type=int, default=3000)
    parser.add_argument("-k", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/dev_runs/compare_battery.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("research/results/dev_runs/compare_battery.csv"),
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        choices=["developmental", "curriculum_only", "chat_bootstrap"],
        default=["developmental", "curriculum_only", "chat_bootstrap"],
    )
    args = parser.parse_args()

    from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (
        default_holdout_set,
    )
    from neural_assemblies.assembly_calculus.emergent.training.perf import (
        resolve_engine,
        warn_engine_scale_mismatch,
    )

    engine = resolve_engine("auto", n_hint=args.n)
    warn_engine_scale_mismatch(engine, args.n, where="compare_training_paths")
    holdout = default_holdout_set()

    runners = {
        "developmental": _run_developmental,
        "curriculum_only": _run_curriculum_only,
        "chat_bootstrap": _run_chat_bootstrap,
    }

    rows: List[Dict[str, object]] = []
    t_all = time.perf_counter()
    for seed in args.seeds:
        for name in args.paths:
            print(f"\n=== {name} seed={seed} ===", flush=True)
            row = runners[name](args.n, args.k, seed, holdout)
            rows.append(row)
            print(
                f"  science={row['science_score']:.3f} "
                f"strain={row['novel_strain']:.1%} "
                f"bridge_oov={row['bridge_oov_top5']:.1%} "
                f"bridge_dir={row.get('bridge_direct_top5', 0):.1%} "
                f"train={row['train_seconds']}s "
                f"remedial={row.get('remedial_sentences', 0):.0f}",
                flush=True,
            )
            if row.get("blocked_at_stage"):
                print(f"  BLOCKED at {row['blocked_at_stage']}", flush=True)

    summary = _summarize_by_path(rows)
    payload = {
        "n": args.n,
        "k": args.k,
        "seeds": args.seeds,
        "engine": engine,
        "wall_seconds": round(time.perf_counter() - t_all, 3),
        "rows": rows,
        "summary": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new_document(args.output, payload)

    fieldnames = ["path", "seed"] + list(BATTERY_KEYS) + ["blocked_at_stage", "remedial_sentences"]
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {args.output} and {args.csv}", flush=True)
    print("\nSummary (science_score_mean):", flush=True)
    for entry in summary:
        mean = float(entry.get("science_score_mean", 0.0))
        std = entry.get("science_score_std")
        std_s = f" ± {std:.3f}" if std is not None else ""
        train_m = float(entry.get("train_seconds_mean", 0.0))
        print(
            f"  {entry['path']}: {mean:.3f}{std_s} "
            f"(train {train_m:.1f}s, n={entry['n_seeds']})",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
