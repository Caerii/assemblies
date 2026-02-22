"""
Parameter Exploration for Forward Prediction

Systematically varies parameters to understand the boundaries of the
category-level prediction effect (Hebbian bridges via co-projection).

Explorations:
  1. Training repetitions: 1, 3, 5, 10
  2. Beta (Hebbian rate): 0.05, 0.10, 0.15, 0.20
  3. Rounds per co-projection pair: 1, 3, 5, 10
  4. Area size (n): 2000, 5000, 10000

Usage:
    uv run python research/experiments/primitives/explore_forward_prediction_params.py
    uv run python research/experiments/primitives/explore_forward_prediction_params.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from research.experiments.primitives.test_forward_prediction import (
    PredictionConfig,
    run_trial,
)
import numpy as np
from research.experiments.base import summarize, ttest_vs_null


def run_sweep_point(label, cfg, n_seeds, base_seed=42):
    """Run one parameter configuration and return summary metrics."""
    verb_adv_vals = []
    obj_adv_vals = []
    verb_correct_vals = []
    obj_correct_vals = []
    verb_top1_vals = []
    obj_top1_vals = []

    for s in range(n_seeds):
        result = run_trial(cfg, base_seed + s)
        verb_adv_vals.append(result["verb_pos_advantage"])
        obj_adv_vals.append(result["obj_pos_advantage"])
        verb_correct_vals.append(result["verb_pos_correct_mean"])
        obj_correct_vals.append(result["obj_pos_correct_mean"])
        verb_top1_vals.append(result["verb_pos_top1_acc"])
        obj_top1_vals.append(result["obj_pos_top1_acc"])

    verb_test = ttest_vs_null(verb_adv_vals, 0.0)
    obj_test = ttest_vs_null(obj_adv_vals, 0.0)

    # Combined advantage (mean of verb + object)
    combined_adv = [(v + o) / 2 for v, o in zip(verb_adv_vals, obj_adv_vals)]
    combined_test = ttest_vs_null(combined_adv, 0.0)

    return {
        "label": label,
        "config": asdict(cfg),
        "verb_advantage": summarize(verb_adv_vals),
        "verb_advantage_test": verb_test,
        "verb_correct_overlap": summarize(verb_correct_vals),
        "verb_top1": summarize(verb_top1_vals),
        "obj_advantage": summarize(obj_adv_vals),
        "obj_advantage_test": obj_test,
        "obj_correct_overlap": summarize(obj_correct_vals),
        "obj_top1": summarize(obj_top1_vals),
        "combined_advantage": summarize(combined_adv),
        "combined_test": combined_test,
        "summary": {
            "verb_adv": float(np.mean(verb_adv_vals)),
            "obj_adv": float(np.mean(obj_adv_vals)),
            "combined_adv": float(np.mean(combined_adv)),
            "verb_top1": float(np.mean(verb_top1_vals)),
            "obj_top1": float(np.mean(obj_top1_vals)),
            "d": combined_test["d"],
            "p": combined_test["p"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parameter exploration for forward prediction")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        n_seeds = 5
        base_n = 5000
        base_k = 50
    else:
        n_seeds = 10
        base_n = 10000
        base_k = 100

    results = {}

    # Baseline
    print("=" * 70)
    print("BASELINE")
    print("=" * 70)
    base_cfg = PredictionConfig(n=base_n, k=base_k)
    r = run_sweep_point("baseline", base_cfg, n_seeds)
    results["baseline"] = r
    s = r["summary"]
    print(f"  VerbAdv={s['verb_adv']:.4f}  ObjAdv={s['obj_adv']:.4f}  "
          f"VerbT1={s['verb_top1']:.2f}  ObjT1={s['obj_top1']:.2f}  "
          f"d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 1: Training repetitions
    print("\n" + "=" * 70)
    print("SWEEP: Training repetitions")
    print("=" * 70)
    for reps in [1, 5, 10, 20]:
        label = f"reps_{reps}"
        cfg = PredictionConfig(n=base_n, k=base_k, training_reps=reps)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  reps={reps:2d}: VerbAdv={s['verb_adv']:.4f}  "
              f"ObjAdv={s['obj_adv']:.4f}  "
              f"VerbT1={s['verb_top1']:.2f}  ObjT1={s['obj_top1']:.2f}  "
              f"d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 2: Beta
    print("\n" + "=" * 70)
    print("SWEEP: Beta (Hebbian learning rate)")
    print("=" * 70)
    for beta in [0.05, 0.15, 0.20, 0.30]:
        label = f"beta_{beta}"
        cfg = PredictionConfig(n=base_n, k=base_k, beta=beta)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  beta={beta:.2f}: VerbAdv={s['verb_adv']:.4f}  "
              f"ObjAdv={s['obj_adv']:.4f}  "
              f"VerbT1={s['verb_top1']:.2f}  ObjT1={s['obj_top1']:.2f}  "
              f"d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 3: Rounds per co-projection pair
    print("\n" + "=" * 70)
    print("SWEEP: Rounds per co-projection pair")
    print("=" * 70)
    for rounds in [1, 3, 10, 15]:
        label = f"rounds_{rounds}"
        cfg = PredictionConfig(n=base_n, k=base_k, train_rounds_per_pair=rounds)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  rounds={rounds:2d}: VerbAdv={s['verb_adv']:.4f}  "
              f"ObjAdv={s['obj_adv']:.4f}  "
              f"VerbT1={s['verb_top1']:.2f}  ObjT1={s['obj_top1']:.2f}  "
              f"d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 4: Area size
    print("\n" + "=" * 70)
    print("SWEEP: Area size (n)")
    print("=" * 70)
    for area_n, area_k in [(2000, 20), (5000, 50), (20000, 200)]:
        label = f"n_{area_n}"
        cfg = PredictionConfig(n=area_n, k=area_k)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  n={area_n:5d} k={area_k:3d}: VerbAdv={s['verb_adv']:.4f}  "
              f"ObjAdv={s['obj_adv']:.4f}  "
              f"VerbT1={s['verb_top1']:.2f}  ObjT1={s['obj_top1']:.2f}  "
              f"d={s['d']:.2f}  p={s['p']:.4f}")

    # Print comparison table
    print("\n\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    header = (f"{'Config':<16}{'VerbAdv':>9}{'ObjAdv':>9}{'VT1':>6}{'OT1':>6}"
              f"{'d':>8}{'p':>10}")
    print(header)
    print("-" * 64)

    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<16}{s['verb_adv']:>9.4f}{s['obj_adv']:>9.4f}"
              f"{s['verb_top1']:>6.2f}{s['obj_top1']:>6.2f}"
              f"{s['d']:>8.2f}{s['p']:>10.4f}")

    # Save
    results_dir = Path(__file__).parent.parent.parent / "results" / "primitives"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_quick" if args.quick else ""
    output_path = results_dir / f"forward_prediction_exploration_{timestamp}{suffix}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
