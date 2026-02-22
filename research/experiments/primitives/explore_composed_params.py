"""
Parameter Exploration for Composed ERP Pipeline

Systematically varies parameters to understand the boundaries of the
N400/P600 double dissociation effect from the composed prediction+binding
pipeline.

Explorations:
  1. Training repetitions: 1, 3, 5, 10
  2. Beta (Hebbian rate): 0.05, 0.10, 0.15, 0.20
  3. Binding rounds: 5, 10, 20
  4. Prediction rounds per pair: 1, 3, 5
  5. Area size (n): 5000, 10000, 15000
  6. Settling rounds (P600 measurement): 3, 5, 10

Usage:
    uv run python research/experiments/primitives/explore_composed_params.py
    uv run python research/experiments/primitives/explore_composed_params.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from research.experiments.primitives.test_composed_erp import (
    ComposedConfig,
    run_trial,
)
import numpy as np
from research.experiments.base import summarize, paired_ttest


def run_sweep_point(label, cfg, n_seeds, base_seed=42):
    """Run one parameter configuration and return summary metrics."""
    n400_gram_vals = []
    n400_catviol_vals = []
    n400_novel_vals = []
    p600_gram_vals = []
    p600_catviol_vals = []
    p600_novel_vals = []

    for s in range(n_seeds):
        result = run_trial(cfg, base_seed + s)
        n400_gram_vals.append(result["n400_gram_mean"])
        n400_catviol_vals.append(result["n400_catviol_mean"])
        n400_novel_vals.append(result["n400_novel_mean"])
        p600_gram_vals.append(result["p600_gram_mean"])
        p600_catviol_vals.append(result["p600_catviol_mean"])
        p600_novel_vals.append(result["p600_novel_mean"])

    # N400 tests
    n400_cv_g = paired_ttest(n400_catviol_vals, n400_gram_vals)
    n400_nv_g = paired_ttest(n400_novel_vals, n400_gram_vals)
    # P600 tests
    p600_cv_g = paired_ttest(p600_catviol_vals, p600_gram_vals)
    p600_nv_g = paired_ttest(p600_novel_vals, p600_gram_vals)

    return {
        "label": label,
        "config": asdict(cfg),
        "n400_gram": summarize(n400_gram_vals),
        "n400_catviol": summarize(n400_catviol_vals),
        "n400_novel": summarize(n400_novel_vals),
        "p600_gram": summarize(p600_gram_vals),
        "p600_catviol": summarize(p600_catviol_vals),
        "p600_novel": summarize(p600_novel_vals),
        "n400_catviol_vs_gram": n400_cv_g,
        "n400_novel_vs_gram": n400_nv_g,
        "p600_catviol_vs_gram": p600_cv_g,
        "p600_novel_vs_gram": p600_nv_g,
        "summary": {
            "n400_gram": float(np.mean(n400_gram_vals)),
            "n400_catviol": float(np.mean(n400_catviol_vals)),
            "n400_novel": float(np.mean(n400_novel_vals)),
            "p600_gram": float(np.mean(p600_gram_vals)),
            "p600_catviol": float(np.mean(p600_catviol_vals)),
            "p600_novel": float(np.mean(p600_novel_vals)),
            "n400_d": n400_cv_g["d"],
            "p600_d": p600_cv_g["d"],
        },
    }


def _print_row(label, s):
    """Print one row of the comparison table."""
    print(f"  {label:<20s} "
          f"N400: g={s['n400_gram']:.3f} c={s['n400_catviol']:.3f} "
          f"n={s['n400_novel']:.3f} d={s['n400_d']:+.2f}  |  "
          f"P600: g={s['p600_gram']:.3f} c={s['p600_catviol']:.3f} "
          f"n={s['p600_novel']:.3f} d={s['p600_d']:+.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter exploration for composed ERP pipeline")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        n_seeds = 3
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
    base_cfg = ComposedConfig(n=base_n, k=base_k)
    r = run_sweep_point("baseline", base_cfg, n_seeds)
    results["baseline"] = r
    _print_row("baseline", r["summary"])

    # Sweep 1: Training repetitions
    print("\n" + "=" * 70)
    print("SWEEP: Training repetitions")
    print("=" * 70)
    for reps in [1, 3, 5, 10]:
        label = f"reps_{reps}"
        cfg = ComposedConfig(n=base_n, k=base_k, training_reps=reps)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        _print_row(label, r["summary"])

    # Sweep 2: Beta
    print("\n" + "=" * 70)
    print("SWEEP: Beta (Hebbian learning rate)")
    print("=" * 70)
    for beta in [0.05, 0.10, 0.15, 0.20]:
        label = f"beta_{beta}"
        cfg = ComposedConfig(n=base_n, k=base_k, beta=beta)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        _print_row(label, r["summary"])

    # Sweep 3: Binding rounds
    print("\n" + "=" * 70)
    print("SWEEP: Binding rounds")
    print("=" * 70)
    for br in [5, 10, 20]:
        label = f"bind_{br}"
        cfg = ComposedConfig(n=base_n, k=base_k, binding_rounds=br)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        _print_row(label, r["summary"])

    # Sweep 4: Prediction rounds per pair
    print("\n" + "=" * 70)
    print("SWEEP: Prediction rounds per pair")
    print("=" * 70)
    for pr in [1, 3, 5]:
        label = f"pred_{pr}"
        cfg = ComposedConfig(n=base_n, k=base_k, train_rounds_per_pair=pr)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        _print_row(label, r["summary"])

    # Sweep 5: Area size
    print("\n" + "=" * 70)
    print("SWEEP: Area size (n)")
    print("=" * 70)
    for area_n, area_k in [(5000, 50), (10000, 100), (15000, 150)]:
        label = f"n_{area_n}"
        cfg = ComposedConfig(n=area_n, k=area_k)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        _print_row(label, r["summary"])

    # Sweep 6: Settling rounds (P600 measurement)
    print("\n" + "=" * 70)
    print("SWEEP: Settling rounds (P600)")
    print("=" * 70)
    for sr in [3, 5, 10]:
        label = f"settle_{sr}"
        cfg = ComposedConfig(n=base_n, k=base_k, n_settling_rounds=sr)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        _print_row(label, r["summary"])

    # Comparison table
    print("\n\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    header = (f"{'Config':<20s} "
              f"{'N400_g':>7}{'N400_c':>7}{'N400_n':>7}{'N4_d':>7} | "
              f"{'P600_g':>7}{'P600_c':>7}{'P600_n':>7}{'P6_d':>7}")
    print(header)
    print("-" * 88)

    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<20s} "
              f"{s['n400_gram']:>7.3f}{s['n400_catviol']:>7.3f}"
              f"{s['n400_novel']:>7.3f}{s['n400_d']:>+7.2f} | "
              f"{s['p600_gram']:>7.3f}{s['p600_catviol']:>7.3f}"
              f"{s['p600_novel']:>7.3f}{s['p600_d']:>+7.2f}")

    # Save
    results_dir = Path(__file__).parent.parent.parent / "results" / "primitives"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_quick" if args.quick else ""
    output_path = results_dir / f"composed_erp_exploration_{timestamp}{suffix}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
