"""
Parameter Exploration for Area Self-Organization

Systematically varies parameters to understand the boundaries of the
self-organization effect. All runs use Condition B (modality features)
since Condition A (specific features only) showed marginal results.

Explorations:
  1. Beta (Hebbian rate): 0.05, 0.10, 0.15, 0.20
  2. Training repetitions: 3, 5, 10, 20
  3. Number of core areas: 4, 6, 8, 10, 12

Usage:
    uv run python research/experiments/primitives/explore_self_org_params.py
    uv run python research/experiments/primitives/explore_self_org_params.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from research.experiments.primitives.test_area_self_organization import (
    SelfOrganizationExperiment,
    SelfOrgConfig,
    get_ground_truth,
    compute_chance_purity,
    run_trial,
    compute_purity,
    compute_completeness,
    build_contingency_table,
)
import numpy as np
from research.experiments.base import summarize, ttest_vs_null


def run_sweep_point(label, cfg, n_seeds, base_seed=42):
    """Run one parameter configuration and return summary metrics."""
    ground_truth = get_ground_truth()
    chance_pur = compute_chance_purity(ground_truth, cfg.n_core_areas)
    core_areas = [f"CORE_{i}" for i in range(cfg.n_core_areas)]

    purity_vals = []
    complete_vals = []
    active_vals = []

    for s in range(n_seeds):
        result = run_trial(cfg, base_seed + s)
        purity_vals.append(result["purity"])
        complete_vals.append(result["completeness"])
        active_vals.append(result["n_active_areas"])

    pur_mean = float(np.mean(purity_vals))
    comp_mean = float(np.mean(complete_vals))
    active_mean = float(np.mean(active_vals))
    pur_test = ttest_vs_null(purity_vals, chance_pur)

    return {
        "label": label,
        "config": asdict(cfg),
        "purity": summarize(purity_vals),
        "purity_vs_chance": pur_test,
        "completeness": summarize(complete_vals),
        "n_active_areas": summarize(active_vals),
        "chance_purity": chance_pur,
        "summary": {
            "purity": pur_mean,
            "completeness": comp_mean,
            "active_areas": active_mean,
            "d": pur_test["d"],
            "p": pur_test["p"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parameter exploration for self-organization")
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
    base_cfg = SelfOrgConfig(
        n=base_n, k=base_k, training_reps=10,
        add_modality_features=True)
    r = run_sweep_point("baseline", base_cfg, n_seeds)
    results["baseline"] = r
    s = r["summary"]
    print(f"  Purity={s['purity']:.3f}  Complete={s['completeness']:.3f}  "
          f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 1: Beta
    print("\n" + "=" * 70)
    print("SWEEP: Beta (Hebbian learning rate)")
    print("=" * 70)
    for beta in [0.05, 0.15, 0.20, 0.30]:
        label = f"beta_{beta}"
        cfg = SelfOrgConfig(
            n=base_n, k=base_k, beta=beta, training_reps=10,
            add_modality_features=True)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  beta={beta:.2f}: Purity={s['purity']:.3f}  "
              f"Complete={s['completeness']:.3f}  "
              f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 2: Training repetitions
    print("\n" + "=" * 70)
    print("SWEEP: Training repetitions")
    print("=" * 70)
    for reps in [1, 3, 5, 20]:
        label = f"reps_{reps}"
        cfg = SelfOrgConfig(
            n=base_n, k=base_k, training_reps=reps,
            add_modality_features=True)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  reps={reps:2d}: Purity={s['purity']:.3f}  "
              f"Complete={s['completeness']:.3f}  "
              f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 3: Number of core areas
    print("\n" + "=" * 70)
    print("SWEEP: Number of core areas")
    print("=" * 70)
    for n_areas in [4, 6, 10, 12]:
        label = f"areas_{n_areas}"
        cfg = SelfOrgConfig(
            n=base_n, k=base_k, n_core_areas=n_areas, training_reps=10,
            add_modality_features=True)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        chance = r["chance_purity"]
        print(f"  areas={n_areas:2d}: Purity={s['purity']:.3f} "
              f"(chance={chance:.3f})  "
              f"Complete={s['completeness']:.3f}  "
              f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Print comparison table
    print("\n\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    header = (f"{'Config':<16}{'Purity':>8}{'Compl':>8}{'Active':>8}"
              f"{'d':>8}{'p':>10}")
    print(header)
    print("-" * 58)

    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<16}{s['purity']:>8.3f}{s['completeness']:>8.3f}"
              f"{s['active_areas']:>8.1f}{s['d']:>8.2f}{s['p']:>10.4f}")

    # Save
    results_dir = Path(__file__).parent.parent.parent / "results" / "primitives"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_quick" if args.quick else ""
    output_path = results_dir / f"self_org_exploration_{timestamp}{suffix}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
