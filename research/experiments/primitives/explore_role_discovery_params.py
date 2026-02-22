"""
Parameter Exploration for Role Discovery

Systematically varies parameters to understand the boundaries of the
unsupervised role separation effect (shared marker + LRI + MI).

Explorations:
  1. Number of structural areas: 2, 3, 4, 6
  2. Stabilization rounds: 1, 3, 5, 7
  3. Inhibition strength: 0.5, 1.0, 2.0, 5.0
  4. Area size (n): 5000, 10000, 20000

Usage:
    uv run python research/experiments/primitives/explore_role_discovery_params.py
    uv run python research/experiments/primitives/explore_role_discovery_params.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from research.experiments.primitives.test_role_self_organization import (
    RoleDiscoveryConfig,
    run_trial,
)
import numpy as np
from research.experiments.base import summarize, ttest_vs_null


def run_sweep_point(label, cfg, n_seeds, base_seed=42):
    """Run one parameter configuration and return summary metrics."""
    chance_purity = 1.0 / cfg.n_struct_areas + 0.5 * (
        1.0 - 1.0 / cfg.n_struct_areas)

    sep_vals = []
    purity_vals = []
    p1_consist_vals = []
    p2_consist_vals = []
    active_vals = []

    for s in range(n_seeds):
        result = run_trial(cfg, base_seed + s)
        sep_vals.append(result["separation_rate"])
        purity_vals.append(result["position_purity"])
        p1_consist_vals.append(result["pos1_consistency"])
        p2_consist_vals.append(result["pos2_consistency"])
        active_vals.append(result["n_active_struct"])

    sep_mean = float(np.mean(sep_vals))
    pur_mean = float(np.mean(purity_vals))
    p1_mean = float(np.mean(p1_consist_vals))
    p2_mean = float(np.mean(p2_consist_vals))
    active_mean = float(np.mean(active_vals))
    pur_test = ttest_vs_null(purity_vals, chance_purity)

    return {
        "label": label,
        "config": asdict(cfg),
        "separation_rate": summarize(sep_vals),
        "purity": summarize(purity_vals),
        "purity_vs_chance": pur_test,
        "pos1_consistency": summarize(p1_consist_vals),
        "pos2_consistency": summarize(p2_consist_vals),
        "n_active_areas": summarize(active_vals),
        "chance_purity": chance_purity,
        "summary": {
            "separation": sep_mean,
            "purity": pur_mean,
            "p1_consist": p1_mean,
            "p2_consist": p2_mean,
            "active_areas": active_mean,
            "d": pur_test["d"],
            "p": pur_test["p"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parameter exploration for role discovery")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        n_seeds = 5
        base_n = 5000
        base_k = 50
        n_test = 30
    else:
        n_seeds = 10
        base_n = 10000
        base_k = 100
        n_test = 50

    results = {}

    # Baseline
    print("=" * 70)
    print("BASELINE")
    print("=" * 70)
    base_cfg = RoleDiscoveryConfig(
        n=base_n, k=base_k, n_test_sentences=n_test)
    r = run_sweep_point("baseline", base_cfg, n_seeds)
    results["baseline"] = r
    s = r["summary"]
    print(f"  Sep={s['separation']:.3f}  Purity={s['purity']:.3f}  "
          f"P1={s['p1_consist']:.3f}  P2={s['p2_consist']:.3f}  "
          f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 1: Number of structural areas
    print("\n" + "=" * 70)
    print("SWEEP: Number of structural areas")
    print("=" * 70)
    for n_areas in [2, 3, 6, 8]:
        label = f"areas_{n_areas}"
        cfg = RoleDiscoveryConfig(
            n=base_n, k=base_k, n_struct_areas=n_areas,
            n_test_sentences=n_test)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        chance = r["chance_purity"]
        print(f"  areas={n_areas}: Sep={s['separation']:.3f}  "
              f"Purity={s['purity']:.3f} (chance={chance:.3f})  "
              f"P1={s['p1_consist']:.3f}  P2={s['p2_consist']:.3f}  "
              f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 2: Stabilization rounds
    print("\n" + "=" * 70)
    print("SWEEP: Stabilization rounds")
    print("=" * 70)
    for stab in [1, 5, 7, 10]:
        label = f"stab_{stab}"
        cfg = RoleDiscoveryConfig(
            n=base_n, k=base_k, stabilize_rounds=stab,
            n_test_sentences=n_test)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  stab={stab:2d}: Sep={s['separation']:.3f}  "
              f"Purity={s['purity']:.3f}  "
              f"P1={s['p1_consist']:.3f}  P2={s['p2_consist']:.3f}  "
              f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 3: Inhibition strength
    print("\n" + "=" * 70)
    print("SWEEP: Inhibition strength")
    print("=" * 70)
    for inh in [0.5, 2.0, 5.0, 10.0]:
        label = f"inhib_{inh}"
        cfg = RoleDiscoveryConfig(
            n=base_n, k=base_k, inhibition_strength=inh,
            n_test_sentences=n_test)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  inhib={inh:5.1f}: Sep={s['separation']:.3f}  "
              f"Purity={s['purity']:.3f}  "
              f"P1={s['p1_consist']:.3f}  P2={s['p2_consist']:.3f}  "
              f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Sweep 4: Area size (n)
    print("\n" + "=" * 70)
    print("SWEEP: Area size (n)")
    print("=" * 70)
    for area_n, area_k in [(2000, 20), (5000, 50), (20000, 200)]:
        label = f"n_{area_n}"
        cfg = RoleDiscoveryConfig(
            n=area_n, k=area_k, n_test_sentences=n_test)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  n={area_n:5d} k={area_k:3d}: Sep={s['separation']:.3f}  "
              f"Purity={s['purity']:.3f}  "
              f"P1={s['p1_consist']:.3f}  P2={s['p2_consist']:.3f}  "
              f"Active={s['active_areas']:.1f}  d={s['d']:.2f}  p={s['p']:.4f}")

    # Print comparison table
    print("\n\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    header = (f"{'Config':<16}{'Sep':>6}{'Purity':>8}{'P1':>8}{'P2':>8}"
              f"{'Active':>8}{'d':>8}{'p':>10}")
    print(header)
    print("-" * 72)

    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<16}{s['separation']:>6.3f}{s['purity']:>8.3f}"
              f"{s['p1_consist']:>8.3f}{s['p2_consist']:>8.3f}"
              f"{s['active_areas']:>8.1f}{s['d']:>8.2f}{s['p']:>10.4f}")

    # Save
    results_dir = Path(__file__).parent.parent.parent / "results" / "primitives"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_quick" if args.quick else ""
    output_path = results_dir / f"role_discovery_exploration_{timestamp}{suffix}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
