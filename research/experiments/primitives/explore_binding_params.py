"""
Parameter Exploration for Variable Binding

Systematically varies parameters to understand the boundaries of the
binding and selective retrieval mechanism (co-projection + deferred
connectome initialisation).

Explorations:
  1. Binding rounds: 5, 10, 20, 30
  2. Beta (Hebbian rate): 0.05, 0.10, 0.15, 0.20
  3. Area size (n): 5000, 10000, 20000
  4. Lexicon rounds: 10, 20, 30

Usage:
    uv run python research/experiments/primitives/explore_binding_params.py
    uv run python research/experiments/primitives/explore_binding_params.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from research.experiments.primitives.test_binding_retrieval import (
    BindingConfig,
    run_capacity_trial,
    run_two_role_trial,
)
import numpy as np
from research.experiments.base import summarize, ttest_vs_null


def run_sweep_point(label, cfg, n_seeds, base_seed=42, capacity_m=3):
    """Run one parameter configuration and return summary metrics."""
    # Single-ROLE capacity trial at M=capacity_m
    correct_vals = []
    incorrect_vals = []
    selectivity_vals = []
    top1_vals = []

    for s in range(n_seeds):
        result = run_capacity_trial(cfg, capacity_m, base_seed + s)
        correct_vals.append(result["correct_overlap_mean"])
        incorrect_vals.append(result["incorrect_overlap_mean"])
        selectivity_vals.append(result["selectivity"])
        top1_vals.append(result["top1_accuracy"])

    sel_test = ttest_vs_null(selectivity_vals, 0.0)

    # Two-role trial
    agent_sel_vals = []
    patient_sel_vals = []
    agent_acc_vals = []
    patient_acc_vals = []

    for s in range(n_seeds):
        result = run_two_role_trial(cfg, base_seed + s)
        agent_sel_vals.append(result["agent_selectivity"])
        patient_sel_vals.append(result["patient_selectivity"])
        agent_acc_vals.append(result["agent_accuracy"])
        patient_acc_vals.append(result["patient_accuracy"])

    agent_test = ttest_vs_null(agent_sel_vals, 0.0)
    patient_test = ttest_vs_null(patient_sel_vals, 0.0)

    return {
        "label": label,
        "config": asdict(cfg),
        "capacity_m": capacity_m,
        "selectivity": summarize(selectivity_vals),
        "selectivity_test": sel_test,
        "top1_accuracy": summarize(top1_vals),
        "correct_overlap": summarize(correct_vals),
        "agent_selectivity": summarize(agent_sel_vals),
        "agent_test": agent_test,
        "agent_accuracy": summarize(agent_acc_vals),
        "patient_selectivity": summarize(patient_sel_vals),
        "patient_test": patient_test,
        "patient_accuracy": summarize(patient_acc_vals),
        "summary": {
            "sel": float(np.mean(selectivity_vals)),
            "top1": float(np.mean(top1_vals)),
            "agent_sel": float(np.mean(agent_sel_vals)),
            "patient_sel": float(np.mean(patient_sel_vals)),
            "agent_acc": float(np.mean(agent_acc_vals)),
            "patient_acc": float(np.mean(patient_acc_vals)),
            "d_sel": sel_test["d"],
            "p_sel": sel_test["p"],
            "d_agent": agent_test["d"],
            "d_patient": patient_test["d"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parameter exploration for binding retrieval")
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
    base_cfg = BindingConfig(n=base_n, k=base_k)
    r = run_sweep_point("baseline", base_cfg, n_seeds)
    results["baseline"] = r
    s = r["summary"]
    print(f"  M=3 sel={s['sel']:.4f} d={s['d_sel']:.2f} "
          f"top1={s['top1']:.3f}  "
          f"AgentSel={s['agent_sel']:.3f} d={s['d_agent']:.2f}  "
          f"PatientSel={s['patient_sel']:.3f} d={s['d_patient']:.2f}")

    # Sweep 1: Binding rounds
    print("\n" + "=" * 70)
    print("SWEEP: Binding rounds")
    print("=" * 70)
    for rounds in [5, 20, 30, 50]:
        label = f"bind_{rounds}"
        cfg = BindingConfig(n=base_n, k=base_k, binding_rounds=rounds)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  rounds={rounds:2d}: M3 sel={s['sel']:.4f} d={s['d_sel']:.2f}  "
              f"top1={s['top1']:.3f}  "
              f"AgSel={s['agent_sel']:.3f} d={s['d_agent']:.2f}  "
              f"PaSel={s['patient_sel']:.3f} d={s['d_patient']:.2f}")

    # Sweep 2: Beta
    print("\n" + "=" * 70)
    print("SWEEP: Beta (Hebbian learning rate)")
    print("=" * 70)
    for beta in [0.05, 0.15, 0.20, 0.30]:
        label = f"beta_{beta}"
        cfg = BindingConfig(n=base_n, k=base_k, beta=beta)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  beta={beta:.2f}: M3 sel={s['sel']:.4f} d={s['d_sel']:.2f}  "
              f"top1={s['top1']:.3f}  "
              f"AgSel={s['agent_sel']:.3f} d={s['d_agent']:.2f}  "
              f"PaSel={s['patient_sel']:.3f} d={s['d_patient']:.2f}")

    # Sweep 3: Area size
    print("\n" + "=" * 70)
    print("SWEEP: Area size (n)")
    print("=" * 70)
    for area_n, area_k in [(2000, 20), (5000, 50), (20000, 200)]:
        label = f"n_{area_n}"
        cfg = BindingConfig(n=area_n, k=area_k)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  n={area_n:5d} k={area_k:3d}: M3 sel={s['sel']:.4f} d={s['d_sel']:.2f}  "
              f"top1={s['top1']:.3f}  "
              f"AgSel={s['agent_sel']:.3f} d={s['d_agent']:.2f}  "
              f"PaSel={s['patient_sel']:.3f} d={s['d_patient']:.2f}")

    # Sweep 4: Lexicon rounds
    print("\n" + "=" * 70)
    print("SWEEP: Lexicon rounds")
    print("=" * 70)
    for lex_rounds in [10, 30, 50]:
        label = f"lex_{lex_rounds}"
        cfg = BindingConfig(n=base_n, k=base_k, lexicon_rounds=lex_rounds)
        r = run_sweep_point(label, cfg, n_seeds)
        results[label] = r
        s = r["summary"]
        print(f"  lex={lex_rounds:2d}: M3 sel={s['sel']:.4f} d={s['d_sel']:.2f}  "
              f"top1={s['top1']:.3f}  "
              f"AgSel={s['agent_sel']:.3f} d={s['d_agent']:.2f}  "
              f"PaSel={s['patient_sel']:.3f} d={s['d_patient']:.2f}")

    # Comparison table
    print("\n\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    header = (f"{'Config':<16}{'M3-Sel':>8}{'M3-d':>7}{'Top1':>6}"
              f"{'AgSel':>8}{'Ag-d':>7}{'PaSel':>8}{'Pa-d':>7}")
    print(header)
    print("-" * 67)

    for name, r in results.items():
        s = r["summary"]
        print(f"{name:<16}{s['sel']:>8.4f}{s['d_sel']:>7.2f}{s['top1']:>6.3f}"
              f"{s['agent_sel']:>8.3f}{s['d_agent']:>7.2f}"
              f"{s['patient_sel']:>8.3f}{s['d_patient']:>7.2f}")

    # Save
    results_dir = Path(__file__).parent.parent.parent / "results" / "primitives"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_quick" if args.quick else ""
    output_path = results_dir / f"binding_exploration_{timestamp}{suffix}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
