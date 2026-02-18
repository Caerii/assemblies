"""
Systematic Parameter Exploration for Agreement NUMBER Experiment

Varies one parameter at a time from the default configuration and compares
key metrics across runs. Uses the refactored composable infrastructure —
each exploration is just a config change, not a copy of the experiment.

Usage:
    uv run python research/experiments/applications/explore_agreement_params.py
    uv run python research/experiments/applications/explore_agreement_params.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from research.experiments.applications.test_agreement_number import (
    AgreementNumberExperiment,
    AgreementNumberConfig,
)


EXPLORATIONS = {
    "baseline":     {},
    "seeds_10":     {"n_seeds": 10},
    "seeds_15":     {"n_seeds": 15},
    "consol_2":     {"consolidation_passes": 2},
    "consol_3":     {"consolidation_passes": 3},
    "settle_5":     {"p600_settling_rounds": 5},
    "settle_20":    {"p600_settling_rounds": 20},
    "k_150":        {"k": 150},
    "k_200":        {"k": 200},
}

QUICK_EXPLORATIONS = {
    "baseline":     {},
    "seeds_8":      {"n_seeds": 8},
    "consol_2":     {"consolidation_passes": 2},
    "settle_5":     {"p600_settling_rounds": 5},
    "k_150":        {"k": 150},
}


def extract_key_metrics(result):
    """Extract the most important metrics from an experiment result."""
    m = result.metrics
    row = {
        "verb_inst_d": None,
        "verb_inst_p": None,
        "verb_margin_d": None,
        "verb_margin_p": None,
        "obj_vp_dist_d": None,
        "obj_vp_dist_p": None,
        "verb_inst_success": m.get("verb_inst_success", False),
        "verb_margin_success": m.get("verb_margin_success", False),
    }

    if "verb_agree_vs_gram" in m:
        t = m["verb_agree_vs_gram"].get("test", {})
        row["verb_inst_d"] = t.get("d")
        row["verb_inst_p"] = t.get("p")

    if "margin_verb_gram_vs_agree" in m:
        t = m["margin_verb_gram_vs_agree"].get("test", {})
        row["verb_margin_d"] = t.get("d")
        row["verb_margin_p"] = t.get("p")

    if "vp_dist_cat_vs_agree" in m:
        t = m["vp_dist_cat_vs_agree"].get("test", {})
        row["obj_vp_dist_d"] = t.get("d")
        row["obj_vp_dist_p"] = t.get("p")

    return row


def fmt(v, width=8):
    """Format a value for the comparison table."""
    if v is None:
        return "-".center(width)
    if isinstance(v, bool):
        return ("YES" if v else "NO").center(width)
    if isinstance(v, float):
        return f"{v:.3f}".rjust(width)
    return str(v).rjust(width)


def main():
    parser = argparse.ArgumentParser(
        description="Systematic parameter exploration for agreement NUMBER")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run reduced exploration set with fewer seeds")
    args = parser.parse_args()

    explorations = QUICK_EXPLORATIONS if args.quick else EXPLORATIONS

    results = {}
    for name, overrides in explorations.items():
        print(f"\n{'='*70}")
        print(f"EXPLORATION: {name}  overrides={overrides or 'none (baseline)'}")
        print(f"{'='*70}")

        cfg = AgreementNumberConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)

        exp = AgreementNumberExperiment(verbose=True)
        result = exp.run(config=cfg)
        results[name] = {
            "config": asdict(cfg),
            "key_metrics": extract_key_metrics(result),
            "full_result_file": str(result.experiment_name),
        }

    # Print comparison table
    print(f"\n\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")

    header = (f"{'Config':<16}"
              f"{'VerbI_d':>8}{'VerbI_p':>8}"
              f"{'VerbM_d':>8}{'VerbM_p':>8}"
              f"{'VPdst_d':>8}{'VPdst_p':>8}"
              f"{'Inst?':>8}{'Marg?':>8}")
    print(header)
    print("-" * 90)

    for name in explorations:
        if name not in results:
            continue
        km = results[name]["key_metrics"]
        row = (f"{name:<16}"
               f"{fmt(km['verb_inst_d'])}{fmt(km['verb_inst_p'])}"
               f"{fmt(km['verb_margin_d'])}{fmt(km['verb_margin_p'])}"
               f"{fmt(km['obj_vp_dist_d'])}{fmt(km['obj_vp_dist_p'])}"
               f"{fmt(km['verb_inst_success'])}{fmt(km['verb_margin_success'])}")
        print(row)

    print(f"\nLegend:")
    print(f"  VerbI = Verb instability agree>gram (d, p)")
    print(f"  VerbM = Verb margin gram>agree (d, p)")
    print(f"  VPdst = Object VP distance cat>agree (d, p)")
    print(f"  Inst? = Verb instability success")
    print(f"  Marg? = Verb margin success")

    # Save full results
    results_dir = Path(__file__).parent.parent.parent / "results" / "applications"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_quick" if args.quick else ""
    output_path = results_dir / f"agreement_exploration_{timestamp}{suffix}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
