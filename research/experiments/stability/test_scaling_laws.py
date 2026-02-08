"""
Scaling Laws for Assembly Formation and Attractor Persistence

Characterizes how convergence time and attractor persistence scale with
network size n, at fixed sparsity k = sqrt(n).

Protocol:
1. Establish: project({"s": ["A"]}, {}) -- initial stimulus activation.
2. Train with convergence detection: project({"s": ["A"]}, {"A": ["A"]})
   x up to 100 rounds. Convergence = 3 consecutive rounds with
   step-to-step overlap > 0.98.
3. Test autonomous persistence: project({}, {"A": ["A"]}) x 20 rounds.
   Measure overlap between current winners and the trained assembly.

Parameters: p=0.05, beta=0.10, w_max=20.0, max_train=100, test_rounds=20.

Hypotheses:

H1/H2: Convergence time and persistence vs network size at k=sqrt(n).
    Null: persistence equals chance k/n.

H3: Scaling law fit -- convergence time scales as O(log n).
    Null: no relationship (slope=0).

Statistical methodology:
- N_SEEDS=10 independent seeds per condition.
- One-sample t-test against null k/n.
- Cohen's d effect sizes. Mean +/- SEM.
- Scaling law fit via linear regression of T vs log10(n).

References:
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
- Dabagia et al., "Coin-Flipping in the Brain", 2024 (weight saturation)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from scipy import stats

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    ttest_vs_null,
)

from src.core.brain import Brain

N_SEEDS = 10


@dataclass
class ScalingConfig:
    """Configuration for scaling trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    max_train_rounds: int = 100
    test_rounds: int = 20


# -- Core trial runner ---------------------------------------------------------


def run_scaling_trial(
    cfg: ScalingConfig, seed: int,
) -> Dict[str, Any]:
    """
    Train stim+self with convergence detection, then test autonomous persistence.
    Returns convergence time and persistence.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

    # Phase 1: initial stimulus activation
    b.project({"s": ["A"]}, {})

    # Phase 2: train stim+self with convergence detection
    winner_history = []
    converged_at = cfg.max_train_rounds

    for r in range(cfg.max_train_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})
        winners = np.array(b.areas["A"].winners, dtype=np.uint32)
        winner_history.append(winners.copy())

        if len(winner_history) >= 4:
            overlaps = [
                measure_overlap(winner_history[-i - 1], winner_history[-i - 2])
                for i in range(3)
            ]
            if all(o > 0.98 for o in overlaps):
                converged_at = r + 1
                break

    trained = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Phase 3: autonomous persistence test
    for _ in range(cfg.test_rounds):
        b.project({}, {"A": ["A"]})

    persistence = measure_overlap(
        trained, np.array(b.areas["A"].winners, dtype=np.uint32)
    )

    return {"convergence_time": float(converged_at), "persistence": persistence}


# -- Main experiment -----------------------------------------------------------


class ScalingLawsExperiment(ExperimentBase):
    """Test scaling laws: convergence time and persistence vs network size."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="scaling_laws",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose,
        )

    def run(
        self,
        p: float = 0.05,
        beta: float = 0.10,
        w_max: float = 20.0,
        n_seeds: int = N_SEEDS,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()
        seeds = list(range(n_seeds))

        n_values = [100, 200, 500, 1000, 2000, 5000]

        self.log("=" * 60)
        self.log("Scaling Laws Experiment")
        self.log(f"  n_values={n_values}")
        self.log(f"  p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}

        # ================================================================
        # H1/H2: Convergence + Persistence vs Network Size (k=sqrt(n))
        # ================================================================
        self.log("\nH1/H2: Convergence + Persistence vs Network Size (k=sqrt(n))")

        scaling_results = []

        for n_val in n_values:
            k_val = int(np.sqrt(n_val))
            null = chance_overlap(k_val, n_val)
            cfg = ScalingConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)

            conv_times = []
            persist_vals = []

            for s in seeds:
                trial = run_scaling_trial(cfg, seed=self.seed + s)
                conv_times.append(trial["convergence_time"])
                persist_vals.append(trial["persistence"])

            row = {
                "n": n_val,
                "k": k_val,
                "k_over_n": k_val / n_val,
                "log10_n": float(np.log10(n_val)),
                "null_overlap": null,
                "convergence_time": summarize(conv_times),
                "persistence": summarize(persist_vals),
                "test_vs_null": ttest_vs_null(persist_vals, null),
            }
            scaling_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"T={row['convergence_time']['mean']:.1f}+/-{row['convergence_time']['sem']:.1f}  "
                f"persist={row['persistence']['mean']:.3f}+/-{row['persistence']['sem']:.3f}  "
                f"d={row['test_vs_null']['d']:.1f}"
            )

        metrics["scaling_results"] = scaling_results

        # ================================================================
        # H3: Scaling Law Fit
        # ================================================================
        self.log("\nH3: Scaling Law Fit")

        log_n = np.array([r["log10_n"] for r in scaling_results])
        mean_t = np.array([r["convergence_time"]["mean"] for r in scaling_results])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, mean_t)

        if abs(slope) < 0.5:
            scaling_type = "O(1) - constant"
        elif slope < 2:
            scaling_type = "O(log n) - logarithmic"
        elif slope < 5:
            scaling_type = "O(log^2 n) - polylogarithmic"
        else:
            scaling_type = "O(n^alpha) - polynomial"

        metrics["scaling_fit"] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "scaling_type": scaling_type,
            "equation": f"T = {slope:.2f} * log10(n) + {intercept:.2f}",
        }

        self.log(f"  Fit: {metrics['scaling_fit']['equation']}  R²={r_value**2:.3f}")
        self.log(f"  Scaling type: {scaling_type}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_seeds": n_seeds,
                "n_values": n_values,
                "base_p": p,
                "base_beta": beta,
                "base_wmax": w_max,
                "max_train_rounds": 100,
                "test_rounds": 20,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scaling Laws Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = ScalingLawsExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
