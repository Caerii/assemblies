"""
Phase Diagram of Assembly Attractor Formation

Maps the boundary in (k/n, beta) parameter space where assemblies transition
from drifting activations to stable fixed-point attractors under autonomous
recurrence.

Protocol:
1. Establish: project({"s": ["A"]}, {}) -- initial stimulus activation.
2. Train: project({"s": ["A"]}, {"A": ["A"]}) x 30 rounds -- stim+self.
3. Test: project({}, {"A": ["A"]}) x 20 rounds -- autonomous recurrence.
4. Measure: persistence = overlap(trained, current) after 20 autonomous rounds.

Stability criterion: persistence >= 0.95 = stable [S], otherwise drifting [D].

Hypotheses:

H1/H2: Sparsity x Beta phase diagram -- There exists a sharp phase boundary
    in (k/n, beta) space separating stable from drifting attractors.
    Null: persistence equals chance k/n at all (k/n, beta).

H3: Connection probability effect -- persistence increases with p above a
    critical threshold.
    Null: persistence is independent of p.

Parameters: n=1000, p=0.05, w_max=20.0, train_rounds=30, test_rounds=20.

Statistical methodology:
- N_SEEDS=10 independent seeds per condition.
- One-sample t-test against null k/n.
- Cohen's d effect sizes. Mean +/- SEM.

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
class PhaseConfig:
    """Configuration for phase diagram trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    train_rounds: int = 30
    test_rounds: int = 20


# -- Core trial runner ---------------------------------------------------------


def run_phase_trial(
    cfg: PhaseConfig, seed: int,
) -> float:
    """
    Train stim+self, then test autonomous persistence.
    Returns persistence (overlap between trained and final assembly).
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

    # Phase 1: initial stimulus activation
    b.project({"s": ["A"]}, {})

    # Phase 2: stim+self training
    for _ in range(cfg.train_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})

    trained = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Phase 3: autonomous persistence test
    for _ in range(cfg.test_rounds):
        b.project({}, {"A": ["A"]})

    return measure_overlap(trained, np.array(b.areas["A"].winners, dtype=np.uint32))


# -- Main experiment -----------------------------------------------------------


class PhaseDiagramExperiment(ExperimentBase):
    """Map the phase diagram of assembly attractor formation."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="phase_diagram",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose,
        )

    def run(
        self,
        n: int = 1000,
        p: float = 0.05,
        w_max: float = 20.0,
        n_seeds: int = N_SEEDS,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()
        seeds = list(range(n_seeds))

        sparsities = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
        betas = [0.01, 0.02, 0.05, 0.10, 0.20]

        self.log("=" * 60)
        self.log("Phase Diagram Experiment")
        self.log(f"  n={n}, p={p}, w_max={w_max}")
        self.log(f"  sparsities={sparsities}")
        self.log(f"  betas={betas}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}

        # ================================================================
        # H1/H2: Sparsity x Beta Phase Grid
        # ================================================================
        self.log("\nH1/H2: Sparsity x Beta Phase Diagram")

        grid_results = []

        for sparsity in sparsities:
            k_val = int(sparsity * n)
            null = chance_overlap(k_val, n)

            for beta in betas:
                cfg = PhaseConfig(n=n, k=k_val, p=p, beta=beta, w_max=w_max)

                persist_vals = []
                for s in seeds:
                    persist_vals.append(run_phase_trial(cfg, seed=self.seed + s))

                row = {
                    "sparsity": sparsity,
                    "k": k_val,
                    "beta": beta,
                    "null_overlap": null,
                    "persistence": summarize(persist_vals),
                    "test_vs_null": ttest_vs_null(persist_vals, null),
                }
                grid_results.append(row)

                stable = "[S]" if row["persistence"]["mean"] >= 0.95 else "[D]"
                self.log(
                    f"  k/n={sparsity:.2f} beta={beta:.2f}: "
                    f"{row['persistence']['mean']:.3f}+/-{row['persistence']['sem']:.3f} "
                    f"{stable}"
                )

        metrics["sparsity_beta_grid"] = grid_results

        # ================================================================
        # Phase boundary: min beta for stability at each sparsity
        # ================================================================
        phase_boundary = {}
        for sparsity in sparsities:
            entries = [r for r in grid_results if r["sparsity"] == sparsity]
            for entry in entries:
                if entry["persistence"]["mean"] >= 0.95:
                    phase_boundary[str(sparsity)] = {
                        "beta": entry["beta"],
                        "persistence": entry["persistence"]["mean"],
                    }
                    break

        metrics["phase_boundary"] = phase_boundary

        self.log("\nPhase boundary (min beta for stability >= 0.95):")
        for s_str, info in phase_boundary.items():
            self.log(f"  k/n={s_str}: beta={info['beta']:.2f} (persist={info['persistence']:.3f})")

        # ================================================================
        # H3: Connection Probability Effect
        # ================================================================
        self.log("\nH3: Connection Probability Effect (n=1000, k=100, beta=0.10)")

        p_values = [0.01, 0.02, 0.05, 0.10, 0.20]
        null_h3 = chance_overlap(100, n)
        h3_results = []

        for p_val in p_values:
            cfg = PhaseConfig(n=n, k=100, p=p_val, beta=0.10, w_max=w_max)

            persist_vals = []
            for s in seeds:
                persist_vals.append(run_phase_trial(cfg, seed=self.seed + s))

            row = {
                "p": p_val,
                "persistence": summarize(persist_vals),
                "test_vs_null": ttest_vs_null(persist_vals, null_h3),
            }
            h3_results.append(row)

            self.log(
                f"  p={p_val:.2f}: "
                f"{row['persistence']['mean']:.3f}+/-{row['persistence']['sem']:.3f}  "
                f"d={row['test_vs_null']['d']:.1f}"
            )

        metrics["p_effect"] = h3_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_seeds": n_seeds,
                "base_n": n,
                "base_p": p,
                "base_wmax": w_max,
                "train_rounds": 30,
                "test_rounds": 20,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase Diagram Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = PhaseDiagramExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
