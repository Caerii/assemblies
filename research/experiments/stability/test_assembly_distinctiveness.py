"""
Assembly Distinctiveness Under Competing Stimuli

Tests whether different stimuli produce distinct, recoverable assemblies in
the same brain area using the stim-only Papadimitriou protocol (no
self-connectome training).

Stim-only projection strengthens stimulus-to-area connections via Hebbian
learning but never trains the A->A self-connectome. This allows multiple
distinct assemblies to coexist in one area without competing attractors.

Training: For each stimulus s_i, apply project({"s_i": ["A"]}, {}) x 30
rounds. Hebbian learning strengthens the pathway from s_i to its preferred
neurons in A.

Recovery: After training all stimuli, reactivate each stimulus with 5
rounds of stim-only projection and measure overlap with its originally
trained assembly.

Hypotheses:

H1: Stim-only distinctiveness -- Different stimuli produce assemblies with
    pairwise overlap near chance (k/n), far from identical (1.0).
    Null vs chance: overlap equals k/n.
    Null vs identical: overlap equals 1.0.

H2: Recovery via reactivation -- Each stimulus recovers its original
    assembly after other stimuli have been active.
    Null: recovery equals chance k/n.

H3: Capacity vs assembly size k -- Pairwise overlap scales linearly with
    k/n, confirming assemblies are as distinct as random k-subsets.
    Null: overlap equals k/n.

H4: Distinctiveness vs network size -- At k=sqrt(n), excess overlap above
    chance decreases with n.
    Null: overlap equals k/n.

Statistical methodology:
- N_SEEDS=10 independent seeds per condition.
- One-sample t-test against null k/n (and against 1.0 for H1).
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
from itertools import combinations

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
class DistinctivenessConfig:
    """Configuration for distinctiveness trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    train_rounds: int = 30


# -- Core trial runners --------------------------------------------------------


def run_distinctiveness_trial(
    cfg: DistinctivenessConfig, n_stimuli: int, seed: int,
) -> Dict[str, Any]:
    """
    Train n_stimuli with stim-only, measure pairwise overlaps and recovery.
    Returns mean pairwise overlap and mean recovery.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)

    stim_names = [f"s{i}" for i in range(n_stimuli)]
    for name in stim_names:
        b.add_stimulus(name, cfg.k)

    # Train each stimulus with stim-only
    assemblies = {}
    for name in stim_names:
        for _ in range(cfg.train_rounds):
            b.project({name: ["A"]}, {})
        assemblies[name] = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Pairwise overlaps
    pairwise = []
    for s1, s2 in combinations(stim_names, 2):
        pairwise.append(measure_overlap(assemblies[s1], assemblies[s2]))
    mean_pairwise = float(np.mean(pairwise)) if pairwise else 0.0

    # Recovery: reactivate each stimulus with 5 rounds of stim-only
    recoveries = []
    for name in stim_names:
        for _ in range(5):
            b.project({name: ["A"]}, {})
        current = np.array(b.areas["A"].winners, dtype=np.uint32)
        recoveries.append(measure_overlap(assemblies[name], current))
    mean_recovery = float(np.mean(recoveries))

    return {"mean_pairwise_overlap": mean_pairwise, "mean_recovery": mean_recovery}


def run_capacity_trial(
    n: int, k: int, p: float, beta: float, w_max: float,
    n_stimuli: int, train_rounds: int, seed: int,
) -> float:
    """
    Train n_stimuli with stim-only at given k, return mean pairwise overlap.
    """
    b = Brain(p=p, seed=seed, w_max=w_max)
    b.add_area("A", n, k, beta, explicit=True)

    stim_names = [f"s{i}" for i in range(n_stimuli)]
    for name in stim_names:
        b.add_stimulus(name, k)

    assemblies = {}
    for name in stim_names:
        for _ in range(train_rounds):
            b.project({name: ["A"]}, {})
        assemblies[name] = np.array(b.areas["A"].winners, dtype=np.uint32)

    pairwise = []
    for s1, s2 in combinations(stim_names, 2):
        pairwise.append(measure_overlap(assemblies[s1], assemblies[s2]))

    return float(np.mean(pairwise)) if pairwise else 0.0


# -- Main experiment -----------------------------------------------------------


class AssemblyDistinctivenessExperiment(ExperimentBase):
    """Test assembly distinctiveness under competing stimuli."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="assembly_distinctiveness",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose,
        )

    def run(
        self,
        n: int = 1000,
        k: int = 100,
        p: float = 0.05,
        beta: float = 0.10,
        w_max: float = 20.0,
        n_seeds: int = N_SEEDS,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()
        seeds = list(range(n_seeds))

        null = chance_overlap(k, n)
        cfg = DistinctivenessConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)

        self.log("=" * 60)
        self.log("Assembly Distinctiveness Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}

        # ================================================================
        # H1: Stim-Only Distinctiveness vs Number of Stimuli
        # ================================================================
        self.log("\nH1: Distinctiveness vs Number of Stimuli")

        n_stimuli_list = [2, 3, 5, 8]
        h1_results = []

        for n_stim in n_stimuli_list:
            overlap_vals = []
            for s in seeds:
                trial = run_distinctiveness_trial(cfg, n_stim, seed=self.seed + s)
                overlap_vals.append(trial["mean_pairwise_overlap"])

            row = {
                "n_stimuli": n_stim,
                "pairwise_overlap": summarize(overlap_vals),
                "test_vs_identical": ttest_vs_null(overlap_vals, 1.0),
                "test_vs_chance": ttest_vs_null(overlap_vals, null),
            }
            h1_results.append(row)

            self.log(
                f"  {n_stim} stimuli: "
                f"overlap={row['pairwise_overlap']['mean']:.3f}+/-{row['pairwise_overlap']['sem']:.3f}  "
                f"vs_chance d={row['test_vs_chance']['d']:.1f}  "
                f"vs_identical d={row['test_vs_identical']['d']:.1f}"
            )

        metrics["h1_distinctiveness"] = h1_results

        # ================================================================
        # H2: Recovery via Stim-Only Reactivation
        # ================================================================
        self.log("\nH2: Recovery via Reactivation")

        h2_results = []

        for n_stim in n_stimuli_list:
            recovery_vals = []
            for s in seeds:
                trial = run_distinctiveness_trial(cfg, n_stim, seed=self.seed + s)
                recovery_vals.append(trial["mean_recovery"])

            row = {
                "n_stimuli": n_stim,
                "recovery_overlap": summarize(recovery_vals),
                "test_recovery_vs_chance": ttest_vs_null(recovery_vals, null),
            }
            h2_results.append(row)

            self.log(
                f"  {n_stim} stimuli: "
                f"recovery={row['recovery_overlap']['mean']:.3f}+/-{row['recovery_overlap']['sem']:.3f}  "
                f"d={row['test_recovery_vs_chance']['d']:.1f}"
            )

        metrics["h2_recovery"] = h2_results

        # ================================================================
        # H3: Capacity vs Assembly Size k
        # ================================================================
        self.log("\nH3: Capacity vs Assembly Size k (n=1000, 5 stimuli)")

        k_values = [10, 20, 50, 100, 200]
        h3_results = []

        for k_val in k_values:
            null_h3 = chance_overlap(k_val, n)
            vals = []
            for s in seeds:
                vals.append(
                    run_capacity_trial(n, k_val, p, beta, w_max, 5, 30, self.seed + s)
                )

            row = {
                "k": k_val,
                "k_over_n": k_val / n,
                "null_overlap": null_h3,
                "theoretical_capacity": n // k_val,
                "pairwise_overlap": summarize(vals),
                "test_vs_chance": ttest_vs_null(vals, null_h3),
            }
            h3_results.append(row)

            self.log(
                f"  k={k_val:3d} (k/n={k_val/n:.3f}): "
                f"overlap={row['pairwise_overlap']['mean']:.3f}  "
                f"chance={null_h3:.3f}  "
                f"d={row['test_vs_chance']['d']:.1f}"
            )

        metrics["h3_capacity_vs_k"] = h3_results

        # ================================================================
        # H4: Distinctiveness vs Network Size (k=sqrt(n))
        # ================================================================
        self.log("\nH4: Distinctiveness vs Network Size (k=sqrt(n), 5 stimuli)")

        h4_sizes = [200, 500, 1000, 2000]
        h4_results = []

        for n_val in h4_sizes:
            k_val = int(np.sqrt(n_val))
            null_h4 = chance_overlap(k_val, n_val)

            vals = []
            for s in seeds:
                vals.append(
                    run_capacity_trial(n_val, k_val, p, beta, w_max, 5, 30, self.seed + s)
                )

            row = {
                "n": n_val,
                "k": k_val,
                "null_overlap": null_h4,
                "pairwise_overlap": summarize(vals),
                "test_vs_chance": ttest_vs_null(vals, null_h4),
            }
            h4_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"overlap={row['pairwise_overlap']['mean']:.3f}  "
                f"chance={null_h4:.3f}  "
                f"d={row['test_vs_chance']['d']:.1f}"
            )

        metrics["h4_distinctiveness_vs_n"] = h4_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_seeds": n_seeds,
                "base_n": n, "base_k": k, "base_p": p,
                "base_beta": beta, "base_wmax": w_max,
                "train_rounds": cfg.train_rounds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Assembly Distinctiveness Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = AssemblyDistinctivenessExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
