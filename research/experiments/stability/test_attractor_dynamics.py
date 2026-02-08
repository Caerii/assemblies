"""
Attractor Dynamics Under Autonomous Recurrence

Characterizes whether Hebbian-trained assemblies form stable fixed-point
attractors under self-projection (no external input), and identifies the
critical parameters governing the transition from drift to stability.

Hypotheses (tested with proper null distributions):

H1: Training threshold — There exists a critical Hebbian exposure
    (beta * train_rounds) above which assemblies form stable attractors.
    Null: overlap under self-projection equals chance level k/n.

H2: Noise recovery — Trained attractors can recover from partial noise
    injection. There exists a critical noise fraction above which
    recovery fails.
    Null: recovery overlap equals the overlap of a random k-subset.

H3: Network scaling — Attractor stability improves with network size n
    (at fixed k = sqrt(n), p, beta).
    Null: persistence is independent of n.

H4: Weight ceiling transition — There exists a critical w_max below which
    attractors cannot form (weights too weak to overcome random drift).
    Null: persistence is independent of w_max.

Statistical methodology:
- Each condition is replicated across N_SEEDS independent random seeds.
- Null distributions are computed analytically or by simulation.
- Significance is assessed via one-sample t-test against the null mean.
- Effect sizes are reported as Cohen's d.
- Results report mean +/- SEM with 95% confidence intervals.

References:
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
- Dabagia et al., "Coin-Flipping in the Brain", 2024 (weight saturation)
- Litwin-Kumar & Doiron, Nature Comms, 2014 (assembly maintenance)
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

# Number of independent seeds per experimental condition.
N_SEEDS = 10


@dataclass
class AttractorConfig:
    """Configuration for one attractor dynamics trial."""
    n: int              # neurons per area
    k: int              # assembly size
    p: float            # connection probability
    beta: float         # Hebbian plasticity rate
    w_max: float        # weight saturation ceiling
    train_rounds: int   # rounds of stim+self training
    test_rounds: int    # rounds of pure self-projection
    stim_size: int = 0  # stimulus size (defaults to k)

    def __post_init__(self):
        if self.stim_size == 0:
            self.stim_size = self.k


def run_single_trial(cfg: AttractorConfig, seed: int) -> Dict[str, Any]:
    """Run one attractor dynamics trial and return raw measurements."""
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.stim_size)

    # Phase 1: establish assembly via stimulus
    b.project({"s": ["A"]}, {})

    # Phase 2: train with stimulus + self-projection
    for _ in range(cfg.train_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})

    trained_winners = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Measure weight structure
    conn = b.connectomes["A"]["A"].weights
    trained_set = set(trained_winners.tolist())
    non_winners = [i for i in range(cfg.n) if i not in trained_set]

    # Intra-assembly mean weight (full)
    ww = np.mean([conn[i, j] for i in trained_set for j in trained_set])

    # Inter-assembly mean weight (sampled for efficiency)
    rng = np.random.default_rng(seed + 99999)
    sample_nw = rng.choice(non_winners, min(100, len(non_winners)), replace=False) if non_winners else []
    nn = float(np.mean([conn[i, j] for i in sample_nw for j in sample_nw])) if len(sample_nw) > 1 else 0.0

    # Phase 3: pure self-projection (autonomous recurrence)
    overlaps_with_trained = []
    step_overlaps = []
    for _ in range(cfg.test_rounds):
        prev_winners = np.array(b.areas["A"].winners, dtype=np.uint32)
        b.project({}, {"A": ["A"]})
        curr_winners = np.array(b.areas["A"].winners, dtype=np.uint32)
        overlaps_with_trained.append(measure_overlap(trained_winners, curr_winners))
        step_overlaps.append(measure_overlap(prev_winners, curr_winners))

    return {
        "overlaps_with_trained": overlaps_with_trained,
        "step_overlaps": step_overlaps,
        "weight_intra": float(ww),
        "weight_inter": float(nn),
        "weight_ratio": float(ww / nn) if nn > 1e-10 else float("inf"),
        "final_persistence": overlaps_with_trained[-1] if overlaps_with_trained else 0.0,
        "mean_persistence": float(np.mean(overlaps_with_trained)) if overlaps_with_trained else 0.0,
    }


def run_basin_trial(cfg: AttractorConfig, noise_frac: float, seed: int) -> Dict[str, Any]:
    """Run one noise-injection trial: train, inject noise, self-project, measure recovery."""
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.stim_size)

    # Train
    b.project({"s": ["A"]}, {})
    for _ in range(cfg.train_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})
    trained = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Inject noise
    rng = np.random.default_rng(seed + 77777)
    noisy_winners = trained.copy()
    n_replace = int(noise_frac * len(trained))
    if n_replace > 0:
        replace_idx = rng.choice(len(trained), n_replace, replace=False)
        non_winners = np.array([i for i in range(cfg.n) if i not in set(trained.tolist())])
        new_neurons = rng.choice(non_winners, n_replace, replace=False)
        noisy_winners[replace_idx] = new_neurons.astype(np.uint32)

    initial_overlap = measure_overlap(trained, noisy_winners)
    b.areas["A"].winners = noisy_winners

    # Self-project and measure recovery
    recovery = []
    for _ in range(cfg.test_rounds):
        b.project({}, {"A": ["A"]})
        curr = np.array(b.areas["A"].winners, dtype=np.uint32)
        recovery.append(measure_overlap(trained, curr))

    return {
        "initial_overlap": initial_overlap,
        "recovery_curve": recovery,
        "final_overlap": recovery[-1] if recovery else 0.0,
    }


class AttractorDynamicsExperiment(ExperimentBase):
    """
    Rigorous statistical characterization of assembly attractor dynamics.

    Each experimental condition is replicated across N_SEEDS independent
    random seeds. Results are tested against null distributions derived
    from the chance overlap k/n.
    """

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="attractor_dynamics",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose,
        )

    def run(self, **kwargs) -> ExperimentResult:
        """Run the full experiment suite with proper statistical controls."""
        self._start_timer()
        seeds = list(range(N_SEEDS))

        # ================================================================
        # H1: Training threshold
        # ================================================================
        self.log("=" * 60)
        self.log("H1: Training threshold (beta x rounds for attractor formation)")
        self.log("=" * 60)

        base_n, base_k, base_p, base_wmax = 1000, 100, 0.05, 20.0
        null_overlap = chance_overlap(base_k, base_n)  # k/n = 0.10
        self.log(f"Null (chance) overlap: {null_overlap:.3f}")

        h1_results = []
        for beta in [0.01, 0.05, 0.1, 0.2]:
            for train_rounds in [1, 5, 10, 20, 50]:
                cfg = AttractorConfig(
                    n=base_n, k=base_k, p=base_p, beta=beta, w_max=base_wmax,
                    train_rounds=train_rounds, test_rounds=20,
                )
                # Run across seeds
                persistences = []
                weight_ratios = []
                for s in seeds:
                    trial = run_single_trial(cfg, seed=s)
                    persistences.append(trial["final_persistence"])
                    weight_ratios.append(trial["weight_ratio"])

                persist_stats = summarize(persistences)
                wratio_stats = summarize(weight_ratios)
                hypothesis_test = ttest_vs_null(persistences, null_overlap)

                row = {
                    "beta": beta,
                    "train_rounds": train_rounds,
                    "beta_x_rounds": beta * train_rounds,
                    "persistence": persist_stats,
                    "weight_ratio": wratio_stats,
                    "test_vs_null": hypothesis_test,
                }
                h1_results.append(row)

                sig = "*" if hypothesis_test["significant"] else " "
                self.log(
                    f"  beta={beta:.2f} rounds={train_rounds:3d} "
                    f"(bxr={beta*train_rounds:.2f}): "
                    f"persist={persist_stats['mean']:.3f}±{persist_stats['sem']:.3f} "
                    f"w_ratio={wratio_stats['mean']:.1f} "
                    f"d={hypothesis_test['d']:.1f} p={hypothesis_test['p']:.4f}{sig}"
                )

        # ================================================================
        # H2: Noise recovery (attractor basin)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H2: Attractor basin (noise tolerance)")
        self.log("=" * 60)

        basin_cfg = AttractorConfig(
            n=1000, k=100, p=0.05, beta=0.1, w_max=20.0,
            train_rounds=30, test_rounds=20,
        )

        h2_results = []
        for noise_frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            recoveries = []
            for s in seeds:
                trial = run_basin_trial(basin_cfg, noise_frac, seed=s)
                recoveries.append(trial["final_overlap"])

            recovery_stats = summarize(recoveries)
            hypothesis_test = ttest_vs_null(recoveries, null_overlap)

            row = {
                "noise_frac": noise_frac,
                "initial_overlap": 1.0 - noise_frac,
                "final_overlap": recovery_stats,
                "test_vs_null": hypothesis_test,
            }
            h2_results.append(row)

            sig = "*" if hypothesis_test["significant"] else " "
            self.log(
                f"  noise={noise_frac:.1f}: "
                f"final={recovery_stats['mean']:.3f}±{recovery_stats['sem']:.3f} "
                f"[{recovery_stats['min']:.2f}, {recovery_stats['max']:.2f}] "
                f"d={hypothesis_test['d']:.1f}{sig}"
            )

        # ================================================================
        # H3: Network scaling (k = sqrt(n))
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H3: Scaling with network size (k = sqrt(n))")
        self.log("=" * 60)

        h3_results = []
        for n in [100, 200, 500, 1000, 2000, 5000]:
            k = max(int(np.sqrt(n)), 5)  # floor, minimum 5
            null_ov = chance_overlap(k, n)

            cfg = AttractorConfig(
                n=n, k=k, p=0.05, beta=0.1, w_max=20.0,
                train_rounds=30, test_rounds=20,
            )
            persistences = []
            for s in seeds:
                trial = run_single_trial(cfg, seed=s)
                persistences.append(trial["final_persistence"])

            persist_stats = summarize(persistences)
            hypothesis_test = ttest_vs_null(persistences, null_ov)

            row = {
                "n": n, "k": k, "k_over_n": k / n,
                "null_overlap": null_ov,
                "persistence": persist_stats,
                "test_vs_null": hypothesis_test,
            }
            h3_results.append(row)

            sig = "*" if hypothesis_test["significant"] else " "
            self.log(
                f"  n={n:5d} k={k:3d} k/n={k/n:.3f}: "
                f"persist={persist_stats['mean']:.3f}±{persist_stats['sem']:.3f} "
                f"null={null_ov:.3f} d={hypothesis_test['d']:.1f}{sig}"
            )

        # ================================================================
        # H4: w_max phase transition
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H4: Weight ceiling (w_max) phase transition")
        self.log("=" * 60)

        null_ov = chance_overlap(100, 1000)

        h4_results = []
        for w_max in [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
            cfg = AttractorConfig(
                n=1000, k=100, p=0.05, beta=0.1, w_max=w_max,
                train_rounds=30, test_rounds=20,
            )
            persistences = []
            weight_ratios = []
            for s in seeds:
                trial = run_single_trial(cfg, seed=s)
                persistences.append(trial["final_persistence"])
                weight_ratios.append(trial["weight_ratio"])

            persist_stats = summarize(persistences)
            wratio_stats = summarize(weight_ratios)
            hypothesis_test = ttest_vs_null(persistences, null_ov)

            row = {
                "w_max": w_max,
                "persistence": persist_stats,
                "weight_ratio": wratio_stats,
                "test_vs_null": hypothesis_test,
            }
            h4_results.append(row)

            sig = "*" if hypothesis_test["significant"] else " "
            self.log(
                f"  w_max={w_max:5.1f}: "
                f"persist={persist_stats['mean']:.3f}±{persist_stats['sem']:.3f} "
                f"w_ratio={wratio_stats['mean']:.1f}±{wratio_stats['sem']:.1f} "
                f"d={hypothesis_test['d']:.1f}{sig}"
            )

        duration = self._stop_timer()

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_seeds": N_SEEDS,
                "base_n": base_n, "base_k": base_k,
                "base_p": base_p, "base_wmax": base_wmax,
                "null_overlap_base": null_overlap,
            },
            metrics={
                "h1_training_threshold": h1_results,
                "h2_attractor_basin": h2_results,
                "h3_scaling": h3_results,
                "h4_wmax_transition": h4_results,
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    """Run attractor dynamics experiment with full statistical analysis."""
    exp = AttractorDynamicsExperiment(verbose=True)
    result = exp.run()

    print("\n" + "=" * 70)
    print("FINDINGS SUMMARY")
    print("=" * 70)

    # H1 summary: find critical beta*rounds
    print("\nH1 — Critical Hebbian exposure for attractor formation:")
    print(f"  Null baseline (chance overlap): {result.parameters['null_overlap_base']:.3f}")
    for r in result.metrics["h1_training_threshold"]:
        if r["test_vs_null"]["significant"] and r["persistence"]["mean"] >= 0.95:
            print(f"  STABLE: beta={r['beta']:.2f} x {r['train_rounds']} rounds "
                  f"(beta*T={r['beta_x_rounds']:.2f}): "
                  f"{r['persistence']['mean']:.3f}±{r['persistence']['sem']:.3f}")

    # H2 summary
    print("\nH2 — Attractor basin (critical noise for recovery failure):")
    for r in result.metrics["h2_attractor_basin"]:
        recovered = r["final_overlap"]["mean"] >= 0.95
        status = "RECOVERS" if recovered else "FAILS"
        print(f"  noise={r['noise_frac']:.1f}: "
              f"{r['final_overlap']['mean']:.3f}±{r['final_overlap']['sem']:.3f} [{status}]")

    # H3 summary
    print("\nH3 — Scaling with n:")
    for r in result.metrics["h3_scaling"]:
        print(f"  n={r['n']:5d} k={r['k']:3d}: "
              f"{r['persistence']['mean']:.3f}±{r['persistence']['sem']:.3f} "
              f"(null={r['null_overlap']:.3f})")

    # H4 summary
    print("\nH4 — w_max phase transition:")
    for r in result.metrics["h4_wmax_transition"]:
        stable = r["persistence"]["mean"] >= 0.95
        status = "STABLE" if stable else "DRIFT"
        print(f"  w_max={r['w_max']:5.1f}: "
              f"{r['persistence']['mean']:.3f}±{r['persistence']['sem']:.3f} [{status}]")

    print(f"\nTotal time: {result.duration_seconds:.1f}s "
          f"({N_SEEDS} seeds per condition)")


if __name__ == "__main__":
    main()
