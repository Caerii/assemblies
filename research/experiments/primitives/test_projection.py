"""
Projection Primitive: Assembly Formation, Convergence, and Persistence

Tests the fundamental projection operation: stimulus activates an area,
Hebbian learning strengthens connections, and a stable assembly emerges.
This is the foundational operation of Assembly Calculus.

Protocol:
1. Train with convergence detection:
   project({"s": ["A"]}, {"A": ["A"]}) x up to 100 rounds (stim+self).
   Convergence = 3 consecutive rounds with step-to-step overlap > 0.98.
2. Test autonomous persistence:
   project({}, {"A": ["A"]}) x 20 rounds (self-only, no stimulus).
   Measure overlap between current winners and the trained assembly.

The stim+self protocol trains both the stimulus->A pathway and the A->A
self-connectome. The autonomous persistence test measures whether the
self-connectome alone can maintain the assembly as a fixed-point attractor.

Hypotheses:

H1: Convergence and persistence vs network size -- At k=sqrt(n),
    convergence time and autonomous persistence should be characterized
    across network sizes.
    Null: persistence equals chance k/n.

H2: Stim+self vs stim-only training -- Stim+self (which trains the
    self-connectome) produces higher persistence than stim-only
    (which only trains the stimulus pathway).
    Null: persistence is independent of training mode.

H3: Cross-area fidelity -- Train A via stim+self, then project A->B.
    The projected assembly in B should faithfully represent A.
    Null: recovery equals chance k/n.

H4: Weight dynamics vs training rounds -- How do self-connectome
    weights and persistence evolve with training duration?
    Null: weight ratio equals 1.0 (no Hebbian effect).

Statistical methodology:
- N_SEEDS=10 independent random seeds per condition.
- One-sample t-test against null k/n.
- Paired t-test for H2.
- Cohen's d effect sizes. Mean +/- SEM.
- Linear regression for scaling fit.

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
    paired_ttest,
)

from src.core.brain import Brain

N_SEEDS = 10


@dataclass
class ProjConfig:
    """Configuration for projection trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    train_rounds: int = 30
    test_rounds: int = 20
    max_train_rounds: int = 100


# -- Core trial runners -------------------------------------------------------


def run_convergence_trial(
    cfg: ProjConfig, seed: int,
) -> Dict[str, Any]:
    """
    Train stim+self with convergence detection, then test autonomous persistence.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

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

    # Autonomous persistence
    for _ in range(cfg.test_rounds):
        b.project({}, {"A": ["A"]})

    persistence = measure_overlap(trained, np.array(b.areas["A"].winners, dtype=np.uint32))

    return {"convergence_time": converged_at, "persistence": persistence}


def run_training_mode_trial(
    cfg: ProjConfig, seed: int, mode: str,
) -> float:
    """Train in stim_self or stim_only mode, then test autonomous persistence."""
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

    for _ in range(cfg.train_rounds):
        if mode == "stim_self":
            b.project({"s": ["A"]}, {"A": ["A"]})
        else:
            b.project({"s": ["A"]}, {})

    trained = np.array(b.areas["A"].winners, dtype=np.uint32)

    for _ in range(cfg.test_rounds):
        b.project({}, {"A": ["A"]})

    return measure_overlap(trained, np.array(b.areas["A"].winners, dtype=np.uint32))


def run_crossarea_trial(
    cfg: ProjConfig, seed: int,
) -> float:
    """Train A, project A->B, corrupt B, recover, measure fidelity."""
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_area("B", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

    # Establish A
    for _ in range(cfg.train_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})

    # Project A->B
    for _ in range(cfg.train_rounds):
        b.project({"s": ["A"]}, {"A": ["B"]})
    trained_b = np.array(b.areas["B"].winners, dtype=np.uint32)

    # Corrupt B
    rng = np.random.default_rng(seed + 77777)
    b.areas["B"].winners = rng.choice(cfg.n, cfg.k, replace=False).tolist()

    # Recover
    for _ in range(cfg.test_rounds):
        b.project({"s": ["A"]}, {"A": ["B"]})

    return measure_overlap(trained_b, np.array(b.areas["B"].winners, dtype=np.uint32))


def run_weight_dynamics_trial(
    n: int, k: int, p: float, beta: float, w_max: float,
    train_rounds: int, test_rounds: int, seed: int,
) -> Dict[str, float]:
    """Measure weight ratio and persistence after T training rounds."""
    b = Brain(p=p, seed=seed, w_max=w_max)
    b.add_area("A", n, k, beta, explicit=True)
    b.add_stimulus("s", k)

    for _ in range(train_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})

    trained = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Weight ratio: mean intra-assembly weight / mean all weights
    area = b.areas["A"]
    if hasattr(area, 'connectomes') and "A" in area.connectomes:
        conn = area.connectomes["A"]
        winners_set = set(trained.tolist())
        intra_weights = []
        all_weights = []
        for i in range(min(n, conn.shape[0])):
            for j in range(min(n, conn.shape[1])):
                w = conn[i, j]
                all_weights.append(w)
                if i in winners_set and j in winners_set:
                    intra_weights.append(w)
        weight_ratio = (np.mean(intra_weights) / np.mean(all_weights)
                        if all_weights and intra_weights else 1.0)
    else:
        weight_ratio = 1.0

    # Autonomous persistence
    for _ in range(test_rounds):
        b.project({}, {"A": ["A"]})

    persistence = measure_overlap(trained, np.array(b.areas["A"].winners, dtype=np.uint32))

    return {"weight_ratio": weight_ratio, "persistence": persistence}


# -- Main experiment -----------------------------------------------------------


class ProjectionExperiment(ExperimentBase):
    """Test projection primitive: convergence, persistence, fidelity."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="projection",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "primitives",
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

        self.log("=" * 60)
        self.log("Projection Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}

        # ================================================================
        # H1: Convergence + persistence vs network size (k=sqrt(n))
        # ================================================================
        self.log("\nH1: Convergence + Persistence vs Network Size (k=sqrt(n))")

        h1_sizes = [100, 200, 500, 1000, 2000, 5000]
        h1_results = []

        for n_val in h1_sizes:
            k_val = int(np.sqrt(n_val))
            cfg = ProjConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null = chance_overlap(k_val, n_val)

            conv_times = []
            persist_vals = []

            for s in seeds:
                trial = run_convergence_trial(cfg, seed=self.seed + s)
                conv_times.append(float(trial["convergence_time"]))
                persist_vals.append(trial["persistence"])

            row = {
                "n": n_val, "k": k_val, "k_over_n": k_val / n_val,
                "convergence_time": summarize(conv_times),
                "persistence": summarize(persist_vals),
                "test_vs_null": ttest_vs_null(persist_vals, null),
            }
            h1_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"T={row['convergence_time']['mean']:.1f}  "
                f"persist={row['persistence']['mean']:.3f}  "
                f"d={row['test_vs_null']['d']:.1f}"
            )

        metrics["convergence_vs_size"] = h1_results

        # Scaling fit
        log_n = np.array([np.log10(r["n"]) for r in h1_results])
        mean_t = np.array([r["convergence_time"]["mean"] for r in h1_results])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, mean_t)

        metrics["scaling_fit"] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "equation": f"T = {slope:.2f} * log10(n) + {intercept:.2f}",
        }

        self.log(f"  Scaling fit: {metrics['scaling_fit']['equation']}  R2={r_value**2:.3f}")

        # ================================================================
        # H2: Stim+self vs stim-only
        # ================================================================
        self.log("\nH2: Stim+Self vs Stim-Only (n=1000, k=100)")

        cfg_h2 = ProjConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null_h2 = chance_overlap(k, n)

        stim_self_vals = []
        stim_only_vals = []

        for s in seeds:
            stim_self_vals.append(run_training_mode_trial(cfg_h2, self.seed + s, "stim_self"))
            stim_only_vals.append(run_training_mode_trial(cfg_h2, self.seed + s, "stim_only"))

        metrics["training_mode_comparison"] = {
            "stim_self": {
                "persistence": summarize(stim_self_vals),
                "test_vs_null": ttest_vs_null(stim_self_vals, null_h2),
            },
            "stim_only": {
                "persistence": summarize(stim_only_vals),
                "test_vs_null": ttest_vs_null(stim_only_vals, null_h2),
            },
            "paired_test": paired_ttest(stim_self_vals, stim_only_vals),
        }

        self.log(f"  Stim+self: {summarize(stim_self_vals)['mean']:.3f}")
        self.log(f"  Stim-only: {summarize(stim_only_vals)['mean']:.3f}")

        # ================================================================
        # H3: Cross-area fidelity (k=sqrt(n))
        # ================================================================
        self.log("\nH3: Cross-Area Fidelity (k=sqrt(n))")

        h3_sizes = [500, 1000, 2000]
        h3_results = []

        for n_val in h3_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h3 = ProjConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h3 = chance_overlap(k_val, n_val)

            recoveries = []
            for s in seeds:
                recoveries.append(run_crossarea_trial(cfg_h3, self.seed + s))

            row = {
                "n": n_val, "k": k_val,
                "recovery": summarize(recoveries),
                "test_vs_null": ttest_vs_null(recoveries, null_h3),
            }
            h3_results.append(row)

            self.log(f"  n={n_val:4d}: {row['recovery']['mean']:.3f}  d={row['test_vs_null']['d']:.1f}")

        metrics["crossarea_fidelity"] = h3_results

        # ================================================================
        # H4: Weight dynamics vs training rounds
        # ================================================================
        self.log("\nH4: Weight Dynamics vs Training Rounds (n=1000, k=100)")

        round_values = [1, 5, 10, 20, 30, 50]
        h4_results = []

        for t_rounds in round_values:
            wr_vals = []
            p_vals = []

            for s in seeds:
                trial = run_weight_dynamics_trial(
                    n, k, p, beta, w_max, t_rounds, 20, self.seed + s
                )
                wr_vals.append(trial["weight_ratio"])
                p_vals.append(trial["persistence"])

            row = {
                "train_rounds": t_rounds,
                "weight_ratio": summarize(wr_vals),
                "persistence": summarize(p_vals),
                "test_ratio_vs_1": ttest_vs_null(wr_vals, 1.0),
            }
            h4_results.append(row)

            self.log(
                f"  rounds={t_rounds:2d}: W_ratio={row['weight_ratio']['mean']:.3f}  "
                f"persist={row['persistence']['mean']:.3f}"
            )

        metrics["weight_dynamics"] = h4_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_seeds": n_seeds,
                "base_n": n, "base_k": k, "base_p": p,
                "base_beta": beta, "base_wmax": w_max,
                "train_rounds": 30, "test_rounds": 20,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Projection Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = ProjectionExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
