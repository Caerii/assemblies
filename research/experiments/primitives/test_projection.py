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
self-connectome can retain activity while learning continues. This does not
establish a frozen fixed-point attractor.

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
    This is A-driven regeneration of B while learning, not autonomous completion.
    Null: recovery equals chance k/n.

H4: Weight dynamics vs training rounds -- How do self-connectome
    weights and persistence evolve with training duration?
    Descriptive ratio only: selection of active neurons invalidates 1.0 as
    a universal beta-zero null. The former constant-1 probe was defective.

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
from typing import Dict, Any
from scipy import stats

from research.experiments.base import (
    resolve_seed_ids, reported_null_test, effect_text,
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    paired_ttest,
)

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.registration import validate_area_registration, validate_round_count

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
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max, engine="numpy_sparse")
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
    if mode not in ("stim_self", "stim_only"):
        raise ValueError("training mode must be stim_self or stim_only")
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max, engine="numpy_sparse")
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
    """A-driven regeneration while learning; B corruption is not a recovery cue.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-projection-measurement
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max, engine="numpy_sparse")
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


def recurrent_weight_ratio(weights, winners) -> float:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-projection-measurement

    Includes absent edges (zeros); this descriptive selection ratio is not a
    beta-zero null or an estimate of learning alone.
    """
    matrix = np.asarray(weights)
    indices = np.asarray(winners)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError("recurrent weights must be a nonempty square matrix")
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
        raise ValueError("recurrent weights must be real numeric values")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("recurrent weights must be finite and nonnegative")
    if (indices.ndim != 1 or not indices.size or
            not np.issubdtype(indices.dtype, np.integer) or
            len(np.unique(indices)) != len(indices) or
            (indices < 0).any() or (indices >= len(matrix)).any()):
        raise ValueError("winners must be unique in-range integer indices")
    mean_all = float(np.mean(matrix, dtype=np.float64))
    if mean_all == 0:
        raise ValueError("weight ratio is undefined for a zero-mean connectome")
    return float(np.mean(matrix[np.ix_(indices, indices)], dtype=np.float64) / mean_all)


def run_weight_dynamics_trial(
    n: int, k: int, p: float, beta: float, w_max: float,
    train_rounds: int, test_rounds: int, seed: int,
) -> Dict[str, float]:
    """Measure weight ratio and persistence after T training rounds."""
    b = Brain(p=p, seed=seed, w_max=w_max, engine="numpy_sparse")
    b.add_area("A", n, k, beta, explicit=True)
    b.add_stimulus("s", k)

    for _ in range(train_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})

    trained = np.array(b.areas["A"].winners, dtype=np.uint32)

    # Measure before autonomous evaluation mutates weights again.
    weight_ratio = recurrent_weight_ratio(b.connectomes["A"]["A"].weights, trained)

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

    # Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#legacy-experiment-configuration
    def run(
        self,
        n: int = 1000,
        k: int = 100,
        p: float = 0.05,
        beta: float = 0.10,
        w_max: float = 20.0,
        n_seeds: int | None = None,
        *, seed_ids=None, train_rounds=30, test_rounds=20, max_train_rounds=100,
        h1_sizes=(100, 200, 500, 1000, 2000, 5000),
        h3_sizes=(500, 1000, 2000), round_values=(1, 5, 10, 20, 30, 50),
    ) -> ExperimentResult:
        seeds = resolve_seed_ids(n_seeds, seed_ids, base_seed=self.seed, default_count=N_SEEDS)
        n_seeds = len(seeds)
        n, k = validate_area_registration('A', n, k)
        train_rounds, test_rounds, max_train_rounds = map(validate_round_count,
                                                        (train_rounds, test_rounds, max_train_rounds))
        h1_sizes, h3_sizes, round_values = map(tuple, (h1_sizes, h3_sizes, round_values))
        for grid in (h1_sizes, h3_sizes, round_values):
            if not grid or len(set(grid)) != len(grid):
                raise ValueError('study grids must be nonempty and unique')
            for value in grid:
                validate_round_count(value)
        if len(h1_sizes) < 2 or any(value < 2 for value in (*h1_sizes, *h3_sizes)):
            raise ValueError('scaling needs two H1 sizes; population sizes must be at least two')
        schedule = dict(train_rounds=train_rounds, test_rounds=test_rounds, max_train_rounds=max_train_rounds)
        self._start_timer()

        self.log("=" * 60)
        self.log("Projection Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}
        raw_data = {"seeds": seeds, "cells": []}

        # ================================================================
        # H1: Convergence + persistence vs network size (k=sqrt(n))
        # ================================================================
        self.log("\nH1: Convergence + Persistence vs Network Size (k=sqrt(n))")

        h1_results = []

        for n_val in h1_sizes:
            k_val = int(np.sqrt(n_val))
            cfg = ProjConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max, **schedule)
            null = chance_overlap(k_val, n_val)

            conv_times = []
            persist_vals = []

            for s in seeds:
                trial = run_convergence_trial(cfg, seed=s)
                conv_times.append(float(trial["convergence_time"]))
                persist_vals.append(trial["persistence"])

            row = {
                "n": n_val, "k": k_val, "k_over_n": k_val / n_val,
                "convergence_time": summarize(conv_times),
                "persistence": summarize(persist_vals),
                "test_vs_null": reported_null_test(persist_vals, null),
            }
            h1_results.append(row)
            raw_data["cells"].append(dict(arm="h1", n=n_val, k=k_val,
                                           values=dict(convergence_time=conv_times, persistence=persist_vals)))

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"T={row['convergence_time']['mean']:.1f}  "
                f"persist={row['persistence']['mean']:.3f}  "
                f"d={effect_text(row['test_vs_null'])}"
            )

        metrics["convergence_vs_size"] = h1_results

        # Scaling fit
        log_n = np.array([np.log10(r["n"]) for r in h1_results])
        mean_t = np.array([r["convergence_time"]["mean"] for r in h1_results])
        if np.ptp(mean_t) == 0:
            metrics["scaling_fit"] = {
                "slope": 0., "intercept": float(mean_t[0]), "r_squared": None,
                "p_value": None, "degenerate": "constant_response",
                "equation": f"T = {float(mean_t[0]):.2f}",
            }
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, mean_t)
            metrics["scaling_fit"] = {
                "slope": float(slope), "intercept": float(intercept),
                "r_squared": float(r_value ** 2), "p_value": float(p_value),
                "equation": f"T = {slope:.2f} * log10(n) + {intercept:.2f}",
            }
        self.log(f"  Scaling fit: {metrics['scaling_fit']['equation']}")

        # ================================================================
        # H2: Stim+self vs stim-only
        # ================================================================
        self.log(f"\nH2: Stim+Self vs Stim-Only (n={n}, k={k})")

        cfg_h2 = ProjConfig(n=n, k=k, p=p, beta=beta, w_max=w_max, **schedule)
        null_h2 = chance_overlap(k, n)

        stim_self_vals = []
        stim_only_vals = []

        for s in seeds:
            stim_self_vals.append(run_training_mode_trial(cfg_h2, s, "stim_self"))
            stim_only_vals.append(run_training_mode_trial(cfg_h2, s, "stim_only"))

        raw_data["cells"].append(dict(arm="h2", n=n, k=k,
                                       values=dict(stim_self=stim_self_vals, stim_only=stim_only_vals)))
        metrics["training_mode_comparison"] = {
            "stim_self": {
                "persistence": summarize(stim_self_vals),
                "test_vs_null": reported_null_test(stim_self_vals, null_h2),
            },
            "stim_only": {
                "persistence": summarize(stim_only_vals),
                "test_vs_null": reported_null_test(stim_only_vals, null_h2),
            },
            "paired_test": paired_ttest(stim_self_vals, stim_only_vals),
        }

        self.log(f"  Stim+self: {summarize(stim_self_vals)['mean']:.3f}")
        self.log(f"  Stim-only: {summarize(stim_only_vals)['mean']:.3f}")

        # ================================================================
        # H3: Cross-area fidelity (k=sqrt(n))
        # ================================================================
        self.log("\nH3: Cross-Area Fidelity (k=sqrt(n))")

        h3_results = []

        for n_val in h3_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h3 = ProjConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max, **schedule)
            null_h3 = chance_overlap(k_val, n_val)

            recoveries = []
            for s in seeds:
                recoveries.append(run_crossarea_trial(cfg_h3, s))

            row = {
                "n": n_val, "k": k_val,
                "recovery": summarize(recoveries),
                "test_vs_null": reported_null_test(recoveries, null_h3),
            }
            h3_results.append(row)
            raw_data["cells"].append(dict(arm="h3", n=n_val, k=k_val, values=recoveries))

            self.log(f"  n={n_val:4d}: {row['recovery']['mean']:.3f}  d={effect_text(row['test_vs_null'])}")

        metrics["crossarea_fidelity"] = h3_results

        # ================================================================
        # H4: Weight dynamics vs training rounds
        # ================================================================
        self.log(f"\nH4: Weight Dynamics vs Training Rounds (n={n}, k={k})")

        h4_results = []

        for t_rounds in round_values:
            wr_vals = []
            p_vals = []

            for s in seeds:
                trial = run_weight_dynamics_trial(
                    n, k, p, beta, w_max, t_rounds, test_rounds, s
                )
                wr_vals.append(trial["weight_ratio"])
                p_vals.append(trial["persistence"])

            row = {
                "train_rounds": t_rounds,
                "weight_ratio": summarize(wr_vals),
                "persistence": summarize(p_vals),
                "weight_ratio_interpretation": "descriptive; no registered learning null",
            }
            h4_results.append(row)
            raw_data["cells"].append(dict(arm="h4", n=n, k=k, train_rounds=t_rounds,
                                           values=dict(weight_ratio=wr_vals, persistence=p_vals)))

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
                **schedule, "h1_sizes": list(h1_sizes), "h3_sizes": list(h3_sizes),
                "round_values": list(round_values), "seed_ids": seeds,
                "primary_engine": "numpy_sparse", "area_engine": "numpy_explicit",
                "evaluation_learning": True, "weight_ratio_definition": "selected-pairs-over-all-pairs-v2",
            },
            metrics=metrics,
            raw_data=raw_data,
            duration_seconds=duration,
        )


def main(argv=None):
    from research.experiments.historical_projection import main as run
    return run(argv)


if __name__ == "__main__":
    main()
