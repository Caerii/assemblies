"""
Scaling Laws for Assembly Formation and Attractor Persistence

Characterizes how convergence time and attractor persistence scale with
network size n, with assembly size k = floor(sqrt(n)); k/n is not fixed.

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

H3: Descriptive regression of observed convergence time against log10(n).
    Descriptive fit only; a fitted coefficient does not establish an
    asymptotic complexity class. Censored observations prevent this fit.

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
from typing import Dict, Any

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    reported_null_test, effect_text,
)

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.assembly import Assembly
from research.experiments._convergence import run_convergence_phase, convergence_scaling_fit, validate_convergence_rule
from research.experiment_config import resolve_seed_ids
from neural_assemblies.core.registration import validate_round_count, validate_area_registration
from research.experiments._explicit import explicit_brain, model_semantics_kwargs

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
    initial_stimulus_rounds: int = 1
    convergence_window: int = 3
    convergence_threshold: float = .98

    def __post_init__(self):
        validate_area_registration("A", self.n, self.k)
        validate_round_count(self.test_rounds)
        validate_round_count(self.initial_stimulus_rounds)
        validate_convergence_rule(self.max_train_rounds, self.convergence_window, self.convergence_threshold)


# -- Core trial runner ---------------------------------------------------------


def run_scaling_trial(
    cfg: ScalingConfig, seed: int, *, model_semantics=None,
) -> Dict[str, Any]:
    """
    Train stim+self with convergence detection, then test autonomous persistence.
    Returns convergence time and persistence.
    """
    b = explicit_brain(Brain, cfg, seed, model_semantics)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

    # Phase 1: initial stimulus activation
    for _ in range(cfg.initial_stimulus_rounds):
        b.project({"s": ["A"]}, {})

    # This phase deliberately excludes the initial activation above.
    observed = run_convergence_phase(b, stimulus="s", area="A", max_rounds=cfg.max_train_rounds,
                                     window=cfg.convergence_window, threshold=cfg.convergence_threshold)
    for _ in range(cfg.test_rounds):
        b.project({}, {"A": ["A"]})
    persistence = measure_overlap(observed.assembly.neuron_ids, Assembly.from_area(b, "A").neuron_ids)
    return {**observed.record(), "persistence": persistence}


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

    # Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#legacy-experiment-configuration
    def run(
        self,
        p: float = 0.05,
        beta: float = 0.10,
        w_max: float = 20.0,
        n_seeds: int | None = None,
        *, seed_ids=None, n_values=(100, 200, 500, 1000, 2000, 5000),
        max_train_rounds=100, test_rounds=20, initial_stimulus_rounds=1,
        convergence_window=3, convergence_threshold=.98,
        model_semantics=None,
    ) -> ExperimentResult:
        seeds = resolve_seed_ids(n_seeds, seed_ids, base_seed=self.seed, default_count=N_SEEDS)
        n_seeds = len(seeds)
        n_values = [validate_round_count(value) for value in n_values]
        if len(n_values) < 2 or len(set(n_values)) != len(n_values) or min(n_values) < 2:
            raise ValueError("scaling requires at least two unique population sizes of at least two")
        schedule = dict(max_train_rounds=validate_round_count(max_train_rounds),
                        test_rounds=validate_round_count(test_rounds),
                        initial_stimulus_rounds=validate_round_count(initial_stimulus_rounds),
                        convergence_window=validate_round_count(convergence_window),
                        convergence_threshold=convergence_threshold)
        configs = [ScalingConfig(n=value, k=int(np.sqrt(value)), p=p, beta=beta, w_max=w_max,
                                 **schedule) for value in n_values]
        self._start_timer()

        self.log("=" * 60)
        self.log("Scaling Laws Experiment")
        self.log(f"  n_values={n_values}")
        self.log(f"  p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}
        raw_data = {"seeds": seeds, "cells": []}
        semantic_kwargs = model_semantics_kwargs(model_semantics)

        # ================================================================
        # H1/H2: Convergence + Persistence vs Network Size (k=sqrt(n))
        # ================================================================
        self.log("\nH1/H2: Convergence + Persistence vs Network Size (k=sqrt(n))")

        scaling_results = []

        for cfg in configs:
            n_val, k_val = cfg.n, cfg.k
            null = chance_overlap(k_val, n_val)

            conv_times = []
            training_counts = []
            converged_flags = []
            persist_vals = []

            for s in seeds:
                trial = run_scaling_trial(cfg, seed=s, **semantic_kwargs)
                conv_times.append(trial["convergence_time"])
                training_counts.append(trial["training_rounds"])
                converged_flags.append(trial["converged"])
                persist_vals.append(trial["persistence"])

            row = {
                "n": n_val,
                "k": k_val,
                "k_over_n": k_val / n_val,
                "log10_n": float(np.log10(n_val)),
                "null_overlap": null,
                "training_rounds": summarize(training_counts),
                "convergence_fraction": summarize([float(flag) for flag in converged_flags]),
                "persistence": summarize(persist_vals),
                "test_vs_null": reported_null_test(persist_vals, null),
            }
            scaling_results.append(row)
            raw_data["cells"].append(dict(n=n_val, k=k_val, values=dict(
                convergence_time=conv_times, training_rounds=training_counts,
                converged=converged_flags, persistence=persist_vals)))

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"T={row['training_rounds']['mean']:.1f}+/-{row['training_rounds']['sem']:.1f}  "
                f"persist={row['persistence']['mean']:.3f}+/-{row['persistence']['sem']:.3f}  "
                f"d={effect_text(row['test_vs_null'])}"
            )

        metrics["scaling_results"] = scaling_results

        # ================================================================
        # H3: Scaling Law Fit
        # ================================================================
        self.log("\nH3: Scaling Law Fit")

        metrics["scaling_fit"] = convergence_scaling_fit(
            n_values, [cell["values"]["convergence_time"] for cell in raw_data["cells"]])
        self.log(f"  Descriptive fit: {metrics['scaling_fit']['equation']}")

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
                **schedule, "seed_ids": seeds, "assembly_size_rule": "floor_sqrt_population",
                "evaluation_learning": True,
                "engine": "numpy_explicit",
            },
            metrics=metrics,
            raw_data=raw_data,
            duration_seconds=duration,
        )


def main(argv=None):
    from research.experiments.historical_scaling import main as run
    return run(argv)


if __name__ == "__main__":
    main()
