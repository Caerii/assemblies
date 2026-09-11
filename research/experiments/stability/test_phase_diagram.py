"""
Phase Diagram of Assembly Attractor Formation

Maps finite persistence during learning-on autonomous recurrence over (k/n, beta).
The grid does not establish frozen fixed points or a physical phase transition.

Protocol:
1. Establish: project({"s": ["A"]}, {}) -- initial stimulus activation.
2. Train: project({"s": ["A"]}, {"A": ["A"]}) x 30 rounds -- stim+self.
3. Test: project({}, {"A": ["A"]}) x 20 rounds -- autonomous recurrence.
4. Measure: persistence = overlap(trained, current) after 20 autonomous rounds.

Descriptive criterion: nominal interval wholly above/below .95, or unresolved.
The lowest sampled above-threshold beta is not a certified phase boundary.

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
from neural_assemblies.core.registration import validate_area_registration, validate_round_count
from research.experiment_config import resolve_seed_ids, resolve_real_grid
from research.experiments._explicit import explicit_brain, model_semantics_kwargs

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
    initial_stimulus_rounds: int = 1

    def __post_init__(self):
        self.n, self.k = validate_area_registration("A", self.n, self.k)
        self.train_rounds = validate_round_count(self.train_rounds)
        self.test_rounds = validate_round_count(self.test_rounds)
        self.initial_stimulus_rounds = validate_round_count(self.initial_stimulus_rounds)
        self.p = resolve_real_grid([self.p], name="connection probability", maximum=1.)[0]
        self.beta = resolve_real_grid([self.beta], name="plasticity")[0]
        self.w_max = resolve_real_grid([self.w_max], name="weight clip")[0]


# -- Core trial runner ---------------------------------------------------------


def run_phase_trial(
    cfg: PhaseConfig, seed: int, *, model_semantics=None,
) -> float:
    """
    Train stim+self, then test autonomous persistence.
    Returns persistence (overlap between trained and final assembly).
    """
    b = explicit_brain(Brain, cfg, seed, model_semantics)
    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("s", cfg.k)

    # Phase 1: initial stimulus activation
    for _ in range(cfg.initial_stimulus_rounds):
        b.project({"s": ["A"]}, {})

    # Phase 2: stim+self training
    for _ in range(cfg.train_rounds):
        b.project({"s": ["A"]}, {"A": ["A"]})

    trained = Assembly.from_area(b, "A")

    # Phase 3: autonomous persistence test
    for _ in range(cfg.test_rounds):
        b.project({}, {"A": ["A"]})

    return measure_overlap(trained.neuron_ids, Assembly.from_area(b, "A").neuron_ids)


def persistence_interval_status(summary, threshold=.95):
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-phase-measurement"""
    if summary["ci95_lo"] >= threshold:
        return "above_threshold"
    if summary["ci95_hi"] < threshold:
        return "below_threshold"
    return "unresolved"


def sampled_threshold_crossings(rows):
    """Descriptive sampled crossings, including unobserved crossings; not phase boundaries."""
    result = {}
    for row in rows:
        key = str(row["sparsity"])
        result.setdefault(key, {"beta": None, "status": "not_observed"})
        previous = result[key]["beta"]
        if row["interval_status"] == "above_threshold" and (previous is None or row["beta"] < previous):
            result[key] = {"beta": row["beta"], "status": "observed_in_sampled_grid"}
    return result


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

    # Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#legacy-experiment-configuration
    def run(
        self,
        n: int = 1000,
        p: float = 0.05,
        w_max: float = 20.0,
        n_seeds: int | None = None,
        *, seed_ids=None, sparsities=(.01, .02, .05, .10, .15, .20, .30),
        betas=(.01, .02, .05, .10, .20), p_values=(.01, .02, .05, .10, .20),
        p_effect_k=100, p_effect_beta=.10, train_rounds=30, test_rounds=20,
        initial_stimulus_rounds=1, persistence_threshold=.95,
        model_semantics=None,
    ) -> ExperimentResult:
        seeds = resolve_seed_ids(n_seeds, seed_ids, base_seed=self.seed, default_count=N_SEEDS)
        n_seeds = len(seeds)
        n, p_effect_k = validate_area_registration("H3", n, p_effect_k)
        p = resolve_real_grid([p], name="connection probability", maximum=1.)[0]
        p_effect_beta = resolve_real_grid([p_effect_beta], name="H3 plasticity")[0]
        w_max = resolve_real_grid([w_max], name="weight clip")[0]
        sparsities = resolve_real_grid(sparsities, name="sparsities", maximum=1.)
        betas = resolve_real_grid(betas, name="betas")
        p_values = resolve_real_grid(p_values, name="connection probabilities", maximum=1.)
        persistence_threshold = resolve_real_grid([persistence_threshold], name="persistence threshold", maximum=1.)[0]
        schedule = dict(train_rounds=validate_round_count(train_rounds), test_rounds=validate_round_count(test_rounds),
                        initial_stimulus_rounds=validate_round_count(initial_stimulus_rounds))
        sizes = [int(sparsity*n) for sparsity in sparsities]
        if len(set(sizes)) != len(sizes):
            raise ValueError("sparsities collide after conversion to integer assembly sizes")
        grid = [(sparsity, PhaseConfig(n, k, p, beta, w_max, **schedule))
                for sparsity, k in zip(sparsities, sizes) for beta in betas]
        p_configs = [PhaseConfig(n, p_effect_k, probability, p_effect_beta, w_max, **schedule)
                     for probability in p_values]
        self._start_timer()

        self.log("=" * 60)
        self.log("Phase Diagram Experiment")
        self.log(f"  n={n}, p={p}, w_max={w_max}")
        self.log(f"  sparsities={sparsities}")
        self.log(f"  betas={betas}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}
        raw_data = {"seeds": seeds, "cells": []}
        semantic_kwargs = model_semantics_kwargs(model_semantics)

        # ================================================================
        # H1/H2: Sparsity x Beta Phase Grid
        # ================================================================
        self.log("\nH1/H2: Sparsity x Beta Phase Diagram")

        grid_results = []

        for sparsity, cfg in grid:
            k_val, beta = cfg.k, cfg.beta
            null = chance_overlap(k_val, n)
            persist_vals = []
            for s in seeds:
                persist_vals.append(run_phase_trial(cfg, seed=s, **semantic_kwargs))

            row = {
                "sparsity": sparsity, "actual_sparsity": k_val/n,
                "k": k_val,
                "beta": beta,
                "null_overlap": null,
                "persistence": summarize(persist_vals),
                "test_vs_null": reported_null_test(persist_vals, null),
            }
            row["interval_status"] = persistence_interval_status(row["persistence"], persistence_threshold)
            grid_results.append(row)
            raw_data["cells"].append(dict(arm="sparsity_beta", n=n, k=k_val,
                                           sparsity=sparsity, beta=beta, p=p, values=persist_vals))

            stable = row["interval_status"]
            self.log(
                f"  k/n={sparsity:.2f} beta={beta:.2f}: "
                f"{row['persistence']['mean']:.3f}+/-{row['persistence']['sem']:.3f} "
                f"{stable}"
            )

        metrics["sparsity_beta_grid"] = grid_results

        # ================================================================
        # Descriptive sampled crossings, including absence at each sparsity
        # ================================================================
        metrics["sampled_threshold_crossings"] = sampled_threshold_crossings(grid_results)
        self.log("Descriptive sampled threshold crossings (not a phase boundary):")
        for sparsity, crossing in metrics["sampled_threshold_crossings"].items():
            self.log(f"  k/n={sparsity}: {crossing}")

        # ================================================================
        # H3: Connection Probability Effect
        # ================================================================
        self.log(f"\nH3: Connection Probability Effect (n={n}, k={p_effect_k}, beta={p_effect_beta})")

        null_h3 = chance_overlap(p_effect_k, n)
        h3_results = []

        for cfg in p_configs:
            p_val = cfg.p

            persist_vals = []
            for s in seeds:
                persist_vals.append(run_phase_trial(cfg, seed=s, **semantic_kwargs))

            row = {
                "p": p_val,
                "persistence": summarize(persist_vals),
                "test_vs_null": reported_null_test(persist_vals, null_h3),
            }
            h3_results.append(row)
            raw_data["cells"].append(dict(arm="p_effect", n=n, k=p_effect_k, beta=p_effect_beta, p=p_val, values=persist_vals))

            self.log(
                f"  p={p_val:.2f}: "
                f"{row['persistence']['mean']:.3f}+/-{row['persistence']['sem']:.3f}  "
                f"d={effect_text(row['test_vs_null'])}"
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
                **schedule, "persistence_threshold": persistence_threshold,
                "sparsities": sparsities, "resolved_assembly_sizes": sizes, "betas": betas,
                "p_values": p_values, "p_effect_k": p_effect_k, "p_effect_beta": p_effect_beta,
                "seed_ids": seeds,
                "engine": "numpy_explicit",
                "evaluation_learning": True,
            },
            metrics=metrics,
            raw_data=raw_data,
            duration_seconds=duration,
        )


def main(argv=None):
    from research.experiments.historical_phase import main as run
    return run(argv)


if __name__ == "__main__":
    main()
