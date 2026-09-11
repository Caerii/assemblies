"""Historical merge: sequential trained overlaps and learning-on driven recovery.

Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-merge-trials

The composition trial trains A->C, then B->C, then joint A/B->C on one brain.
Replacing C winners does not clear learned weights or affect subsequent drive:
C has no outgoing fiber. Earlier training therefore persists into the joint phase.
The reported merge_quality is the average parent overlap; composition_score is
its maximum. Neither certifies that both parents are represented. Keep the
individual overlaps visible, particularly when one parent dominates.

The separate recovery trial skips the isolated-parent C training. After joint
training, it reads from A and then B (20 rounds each by default), while learning
continues. This is not isolated frozen partial-cue completion. The tagged harness retains
explicit settings and all raw overlaps. These observables alone do not establish
an adopted merge result.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from dataclasses import dataclass, replace
from typing import Dict
from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    measure_overlap,
    chance_overlap,
    summarize,
    reported_null_test,
)

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.registration import validate_area_registration, validate_round_count
from research.experiment_config import resolve_seed_ids, resolve_real_grid

N_SEEDS = 10


@dataclass
class MergeConfig:
    """Configuration for merge trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    merge_rounds: int = 30
    test_rounds: int = 20

    def __post_init__(self):
        self.n, self.k = validate_area_registration("A", self.n, self.k)
        self.establish_rounds = validate_round_count(self.establish_rounds)
        self.merge_rounds = validate_round_count(self.merge_rounds)
        self.test_rounds = validate_round_count(self.test_rounds)
        self.p = resolve_real_grid([self.p], name="connection probability", maximum=1.)[0]
        self.beta = resolve_real_grid([self.beta], name="plasticity")[0]
        self.w_max = resolve_real_grid([self.w_max], name="weight clip")[0]


# -- Core trial runners -------------------------------------------------------


# Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-merge-trials
def _establish_sources(cfg, seed):
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max, engine="numpy_sparse")

    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_area("B", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_area("C", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("sa", cfg.k)
    b.add_stimulus("sb", cfg.k)

    # Establish A (stim+self)
    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["A"]}, {"A": ["A"]})

    # Establish B (stim+self)
    for _ in range(cfg.establish_rounds):
        b.project({"sb": ["B"]}, {"B": ["B"]})

    return b


def run_merge_trial(
    cfg: MergeConfig, seed: int,
) -> Dict[str, float]:
    """
    Establish A and B, project each to C separately, then merge.
    Returns overlaps between C_AB, C_A, C_B.
    """
    rng = np.random.default_rng(seed + 88888)
    b = _establish_sources(cfg, seed)

    # Phase 2: A-only -> C
    for _ in range(cfg.merge_rounds):
        b.project({"sa": ["A"]}, {"A": ["C"]})
    c_a = np.array(b.areas["C"].winners, dtype=np.uint32)

    # Replace C winners; learned fibers persist
    b.areas["C"].winners = rng.choice(cfg.n, cfg.k, replace=False).tolist()

    # Phase 3: B-only -> C
    for _ in range(cfg.merge_rounds):
        b.project({"sb": ["B"]}, {"B": ["C"]})
    c_b = np.array(b.areas["C"].winners, dtype=np.uint32)

    # Replace C winners; learned fibers persist
    b.areas["C"].winners = rng.choice(cfg.n, cfg.k, replace=False).tolist()

    # Phase 4: Merge (co-stimulation)
    for _ in range(cfg.merge_rounds):
        b.project({"sa": ["A"], "sb": ["B"]}, {"A": ["C"], "B": ["C"]})
    c_ab = np.array(b.areas["C"].winners, dtype=np.uint32)

    return _merge_overlaps(c_ab, c_a, c_b)


def _merge_overlaps(c_ab, c_a, c_b):
    """Historical average/max overlaps; neither certifies both-parent retention."""
    # Measure overlaps
    overlap_ab_a = measure_overlap(c_ab, c_a)
    overlap_ab_b = measure_overlap(c_ab, c_b)
    overlap_a_b = measure_overlap(c_a, c_b)
    merge_quality = (overlap_ab_a + overlap_ab_b) / 2
    composition_score = max(overlap_ab_a, overlap_ab_b)

    return {
        "merge_quality": merge_quality,
        "composition_score": composition_score,
        "overlap_cab_ca": overlap_ab_a,
        "overlap_cab_cb": overlap_ab_b,
        "overlap_ca_cb": overlap_a_b,
    }


def run_recovery_trial(
    cfg: MergeConfig, seed: int,
) -> Dict[str, float]:
    """Sequential A-only then B-only readout after joint training; both learn."""
    b = _establish_sources(cfg, seed)

    # Merge training (co-stimulation)
    for _ in range(cfg.merge_rounds):
        b.project({"sa": ["A"], "sb": ["B"]}, {"A": ["C"], "B": ["C"]})
    c_merged = np.array(b.areas["C"].winners, dtype=np.uint32)

    # Test: A-only recovery
    rng = np.random.default_rng(seed + 77777)
    b.areas["C"].winners = rng.choice(cfg.n, cfg.k, replace=False).tolist()
    for _ in range(cfg.test_rounds):
        b.project({"sa": ["A"]}, {"A": ["C"]})
    recovery_a = measure_overlap(c_merged, np.array(b.areas["C"].winners, dtype=np.uint32))

    # Test: B-only recovery
    b.areas["C"].winners = rng.choice(cfg.n, cfg.k, replace=False).tolist()
    for _ in range(cfg.test_rounds):
        b.project({"sb": ["B"]}, {"B": ["C"]})
    recovery_b = measure_overlap(c_merged, np.array(b.areas["C"].winners, dtype=np.uint32))

    return {"recovery_from_A": recovery_a, "recovery_from_B": recovery_b}


# -- Main experiment -----------------------------------------------------------


class MergeExperiment(ExperimentBase):
    """Test merge primitive: composition via co-projection."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="merge_composition",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "primitives",
            verbose=verbose,
        )

    # Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-merge-harness
    def run(self, n=1000, k=100, p=.05, beta=.1, w_max=20., n_seeds=None, *,
            seed_ids=None, establish_rounds=30, merge_rounds=30, test_rounds=20,
            round_values=None, h4_sizes=None):
        seeds = resolve_seed_ids(n_seeds, seed_ids, base_seed=self.seed, default_count=N_SEEDS)
        cfg = MergeConfig(n,k,p,beta,w_max,establish_rounds,merge_rounds,test_rounds)
        rounds = [validate_round_count(value) for value in
                  ([1,5,10,20,30,50] if round_values is None else round_values)]
        sizes = [validate_area_registration("A", value, 1)[0] for value in
                 ([200,500,1000,2000] if h4_sizes is None else h4_sizes)]
        if not rounds or len(set(rounds)) != len(rounds) or not sizes or len(set(sizes)) != len(sizes):
            raise ValueError("merge grids must be nonempty and unique")
        duration_configs = [replace(cfg, merge_rounds=value) for value in rounds]
        size_configs = [replace(cfg, n=value, k=int(np.sqrt(value))) for value in sizes]
        self._start_timer()
        cells = []

        def measure(arm, config, trial):
            rows = [trial(config, seed) for seed in seeds]
            # Rename historical arithmetic, without promoting it to a composition test.
            names = {"merge_quality": "mean_parent_overlap", "composition_score": "max_parent_overlap"}
            values = {names.get(key,key): [row[key] for row in rows] for key in rows[0]}
            cells.append({"arm": arm, "n": config.n, "k": config.k,
                          "merge_rounds": config.merge_rounds, "values": values})
            summaries = {key: summarize(column) for key,column in values.items()}
            return {"summaries": summaries,
                    "tests_vs_chance": {key: reported_null_test(column, chance_overlap(config.k,config.n))
                                        for key,column in values.items()
                                        if key == "mean_parent_overlap" or key.startswith("recovery_from_")}}

        metrics = {"parent_overlaps": measure("base",cfg,run_merge_trial)}
        metrics["overlaps_vs_rounds"] = [
            {"merge_rounds": item.merge_rounds, **measure("rounds",item,run_merge_trial)}
            for item in duration_configs]
        metrics["driven_recovery"] = measure("recovery",cfg,run_recovery_trial)
        metrics["overlaps_vs_size"] = [
            {"n": item.n, "k": item.k, **measure("size",item,run_merge_trial)} for item in size_configs]
        return ExperimentResult(
            experiment_name=self.name,
            parameters={"n_seeds": len(seeds), "seed_ids": seeds,
                        "base_n": cfg.n, "base_k": cfg.k, "base_p": cfg.p,
                        "base_beta": cfg.beta, "base_wmax": cfg.w_max,
                        "establish_rounds": cfg.establish_rounds, "merge_rounds": cfg.merge_rounds,
                        "test_rounds": cfg.test_rounds, "round_values": rounds, "h4_sizes": sizes,
                        "engine": "numpy_sparse", "area_engine": "numpy_explicit",
                        "reporting_version": "parent-overlaps-v1",
                        "size_assembly_rule": "floor(sqrt(n))",
                        "readout": "learning-on; separate composition and recovery training histories"},
            metrics=metrics, raw_data={"seed_ids": seeds, "cells": cells},
            duration_seconds=self._stop_timer())


def main(argv=None):
    from research.experiments.historical_merge import main as run
    return run(argv)


if __name__ == "__main__":
    main()
