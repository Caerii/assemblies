"""Historical merge: sequential trained overlaps and learning-on driven recovery.

Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#historical-merge-trials

The composition trial trains A->C, then B->C, then joint A/B->C on one brain.
Replacing C winners does not clear learned weights or affect subsequent drive:
C has no outgoing fiber. Earlier training therefore persists into the joint phase.
The reported merge_quality is the average parent overlap; composition_score is
its maximum. Neither certifies that both parents are represented. Keep the
individual overlaps visible, particularly when one parent dominates.

The separate recovery trial skips the isolated-parent C training. After joint
training, it reads from A for 20 rounds and then B for 20 rounds, while learning
continues. This is not isolated frozen partial-cue completion. The historical
outer harness and its scientific hypotheses still require migration; these
observables alone do not establish an adopted merge result.
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
    ttest_vs_null,
)

from neural_assemblies.core.brain import Brain

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
    for _ in range(20):
        b.project({"sa": ["A"]}, {"A": ["C"]})
    recovery_a = measure_overlap(c_merged, np.array(b.areas["C"].winners, dtype=np.uint32))

    # Test: B-only recovery
    b.areas["C"].winners = rng.choice(cfg.n, cfg.k, replace=False).tolist()
    for _ in range(20):
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

    # Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#legacy-experiment-configuration
    def run(
        self,
        n: int = 1000,
        k: int = 100,
        p: float = 0.05,
        beta: float = 0.10,
        w_max: float = 20.0,
        n_seeds: int = N_SEEDS,
    ) -> ExperimentResult:
        self._start_timer()
        seeds = list(range(n_seeds))

        cfg = MergeConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Merge Composition Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  establish_rounds={cfg.establish_rounds}")
        self.log(f"  merge_rounds={cfg.merge_rounds}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}

        # ================================================================
        # H1: Composition quality
        # ================================================================
        self.log("\nH1: Merge Composition Quality")

        mq_vals, cs_vals = [], []
        ab_a_vals, ab_b_vals, a_b_vals = [], [], []

        for s in seeds:
            trial = run_merge_trial(cfg, seed=self.seed + s)
            mq_vals.append(trial["merge_quality"])
            cs_vals.append(trial["composition_score"])
            ab_a_vals.append(trial["overlap_cab_ca"])
            ab_b_vals.append(trial["overlap_cab_cb"])
            a_b_vals.append(trial["overlap_ca_cb"])

        metrics["h1_composition"] = {
            "merge_quality": summarize(mq_vals),
            "composition_score": summarize(cs_vals),
            "overlap_cab_ca": summarize(ab_a_vals),
            "overlap_cab_cb": summarize(ab_b_vals),
            "overlap_ca_cb": summarize(a_b_vals),
            "test_quality_vs_chance": ttest_vs_null(mq_vals, null),
        }

        self.log(f"  Merge quality: {summarize(mq_vals)['mean']:.3f}+/-{summarize(mq_vals)['sem']:.3f}")
        self.log(f"  C_AB vs C_A:   {summarize(ab_a_vals)['mean']:.3f}")
        self.log(f"  C_AB vs C_B:   {summarize(ab_b_vals)['mean']:.3f}")
        self.log(f"  C_A vs C_B:    {summarize(a_b_vals)['mean']:.3f} (should be ~chance)")

        # ================================================================
        # H2: Quality vs merge rounds
        # ================================================================
        self.log("\nH2: Merge Quality vs Training Rounds")

        round_values = [1, 5, 10, 20, 30, 50]
        h2_results = []

        for n_rounds in round_values:
            cfg_h2 = MergeConfig(n=n, k=k, p=p, beta=beta, w_max=w_max,
                                 merge_rounds=n_rounds)
            vals = []
            for s in seeds:
                trial = run_merge_trial(cfg_h2, seed=self.seed + s)
                vals.append(trial["merge_quality"])

            row = {
                "merge_rounds": n_rounds,
                "merge_quality": summarize(vals),
                "test_vs_chance": ttest_vs_null(vals, null),
            }
            h2_results.append(row)

            self.log(
                f"  rounds={n_rounds:2d}: "
                f"{row['merge_quality']['mean']:.3f}+/-{row['merge_quality']['sem']:.3f}  "
                f"d={row['test_vs_chance']['d']:.1f}"
            )

        metrics["h2_quality_vs_rounds"] = h2_results

        # ================================================================
        # H3: Partial recovery
        # ================================================================
        self.log("\nH3: Partial Recovery (single parent recovers merge)")

        rec_a_vals = []
        rec_b_vals = []

        for s in seeds:
            trial = run_recovery_trial(cfg, seed=self.seed + s)
            rec_a_vals.append(trial["recovery_from_A"])
            rec_b_vals.append(trial["recovery_from_B"])

        metrics["h3_recovery"] = {
            "recovery_from_A": {
                "stats": summarize(rec_a_vals),
                "test_vs_chance": ttest_vs_null(rec_a_vals, null),
            },
            "recovery_from_B": {
                "stats": summarize(rec_b_vals),
                "test_vs_chance": ttest_vs_null(rec_b_vals, null),
            },
        }

        self.log(f"  A-only recovery: {summarize(rec_a_vals)['mean']:.3f}")
        self.log(f"  B-only recovery: {summarize(rec_b_vals)['mean']:.3f}")

        # ================================================================
        # H4: Quality vs network size (k=sqrt(n))
        # ================================================================
        self.log("\nH4: Merge Quality vs Network Size (k=sqrt(n))")

        h4_sizes = [200, 500, 1000, 2000]
        h4_results = []

        for n_val in h4_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h4 = MergeConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h4 = chance_overlap(k_val, n_val)

            vals = []
            for s in seeds:
                trial = run_merge_trial(cfg_h4, seed=self.seed + s)
                vals.append(trial["merge_quality"])

            row = {
                "n": n_val, "k": k_val,
                "merge_quality": summarize(vals),
                "test_vs_chance": ttest_vs_null(vals, null_h4),
            }
            h4_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"{row['merge_quality']['mean']:.3f}  d={row['test_vs_chance']['d']:.1f}"
            )

        metrics["h4_quality_vs_size"] = h4_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_seeds": n_seeds,
                "base_n": n, "base_k": k, "base_p": p,
                "base_beta": beta, "base_wmax": w_max,
                "establish_rounds": cfg.establish_rounds,
                "merge_rounds": cfg.merge_rounds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge Composition Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = MergeExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
