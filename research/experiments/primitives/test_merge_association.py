"""
Merge-Association Compositionality

Tests whether merged assemblies are "first-class" — can they serve as
the source or target of associations, composing with other primitives?

This is the compositionality test for the Assembly Calculus. If merge
produces assemblies that fully participate in association, then the three
primitives (project, associate, merge) form a closed algebra: the output
of any operation can serve as input to any other. If merged assemblies
are degraded or can't drive associations, the framework is limited to
flat pairwise operations.

Protocol for merge:
    project({"sa": ["X"], "sb": ["Y"]}, {"X": ["M"], "Y": ["M"]})
    This fires stimuli for A (in area X) and B (in area Y) simultaneously,
    and projects both into area M, where they merge into a combined
    representation.

Hypotheses:

H1: Merge stability — Repeated merge operations produce a stable
    assembly in M that overlaps highly with itself across rounds.
    Null: merged assembly is unstable (overlap with itself is chance k/n).

H2: Merged assembly as association SOURCE — Train merge A+B→M, then
    train association M→D (where D is in a separate area Z). Test:
    re-activate the merge (fire both source stimuli), project through
    M→Z. Does Z recover D's trained assembly?
    Null: recovery at Z equals chance k/n.

H3: Merged assembly as association TARGET — Train association E→M
    (where E is in area W, M holds the merged assembly). Test: activate
    E's stimulus, project W→M. Does M recover the merged assembly?
    Null: recovery at M equals chance k/n.

H4: Full pipeline — Merge A+B→M, then associate M→D. Test: activate
    ONLY A and B's stimuli, project through the full pipeline
    (X→M, Y→M, M→Z). Does D recover at Z?
    This is the end-to-end compositionality test: stimulus → merge →
    association → recovery, with no direct stimulus at the merge area.
    Null: recovery at Z equals chance k/n.

H5: Merge-association vs direct association — Compare recovery at Z
    when the source is a merged assembly (M→Z) vs a direct stimulus-
    established assembly (direct association A→Z). Is the merged source
    weaker, equal, or stronger?
    Null: recovery is independent of source type.

Statistical methodology:
- N_SEEDS=10 independent random seeds per condition.
- One-sample t-test against null k/n.
- Paired t-test for H5.
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
    paired_ttest,
)

from src.core.brain import Brain

N_SEEDS = 10


@dataclass
class MergeAssocConfig:
    """Configuration for merge-association trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    merge_rounds: int = 30
    assoc_rounds: int = 30
    test_rounds: int = 15


# ── Core trial runners ──────────────────────────────────────────────


def run_merge_association_trial(
    cfg: MergeAssocConfig, seed: int
) -> Dict[str, Any]:
    """
    Full merge-association pipeline:
    1. Establish A in X, B in Y, D in Z via stim-only.
    2. Merge A+B into M.
    3. Train association M→Z (using co-stimulation to maintain both
       the merged assembly in M and D's assembly in Z).
    4. Test multiple recall modes.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    # Areas: X (holds A), Y (holds B), M (merge target), Z (assoc target)
    for name in ["X", "Y", "M", "Z"]:
        b.add_area(name, cfg.n, cfg.k, cfg.beta, explicit=True)

    b.add_stimulus("sa", cfg.k)  # drives A in X
    b.add_stimulus("sb", cfg.k)  # drives B in Y
    b.add_stimulus("sd", cfg.k)  # drives D in Z

    # ── Phase 1: Establish assemblies via stim-only ──
    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["X"]}, {})
    trained_a = np.array(b.areas["X"].winners, dtype=np.uint32)

    for _ in range(cfg.establish_rounds):
        b.project({"sb": ["Y"]}, {})
    trained_b = np.array(b.areas["Y"].winners, dtype=np.uint32)

    for _ in range(cfg.establish_rounds):
        b.project({"sd": ["Z"]}, {})
    trained_d = np.array(b.areas["Z"].winners, dtype=np.uint32)

    # ── Phase 2: Merge A+B into M ──
    for _ in range(cfg.merge_rounds):
        b.project({"sa": ["X"], "sb": ["Y"]}, {"X": ["M"], "Y": ["M"]})
    trained_m = np.array(b.areas["M"].winners, dtype=np.uint32)

    # H1: Merge stability — run merge again, check overlap with trained_m
    for _ in range(cfg.merge_rounds):
        b.project({"sa": ["X"], "sb": ["Y"]}, {"X": ["M"], "Y": ["M"]})
    merge_stability = measure_overlap(
        trained_m, np.array(b.areas["M"].winners, dtype=np.uint32)
    )

    # ── Phase 3: Train association M→Z ──
    # Co-stimulate: fire sa, sb to maintain merge in M; fire sd to maintain D in Z.
    # Project X→M, Y→M (maintain merge), M→Z (train association).
    for _ in range(cfg.assoc_rounds):
        b.project(
            {"sa": ["X"], "sb": ["Y"], "sd": ["Z"]},
            {"X": ["M"], "Y": ["M"], "M": ["Z"]},
        )

    # ── Phase 4: Test recall modes ──

    # H2: Merge as source — re-activate merge, project M→Z
    for _ in range(cfg.test_rounds):
        b.project(
            {"sa": ["X"], "sb": ["Y"]},
            {"X": ["M"], "Y": ["M"], "M": ["Z"]},
        )
    h2_recovery_z = measure_overlap(
        trained_d, np.array(b.areas["Z"].winners, dtype=np.uint32)
    )
    h2_recovery_m = measure_overlap(
        trained_m, np.array(b.areas["M"].winners, dtype=np.uint32)
    )

    # H4: Full pipeline — activate ONLY source stimuli, project through
    # full pipeline X→M, Y→M, M→Z. No direct stimulus at M or Z.
    # First re-establish source assemblies cleanly.
    for _ in range(5):
        b.project({"sa": ["X"]}, {})
    for _ in range(5):
        b.project({"sb": ["Y"]}, {})

    for _ in range(cfg.test_rounds):
        b.project(
            {"sa": ["X"], "sb": ["Y"]},
            {"X": ["M"], "Y": ["M"], "M": ["Z"]},
        )
    h4_recovery_z = measure_overlap(
        trained_d, np.array(b.areas["Z"].winners, dtype=np.uint32)
    )

    return {
        "merge_stability": merge_stability,
        "h2_recovery_z": h2_recovery_z,
        "h2_recovery_m": h2_recovery_m,
        "h4_recovery_z": h4_recovery_z,
    }


def run_merge_as_target_trial(
    cfg: MergeAssocConfig, seed: int
) -> Dict[str, float]:
    """
    H3: Merged assembly as association TARGET.
    Train E→M where M holds a merged assembly.
    Test: activate E, project W→M, measure M recovery.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    for name in ["X", "Y", "M", "W"]:
        b.add_area(name, cfg.n, cfg.k, cfg.beta, explicit=True)

    b.add_stimulus("sa", cfg.k)
    b.add_stimulus("sb", cfg.k)
    b.add_stimulus("se", cfg.k)

    # Establish source assemblies
    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["X"]}, {})
    for _ in range(cfg.establish_rounds):
        b.project({"sb": ["Y"]}, {})
    for _ in range(cfg.establish_rounds):
        b.project({"se": ["W"]}, {})

    # Merge A+B into M
    for _ in range(cfg.merge_rounds):
        b.project({"sa": ["X"], "sb": ["Y"]}, {"X": ["M"], "Y": ["M"]})
    trained_m = np.array(b.areas["M"].winners, dtype=np.uint32)

    # Train association E→M (W→M fiber)
    # Co-stimulate: fire se to maintain E in W, fire sa+sb to maintain merge in M
    # Project W→M
    for _ in range(cfg.assoc_rounds):
        b.project(
            {"se": ["W"], "sa": ["X"], "sb": ["Y"]},
            {"W": ["M"], "X": ["M"], "Y": ["M"]},
        )

    # Test: activate ONLY E's stimulus, project W→M (no merge stimuli)
    for _ in range(cfg.test_rounds):
        b.project({"se": ["W"]}, {"W": ["M"]})
    recovery_m = measure_overlap(
        trained_m, np.array(b.areas["M"].winners, dtype=np.uint32)
    )

    return {"recovery_m": recovery_m}


def run_direct_association_trial(
    cfg: MergeAssocConfig, seed: int
) -> Dict[str, float]:
    """
    H5 baseline: Direct association A→Z (no merge involved).
    For comparison with merge-sourced association.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_area("Z", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("sa", cfg.k)
    b.add_stimulus("sd", cfg.k)

    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["A"]}, {})

    for _ in range(cfg.establish_rounds):
        b.project({"sd": ["Z"]}, {})
    trained_d = np.array(b.areas["Z"].winners, dtype=np.uint32)

    # Train direct association A→Z
    for _ in range(cfg.assoc_rounds):
        b.project({"sa": ["A"], "sd": ["Z"]}, {"A": ["Z"]})

    # Test
    for _ in range(cfg.test_rounds):
        b.project({"sa": ["A"]}, {"A": ["Z"]})
    recovery = measure_overlap(
        trained_d, np.array(b.areas["Z"].winners, dtype=np.uint32)
    )

    return {"recovery_z": recovery}


# ── Main experiment ──────────────────────────────────────────────────


class MergeAssociationExperiment(ExperimentBase):
    """Test whether merged assemblies compose with associations."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="merge_association",
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

        cfg = MergeAssocConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Merge-Association Compositionality Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics = {}

        # ================================================================
        # H1/H2/H4: Merge stability, merge-as-source, full pipeline
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H1/H2/H4: Merge → Association pipeline")
        self.log("=" * 60)

        merge_stab = []
        h2_rec_z = []
        h2_rec_m = []
        h4_rec_z = []

        for s in seeds:
            trial = run_merge_association_trial(cfg, seed=self.seed + s)
            merge_stab.append(trial["merge_stability"])
            h2_rec_z.append(trial["h2_recovery_z"])
            h2_rec_m.append(trial["h2_recovery_m"])
            h4_rec_z.append(trial["h4_recovery_z"])

        stab_stats = summarize(merge_stab)
        h2z_stats = summarize(h2_rec_z)
        h2m_stats = summarize(h2_rec_m)
        h4z_stats = summarize(h4_rec_z)

        stab_test = ttest_vs_null(merge_stab, null)
        h2z_test = ttest_vs_null(h2_rec_z, null)
        h4z_test = ttest_vs_null(h4_rec_z, null)

        metrics["h1_merge_stability"] = {"stats": stab_stats, "test_vs_chance": stab_test}
        metrics["h2_merge_as_source"] = {
            "recovery_z": {"stats": h2z_stats, "test_vs_chance": h2z_test},
            "recovery_m": {"stats": h2m_stats},
        }
        metrics["h4_full_pipeline"] = {"stats": h4z_stats, "test_vs_chance": h4z_test}

        self.log(f"  H1 Merge stability:     {stab_stats['mean']:.3f}+/-{stab_stats['sem']:.3f}  d={stab_test['d']:.1f}")
        self.log(f"  H2 Merge→assoc (at Z):  {h2z_stats['mean']:.3f}+/-{h2z_stats['sem']:.3f}  d={h2z_test['d']:.1f}")
        self.log(f"     Merge maint (at M):   {h2m_stats['mean']:.3f}+/-{h2m_stats['sem']:.3f}")
        self.log(f"  H4 Full pipeline (at Z): {h4z_stats['mean']:.3f}+/-{h4z_stats['sem']:.3f}  d={h4z_test['d']:.1f}")

        # ================================================================
        # H3: Merged assembly as association TARGET
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H3: Merged assembly as association target (E→M)")
        self.log("=" * 60)

        h3_rec = []
        for s in seeds:
            trial = run_merge_as_target_trial(cfg, seed=self.seed + s)
            h3_rec.append(trial["recovery_m"])

        h3_stats = summarize(h3_rec)
        h3_test = ttest_vs_null(h3_rec, null)

        metrics["h3_merge_as_target"] = {"stats": h3_stats, "test_vs_chance": h3_test}

        self.log(
            f"  E→M recovery:  {h3_stats['mean']:.3f}+/-{h3_stats['sem']:.3f}  "
            f"d={h3_test['d']:.1f}"
        )

        # ================================================================
        # H5: Merge-sourced vs direct association
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H5: Merge-sourced vs direct association quality")
        self.log("=" * 60)

        direct_rec = []
        for s in seeds:
            trial = run_direct_association_trial(cfg, seed=self.seed + s)
            direct_rec.append(trial["recovery_z"])

        direct_stats = summarize(direct_rec)
        direct_test = ttest_vs_null(direct_rec, null)

        paired = paired_ttest(h2_rec_z, direct_rec)

        metrics["h5_merge_vs_direct"] = {
            "merge_sourced": {"stats": h2z_stats, "test_vs_chance": h2z_test},
            "direct": {"stats": direct_stats, "test_vs_chance": direct_test},
            "paired_test": paired,
        }

        self.log(f"  Merge-sourced (M→Z): {h2z_stats['mean']:.3f}+/-{h2z_stats['sem']:.3f}")
        self.log(f"  Direct (A→Z):        {direct_stats['mean']:.3f}+/-{direct_stats['sem']:.3f}")
        self.log(
            f"  Paired test: t={paired['t']:.2f}, p={paired['p']:.4f}  "
            f"{'SIGNIFICANT' if paired['sig'] else 'no significant difference'}"
        )

        # ================================================================
        # H6: Size scaling (k=sqrt(n))
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H6: Merge-association at sparse coding (k=sqrt(n))")
        self.log("=" * 60)

        h6_sizes = [500, 1000, 2000, 5000]
        h6_results = []

        for n_val in h6_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h6 = MergeAssocConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h6 = chance_overlap(k_val, n_val)

            rec_vals = []
            for s in seeds:
                trial = run_merge_association_trial(cfg_h6, seed=self.seed + s)
                rec_vals.append(trial["h2_recovery_z"])

            rec_s = summarize(rec_vals)
            rec_t = ttest_vs_null(rec_vals, null_h6)

            row = {"n": n_val, "k": k_val, "stats": rec_s, "test_vs_chance": rec_t}
            h6_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"merge→assoc={rec_s['mean']:.3f}+/-{rec_s['sem']:.3f}  "
                f"d={rec_t['d']:.1f}"
            )

        metrics["h6_size_scaling"] = h6_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": n, "k": k, "p": p, "beta": beta, "w_max": w_max,
                "establish_rounds": cfg.establish_rounds,
                "merge_rounds": cfg.merge_rounds,
                "assoc_rounds": cfg.assoc_rounds,
                "test_rounds": cfg.test_rounds,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge-Association Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = MergeAssociationExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("MERGE-ASSOCIATION SUMMARY")
    print("=" * 70)

    null = result.parameters["k"] / result.parameters["n"]
    print(f"\nchance = {null:.3f}")

    print(f"\nH1 Merge stability:       {result.metrics['h1_merge_stability']['stats']['mean']:.3f}")
    print(f"H2 Merge as source (Z):   {result.metrics['h2_merge_as_source']['recovery_z']['stats']['mean']:.3f}")
    print(f"H3 Merge as target (M):   {result.metrics['h3_merge_as_target']['stats']['mean']:.3f}")
    print(f"H4 Full pipeline (Z):     {result.metrics['h4_full_pipeline']['stats']['mean']:.3f}")

    h5 = result.metrics["h5_merge_vs_direct"]
    print(f"\nH5 Merge-sourced: {h5['merge_sourced']['stats']['mean']:.3f}  "
          f"Direct: {h5['direct']['stats']['mean']:.3f}  "
          f"sig={h5['paired_test']['sig']}")

    print("\nH6 Size scaling:")
    for r in result.metrics["h6_size_scaling"]:
        print(f"  n={r['n']:4d}: {r['stats']['mean']:.3f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
