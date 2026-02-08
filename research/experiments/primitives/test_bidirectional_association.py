"""
Bidirectional Association

Tests whether cross-area associations are inherently directional or can
support symmetric recall. After training A→B (unidirectional), does
activating B recover A? After explicitly training both A→B and B→A,
do both directions coexist without interference?

Biological context: Associative memory is typically symmetric — seeing a
face recalls a name, hearing a name recalls a face. If Assembly Calculus
associations are inherently directional (only the trained fiber direction
works), the framework needs explicit bidirectional training for symmetric
recall. If reverse recall emerges implicitly from co-stimulation, the
framework naturally supports the symmetric associative memory observed
in hippocampal-cortical circuits.

Protocol:
1. Create two areas A, B with independent stimulus pathways.
2. Establish assemblies via stim-only (30 rounds each).
3. Train associations (unidirectional or bidirectional) via co-stimulation.
4. Test forward recall: activate A's stimulus, project A→B, measure B.
5. Test reverse recall: activate B's stimulus, project B→A, measure A.

Hypotheses:

H1: Forward recall (baseline) — Training A→B via co-stimulation
    produces high recovery at B when A's stimulus is activated.
    Null: recovery equals chance k/n.

H2: Implicit reverse recall — After training ONLY A→B (unidirectional
    fiber), does activating B's stimulus and projecting B→A recover A's
    assembly? The B→A fiber was never explicitly trained, but both
    assemblies were co-active during training.
    Null: reverse recovery equals chance k/n.
    Prediction: reverse recall should be AT CHANCE because the B→A
    fiber connections were never strengthened — only A→B fiber was
    in the projection map during training.

H3: Explicit bidirectional — Train both A→B and B→A fibers
    (either sequentially or simultaneously). Both directions should
    show high recovery.
    Null: recovery in either direction equals chance k/n.

H4: Bidirectional vs unidirectional quality — Does training both
    directions degrade the forward direction compared to unidirectional-
    only training? The A→B connectome is trained identically in both
    cases; the question is whether additional B→A training somehow
    interferes.
    Null: forward recovery is independent of whether reverse was trained.

Statistical methodology:
- N_SEEDS=10 independent random seeds per condition.
- One-sample t-test against null k/n.
- Paired t-test for H4 (bidirectional vs unidirectional forward).
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
class BidirConfig:
    """Configuration for bidirectional association trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    assoc_rounds: int = 30
    test_rounds: int = 15


# ── Core trial runners ──────────────────────────────────────────────


def run_unidirectional_trial(
    cfg: BidirConfig, seed: int
) -> Dict[str, float]:
    """
    Train ONLY A→B fiber. Test both forward (A→B) and reverse (B→A).
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_area("B", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("sa", cfg.k)
    b.add_stimulus("sb", cfg.k)

    # Establish assemblies
    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["A"]}, {})
    trained_a = np.array(b.areas["A"].winners, dtype=np.uint32)

    for _ in range(cfg.establish_rounds):
        b.project({"sb": ["B"]}, {})
    trained_b = np.array(b.areas["B"].winners, dtype=np.uint32)

    # Train A→B only (unidirectional)
    for _ in range(cfg.assoc_rounds):
        b.project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]})

    # Test forward: activate A, project A→B
    for _ in range(cfg.test_rounds):
        b.project({"sa": ["A"]}, {"A": ["B"]})
    forward = measure_overlap(trained_b, np.array(b.areas["B"].winners, dtype=np.uint32))

    # Test reverse: activate B, project B→A
    for _ in range(cfg.test_rounds):
        b.project({"sb": ["B"]}, {"B": ["A"]})
    reverse = measure_overlap(trained_a, np.array(b.areas["A"].winners, dtype=np.uint32))

    return {"forward": forward, "reverse": reverse}


def run_bidirectional_trial(
    cfg: BidirConfig, seed: int, simultaneous: bool = True
) -> Dict[str, float]:
    """
    Train BOTH A→B and B→A fibers. Test both directions.

    If simultaneous=True, train both fibers in each round:
        project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"], "B": ["A"]})
    If simultaneous=False, train A→B first, then B→A:
        project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]}) x assoc_rounds
        project({"sa": ["A"], "sb": ["B"]}, {"B": ["A"]}) x assoc_rounds
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    b.add_area("A", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_area("B", cfg.n, cfg.k, cfg.beta, explicit=True)
    b.add_stimulus("sa", cfg.k)
    b.add_stimulus("sb", cfg.k)

    # Establish assemblies
    for _ in range(cfg.establish_rounds):
        b.project({"sa": ["A"]}, {})
    trained_a = np.array(b.areas["A"].winners, dtype=np.uint32)

    for _ in range(cfg.establish_rounds):
        b.project({"sb": ["B"]}, {})
    trained_b = np.array(b.areas["B"].winners, dtype=np.uint32)

    # Train bidirectional
    if simultaneous:
        for _ in range(cfg.assoc_rounds):
            b.project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"], "B": ["A"]})
    else:
        # Sequential: A→B first, then B→A
        for _ in range(cfg.assoc_rounds):
            b.project({"sa": ["A"], "sb": ["B"]}, {"A": ["B"]})
        for _ in range(cfg.assoc_rounds):
            b.project({"sa": ["A"], "sb": ["B"]}, {"B": ["A"]})

    # Test forward: activate A, project A→B
    for _ in range(cfg.test_rounds):
        b.project({"sa": ["A"]}, {"A": ["B"]})
    forward = measure_overlap(trained_b, np.array(b.areas["B"].winners, dtype=np.uint32))

    # Test reverse: activate B, project B→A
    for _ in range(cfg.test_rounds):
        b.project({"sb": ["B"]}, {"B": ["A"]})
    reverse = measure_overlap(trained_a, np.array(b.areas["A"].winners, dtype=np.uint32))

    return {"forward": forward, "reverse": reverse}


# ── Main experiment ──────────────────────────────────────────────────


class BidirectionalAssociationExperiment(ExperimentBase):
    """Test directionality of cross-area associations."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="bidirectional_association",
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

        cfg = BidirConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Bidirectional Association Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  establish_rounds={cfg.establish_rounds}")
        self.log(f"  assoc_rounds={cfg.assoc_rounds}")
        self.log(f"  test_rounds={cfg.test_rounds}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics = {}

        # ================================================================
        # H1/H2: Unidirectional training — forward and reverse recall
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H1/H2: Unidirectional training (A→B fiber only)")
        self.log("=" * 60)

        uni_forward = []
        uni_reverse = []

        for s in seeds:
            trial = run_unidirectional_trial(cfg, seed=self.seed + s)
            uni_forward.append(trial["forward"])
            uni_reverse.append(trial["reverse"])

        fwd_stats = summarize(uni_forward)
        rev_stats = summarize(uni_reverse)
        fwd_test = ttest_vs_null(uni_forward, null)
        rev_test = ttest_vs_null(uni_reverse, null)

        metrics["h1_forward_unidirectional"] = {
            "stats": fwd_stats, "test_vs_chance": fwd_test,
        }
        metrics["h2_reverse_unidirectional"] = {
            "stats": rev_stats, "test_vs_chance": rev_test,
        }

        self.log(
            f"  Forward (A→B):  {fwd_stats['mean']:.3f}+/-{fwd_stats['sem']:.3f}  "
            f"d={fwd_test['d']:.1f}"
        )
        self.log(
            f"  Reverse (B→A):  {rev_stats['mean']:.3f}+/-{rev_stats['sem']:.3f}  "
            f"d={rev_test['d']:.1f}  "
            f"{'ABOVE CHANCE' if rev_test['sig'] else 'at chance'}"
        )

        # ================================================================
        # H3: Explicit bidirectional training (simultaneous)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H3: Bidirectional training (simultaneous A→B + B→A)")
        self.log("=" * 60)

        bidir_sim_forward = []
        bidir_sim_reverse = []

        for s in seeds:
            trial = run_bidirectional_trial(cfg, seed=self.seed + s, simultaneous=True)
            bidir_sim_forward.append(trial["forward"])
            bidir_sim_reverse.append(trial["reverse"])

        sim_fwd_stats = summarize(bidir_sim_forward)
        sim_rev_stats = summarize(bidir_sim_reverse)
        sim_fwd_test = ttest_vs_null(bidir_sim_forward, null)
        sim_rev_test = ttest_vs_null(bidir_sim_reverse, null)

        metrics["h3_bidirectional_simultaneous"] = {
            "forward": {"stats": sim_fwd_stats, "test_vs_chance": sim_fwd_test},
            "reverse": {"stats": sim_rev_stats, "test_vs_chance": sim_rev_test},
        }

        self.log(
            f"  Forward (A→B):  {sim_fwd_stats['mean']:.3f}+/-{sim_fwd_stats['sem']:.3f}  "
            f"d={sim_fwd_test['d']:.1f}"
        )
        self.log(
            f"  Reverse (B→A):  {sim_rev_stats['mean']:.3f}+/-{sim_rev_stats['sem']:.3f}  "
            f"d={sim_rev_test['d']:.1f}"
        )

        # Also test sequential bidirectional training
        self.log("")
        self.log("  Sequential bidirectional (A→B first, then B→A):")

        bidir_seq_forward = []
        bidir_seq_reverse = []

        for s in seeds:
            trial = run_bidirectional_trial(cfg, seed=self.seed + s, simultaneous=False)
            bidir_seq_forward.append(trial["forward"])
            bidir_seq_reverse.append(trial["reverse"])

        seq_fwd_stats = summarize(bidir_seq_forward)
        seq_rev_stats = summarize(bidir_seq_reverse)
        seq_fwd_test = ttest_vs_null(bidir_seq_forward, null)
        seq_rev_test = ttest_vs_null(bidir_seq_reverse, null)

        metrics["h3_bidirectional_sequential"] = {
            "forward": {"stats": seq_fwd_stats, "test_vs_chance": seq_fwd_test},
            "reverse": {"stats": seq_rev_stats, "test_vs_chance": seq_rev_test},
        }

        self.log(
            f"  Forward (A→B):  {seq_fwd_stats['mean']:.3f}+/-{seq_fwd_stats['sem']:.3f}  "
            f"d={seq_fwd_test['d']:.1f}"
        )
        self.log(
            f"  Reverse (B→A):  {seq_rev_stats['mean']:.3f}+/-{seq_rev_stats['sem']:.3f}  "
            f"d={seq_rev_test['d']:.1f}"
        )

        # ================================================================
        # H4: Bidirectional vs unidirectional forward quality
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H4: Does bidirectional training degrade forward recall?")
        self.log("=" * 60)

        # Compare unidirectional forward vs simultaneous bidirectional forward
        paired = paired_ttest(uni_forward, bidir_sim_forward)
        metrics["h4_bidir_vs_unidir"] = {
            "unidirectional_forward": fwd_stats,
            "bidirectional_forward": sim_fwd_stats,
            "paired_test": paired,
        }

        self.log(
            f"  Unidirectional forward:  {fwd_stats['mean']:.3f}+/-{fwd_stats['sem']:.3f}"
        )
        self.log(
            f"  Bidirectional forward:   {sim_fwd_stats['mean']:.3f}+/-{sim_fwd_stats['sem']:.3f}"
        )
        self.log(
            f"  Paired t-test: t={paired['t']:.2f}, p={paired['p']:.4f}, "
            f"d={paired['d']:.2f}  "
            f"{'SIGNIFICANT DIFFERENCE' if paired['sig'] else 'no significant difference'}"
        )

        # ================================================================
        # H5: Size scaling at k=sqrt(n)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H5: Bidirectional recall vs network size (k=sqrt(n))")
        self.log("=" * 60)

        h5_sizes = [500, 1000, 2000, 5000]
        h5_results = []

        for n_val in h5_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h5 = BidirConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h5 = chance_overlap(k_val, n_val)

            fwd_vals = []
            rev_vals = []

            for s in seeds:
                trial = run_bidirectional_trial(cfg_h5, seed=self.seed + s, simultaneous=True)
                fwd_vals.append(trial["forward"])
                rev_vals.append(trial["reverse"])

            fwd_s = summarize(fwd_vals)
            rev_s = summarize(rev_vals)
            fwd_t = ttest_vs_null(fwd_vals, null_h5)
            rev_t = ttest_vs_null(rev_vals, null_h5)

            row = {
                "n": n_val, "k": k_val,
                "forward": {"stats": fwd_s, "test_vs_chance": fwd_t},
                "reverse": {"stats": rev_s, "test_vs_chance": rev_t},
            }
            h5_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"fwd={fwd_s['mean']:.3f}+/-{fwd_s['sem']:.3f}  "
                f"rev={rev_s['mean']:.3f}+/-{rev_s['sem']:.3f}"
            )

        metrics["h5_size_scaling"] = h5_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": n, "k": k, "p": p, "beta": beta, "w_max": w_max,
                "establish_rounds": cfg.establish_rounds,
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

    parser = argparse.ArgumentParser(description="Bidirectional Association Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = BidirectionalAssociationExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("BIDIRECTIONAL ASSOCIATION SUMMARY")
    print("=" * 70)

    null = result.parameters["k"] / result.parameters["n"]

    print(f"\nH1/H2 -- Unidirectional training (chance={null:.3f}):")
    h1 = result.metrics["h1_forward_unidirectional"]
    h2 = result.metrics["h2_reverse_unidirectional"]
    print(f"  Forward (A→B): {h1['stats']['mean']:.3f}")
    print(f"  Reverse (B→A): {h2['stats']['mean']:.3f}")

    print("\nH3 -- Bidirectional training:")
    h3s = result.metrics["h3_bidirectional_simultaneous"]
    h3q = result.metrics["h3_bidirectional_sequential"]
    print(f"  Simultaneous: fwd={h3s['forward']['stats']['mean']:.3f}  rev={h3s['reverse']['stats']['mean']:.3f}")
    print(f"  Sequential:   fwd={h3q['forward']['stats']['mean']:.3f}  rev={h3q['reverse']['stats']['mean']:.3f}")

    print("\nH4 -- Bidirectional vs unidirectional forward:")
    h4 = result.metrics["h4_bidir_vs_unidir"]
    print(f"  Unidirectional: {h4['unidirectional_forward']['mean']:.3f}")
    print(f"  Bidirectional:  {h4['bidirectional_forward']['mean']:.3f}")
    print(f"  Difference significant: {h4['paired_test']['sig']}")

    print("\nH5 -- Size scaling (k=sqrt(n), bidirectional):")
    for r in result.metrics["h5_size_scaling"]:
        print(f"  n={r['n']:4d}: fwd={r['forward']['stats']['mean']:.3f}  rev={r['reverse']['stats']['mean']:.3f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
