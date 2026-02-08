"""
Inter-Area Inhibition Primitives

Tests the two inter-area inhibition mechanisms that complete the six
principles of the Nemo framework (Mitropolsky & Papadimitriou, 2025):

1. Trigger inhibition (brain.inhibit_areas): Suppress all activity in
   specified areas for one projection step, clearing residual patterns.

2. Mutual inhibition (brain.add_mutual_inhibition): When competing
   areas receive simultaneous input, only the area with the greatest
   total synaptic drive fires; all others are silenced.

Together with excitatory neurons, brain areas, random synapses, Hebbian
plasticity, and local inhibition (k-cap), these complete the Nemo model.

Hypotheses:

H1: Trigger inhibition clears assembly activity — after triggering,
    the inhibited area has no active neurons.

H2: Trigger inhibition preserves learned connections — re-stimulating
    the area after trigger reset recovers the original assembly to
    near-perfect overlap, proving the connectome is undamaged.

H3: Mutual inhibition selects the area with stronger drive — when two
    areas in mutual inhibition both receive input, the one with more
    potentiated connections fires and the other is completely silenced.

H4: Trigger-assisted switching in recurrent circuit — in a 2-area
    bidirectional loop with two trained patterns, trigger reset before
    switching dramatically improves the new pattern's takeover quality
    compared to direct switching.

H5: Trigger-assisted switching at k=sqrt(n) — scaling behavior.

Statistical methodology:
- N_SEEDS=10 independent random seeds per condition.
- One-sample t-test against null k/n.
- Paired t-test for H4 (baseline vs trigger, same seed).
- Cohen's d effect sizes. Mean +/- SEM.

References:
- Mitropolsky & Papadimitriou, arXiv:2507.11788, 2025
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
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
class InhibitionConfig:
    """Configuration for inhibition experiments."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    kick_rounds: int = 15
    autonomous_rounds: int = 20


# ── Core trial runners ──────────────────────────────────────────────


def build_bidir_loop(
    n: int, k: int, p: float, beta: float, w_max: float,
    establish_rounds: int, seed: int,
) -> tuple:
    """Build a 2-area bidirectional loop with 2 trained patterns.

    Returns (brain, trained_1x, trained_2x, loop_map).
    trained_Nx is the pattern N assembly in area X after establishment.
    """
    b = Brain(p=p, seed=seed, w_max=w_max)
    b.add_area("X", n, k, beta, explicit=True)
    b.add_area("Y", n, k, beta, explicit=True)
    b.add_stimulus("s1x", k)
    b.add_stimulus("s1y", k)
    b.add_stimulus("s2x", k)
    b.add_stimulus("s2y", k)

    loop_map = {"X": ["Y"], "Y": ["X"]}

    # Establish assemblies for both patterns
    for _ in range(establish_rounds):
        b.project({"s1x": ["X"]}, {})
    trained_1x = np.array(b.areas["X"].winners, dtype=np.uint32)

    for _ in range(establish_rounds):
        b.project({"s1y": ["Y"]}, {})

    for _ in range(establish_rounds):
        b.project({"s2x": ["X"]}, {})
    trained_2x = np.array(b.areas["X"].winners, dtype=np.uint32)

    for _ in range(establish_rounds):
        b.project({"s2y": ["Y"]}, {})

    # Train loop associations for each pattern
    for _ in range(establish_rounds):
        b.project({"s1x": ["X"], "s1y": ["Y"]}, loop_map)

    for _ in range(establish_rounds):
        b.project({"s2x": ["X"], "s2y": ["Y"]}, loop_map)

    return b, trained_1x, trained_2x, loop_map


def run_switch_trial(
    b: Brain, loop_map: Dict, kick_rounds: int, autonomous_rounds: int,
    use_trigger: bool,
) -> tuple:
    """Kick pattern 1, run autonomous, switch to pattern 2, measure.

    Args:
        b: Brain with trained patterns (areas X, Y with stimuli s1x, s2x).
        loop_map: Bidirectional loop projection map.
        kick_rounds: Rounds to drive with stimulus during kick-start.
        autonomous_rounds: Rounds of autonomous circulation after kick.
        use_trigger: If True, inhibit both areas before switching.

    Returns:
        (final_X_winners,) — winners in area X after pattern 2 autonomous.
    """
    # Kick-start pattern 1
    for _ in range(kick_rounds):
        b.project({"s1x": ["X"]}, loop_map)
    # Autonomous circulation of pattern 1
    for _ in range(autonomous_rounds):
        b.project({}, loop_map)

    # Switch to pattern 2
    if use_trigger:
        b.inhibit_areas(["X", "Y"])
        b.project({}, loop_map)

    for _ in range(kick_rounds):
        b.project({"s2x": ["X"]}, loop_map)
    for _ in range(autonomous_rounds):
        b.project({}, loop_map)

    return np.array(b.areas["X"].winners, dtype=np.uint32)


# ── Main experiment ──────────────────────────────────────────────────


class InhibitionExperiment(ExperimentBase):
    """Test inter-area inhibition primitives."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="inhibition",
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

        cfg = InhibitionConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Inter-Area Inhibition Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  establish_rounds={cfg.establish_rounds}")
        self.log(f"  kick_rounds={cfg.kick_rounds}")
        self.log(f"  autonomous_rounds={cfg.autonomous_rounds}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics = {}

        # ================================================================
        # H1: Trigger inhibition clears assembly activity
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H1: Trigger inhibition clears active assembly")
        self.log("=" * 60)

        h1_pre = []
        h1_post = []

        for s in seeds:
            b = Brain(p=p, seed=self.seed + s, w_max=w_max)
            b.add_area("X", n, k, beta, explicit=True)
            b.add_stimulus("stim", k)

            # Establish assembly
            for _ in range(cfg.establish_rounds):
                b.project({"stim": ["X"]}, {})
            trained = np.array(b.areas["X"].winners, dtype=np.uint32)

            # Verify stable
            b.project({"stim": ["X"]}, {})
            h1_pre.append(measure_overlap(trained, b.areas["X"].winners))

            # Trigger inhibit
            b.inhibit_areas(["X"])
            b.project({}, {})

            # Post-inhibition: area should have no active neurons
            winners_after = b.areas["X"].winners
            if len(winners_after) == 0:
                h1_post.append(0.0)
            else:
                h1_post.append(measure_overlap(trained, winners_after))

        h1_pre_stats = summarize(h1_pre)
        h1_post_stats = summarize(h1_post)
        metrics["h1_trigger_clear"] = {
            "pre_inhibit": h1_pre_stats,
            "post_inhibit": h1_post_stats,
        }

        self.log(
            f"  Pre-inhibit overlap:  {h1_pre_stats['mean']:.3f}+/-{h1_pre_stats['sem']:.3f}"
        )
        self.log(
            f"  Post-inhibit overlap: {h1_post_stats['mean']:.3f}+/-{h1_post_stats['sem']:.3f}"
        )

        # ================================================================
        # H2: Trigger preserves connectome (reversible)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H2: Trigger inhibition preserves learned connections")
        self.log("=" * 60)

        h2_recovery = []

        for s in seeds:
            b = Brain(p=p, seed=self.seed + s, w_max=w_max)
            b.add_area("X", n, k, beta, explicit=True)
            b.add_stimulus("stim", k)

            for _ in range(cfg.establish_rounds):
                b.project({"stim": ["X"]}, {})
            trained = np.array(b.areas["X"].winners, dtype=np.uint32)

            # Inhibit
            b.inhibit_areas(["X"])
            b.project({}, {})

            # Re-stimulate — assembly should recover
            for _ in range(5):
                b.project({"stim": ["X"]}, {})
            h2_recovery.append(measure_overlap(trained, b.areas["X"].winners))

        h2_stats = summarize(h2_recovery)
        h2_test = ttest_vs_null(h2_recovery, null)
        metrics["h2_recovery"] = {"stats": h2_stats, "test_vs_chance": h2_test}

        self.log(
            f"  Recovery overlap: {h2_stats['mean']:.3f}+/-{h2_stats['sem']:.3f}  "
            f"d={h2_test['d']:.1f}"
        )

        # ================================================================
        # H3: Mutual inhibition selects stronger drive
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H3: Mutual inhibition selects area with stronger drive")
        self.log("=" * 60)

        h3_winner = []
        h3_loser_active = []

        for s in seeds:
            b = Brain(p=p, seed=self.seed + s, w_max=w_max)
            b.add_area("A", n, k, beta, explicit=True)
            b.add_area("B", n, k, beta, explicit=True)
            b.add_stimulus("stim_a", k)
            b.add_stimulus("stim_b", k)

            # Establish A (potentiates stim_a→A weights)
            for _ in range(cfg.establish_rounds):
                b.project({"stim_a": ["A"]}, {})
            trained_A = np.array(b.areas["A"].winners, dtype=np.uint32)

            # B is left untrained (fresh random weights)

            # Declare mutual inhibition
            b.add_mutual_inhibition(["A", "B"])

            # Fire both stimuli simultaneously
            b.project({"stim_a": ["A"], "stim_b": ["B"]}, {})

            # A should win (potentiated weights give more total input)
            h3_winner.append(measure_overlap(trained_A, b.areas["A"].winners))

            # B should be suppressed (empty winners)
            b_winners = b.areas["B"].winners
            h3_loser_active.append(float(len(b_winners)))

        h3_win_stats = summarize(h3_winner)
        h3_lose_stats = summarize(h3_loser_active)
        metrics["h3_mutual_inhibition"] = {
            "winner_overlap": h3_win_stats,
            "loser_active_neurons": h3_lose_stats,
        }

        self.log(
            f"  Winner (A) overlap: {h3_win_stats['mean']:.3f}+/-{h3_win_stats['sem']:.3f}"
        )
        self.log(
            f"  Loser (B) active neurons: {h3_lose_stats['mean']:.0f} (expect 0)"
        )

        # ================================================================
        # H4: Trigger-assisted switching in recurrent circuit
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H4: Trigger-assisted switching in 2-area recurrent loop")
        self.log("=" * 60)

        h4_baseline_ov1 = []  # overlap with pattern 1 after switching (residual)
        h4_baseline_ov2 = []  # overlap with pattern 2 after switching
        h4_trigger_ov1 = []
        h4_trigger_ov2 = []

        for s in seeds:
            # Baseline: direct switch (no trigger)
            b, tr_1x, tr_2x, lmap = build_bidir_loop(
                n, k, p, beta, w_max, cfg.establish_rounds, self.seed + s
            )
            final = run_switch_trial(
                b, lmap, cfg.kick_rounds, cfg.autonomous_rounds,
                use_trigger=False,
            )
            h4_baseline_ov1.append(measure_overlap(tr_1x, final))
            h4_baseline_ov2.append(measure_overlap(tr_2x, final))

            # Trigger: inhibit both areas before switching
            b2, tr_1x2, tr_2x2, lmap2 = build_bidir_loop(
                n, k, p, beta, w_max, cfg.establish_rounds, self.seed + s
            )
            final2 = run_switch_trial(
                b2, lmap2, cfg.kick_rounds, cfg.autonomous_rounds,
                use_trigger=True,
            )
            h4_trigger_ov1.append(measure_overlap(tr_1x2, final2))
            h4_trigger_ov2.append(measure_overlap(tr_2x2, final2))

        h4_bl_1 = summarize(h4_baseline_ov1)
        h4_bl_2 = summarize(h4_baseline_ov2)
        h4_tr_1 = summarize(h4_trigger_ov1)
        h4_tr_2 = summarize(h4_trigger_ov2)
        h4_paired = paired_ttest(h4_trigger_ov2, h4_baseline_ov2)

        metrics["h4_trigger_switching"] = {
            "baseline": {"overlap_pattern1": h4_bl_1, "overlap_pattern2": h4_bl_2},
            "trigger": {"overlap_pattern1": h4_tr_1, "overlap_pattern2": h4_tr_2},
            "paired_test_pattern2": h4_paired,
        }

        self.log(
            f"  Baseline: residual(1)={h4_bl_1['mean']:.3f}  "
            f"target(2)={h4_bl_2['mean']:.3f}"
        )
        self.log(
            f"  Trigger:  residual(1)={h4_tr_1['mean']:.3f}  "
            f"target(2)={h4_tr_2['mean']:.3f}"
        )
        self.log(
            f"  Trigger improvement on pattern 2: "
            f"t={h4_paired['t']:.2f}, p={h4_paired['p']:.4f}"
        )

        # ================================================================
        # H5: Trigger-assisted switching at k=sqrt(n)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H5: Trigger-assisted switching at k=sqrt(n)")
        self.log("=" * 60)

        h5_sizes = [500, 1000, 2000, 5000]
        h5_results = []

        for n_val in h5_sizes:
            k_val = int(np.sqrt(n_val))
            null_h5 = chance_overlap(k_val, n_val)
            k2p = k_val * k_val * p

            baseline_vals = []
            trigger_vals = []

            for s in seeds:
                # Baseline
                b, _, tr_2x, lmap = build_bidir_loop(
                    n_val, k_val, p, beta, w_max,
                    cfg.establish_rounds, self.seed + s,
                )
                final = run_switch_trial(
                    b, lmap, cfg.kick_rounds, cfg.autonomous_rounds,
                    use_trigger=False,
                )
                baseline_vals.append(measure_overlap(tr_2x, final))

                # Trigger
                b2, _, tr_2x2, lmap2 = build_bidir_loop(
                    n_val, k_val, p, beta, w_max,
                    cfg.establish_rounds, self.seed + s,
                )
                final2 = run_switch_trial(
                    b2, lmap2, cfg.kick_rounds, cfg.autonomous_rounds,
                    use_trigger=True,
                )
                trigger_vals.append(measure_overlap(tr_2x2, final2))

            bl_stats = summarize(baseline_vals)
            tr_stats = summarize(trigger_vals)
            paired = paired_ttest(trigger_vals, baseline_vals)

            row = {
                "n": n_val, "k": k_val, "k2p": k2p,
                "baseline": bl_stats,
                "trigger": tr_stats,
                "paired_test": paired,
            }
            h5_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}, k2p={k2p:5.0f}: "
                f"baseline={bl_stats['mean']:.3f}  "
                f"trigger={tr_stats['mean']:.3f}  "
                f"delta={tr_stats['mean'] - bl_stats['mean']:+.3f}"
            )

        metrics["h5_scaling"] = h5_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": n, "k": k, "p": p, "beta": beta, "w_max": w_max,
                "establish_rounds": cfg.establish_rounds,
                "kick_rounds": cfg.kick_rounds,
                "autonomous_rounds": cfg.autonomous_rounds,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inhibition Primitives Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = InhibitionExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("INHIBITION PRIMITIVES SUMMARY")
    print("=" * 70)

    m = result.metrics

    h1 = m["h1_trigger_clear"]
    print(f"\nH1 Trigger clears: pre={h1['pre_inhibit']['mean']:.3f}  "
          f"post={h1['post_inhibit']['mean']:.3f}")

    print(f"H2 Recovery after trigger: {m['h2_recovery']['stats']['mean']:.3f}")

    h3 = m["h3_mutual_inhibition"]
    print(f"H3 Mutual inhibition: winner={h3['winner_overlap']['mean']:.3f}  "
          f"loser neurons={h3['loser_active_neurons']['mean']:.0f}")

    h4 = m["h4_trigger_switching"]
    print(f"\nH4 Trigger switching (2-area loop):")
    print(f"  Baseline: residual(1)={h4['baseline']['overlap_pattern1']['mean']:.3f}  "
          f"target(2)={h4['baseline']['overlap_pattern2']['mean']:.3f}")
    print(f"  Trigger:  residual(1)={h4['trigger']['overlap_pattern1']['mean']:.3f}  "
          f"target(2)={h4['trigger']['overlap_pattern2']['mean']:.3f}")
    print(f"  p={h4['paired_test_pattern2']['p']:.4f}")

    print(f"\nH5 Scaling:")
    for r in m["h5_scaling"]:
        print(f"  n={r['n']:4d}, k2p={r['k2p']:5.0f}: "
              f"baseline={r['baseline']['mean']:.3f}  "
              f"trigger={r['trigger']['mean']:.3f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
