"""
Multi-Pattern Loop Switching (Working Memory Capacity)

Tests whether recurrent loops can store and switch between multiple
distinct circulating patterns. The single-pattern recurrent loop
experiment showed zero-decay fixed-point attractors. This experiment
asks: can we break out of one attractor and enter another?

Biological context: Working memory holds multiple items and can switch
attention between them. Prefrontal recurrent circuits must support
rapid switching between maintained representations. If the Assembly
Calculus loop can only sustain ONE pattern (single-item buffer), it
models sustained attention but not multi-item working memory. If it
can switch cleanly between trained patterns, it supports flexible
working memory with capacity limits determined by interference.

Protocol:
1. Create a 3-area loop (X0, X1, X2).
2. Establish N distinct assembly sets (patterns), each with its own
   stimuli: pattern i has stimuli si0, si1, si2 driving assemblies
   in X0, X1, X2.
3. For each pattern i, train loop associations via co-stimulation:
   X0(i)→X1(i), X1(i)→X2(i), X2(i)→X0(i).
4. Test switching: kick-start pattern A, run autonomous, then kick-
   start pattern B, run autonomous. Measure overlap with both A and B
   after each phase.

Hypotheses:

H1: Single-pattern baseline — One trained pattern sustains autonomous
    circulation at ~0.99 overlap (replicates recurrent loop H1).
    Null: overlap equals chance k/n.

H2: Pattern switching — After training two patterns in the same loop,
    kick-start pattern A (verify circulation), then kick-start pattern B.
    Does B replace A cleanly? Does A's overlap drop to chance while B's
    rises to high overlap?
    Null: kick-starting B does not change the overlap with A.

H3: Interference from dual training — Does training a second pattern
    in the same loop degrade the first pattern's autonomous circulation
    compared to single-pattern baseline?
    Null: pattern A's circulation quality is independent of whether
    pattern B was also trained.

H4: Pattern capacity — How many distinct patterns can be trained in
    one loop before switching reliability degrades? Test 1, 2, 3, 5
    patterns.
    Null: switching quality is independent of the number of trained
    patterns.

Statistical methodology:
- N_SEEDS=10 independent random seeds per condition.
- One-sample t-test against null k/n.
- Paired t-test for H3 (single vs dual training).
- Cohen's d effect sizes. Mean +/- SEM.

References:
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
- Dabagia et al., "Coin-Flipping in the Brain", 2024
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
class LoopSwitchConfig:
    """Configuration for loop switching trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    assoc_rounds: int = 30
    kick_rounds: int = 15
    autonomous_rounds: int = 20


# ── Core trial runners ──────────────────────────────────────────────


def build_loop_with_patterns(
    cfg: LoopSwitchConfig, n_patterns: int, seed: int,
) -> tuple:
    """
    Build a 3-area loop and train n_patterns distinct assembly sets
    with loop associations.

    Returns (brain, trained_assemblies, stim_names_by_pattern, area_names).
    trained_assemblies[pattern_idx][area_name] = np.array of winners.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    loop_size = 3
    area_names = [f"X{i}" for i in range(loop_size)]

    for name in area_names:
        b.add_area(name, cfg.n, cfg.k, cfg.beta, explicit=True)

    # Create stimuli for each pattern
    stim_names = {}  # stim_names[pattern_idx] = ["s0_0", "s0_1", "s0_2"]
    for p_idx in range(n_patterns):
        pattern_stims = []
        for a_idx in range(loop_size):
            sname = f"s{p_idx}_{a_idx}"
            b.add_stimulus(sname, cfg.k)
            pattern_stims.append(sname)
        stim_names[p_idx] = pattern_stims

    # Establish all assemblies via stim-only
    trained = {}
    for p_idx in range(n_patterns):
        trained[p_idx] = {}
        for a_idx, area in enumerate(area_names):
            sname = stim_names[p_idx][a_idx]
            for _ in range(cfg.establish_rounds):
                b.project({sname: [area]}, {})
            trained[p_idx][area] = np.array(b.areas[area].winners, dtype=np.uint32)

    # Train loop associations for each pattern via co-stimulation
    loop_map = {}
    for i in range(loop_size):
        src = area_names[i]
        dst = area_names[(i + 1) % loop_size]
        loop_map[src] = [dst]

    for p_idx in range(n_patterns):
        for _ in range(cfg.assoc_rounds):
            stim_map = {}
            for a_idx, area in enumerate(area_names):
                stim_map[stim_names[p_idx][a_idx]] = [area]
            b.project(stim_map, loop_map)

    return b, trained, stim_names, area_names, loop_map


def kick_and_run(
    b: Brain, stim_names: List[str], area_names: List[str],
    loop_map: Dict, kick_rounds: int, autonomous_rounds: int
) -> List[Dict[str, Any]]:
    """
    Kick-start a pattern and run autonomous circulation.
    Returns per-round state (winners at each area).
    """
    # Kick-start: fire pattern's first stimulus into the loop
    for _ in range(kick_rounds):
        b.project({stim_names[0]: [area_names[0]]}, loop_map)

    # Autonomous circulation
    trajectory = []
    for _ in range(autonomous_rounds):
        b.project({}, loop_map)
        state = {}
        for area in area_names:
            state[area] = np.array(b.areas[area].winners, dtype=np.uint32)
        trajectory.append(state)

    return trajectory


def measure_pattern_overlap(
    state: Dict[str, np.ndarray],
    trained: Dict[str, np.ndarray],
    area_names: List[str],
) -> float:
    """Mean overlap across all areas between current state and trained pattern."""
    overlaps = []
    for area in area_names:
        overlaps.append(measure_overlap(trained[area], state[area]))
    return float(np.mean(overlaps))


# ── Main experiment ──────────────────────────────────────────────────


class LoopSwitchingExperiment(ExperimentBase):
    """Test multi-pattern loop switching and working memory capacity."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="loop_switching",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
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

        cfg = LoopSwitchConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Multi-Pattern Loop Switching Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  establish_rounds={cfg.establish_rounds}")
        self.log(f"  assoc_rounds={cfg.assoc_rounds}")
        self.log(f"  kick_rounds={cfg.kick_rounds}")
        self.log(f"  autonomous_rounds={cfg.autonomous_rounds}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics = {}

        # ================================================================
        # H1: Single-pattern baseline
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H1: Single-pattern baseline (1 pattern trained)")
        self.log("=" * 60)

        h1_overlaps = []
        for s in seeds:
            b, trained, stims, areas, lmap = build_loop_with_patterns(
                cfg, n_patterns=1, seed=self.seed + s
            )
            traj = kick_and_run(b, stims[0], areas, lmap,
                                cfg.kick_rounds, cfg.autonomous_rounds)
            final_overlap = measure_pattern_overlap(traj[-1], trained[0], areas)
            h1_overlaps.append(final_overlap)

        h1_stats = summarize(h1_overlaps)
        h1_test = ttest_vs_null(h1_overlaps, null)
        metrics["h1_single_pattern"] = {"stats": h1_stats, "test_vs_chance": h1_test}

        self.log(
            f"  Final overlap: {h1_stats['mean']:.3f}+/-{h1_stats['sem']:.3f}  "
            f"d={h1_test['d']:.1f}"
        )

        # ================================================================
        # H2: Pattern switching (2 patterns)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H2: Pattern switching (train 2, kick A then B)")
        self.log("=" * 60)

        h2_results = {"after_A": [], "after_B_overlap_A": [], "after_B_overlap_B": []}

        for s in seeds:
            b, trained, stims, areas, lmap = build_loop_with_patterns(
                cfg, n_patterns=2, seed=self.seed + s
            )

            # Kick-start pattern A, run autonomous
            traj_a = kick_and_run(b, stims[0], areas, lmap,
                                  cfg.kick_rounds, cfg.autonomous_rounds)
            overlap_a_after_a = measure_pattern_overlap(traj_a[-1], trained[0], areas)
            h2_results["after_A"].append(overlap_a_after_a)

            # Now kick-start pattern B, run autonomous
            traj_b = kick_and_run(b, stims[1], areas, lmap,
                                  cfg.kick_rounds, cfg.autonomous_rounds)
            overlap_a_after_b = measure_pattern_overlap(traj_b[-1], trained[0], areas)
            overlap_b_after_b = measure_pattern_overlap(traj_b[-1], trained[1], areas)
            h2_results["after_B_overlap_A"].append(overlap_a_after_b)
            h2_results["after_B_overlap_B"].append(overlap_b_after_b)

        after_a_stats = summarize(h2_results["after_A"])
        b_ov_a_stats = summarize(h2_results["after_B_overlap_A"])
        b_ov_b_stats = summarize(h2_results["after_B_overlap_B"])
        b_ov_b_test = ttest_vs_null(h2_results["after_B_overlap_B"], null)

        metrics["h2_switching"] = {
            "after_kick_A": {"overlap_with_A": after_a_stats},
            "after_kick_B": {
                "overlap_with_A": b_ov_a_stats,
                "overlap_with_B": {"stats": b_ov_b_stats, "test_vs_chance": b_ov_b_test},
            },
        }

        self.log(f"  After kick A:  overlap(A)={after_a_stats['mean']:.3f}")
        self.log(f"  After kick B:  overlap(A)={b_ov_a_stats['mean']:.3f}  "
                 f"overlap(B)={b_ov_b_stats['mean']:.3f}  "
                 f"d(B)={b_ov_b_test['d']:.1f}")

        # ================================================================
        # H3: Interference from dual training
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H3: Does dual training degrade pattern A's circulation?")
        self.log("=" * 60)

        # Compare H1 (single-pattern) vs H2 after_A (dual-pattern, testing A)
        paired = paired_ttest(h1_overlaps, h2_results["after_A"])
        metrics["h3_interference"] = {
            "single_pattern": h1_stats,
            "dual_pattern_A": after_a_stats,
            "paired_test": paired,
        }

        self.log(f"  Single-pattern A: {h1_stats['mean']:.3f}")
        self.log(f"  Dual-pattern A:   {after_a_stats['mean']:.3f}")
        self.log(
            f"  Paired test: t={paired['t']:.2f}, p={paired['p']:.4f}  "
            f"{'SIGNIFICANT DEGRADATION' if paired['sig'] else 'no significant difference'}"
        )

        # ================================================================
        # H4: Pattern capacity (1, 2, 3, 5 patterns)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H4: Pattern capacity — switching quality vs # patterns")
        self.log("=" * 60)

        pattern_counts = [1, 2, 3, 5]
        h4_results = []

        for n_pat in pattern_counts:
            switch_quality = []

            for s in seeds:
                b, trained, stims, areas, lmap = build_loop_with_patterns(
                    cfg, n_patterns=n_pat, seed=self.seed + s
                )

                # Test each pattern: kick-start it, run autonomous, measure overlap
                pattern_overlaps = []
                for p_idx in range(n_pat):
                    traj = kick_and_run(b, stims[p_idx], areas, lmap,
                                        cfg.kick_rounds, cfg.autonomous_rounds)
                    ov = measure_pattern_overlap(traj[-1], trained[p_idx], areas)
                    pattern_overlaps.append(ov)

                # Mean switching quality across all patterns
                switch_quality.append(float(np.mean(pattern_overlaps)))

            sq_stats = summarize(switch_quality)
            sq_test = ttest_vs_null(switch_quality, null)

            row = {
                "n_patterns": n_pat,
                "mean_switching_quality": sq_stats,
                "test_vs_chance": sq_test,
            }
            h4_results.append(row)

            self.log(
                f"  {n_pat} pattern(s): "
                f"switching quality={sq_stats['mean']:.3f}+/-{sq_stats['sem']:.3f}  "
                f"d={sq_test['d']:.1f}"
            )

        metrics["h4_capacity"] = h4_results

        # ================================================================
        # H5: Switching at sparse coding (k=sqrt(n))
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H5: Two-pattern switching at k=sqrt(n)")
        self.log("=" * 60)

        h5_sizes = [500, 1000, 2000, 5000]
        h5_results = []

        for n_val in h5_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h5 = LoopSwitchConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h5 = chance_overlap(k_val, n_val)

            switch_vals = []
            for s in seeds:
                b, trained, stims, areas, lmap = build_loop_with_patterns(
                    cfg_h5, n_patterns=2, seed=self.seed + s
                )
                # Test pattern B after training both
                # Kick A first, then switch to B
                kick_and_run(b, stims[0], areas, lmap,
                             cfg_h5.kick_rounds, cfg_h5.autonomous_rounds)
                traj_b = kick_and_run(b, stims[1], areas, lmap,
                                      cfg_h5.kick_rounds, cfg_h5.autonomous_rounds)
                ov_b = measure_pattern_overlap(traj_b[-1], trained[1], areas)
                switch_vals.append(ov_b)

            sv_stats = summarize(switch_vals)
            sv_test = ttest_vs_null(switch_vals, null_h5)

            row = {"n": n_val, "k": k_val, "stats": sv_stats, "test_vs_chance": sv_test}
            h5_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"switch quality={sv_stats['mean']:.3f}+/-{sv_stats['sem']:.3f}  "
                f"d={sv_test['d']:.1f}"
            )

        metrics["h5_sparse_switching"] = h5_results

        # ================================================================
        # H6: Trigger-assisted switching (inter-area inhibition)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H6: Trigger-assisted switching (inhibit before kick)")
        self.log("=" * 60)

        # H6a: Two-pattern switching with trigger
        h6_after_a = []
        h6_b_ov_a = []
        h6_b_ov_b = []

        for s in seeds:
            b, trained, stims, areas, lmap = build_loop_with_patterns(
                cfg, n_patterns=2, seed=self.seed + s
            )

            # Kick-start pattern A, run autonomous
            traj_a = kick_and_run(b, stims[0], areas, lmap,
                                  cfg.kick_rounds, cfg.autonomous_rounds)
            h6_after_a.append(measure_pattern_overlap(traj_a[-1], trained[0], areas))

            # TRIGGER: inhibit all loop areas before switching
            b.inhibit_areas(areas)
            b.project({}, lmap)

            # Kick-start pattern B, run autonomous
            traj_b = kick_and_run(b, stims[1], areas, lmap,
                                  cfg.kick_rounds, cfg.autonomous_rounds)
            h6_b_ov_a.append(measure_pattern_overlap(traj_b[-1], trained[0], areas))
            h6_b_ov_b.append(measure_pattern_overlap(traj_b[-1], trained[1], areas))

        h6_a_stats = summarize(h6_after_a)
        h6_ba_stats = summarize(h6_b_ov_a)
        h6_bb_stats = summarize(h6_b_ov_b)
        h6_bb_test = ttest_vs_null(h6_b_ov_b, null)

        # Compare with baseline H2 (pattern B overlap)
        h6_vs_h2 = paired_ttest(h6_b_ov_b, h2_results["after_B_overlap_B"])

        self.log(f"  After kick A (trigger):  overlap(A)={h6_a_stats['mean']:.3f}")
        self.log(f"  After trigger + kick B:  overlap(A)={h6_ba_stats['mean']:.3f}  "
                 f"overlap(B)={h6_bb_stats['mean']:.3f}  d(B)={h6_bb_test['d']:.1f}")
        self.log(f"  vs baseline H2 B overlap: t={h6_vs_h2['t']:.2f}, p={h6_vs_h2['p']:.4f}")

        # H6b: Capacity with trigger (1, 2, 3, 5 patterns)
        h6_capacity = []
        for n_pat in pattern_counts:
            sq = []
            for s in seeds:
                b, trained, stims, areas, lmap = build_loop_with_patterns(
                    cfg, n_patterns=n_pat, seed=self.seed + s
                )
                pat_ovs = []
                for p_idx in range(n_pat):
                    # Trigger before each pattern switch
                    b.inhibit_areas(areas)
                    b.project({}, lmap)
                    traj = kick_and_run(b, stims[p_idx], areas, lmap,
                                        cfg.kick_rounds, cfg.autonomous_rounds)
                    pat_ovs.append(measure_pattern_overlap(traj[-1], trained[p_idx], areas))
                sq.append(float(np.mean(pat_ovs)))

            sq_stats = summarize(sq)
            h6_capacity.append({"n_patterns": n_pat, "stats": sq_stats})
            self.log(f"  {n_pat} pattern(s) (trigger): "
                     f"{sq_stats['mean']:.3f}+/-{sq_stats['sem']:.3f}")

        # Comparison table
        self.log("")
        self.log("  ── Baseline vs Trigger comparison ──")
        for i, n_pat in enumerate(pattern_counts):
            baseline = h4_results[i]["mean_switching_quality"]["mean"]
            triggered = h6_capacity[i]["stats"]["mean"]
            delta = triggered - baseline
            self.log(f"  {n_pat} patterns: baseline={baseline:.3f}  "
                     f"trigger={triggered:.3f}  delta={delta:+.3f}")

        metrics["h6_trigger"] = {
            "switching": {
                "after_A": h6_a_stats,
                "after_B_overlap_A": h6_ba_stats,
                "after_B_overlap_B": {"stats": h6_bb_stats, "test_vs_chance": h6_bb_test},
                "vs_baseline_h2": h6_vs_h2,
            },
            "capacity": h6_capacity,
        }

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": n, "k": k, "p": p, "beta": beta, "w_max": w_max,
                "loop_size": 3,
                "establish_rounds": cfg.establish_rounds,
                "assoc_rounds": cfg.assoc_rounds,
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

    parser = argparse.ArgumentParser(description="Loop Switching Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = LoopSwitchingExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("LOOP SWITCHING SUMMARY")
    print("=" * 70)

    null = result.parameters["k"] / result.parameters["n"]

    print(f"\nH1 Single-pattern: {result.metrics['h1_single_pattern']['stats']['mean']:.3f}")

    h2 = result.metrics["h2_switching"]
    print(f"\nH2 Switching:")
    print(f"  After kick A: overlap(A)={h2['after_kick_A']['overlap_with_A']['mean']:.3f}")
    print(f"  After kick B: overlap(A)={h2['after_kick_B']['overlap_with_A']['mean']:.3f}  "
          f"overlap(B)={h2['after_kick_B']['overlap_with_B']['stats']['mean']:.3f}")

    h3 = result.metrics["h3_interference"]
    print(f"\nH3 Interference: single={h3['single_pattern']['mean']:.3f}  "
          f"dual={h3['dual_pattern_A']['mean']:.3f}  "
          f"sig={h3['paired_test']['sig']}")

    print(f"\nH4 Capacity:")
    for r in result.metrics["h4_capacity"]:
        print(f"  {r['n_patterns']} patterns: {r['mean_switching_quality']['mean']:.3f}")

    print(f"\nH5 Sparse switching:")
    for r in result.metrics["h5_sparse_switching"]:
        print(f"  n={r['n']:4d}: {r['stats']['mean']:.3f}")

    h6 = result.metrics["h6_trigger"]
    sw = h6["switching"]
    print(f"\nH6 Trigger-assisted switching:")
    print(f"  After kick A: overlap(A)={sw['after_A']['mean']:.3f}")
    print(f"  After trigger+B: overlap(A)={sw['after_B_overlap_A']['mean']:.3f}  "
          f"overlap(B)={sw['after_B_overlap_B']['stats']['mean']:.3f}")
    print(f"  Capacity (trigger):")
    for r in h6["capacity"]:
        print(f"    {r['n_patterns']} patterns: {r['stats']['mean']:.3f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
