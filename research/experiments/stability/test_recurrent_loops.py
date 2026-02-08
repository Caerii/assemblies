"""
Recurrent Loop Dynamics

Tests whether multi-area recurrent loops (X0→X1→...→X(N-1)→X0) support
stable autonomous circulation of assembly representations after an initial
stimulus kick-start.

This is fundamentally different from:
- Single-area self-projection (A→A): one attractor, one area
- Feedforward chains (A→B→C): one-way propagation, source-driven
- Recurrent loops: sustained multi-area circulation, no external input

Biological context: Cortical circuits are dominated by recurrent loops —
cortico-thalamo-cortical loops, the hippocampal loop (CA3→CA1→EC→DG→CA3),
and recurrent frontal-sensory circuits. If the Assembly Calculus supports
stable recurrence through learned cross-area associations, it can model
working memory and iterative computation distributed across areas. If it
diverges, the framework is limited to feedforward pipelines.

Protocol:
1. Create N areas in a loop: X0, X1, ..., X(N-1).
2. Establish each assembly via stim-only projection (30 rounds).
3. Train N loop associations via co-stimulation (30 rounds each):
   X0→X1, X1→X2, ..., X(N-1)→X0.
4. Kick-start: fire X0's stimulus into the active loop for kick_rounds.
   This injects the correct signal at X0 and lets it propagate around
   the loop until all areas hold their trained assemblies.
5. Autonomous: remove all stimuli, run the loop for test_rounds.
   project({}, {"X0": ["X1"], "X1": ["X2"], ..., "X(N-1)": ["X0"]}).
6. Measure overlap at each area with its trained assembly every round.

Hypotheses:

H1: Loop persistence — After kick-starting a 3-area loop with 15 rounds
    of stimulus-driven circulation and then running 30 rounds of
    autonomous circulation, each area maintains its trained assembly
    (overlap >> k/n).
    Null: overlap after autonomous circulation equals chance k/n.

H2: Persistence vs loop length — Shorter loops (3 areas) may be more
    stable than longer loops (4, 5, 6 areas) because the signal
    recirculates more frequently.
    Null: persistence is independent of loop length.

H3: Temporal dynamics — Track overlap at each area over 50 rounds of
    autonomous circulation. The signal may: (a) converge to a stable
    fixed point, (b) oscillate, (c) decay monotonically to chance.
    Null: overlap is constant over time.

H4: Loop stability vs network size (k=sqrt(n)) — At biologically
    realistic sparsity, does loop persistence scale with n?
    Null: persistence is independent of n.

Statistical methodology:
- N_SEEDS=10 independent random seeds per condition.
- One-sample t-test against null k/n.
- Cohen's d effect sizes. Mean +/- SEM.

References:
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
- Dabagia et al., "Coin-Flipping in the Brain", 2024 (weight saturation)
- Litwin-Kumar & Doiron, Nature Comms, 2014 (recurrent assembly dynamics)
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
)

from src.core.brain import Brain

N_SEEDS = 10


@dataclass
class LoopConfig:
    """Configuration for recurrent loop trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    assoc_rounds: int = 30
    kick_rounds: int = 15
    test_rounds: int = 30


# ── Core trial runner ────────────────────────────────────────────────


def run_loop_trial(
    cfg: LoopConfig, loop_size: int, seed: int
) -> Dict[str, Any]:
    """
    Build a loop of loop_size areas, establish assemblies, train loop
    associations, kick-start with stimulus, then run autonomous circulation.

    Returns per-area recovery at end and full temporal trajectory.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    area_names = [f"X{i}" for i in range(loop_size)]
    stim_names = [f"s{i}" for i in range(loop_size)]

    for name in area_names:
        b.add_area(name, cfg.n, cfg.k, cfg.beta, explicit=True)
    for name in stim_names:
        b.add_stimulus(name, cfg.k)

    # ── Phase 1: Establish all assemblies via stim-only ──
    trained = {}
    for area, stim in zip(area_names, stim_names):
        for _ in range(cfg.establish_rounds):
            b.project({stim: [area]}, {})
        trained[area] = np.array(b.areas[area].winners, dtype=np.uint32)

    # ── Phase 2: Train loop associations via co-stimulation ──
    # X0→X1, X1→X2, ..., X(N-1)→X0
    for i in range(loop_size):
        src = i
        dst = (i + 1) % loop_size
        src_area = area_names[src]
        dst_area = area_names[dst]
        src_stim = stim_names[src]
        dst_stim = stim_names[dst]
        for _ in range(cfg.assoc_rounds):
            b.project(
                {src_stim: [src_area], dst_stim: [dst_area]},
                {src_area: [dst_area]},
            )

    # ── Phase 3: Kick-start with X0's stimulus driving the loop ──
    loop_map = {}
    for i in range(loop_size):
        src_area = area_names[i]
        dst_area = area_names[(i + 1) % loop_size]
        loop_map[src_area] = [dst_area]

    for _ in range(cfg.kick_rounds):
        b.project({stim_names[0]: [area_names[0]]}, loop_map)

    # Measure state after kick-start (before autonomous phase)
    post_kick = {}
    for area in area_names:
        curr = np.array(b.areas[area].winners, dtype=np.uint32)
        post_kick[area] = measure_overlap(trained[area], curr)

    # ── Phase 4: Autonomous circulation (no stimulus) ──
    trajectory = {area: [] for area in area_names}

    for t in range(cfg.test_rounds):
        b.project({}, loop_map)
        for area in area_names:
            curr = np.array(b.areas[area].winners, dtype=np.uint32)
            trajectory[area].append(measure_overlap(trained[area], curr))

    # Final recovery
    final = {}
    for area in area_names:
        final[area] = trajectory[area][-1] if trajectory[area] else 0.0

    return {
        "post_kick": post_kick,
        "final": final,
        "trajectory": trajectory,
    }


# ── Main experiment ──────────────────────────────────────────────────


class RecurrentLoopExperiment(ExperimentBase):
    """Test whether recurrent loops support stable autonomous circulation."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="recurrent_loops",
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

        cfg = LoopConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Recurrent Loop Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  establish_rounds={cfg.establish_rounds}")
        self.log(f"  assoc_rounds={cfg.assoc_rounds}")
        self.log(f"  kick_rounds={cfg.kick_rounds}")
        self.log(f"  test_rounds={cfg.test_rounds}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics = {}

        # ================================================================
        # H1/H2: Loop persistence vs loop size
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H1/H2: Loop persistence vs loop size")
        self.log("=" * 60)

        loop_sizes = [3, 4, 5, 6]
        h1_results = []

        for ls in loop_sizes:
            area_names = [f"X{i}" for i in range(ls)]

            # Collect per-area final recovery and post-kick state
            per_area_final = {a: [] for a in area_names}
            per_area_kick = {a: [] for a in area_names}

            for s in seeds:
                trial = run_loop_trial(cfg, ls, seed=self.seed + s)
                for a in area_names:
                    per_area_final[a].append(trial["final"][a])
                    per_area_kick[a].append(trial["post_kick"][a])

            # Mean across all areas for each seed
            mean_final_per_seed = []
            mean_kick_per_seed = []
            for s_idx in range(len(seeds)):
                mean_final_per_seed.append(
                    float(np.mean([per_area_final[a][s_idx] for a in area_names]))
                )
                mean_kick_per_seed.append(
                    float(np.mean([per_area_kick[a][s_idx] for a in area_names]))
                )

            kick_stats = summarize(mean_kick_per_seed)
            final_stats = summarize(mean_final_per_seed)
            test_final = ttest_vs_null(mean_final_per_seed, null)

            # Per-area detail
            area_detail = []
            for a in area_names:
                a_final_stats = summarize(per_area_final[a])
                a_kick_stats = summarize(per_area_kick[a])
                a_test = ttest_vs_null(per_area_final[a], null)
                area_detail.append({
                    "area": a,
                    "post_kick": a_kick_stats,
                    "final": a_final_stats,
                    "test_vs_chance": a_test,
                })

            row = {
                "loop_size": ls,
                "post_kick_mean": kick_stats,
                "final_mean": final_stats,
                "test_final_vs_chance": test_final,
                "per_area": area_detail,
            }
            h1_results.append(row)

            self.log(
                f"  {ls}-area loop: "
                f"post_kick={kick_stats['mean']:.3f}  "
                f"final(t={cfg.test_rounds})={final_stats['mean']:.3f}+/-{final_stats['sem']:.3f}  "
                f"d={test_final['d']:.1f}"
            )

        metrics["h1h2_loop_persistence"] = h1_results

        # ================================================================
        # H3: Temporal dynamics (3-area loop, 50 autonomous rounds)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H3: Temporal dynamics (3-area loop, 50 autonomous rounds)")
        self.log("=" * 60)

        cfg_h3 = LoopConfig(n=n, k=k, p=p, beta=beta, w_max=w_max,
                            test_rounds=50)
        ls_h3 = 3
        area_names_h3 = [f"X{i}" for i in range(ls_h3)]

        # Collect trajectories across seeds
        trajectories_by_area = {a: [] for a in area_names_h3}
        # Also collect the mean-across-areas trajectory per seed
        mean_trajectories = []

        for s in seeds:
            trial = run_loop_trial(cfg_h3, ls_h3, seed=self.seed + s)
            mean_traj = []
            for t in range(cfg_h3.test_rounds):
                vals = [trial["trajectory"][a][t] for a in area_names_h3]
                mean_traj.append(float(np.mean(vals)))
            mean_trajectories.append(mean_traj)

            for a in area_names_h3:
                trajectories_by_area[a].append(trial["trajectory"][a])

        # Average trajectory across seeds (mean of means at each timestep)
        avg_trajectory = []
        for t in range(cfg_h3.test_rounds):
            vals_at_t = [mean_trajectories[s][t] for s in range(len(seeds))]
            avg_trajectory.append(summarize(vals_at_t))

        # Report at key timepoints
        h3_timepoints = [0, 4, 9, 14, 19, 29, 39, 49]
        h3_results = []
        for t in h3_timepoints:
            if t < len(avg_trajectory):
                entry = {
                    "round": t + 1,  # 1-indexed for clarity
                    "mean_overlap": avg_trajectory[t],
                }
                h3_results.append(entry)
                self.log(
                    f"  t={t+1:2d}: "
                    f"overlap={avg_trajectory[t]['mean']:.3f}+/-{avg_trajectory[t]['sem']:.3f}"
                )

        # Compute decay rate: fit linear regression on log(overlap) vs time
        mean_curve = [avg_trajectory[t]["mean"] for t in range(cfg_h3.test_rounds)]
        if all(v > 0 for v in mean_curve):
            log_curve = [np.log(v) for v in mean_curve]
            times = list(range(cfg_h3.test_rounds))
            slope, intercept, r_val, p_val, std_err = stats.linregress(times, log_curve)
            half_life = -np.log(2) / slope if slope < 0 else float("inf")
            decay_fit = {
                "log_slope": float(slope),
                "r_squared": float(r_val ** 2),
                "p_value": float(p_val),
                "half_life_rounds": float(half_life),
            }
            if half_life == float("inf"):
                self.log(f"  Decay: no significant decay (slope={slope:.6f})")
            else:
                self.log(f"  Decay: half-life={half_life:.1f} rounds "
                         f"(slope={slope:.6f}, R²={r_val**2:.3f})")
        else:
            decay_fit = {"log_slope": 0.0, "r_squared": 0.0,
                        "p_value": 1.0, "half_life_rounds": 0.0}

        metrics["h3_temporal_dynamics"] = {
            "timepoints": h3_results,
            "decay_fit": decay_fit,
            "full_trajectory": [{"round": t + 1, "mean": avg_trajectory[t]["mean"],
                                "sem": avg_trajectory[t]["sem"]}
                               for t in range(cfg_h3.test_rounds)],
        }

        # ================================================================
        # H4: Loop stability vs network size (k=sqrt(n), 3-area loop)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H4: Loop stability vs network size (k=sqrt(n), 3-area loop)")
        self.log("=" * 60)

        h4_sizes = [200, 500, 1000, 2000, 5000]
        h4_loop_size = 3
        h4_results = []

        for n_val in h4_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h4 = LoopConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h4 = chance_overlap(k_val, n_val)

            area_names_h4 = [f"X{i}" for i in range(h4_loop_size)]
            mean_final = []
            mean_kick = []

            for s in seeds:
                trial = run_loop_trial(cfg_h4, h4_loop_size, seed=self.seed + s)
                finals = [trial["final"][a] for a in area_names_h4]
                kicks = [trial["post_kick"][a] for a in area_names_h4]
                mean_final.append(float(np.mean(finals)))
                mean_kick.append(float(np.mean(kicks)))

            kick_stats = summarize(mean_kick)
            final_stats = summarize(mean_final)
            test_final = ttest_vs_null(mean_final, null_h4)

            row = {
                "n": n_val,
                "k": k_val,
                "loop_size": h4_loop_size,
                "post_kick_mean": kick_stats,
                "final_mean": final_stats,
                "test_final_vs_chance": test_final,
            }
            h4_results.append(row)

            self.log(
                f"  n={n_val:4d}, k={k_val:2d}: "
                f"post_kick={kick_stats['mean']:.3f}  "
                f"final={final_stats['mean']:.3f}+/-{final_stats['sem']:.3f}  "
                f"d={test_final['d']:.1f}"
            )

        metrics["h4_size_scaling"] = h4_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": n, "k": k, "p": p, "beta": beta, "w_max": w_max,
                "establish_rounds": cfg.establish_rounds,
                "assoc_rounds": cfg.assoc_rounds,
                "kick_rounds": cfg.kick_rounds,
                "test_rounds": cfg.test_rounds,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Recurrent Loop Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = RecurrentLoopExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RECURRENT LOOP SUMMARY")
    print("=" * 70)

    null = result.parameters["k"] / result.parameters["n"]

    print(f"\nH1/H2 -- Loop persistence vs size (chance={null:.3f}):")
    for r in result.metrics["h1h2_loop_persistence"]:
        ls = r["loop_size"]
        kick = r["post_kick_mean"]["mean"]
        final = r["final_mean"]["mean"]
        print(f"  {ls}-area loop: post_kick={kick:.3f}  final={final:.3f}")

    print("\nH3 -- Temporal dynamics (3-area loop):")
    for entry in result.metrics["h3_temporal_dynamics"]["timepoints"]:
        t = entry["round"]
        ov = entry["mean_overlap"]["mean"]
        print(f"  t={t:2d}: {ov:.3f}")
    df = result.metrics["h3_temporal_dynamics"]["decay_fit"]
    if df["half_life_rounds"] == float("inf") or df["half_life_rounds"] > 1000:
        print(f"  No significant decay")
    else:
        print(f"  Half-life: {df['half_life_rounds']:.1f} rounds")

    print("\nH4 -- Loop stability vs network size (k=sqrt(n), 3-area loop):")
    for r in result.metrics["h4_size_scaling"]:
        n_val = r["n"]
        kick = r["post_kick_mean"]["mean"]
        final = r["final_mean"]["mean"]
        print(f"  n={n_val:4d}, k={r['k']:2d}: post_kick={kick:.3f}  final={final:.3f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
