"""
Transitive Association Chains (Multi-Hop Pattern Completion)

Tests whether single-hop associations compose into multi-hop chains.
After training A→B and B→C associations via co-stimulation, activating
A's stimulus and projecting through the chain should recover C's trained
assembly — transitive pattern completion through learned intermediate
representations.

This is the foundational operation for any computation beyond pairwise
association: language (word→syntax→semantics), reasoning (premise→
intermediate→conclusion), and hierarchical representation.

Protocol:
1. Create a chain of L+1 areas: X0, X1, ..., XL.
2. Establish each assembly via stim-only projection (30 rounds).
3. Train L adjacent associations via co-stimulation (30 rounds each):
   X0→X1, X1→X2, ..., X(L-1)→XL.
4. Test: fire X0's stimulus, project through the full chain simultaneously,
   measure recovery at each area after propagation.

The chain propagation uses simultaneous projection:
   project({"s0": ["X0"]}, {"X0": ["X1"], "X1": ["X2"], ..., "X(L-1)": ["XL"]})
Signal propagates one hop per round (feedforward delay), so L rounds are
needed for the signal to reach XL, plus additional rounds for stabilization.

Hypotheses:

H1: Two-hop chain — Activating X0's stimulus recovers X2's assembly
    through the learned X0→X1→X2 chain, with recovery significantly
    above chance.
    Null: recovery at X2 equals chance k/n.

H2: Chain length scaling — Recovery at the final area degrades with
    chain length. There exists a critical chain length beyond which
    recovery drops below a useful threshold.
    Null: recovery is independent of chain length.

H3: Per-hop signal quality — In a long chain, measure recovery at each
    intermediate area to characterize the propagation profile: uniform
    decay, sharp cutoff, or distance-dependent gradient.
    Null: recovery at each hop equals chance k/n.

H4: Chain capacity vs network size — Larger networks support longer
    chains before signal degrades to chance.
    Null: chain capacity is independent of n.

Statistical methodology:
- N_SEEDS=10 independent random seeds per condition.
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
)

from src.core.brain import Brain

N_SEEDS = 10


@dataclass
class ChainConfig:
    """Configuration for chain propagation trials."""
    n: int
    k: int
    p: float
    beta: float
    w_max: float
    establish_rounds: int = 30
    assoc_rounds: int = 30
    propagation_rounds: int = 15


# ── Core trial runner ────────────────────────────────────────────────


def run_chain_trial(
    cfg: ChainConfig, chain_length: int, seed: int
) -> Dict[str, float]:
    """
    Build a chain of chain_length hops (chain_length+1 areas), establish
    assemblies, train adjacent associations, then test propagation.

    Returns recovery overlap at each area in the chain.
    """
    b = Brain(p=cfg.p, seed=seed, w_max=cfg.w_max)

    n_areas = chain_length + 1
    area_names = [f"X{i}" for i in range(n_areas)]
    stim_names = [f"s{i}" for i in range(n_areas)]

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

    # ── Phase 2: Train adjacent associations via co-stimulation ──
    for i in range(chain_length):
        src_area = area_names[i]
        dst_area = area_names[i + 1]
        src_stim = stim_names[i]
        dst_stim = stim_names[i + 1]
        for _ in range(cfg.assoc_rounds):
            b.project(
                {src_stim: [src_area], dst_stim: [dst_area]},
                {src_area: [dst_area]},
            )

    # ── Phase 3: Test chain propagation ──
    # Fire only the first stimulus and project through the full chain.
    # Signal propagates one hop per round (feedforward delay).
    chain_map = {}
    for i in range(chain_length):
        chain_map[area_names[i]] = [area_names[i + 1]]

    for _ in range(cfg.propagation_rounds):
        b.project({stim_names[0]: [area_names[0]]}, chain_map)

    # Measure recovery at each area
    recovery = {}
    for area in area_names:
        curr = np.array(b.areas[area].winners, dtype=np.uint32)
        recovery[area] = measure_overlap(trained[area], curr)

    return recovery


# ── Main experiment ──────────────────────────────────────────────────


class AssociationChainExperiment(ExperimentBase):
    """Test transitive pattern completion through multi-hop association chains."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="association_chain",
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

        cfg = ChainConfig(n=n, k=k, p=p, beta=beta, w_max=w_max)
        null = chance_overlap(k, n)

        self.log("=" * 60)
        self.log("Association Chain Experiment")
        self.log(f"  n={n}, k={k}, p={p}, beta={beta}, w_max={w_max}")
        self.log(f"  establish_rounds={cfg.establish_rounds}")
        self.log(f"  assoc_rounds={cfg.assoc_rounds}")
        self.log(f"  propagation_rounds={cfg.propagation_rounds}")
        self.log(f"  null overlap (k/n) = {null:.3f}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 60)

        metrics = {}

        # ================================================================
        # H1/H2: Chain length sweep (1 to 5 hops)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H1/H2: Recovery at final area vs chain length")
        self.log("=" * 60)

        chain_lengths = [1, 2, 3, 4, 5]
        h1_results = []

        for L in chain_lengths:
            area_names = [f"X{i}" for i in range(L + 1)]

            # Collect per-area recovery across seeds
            per_area = {a: [] for a in area_names}

            for s in seeds:
                recovery = run_chain_trial(cfg, L, seed=self.seed + s)
                for a in area_names:
                    per_area[a].append(recovery[a])

            # Summarize: source area, final area, all areas
            source_stats = summarize(per_area[area_names[0]])
            final_stats = summarize(per_area[area_names[-1]])
            test_final = ttest_vs_null(per_area[area_names[-1]], null)

            # Per-area detail
            area_detail = []
            for a in area_names:
                a_stats = summarize(per_area[a])
                a_test = ttest_vs_null(per_area[a], null)
                hop = int(a[1:])  # extract hop number from "X0", "X1", etc.
                area_detail.append({
                    "area": a,
                    "hop": hop,
                    "recovery": a_stats,
                    "test_vs_chance": a_test,
                })

            row = {
                "chain_length": L,
                "n_areas": L + 1,
                "source_recovery": source_stats,
                "final_recovery": final_stats,
                "test_final_vs_chance": test_final,
                "per_area": area_detail,
            }
            h1_results.append(row)

            # Log per-area recovery for this chain length
            area_str = "  ".join(
                f"X{i}={summarize(per_area[f'X{i}'])['mean']:.3f}"
                for i in range(L + 1)
            )
            self.log(
                f"  {L}-hop chain ({L+1} areas): {area_str}  "
                f"[final d={test_final['d']:.1f}]"
            )

        metrics["h1h2_chain_length"] = h1_results

        # ================================================================
        # H3: Per-hop signal profile in a 5-hop chain
        # ================================================================
        # Already captured in H1/H2 per_area data for L=5.
        # Extract and present as a dedicated section for clarity.
        self.log("")
        self.log("=" * 60)
        self.log("H3: Per-hop propagation profile (5-hop chain)")
        self.log("=" * 60)

        # Find the L=5 result from H1/H2
        h3_data = None
        for r in h1_results:
            if r["chain_length"] == 5:
                h3_data = r["per_area"]
                break

        if h3_data:
            for entry in h3_data:
                rec = entry["recovery"]["mean"]
                sem = entry["recovery"]["sem"]
                d = entry["test_vs_chance"]["d"]
                self.log(
                    f"  {entry['area']} (hop {entry['hop']}): "
                    f"recovery={rec:.3f}+/-{sem:.3f}  d={d:.1f}"
                )

            # Compute per-hop decay rate
            means = [e["recovery"]["mean"] for e in h3_data]
            if len(means) >= 2:
                hop_decay = []
                for i in range(1, len(means)):
                    if means[i - 1] > 0:
                        hop_decay.append(means[i] / means[i - 1])
                avg_decay = float(np.mean(hop_decay)) if hop_decay else 1.0
                self.log(f"  Average per-hop retention: {avg_decay:.3f}")
                metrics["h3_per_hop_retention"] = avg_decay
            else:
                metrics["h3_per_hop_retention"] = None

        metrics["h3_propagation_profile"] = h3_data

        # ================================================================
        # H4: Chain capacity vs network size (fixed k/n=0.10, 3-hop chain)
        # ================================================================
        self.log("")
        self.log("=" * 60)
        self.log("H4: 3-hop chain recovery vs network size (k/n=0.10)")
        self.log("=" * 60)

        h4_sizes = [200, 500, 1000, 2000]
        h4_chain_length = 3
        h4_results = []

        for n_val in h4_sizes:
            k_val = int(0.10 * n_val)
            cfg_h4 = ChainConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h4 = chance_overlap(k_val, n_val)

            area_names_h4 = [f"X{i}" for i in range(h4_chain_length + 1)]
            per_area = {a: [] for a in area_names_h4}

            for s in seeds:
                recovery = run_chain_trial(cfg_h4, h4_chain_length, seed=self.seed + s)
                for a in area_names_h4:
                    per_area[a].append(recovery[a])

            final_stats = summarize(per_area[area_names_h4[-1]])
            test_final = ttest_vs_null(per_area[area_names_h4[-1]], null_h4)

            # Per-area detail
            area_detail = []
            for a in area_names_h4:
                a_stats = summarize(per_area[a])
                a_test = ttest_vs_null(per_area[a], null_h4)
                area_detail.append({
                    "area": a,
                    "recovery": a_stats,
                    "test_vs_chance": a_test,
                })

            row = {
                "n": n_val,
                "k": k_val,
                "chain_length": h4_chain_length,
                "final_recovery": final_stats,
                "test_final_vs_chance": test_final,
                "per_area": area_detail,
            }
            h4_results.append(row)

            area_str = "  ".join(
                f"X{i}={summarize(per_area[f'X{i}'])['mean']:.3f}"
                for i in range(h4_chain_length + 1)
            )
            self.log(
                f"  n={n_val:4d}, k={k_val:3d}: {area_str}  "
                f"[final d={test_final['d']:.1f}]"
            )

        metrics["h4_size_scaling"] = h4_results

        # ================================================================
        # H5: Sparse chain propagation (k=sqrt(n))
        # ================================================================
        # H1-H3 showed a ceiling effect at k/n=0.10 (lossless propagation).
        # Sparser representations should reveal genuine per-hop decay and
        # the actual maximum useful chain length.
        self.log("")
        self.log("=" * 60)
        self.log("H5: Sparse chain propagation (k=sqrt(n))")
        self.log("=" * 60)

        h5_sizes = [500, 1000, 2000, 5000]
        h5_chain_lengths = [1, 2, 3, 5]
        h5_results = []

        for n_val in h5_sizes:
            k_val = int(np.sqrt(n_val))
            cfg_h5 = ChainConfig(n=n_val, k=k_val, p=p, beta=beta, w_max=w_max)
            null_h5 = chance_overlap(k_val, n_val)

            self.log(f"  n={n_val}, k={k_val}, k/n={k_val/n_val:.3f}, "
                     f"k²p={k_val**2 * p:.0f}")

            size_results = {"n": n_val, "k": k_val, "chains": []}

            for L in h5_chain_lengths:
                area_names_h5 = [f"X{i}" for i in range(L + 1)]
                per_area = {a: [] for a in area_names_h5}

                for s in seeds:
                    recovery = run_chain_trial(cfg_h5, L, seed=self.seed + s)
                    for a in area_names_h5:
                        per_area[a].append(recovery[a])

                final_stats = summarize(per_area[area_names_h5[-1]])
                test_final = ttest_vs_null(per_area[area_names_h5[-1]], null_h5)

                area_detail = []
                for a in area_names_h5:
                    a_stats = summarize(per_area[a])
                    a_test = ttest_vs_null(per_area[a], null_h5)
                    area_detail.append({
                        "area": a,
                        "recovery": a_stats,
                        "test_vs_chance": a_test,
                    })

                chain_row = {
                    "chain_length": L,
                    "final_recovery": final_stats,
                    "test_final_vs_chance": test_final,
                    "per_area": area_detail,
                }
                size_results["chains"].append(chain_row)

                area_str = "  ".join(
                    f"X{i}={summarize(per_area[f'X{i}'])['mean']:.3f}"
                    for i in range(L + 1)
                )
                self.log(f"    {L}-hop: {area_str}")

            h5_results.append(size_results)

        metrics["h5_sparse_chains"] = h5_results

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": n, "k": k, "p": p, "beta": beta, "w_max": w_max,
                "establish_rounds": cfg.establish_rounds,
                "assoc_rounds": cfg.assoc_rounds,
                "propagation_rounds": cfg.propagation_rounds,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Association Chain Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer seeds)")

    args = parser.parse_args()

    exp = AssociationChainExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=5)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ASSOCIATION CHAIN SUMMARY")
    print("=" * 70)

    null = result.parameters["k"] / result.parameters["n"]

    print(f"\nH1/H2 -- Final-area recovery vs chain length (chance={null:.3f}):")
    for r in result.metrics["h1h2_chain_length"]:
        L = r["chain_length"]
        final = r["final_recovery"]["mean"]
        source = r["source_recovery"]["mean"]
        print(f"  {L}-hop: source={source:.3f}  final={final:.3f}")

    print("\nH3 -- Per-hop profile (5-hop chain):")
    if result.metrics["h3_propagation_profile"]:
        for entry in result.metrics["h3_propagation_profile"]:
            rec = entry["recovery"]["mean"]
            print(f"  {entry['area']} (hop {entry['hop']}): {rec:.3f}")
        if result.metrics.get("h3_per_hop_retention"):
            print(f"  Per-hop retention: {result.metrics['h3_per_hop_retention']:.3f}")

    print("\nH4 -- 3-hop chain vs network size:")
    for r in result.metrics["h4_size_scaling"]:
        n_val = r["n"]
        final = r["final_recovery"]["mean"]
        print(f"  n={n_val:4d}: final={final:.3f}")

    print("\nH5 -- Sparse chains (k=sqrt(n)):")
    for size_data in result.metrics.get("h5_sparse_chains", []):
        n_val = size_data["n"]
        k_val = size_data["k"]
        print(f"  n={n_val}, k={k_val}:")
        for chain in size_data["chains"]:
            L = chain["chain_length"]
            final = chain["final_recovery"]["mean"]
            print(f"    {L}-hop: final={final:.3f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
