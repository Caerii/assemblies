"""
N400 Parameter Sensitivity Sweep

Tests whether the global-energy N400 finding is robust across parameter
regimes, or whether it only works at n=50000, k=100, p=0.05, beta=0.05.

Sweeps across a grid of (n, k, p, beta) values, measuring:
- Global pre-k-WTA energy (1c from test_n400_pre_kwta.py)
- Settling dynamics cumulative energy (Path 3 from test_n400_pre_kwta.py)

These are the two metrics that showed correct N400 direction with massive
effect sizes (d=-25 to -31 for global energy, d=-16 to -18 for settling).

If the finding is robust, both metrics should show correct direction
(related < unrelated) across all parameter combinations. If it breaks
at certain parameters, that's informative about the mechanism.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from itertools import product

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from research.experiments.applications.test_n400_pre_kwta import PRIMING_TESTS
from research.experiments.vocab import build_standard_vocab, build_priming_pairs
from research.experiments.metrics import (
    measure_pre_kwta_activation, measure_settling_dynamics,
)
from research.experiments.infrastructure import build_core_lexicon
from src.assembly_calculus.emergent import EmergentParser


@dataclass
class SweepConfig:
    """Parameter grid for the sweep."""
    # Each list defines the values to sweep over
    n_values: List[int] = None
    k_values: List[int] = None
    p_values: List[float] = None
    beta_values: List[float] = None
    rounds: int = 10
    n_seeds: int = 3

    def __post_init__(self):
        if self.n_values is None:
            self.n_values = [10000, 50000]
        if self.k_values is None:
            self.k_values = [50, 100]
        if self.p_values is None:
            self.p_values = [0.01, 0.05]
        if self.beta_values is None:
            self.beta_values = [0.05, 0.10]


def _quick_config() -> SweepConfig:
    """Minimal grid for quick testing."""
    return SweepConfig(
        n_values=[10000, 50000],
        k_values=[50, 100],
        p_values=[0.01, 0.05],
        beta_values=[0.05, 0.10],
        n_seeds=2,
    )


def _full_config() -> SweepConfig:
    """Full grid for thorough testing."""
    return SweepConfig(
        n_values=[10000, 50000, 200000],
        k_values=[50, 100, 317],
        p_values=[0.01, 0.05, 0.10],
        beta_values=[0.01, 0.05, 0.10],
        n_seeds=3,
    )


class N400ParameterSweepExperiment(ExperimentBase):
    """Sweep N400 global-energy metric across parameter grid."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="n400_parameter_sweep",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = _quick_config() if quick else _full_config()
        engine = kwargs.get("engine", "numpy_sparse")

        vocab = build_standard_vocab()
        training = build_priming_pairs(vocab)

        test_words = set()
        for target, rel, unrel in PRIMING_TESTS:
            test_words.update([target, rel, unrel])

        grid = list(product(
            cfg.n_values, cfg.k_values, cfg.p_values, cfg.beta_values))

        self.log(f"Parameter grid: {len(grid)} combinations x "
                 f"{cfg.n_seeds} seeds = {len(grid) * cfg.n_seeds} runs")
        self.log(f"Engine: {engine}")

        all_results = []

        for combo_idx, (n, k, p, beta) in enumerate(grid):
            # Skip invalid combinations (k must be < n)
            if k >= n:
                continue

            combo_label = f"n={n},k={k},p={p},beta={beta}"
            self.log(f"\n--- Combo {combo_idx + 1}/{len(grid)}: {combo_label} ---")

            energy_rel_seeds = []
            energy_unrel_seeds = []
            settling_rel_seeds = []
            settling_unrel_seeds = []

            for seed_idx in range(cfg.n_seeds):
                seed = seed_idx
                try:
                    parser = EmergentParser(
                        n=n, k=k, p=p, beta=beta,
                        seed=seed, rounds=cfg.rounds,
                        vocabulary=vocab, engine=engine,
                    )
                    parser.train(sentences=training)

                    lexicon = build_core_lexicon(
                        parser, list(test_words), cfg.rounds)

                    # Measure global energy for each test pair
                    energy_r, energy_u = [], []
                    settling_r, settling_u = [], []

                    for target, rel_prime, unrel_prime in PRIMING_TESTS:
                        if target not in lexicon or rel_prime not in lexicon:
                            continue

                        m_rel = measure_pre_kwta_activation(
                            parser, rel_prime, target, lexicon, cfg.rounds)
                        m_unrel = measure_pre_kwta_activation(
                            parser, unrel_prime, target, lexicon, cfg.rounds)
                        energy_r.append(m_rel["global_energy"])
                        energy_u.append(m_unrel["global_energy"])

                        s_rel = measure_settling_dynamics(
                            parser, rel_prime, target, lexicon,
                            max_rounds=cfg.rounds)
                        s_unrel = measure_settling_dynamics(
                            parser, unrel_prime, target, lexicon,
                            max_rounds=cfg.rounds)
                        settling_r.append(s_rel["cumulative_energy"])
                        settling_u.append(s_unrel["cumulative_energy"])

                    if energy_r:
                        energy_rel_seeds.append(float(np.mean(energy_r)))
                        energy_unrel_seeds.append(float(np.mean(energy_u)))
                    if settling_r:
                        settling_rel_seeds.append(float(np.mean(settling_r)))
                        settling_unrel_seeds.append(float(np.mean(settling_u)))

                except Exception as e:
                    self.log(f"  Seed {seed_idx} failed: {e}")
                    continue

            # Analyze this combination
            combo_result = {
                "n": n, "k": k, "p": p, "beta": beta,
                "n_seeds_completed": len(energy_rel_seeds),
            }

            if len(energy_rel_seeds) >= 2:
                e_stats = paired_ttest(energy_rel_seeds, energy_unrel_seeds)
                e_rel_s = summarize(energy_rel_seeds)
                e_unrel_s = summarize(energy_unrel_seeds)
                direction = "CORRECT" if e_rel_s["mean"] < e_unrel_s["mean"] else "REVERSED"
                combo_result["global_energy"] = {
                    "related": e_rel_s, "unrelated": e_unrel_s,
                    "test": e_stats, "direction": direction,
                }
                self.log(f"  Energy: rel={e_rel_s['mean']:.1f} "
                         f"unrel={e_unrel_s['mean']:.1f}  "
                         f"d={e_stats['d']:.2f}  p={e_stats['p']:.4f}  "
                         f"{direction}")
            else:
                combo_result["global_energy"] = {"direction": "INSUFFICIENT_DATA"}
                self.log(f"  Energy: insufficient data ({len(energy_rel_seeds)} seeds)")

            if len(settling_rel_seeds) >= 2:
                s_stats = paired_ttest(settling_rel_seeds, settling_unrel_seeds)
                s_rel_s = summarize(settling_rel_seeds)
                s_unrel_s = summarize(settling_unrel_seeds)
                direction = "CORRECT" if s_rel_s["mean"] < s_unrel_s["mean"] else "REVERSED"
                combo_result["settling"] = {
                    "related": s_rel_s, "unrelated": s_unrel_s,
                    "test": s_stats, "direction": direction,
                }
                self.log(f"  Settling: rel={s_rel_s['mean']:.1f} "
                         f"unrel={s_unrel_s['mean']:.1f}  "
                         f"d={s_stats['d']:.2f}  p={s_stats['p']:.4f}  "
                         f"{direction}")
            else:
                combo_result["settling"] = {"direction": "INSUFFICIENT_DATA"}
                self.log(f"  Settling: insufficient data")

            all_results.append(combo_result)

        # Summary
        self.log("\n" + "=" * 70)
        self.log("PARAMETER SWEEP SUMMARY")
        self.log("=" * 70)

        n_correct_energy = sum(
            1 for r in all_results
            if r.get("global_energy", {}).get("direction") == "CORRECT")
        n_correct_settling = sum(
            1 for r in all_results
            if r.get("settling", {}).get("direction") == "CORRECT")
        n_tested = sum(
            1 for r in all_results
            if r.get("global_energy", {}).get("direction") not in
            (None, "INSUFFICIENT_DATA"))

        self.log(f"\nGlobal energy: {n_correct_energy}/{n_tested} correct direction")
        self.log(f"Settling:      {n_correct_settling}/{n_tested} correct direction")

        self.log(f"\n{'n':>6} {'k':>4} {'p':>5} {'beta':>5} | "
                 f"{'Energy_d':>9} {'E_dir':>8} | "
                 f"{'Settl_d':>9} {'S_dir':>8}")
        self.log("-" * 70)
        for r in all_results:
            e = r.get("global_energy", {})
            s = r.get("settling", {})
            e_d = e.get("test", {}).get("d", float("nan"))
            s_d = s.get("test", {}).get("d", float("nan"))
            e_dir = e.get("direction", "N/A")
            s_dir = s.get("direction", "N/A")
            self.log(f"{r['n']:>6} {r['k']:>4} {r['p']:>5.2f} {r['beta']:>5.2f} | "
                     f"{e_d:>9.2f} {e_dir:>8} | "
                     f"{s_d:>9.2f} {s_dir:>8}")

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "grid": {
                    "n_values": cfg.n_values,
                    "k_values": cfg.k_values,
                    "p_values": cfg.p_values,
                    "beta_values": cfg.beta_values,
                },
                "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
                "engine": engine,
            },
            metrics={
                "n_combinations_tested": n_tested,
                "n_correct_energy": n_correct_energy,
                "n_correct_settling": n_correct_settling,
                "fraction_correct_energy": (
                    n_correct_energy / n_tested if n_tested > 0 else 0),
                "fraction_correct_settling": (
                    n_correct_settling / n_tested if n_tested > 0 else 0),
                "per_combination": all_results,
            },
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="N400 Parameter Sensitivity Sweep")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with smaller grid (2x2x2x2 instead of 3x3x3x3)")
    parser.add_argument(
        "--engine", default="numpy_sparse",
        choices=["numpy_sparse", "torch_sparse"],
        help="Engine backend to use")
    args = parser.parse_args()

    exp = N400ParameterSweepExperiment()
    exp.run(quick=args.quick, engine=args.engine)
