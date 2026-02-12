"""
N400 via Pre-k-WTA Activation — Three Measurement Paths

Motivation
----------
Previous N400 experiments (test_n400_amplification.py, test_n400_context.py)
measured POST-k-WTA assembly state — which neurons won the competition — and
consistently found semantic INTERFERENCE (reversed N400 direction across all
13 conditions tested).  This is because k-WTA forces related words to compete
for the same neuron slots.

Literature review reveals the N400 reflects PRE-competition activation:
- Nour Eddine et al. 2024: N400 = lexico-semantic prediction error
- Cheyette & Plaut 2017: N400 = transient over-activation during settling
- Kutas & Federmeier 2011: N400 = ease of semantic memory access

The key insight: the ``all_inputs`` tensor in ``project_into()`` (before
``topk`` selection) is the continuous activation landscape that k-WTA
discards.  The N400 should be measured HERE, not in the post-competition
winner set.

Three Measurement Paths
-----------------------
Path 1 — Pre-k-WTA Activation (4 sub-metrics):
    a) Mean input at target assembly neuron positions
    b) Max input at target positions
    c) Global energy: sum of ALL pre-k-WTA inputs (total synaptic input)
    d) Rank fraction: how many target neurons are in the top-k

Path 2 — Prediction Error:
    Cosine similarity between the prime's recurrence prediction (via
    core→core Hebbian weights) and the target's assembly signature.

Path 3 — Settling Dynamics (transient over-activation):
    Cumulative pre-k-WTA energy across multiple projection rounds.
    Models the Cheyette & Plaut transient over-activation account.

Results (n=50000, k=100, p=0.05, beta=0.05, rounds=10, 3-5 seeds)
------------------------------------------------------------------
Metric               Direction    Cohen's d    p
1a:mean_input        REVERSED      -5.8       0.010
1b:max_input         REVERSED      -6.6       0.008
1c:global_energy     CORRECT      -25.2       0.001   ***
1d:rank_frac         REVERSED      -8.5       0.005
2a:pred_error        CORRECT       -0.1       0.858
2b:cosine_sim        CORRECT        0.1       0.858
3:settling           CORRECT      -16.6       0.001   ***

Key Finding
-----------
The N400 in the Assembly Calculus maps to GLOBAL pre-k-WTA energy
(sum of all_inputs across all neurons), NOT to neuron-specific activation.

- Neuron-specific metrics (1a, 1b, 1d) show reversed direction because
  related primes strengthen shared neurons but create competition for
  target-unique neurons, pulling the mean down.

- Global energy metrics (1c, Path 3) show CORRECT direction with massive
  effect sizes because they capture the total system workload — related
  primes reduce total energy needed via pre-activated shared features.

This maps to the neuroscience: the N400 ERP is a scalp-recorded aggregate
signal reflecting total cortical energy expenditure, not single-neuron activity.

    N400 amplitude ~ sum(all_inputs)  [global pre-k-WTA energy]
    Related prime  -> lower global energy -> smaller N400
    Unrelated prime -> higher global energy -> larger N400

Engine Extension
----------------
This experiment uses ``record_activation=True`` in ``project_into()``,
which populates ``ProjectionResult.pre_kwta_inputs``, ``.pre_kwta_prev_only``,
and ``.pre_kwta_total``.  Added in this commit to all engines.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import List

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from research.experiments.vocab import build_standard_vocab, build_priming_pairs
from research.experiments.metrics import (
    measure_pre_kwta_activation, measure_settling_dynamics,
    measure_prediction_error,
)
from research.experiments.infrastructure import build_core_lexicon
from src.assembly_calculus.emergent import EmergentParser


@dataclass
class PreKwtaConfig:
    """Configuration matching previous N400 experiments."""
    n: int = 50000
    k: int = 100
    n_seeds: int = 5
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10


# Test triplets: (target, related_prime, unrelated_prime)
PRIMING_TESTS = [
    ("cat",   "dog",   "table"),
    ("bird",  "cat",   "chair"),
    ("horse", "fish",  "book"),
    ("fish",  "bird",  "car"),
    ("dog",   "horse", "ball"),
    ("table", "chair", "dog"),
    ("chair", "table", "cat"),
    ("book",  "ball",  "bird"),
]


# -- Experiment class ---------------------------------------------------------

class N400PreKwtaExperiment(ExperimentBase):
    """Test N400 via pre-k-WTA activation (3 measurement paths)."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="n400_pre_kwta",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = PreKwtaConfig()
        if quick:
            cfg.n_seeds = 3
        engine = kwargs.get("engine", "numpy_sparse")

        vocab = build_standard_vocab()
        training = build_priming_pairs(vocab)
        seeds = list(range(cfg.n_seeds))

        test_words = set()
        for target, rel, unrel in PRIMING_TESTS:
            test_words.update([target, rel, unrel])

        self.log(f"Engine: {engine}")

        # Collect per-seed, per-pair measurements
        # Path 1: multiple sub-metrics
        p1_mean_rel, p1_mean_unrel = [], []
        p1_max_rel, p1_max_unrel = [], []
        p1_energy_rel, p1_energy_unrel = [], []
        p1_rank_rel, p1_rank_unrel = [], []
        p2_related_all, p2_unrelated_all = [], []
        p2_cosine_rel, p2_cosine_unrel = [], []
        p3_related_all, p3_unrelated_all = [], []

        for seed_idx, seed in enumerate(seeds):
            self.log(f"=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
                engine=engine,
            )
            parser.train(sentences=training)

            # Build core lexicon (compact indices for each word)
            lexicon = build_core_lexicon(
                parser, list(test_words), cfg.rounds)

            # -- Path 1: Pre-k-WTA Activation --
            self.log("  Path 1: Pre-k-WTA activation...")
            s_mean_r, s_mean_u = [], []
            s_max_r, s_max_u = [], []
            s_energy_r, s_energy_u = [], []
            s_rank_r, s_rank_u = [], []
            for target, rel_prime, unrel_prime in PRIMING_TESTS:
                if target not in lexicon or rel_prime not in lexicon:
                    continue

                m_rel = measure_pre_kwta_activation(
                    parser, rel_prime, target, lexicon, cfg.rounds)
                m_unrel = measure_pre_kwta_activation(
                    parser, unrel_prime, target, lexicon, cfg.rounds)

                s_mean_r.append(m_rel["mean_input"])
                s_mean_u.append(m_unrel["mean_input"])
                s_max_r.append(m_rel["max_input"])
                s_max_u.append(m_unrel["max_input"])
                s_energy_r.append(m_rel["global_energy"])
                s_energy_u.append(m_unrel["global_energy"])
                s_rank_r.append(m_rel["target_rank_frac"])
                s_rank_u.append(m_unrel["target_rank_frac"])

            if s_mean_r:
                p1_mean_rel.append(float(np.mean(s_mean_r)))
                p1_mean_unrel.append(float(np.mean(s_mean_u)))
                p1_max_rel.append(float(np.mean(s_max_r)))
                p1_max_unrel.append(float(np.mean(s_max_u)))
                p1_energy_rel.append(float(np.mean(s_energy_r)))
                p1_energy_unrel.append(float(np.mean(s_energy_u)))
                p1_rank_rel.append(float(np.mean(s_rank_r)))
                p1_rank_unrel.append(float(np.mean(s_rank_u)))
            self.log(f"    mean:  rel={np.mean(s_mean_r):.4f}  "
                     f"unrel={np.mean(s_mean_u):.4f}")
            self.log(f"    max:   rel={np.mean(s_max_r):.4f}  "
                     f"unrel={np.mean(s_max_u):.4f}")
            self.log(f"    rank:  rel={np.mean(s_rank_r):.4f}  "
                     f"unrel={np.mean(s_rank_u):.4f}")

            # -- Path 2: Prediction Error --
            self.log("  Path 2: Prediction error...")
            p2_rel_seed, p2_unrel_seed = [], []
            p2_err_r, p2_err_u = [], []
            p2_cos_r, p2_cos_u = [], []
            for target, rel_prime, unrel_prime in PRIMING_TESTS:
                if target not in lexicon:
                    continue

                m_rel = measure_prediction_error(
                    parser, rel_prime, target, lexicon, cfg.rounds)
                m_unrel = measure_prediction_error(
                    parser, unrel_prime, target, lexicon, cfg.rounds)

                if not np.isnan(m_rel["prediction_error"]):
                    p2_err_r.append(m_rel["prediction_error"])
                    p2_err_u.append(m_unrel["prediction_error"])
                    p2_cos_r.append(m_rel["cosine_sim"])
                    p2_cos_u.append(m_unrel["cosine_sim"])

            if p2_err_r:
                p2_related_all.append(float(np.mean(p2_err_r)))
                p2_unrelated_all.append(float(np.mean(p2_err_u)))
                p2_cosine_rel.append(float(np.mean(p2_cos_r)))
                p2_cosine_unrel.append(float(np.mean(p2_cos_u)))
                self.log(f"    error: rel={np.mean(p2_err_r):.6f}  "
                         f"unrel={np.mean(p2_err_u):.6f}")
                self.log(f"    cosine: rel={np.mean(p2_cos_r):.6f}  "
                         f"unrel={np.mean(p2_cos_u):.6f}")
            else:
                self.log("    (no valid prediction error data)")

            # -- Path 3: Settling Dynamics --
            self.log("  Path 3: Settling dynamics...")
            p3_rel_seed, p3_unrel_seed = [], []
            for target, rel_prime, unrel_prime in PRIMING_TESTS:
                if target not in lexicon:
                    continue

                m_rel = measure_settling_dynamics(
                    parser, rel_prime, target, lexicon,
                    max_rounds=cfg.rounds)
                m_unrel = measure_settling_dynamics(
                    parser, unrel_prime, target, lexicon,
                    max_rounds=cfg.rounds)

                p3_rel_seed.append(m_rel["cumulative_energy"])
                p3_unrel_seed.append(m_unrel["cumulative_energy"])

            if p3_rel_seed:
                p3_related_all.append(float(np.mean(p3_rel_seed)))
                p3_unrelated_all.append(float(np.mean(p3_unrel_seed)))
            self.log(f"    related={np.mean(p3_rel_seed):.2f}  "
                     f"unrelated={np.mean(p3_unrel_seed):.2f}")

        # -- Statistical analysis --
        self.log("\n" + "=" * 60)
        self.log("RESULTS")
        self.log("=" * 60)

        metrics = {}

        # Path 1 — multiple sub-metrics
        self.log("\nPath 1: Pre-k-WTA Activation")
        p1_sub = {}
        for label, rel_list, unrel_list, pred_dir in [
            ("1a:mean_input", p1_mean_rel, p1_mean_unrel, ">"),
            ("1b:max_input", p1_max_rel, p1_max_unrel, ">"),
            ("1c:global_energy", p1_energy_rel, p1_energy_unrel, "<"),
            ("1d:rank_frac", p1_rank_rel, p1_rank_unrel, ">"),
        ]:
            if len(rel_list) < 2:
                continue
            stats = paired_ttest(rel_list, unrel_list)
            rel_s = summarize(rel_list)
            unrel_s = summarize(unrel_list)
            if pred_dir == ">":
                direction = "CORRECT" if rel_s["mean"] > unrel_s["mean"] else "REVERSED"
            else:
                direction = "CORRECT" if rel_s["mean"] < unrel_s["mean"] else "REVERSED"
            self.log(f"  {label}: rel={rel_s['mean']:.4f} "
                     f"unrel={unrel_s['mean']:.4f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}  {direction}")
            p1_sub[label] = {
                "related": rel_s, "unrelated": unrel_s,
                "test": stats, "direction": direction,
            }
        metrics["path1"] = p1_sub

        # Path 2
        self.log("\nPath 2: Prediction Error (recurrence prediction vs target)")
        p2_sub = {}
        for label, rel_list, unrel_list, pred_dir in [
            ("2a:pred_error", p2_related_all, p2_unrelated_all, "<"),
            ("2b:cosine_sim", p2_cosine_rel, p2_cosine_unrel, ">"),
        ]:
            if len(rel_list) < 2:
                self.log(f"  {label}: SKIPPED (no valid data)")
                continue
            stats = paired_ttest(rel_list, unrel_list)
            rel_s = summarize(rel_list)
            unrel_s = summarize(unrel_list)
            if pred_dir == ">":
                direction = "CORRECT" if rel_s["mean"] > unrel_s["mean"] else "REVERSED"
            else:
                direction = "CORRECT" if rel_s["mean"] < unrel_s["mean"] else "REVERSED"
            self.log(f"  {label}: rel={rel_s['mean']:.6f} "
                     f"unrel={unrel_s['mean']:.6f}  "
                     f"d={stats['d']:.3f}  p={stats['p']:.4f}  {direction}")
            p2_sub[label] = {
                "related": rel_s, "unrelated": unrel_s,
                "test": stats, "direction": direction,
            }
        metrics["path2"] = p2_sub

        # Path 3
        self.log("\nPath 3: Settling Dynamics (cumulative pre-k-WTA energy)")
        self.log(f"  Prediction: related < unrelated (less energy)")
        p3_stats = paired_ttest(p3_related_all, p3_unrelated_all)
        p3_rel_summary = summarize(p3_related_all)
        p3_unrel_summary = summarize(p3_unrelated_all)
        direction = "CORRECT" if p3_rel_summary["mean"] < p3_unrel_summary["mean"] else "REVERSED"
        self.log(f"  Related:   {p3_rel_summary['mean']:.2f} "
                 f"+/- {p3_rel_summary['sem']:.2f}")
        self.log(f"  Unrelated: {p3_unrel_summary['mean']:.2f} "
                 f"+/- {p3_unrel_summary['sem']:.2f}")
        self.log(f"  Direction: {direction}")
        self.log(f"  Cohen's d: {p3_stats['d']:.3f}  p={p3_stats['p']:.4f}")
        metrics["path3"] = {
            "related": p3_rel_summary,
            "unrelated": p3_unrel_summary,
            "test": p3_stats,
            "direction": direction,
        }

        # Summary table
        self.log("\n" + "-" * 60)
        self.log("SUMMARY")
        self.log("-" * 60)
        self.log(f"{'Metric':<20} {'Direction':<10} {'d':<8} {'p':<8}")
        for path_key in ["path1", "path2", "path3"]:
            m = metrics[path_key]
            if isinstance(m, dict) and any(
                    isinstance(v, dict) and "test" in v for v in m.values()):
                for sub_key, sub in m.items():
                    if "test" in sub:
                        self.log(f"{sub_key:<20} {sub['direction']:<10} "
                                 f"{sub['test']['d']:<8.3f} "
                                 f"{sub['test']['p']:<8.4f}")
            elif "test" in m:
                self.log(f"{path_key:<20} {m['direction']:<10} "
                         f"{m['test']['d']:<8.3f} {m['test']['p']:<8.4f}")

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p,
                "beta": cfg.beta, "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds, "engine": engine,
            },
            metrics=metrics,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


# -- CLI entry point ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="N400 Pre-k-WTA Activation Experiment")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    parser.add_argument(
        "--engine", default="numpy_sparse",
        choices=["numpy_sparse", "torch_sparse"],
        help="Engine backend to use")
    args = parser.parse_args()

    exp = N400PreKwtaExperiment()
    exp.run(quick=args.quick, engine=args.engine)
