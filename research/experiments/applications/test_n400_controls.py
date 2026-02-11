"""
N400 Control Conditions — Ruling Out Confounds

Tests whether the global-energy N400 effect is truly semantic by running
control conditions that should produce specific, known outcomes:

Condition A — Semantic Priming (baseline):
    Related vs unrelated prime (same as test_n400_pre_kwta.py).
    Expected: related < unrelated (N400 facilitation).

Condition B — Repetition Priming:
    Prime == target. Should show stronger facilitation than semantic priming.
    The N400 literature consistently shows repetition > semantic priming.
    Expected: repetition < related < unrelated.

Condition C — Shuffled Control:
    Same words but randomly permute prime-target pairings within category.
    If the effect is truly semantic (from trained Hebbian weights between
    related words), shuffled pairings should show no systematic effect.
    Expected: no significant difference (null effect).

Condition D — Cross-Category Association:
    Pairs that co-occurred in training but are from different categories
    (e.g., "dog" and "ball" from "The dog finds the ball").
    N400 literature shows partial facilitation for associated words.
    Expected: cross-category < unrelated, but cross-category > related.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from research.experiments.applications.test_n400_pre_kwta import (
    _build_vocab, _build_training,
    build_core_lexicon, measure_pre_kwta_activation,
)
from src.assembly_calculus.emergent import EmergentParser


@dataclass
class ControlConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5


# -- Test triplets for each condition -----------------------------------------

# Condition A: Semantic priming (same as test_n400_pre_kwta.py)
SEMANTIC_TESTS = [
    ("cat",   "dog",   "table"),
    ("bird",  "cat",   "chair"),
    ("horse", "fish",  "book"),
    ("fish",  "bird",  "car"),
    ("dog",   "horse", "ball"),
    ("table", "chair", "dog"),
    ("chair", "table", "cat"),
    ("book",  "ball",  "bird"),
]

# Condition B: Repetition priming — prime == target
# Use same targets as semantic tests, compare repetition vs semantic vs unrelated
REPETITION_TARGETS = ["cat", "bird", "horse", "fish", "dog", "table", "chair", "book"]

# Condition C: Shuffled — pair each target with a prime from a DIFFERENT
# category that has NO shared features and NO co-occurrence in training.
# This is the true null control: the prime has zero semantic relation.
# Compare shuffled_prime vs another_unrelated_prime — both should be
# equally unrelated, so we expect no significant difference.
SHUFFLED_TESTS = [
    # (target, unrelated_prime_A, unrelated_prime_B)
    # Both primes are unrelated to target — expect null difference
    ("cat",   "table",  "book"),    # animal target, two object primes
    ("bird",  "chair",  "ball"),    # animal target, two object primes
    ("horse", "book",   "car"),     # animal target, two object primes
    ("fish",  "ball",   "cup"),     # animal target, two object primes
    ("dog",   "car",    "chair"),   # animal target, two object primes
    ("table", "dog",    "bird"),    # object target, two animal primes
    ("chair", "cat",    "fish"),    # object target, two animal primes
    ("book",  "horse",  "mouse"),   # object target, two animal primes
]

# Condition D: Cross-category association (co-occurred in training sentences)
# Training had: "dog finds ball", "cat sees book", "bird finds car",
# "horse sees table", "dog likes chair", "cat likes cup"
CROSS_CATEGORY_TESTS = [
    # (target, associated_prime, unrelated_prime)
    ("ball",  "dog",   "cat"),     # "dog finds ball" in training
    ("book",  "cat",   "horse"),   # "cat sees book" in training
    ("car",   "bird",  "fish"),    # "bird finds car" in training
    ("table", "horse", "dog"),     # "horse sees table" in training
    ("chair", "dog",   "bird"),    # "dog likes chair" in training
    ("cup",   "cat",   "fish"),    # "cat likes cup" in training
]


class N400ControlsExperiment(ExperimentBase):
    """Test N400 control conditions."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="n400_controls",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def _measure_condition(
        self, parser, pairs, lexicon, rounds, label,
    ) -> Dict[str, List[float]]:
        """Measure global energy for a set of (target, prime_a, prime_b) triplets.

        Returns dict with 'a' and 'b' lists of per-pair global energy values.
        """
        a_vals, b_vals = [], []
        for triplet in pairs:
            target, prime_a, prime_b = triplet
            if target not in lexicon or prime_a not in lexicon:
                continue
            if prime_b not in lexicon:
                continue

            m_a = measure_pre_kwta_activation(
                parser, prime_a, target, lexicon, rounds)
            m_b = measure_pre_kwta_activation(
                parser, prime_b, target, lexicon, rounds)
            a_vals.append(m_a["global_energy"])
            b_vals.append(m_b["global_energy"])

        return {"a": a_vals, "b": b_vals}

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = ControlConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)

        test_words = set()
        for tests in [SEMANTIC_TESTS, SHUFFLED_TESTS, CROSS_CATEGORY_TESTS]:
            for triplet in tests:
                test_words.update(triplet)
        test_words.update(REPETITION_TARGETS)

        seeds = list(range(cfg.n_seeds))

        # Per-seed accumulators for each condition
        # A: semantic (related vs unrelated)
        a_rel_seeds, a_unrel_seeds = [], []
        # B: repetition vs related vs unrelated
        b_rep_seeds, b_sem_seeds, b_unrel_seeds = [], [], []
        # C: shuffled (shuffled_related vs unrelated)
        c_shuf_seeds, c_unrel_seeds = [], []
        # D: cross-category (associated vs unrelated)
        d_assoc_seeds, d_unrel_seeds = [], []

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            lexicon = build_core_lexicon(
                parser, list(test_words), cfg.rounds)

            # -- Condition A: Semantic priming --
            self.log("  Condition A: Semantic priming...")
            res_a = self._measure_condition(
                parser, SEMANTIC_TESTS, lexicon, cfg.rounds, "semantic")
            if res_a["a"]:
                a_rel_seeds.append(float(np.mean(res_a["a"])))
                a_unrel_seeds.append(float(np.mean(res_a["b"])))
                self.log(f"    related={np.mean(res_a['a']):.1f}  "
                         f"unrelated={np.mean(res_a['b']):.1f}")

            # -- Condition B: Repetition priming --
            self.log("  Condition B: Repetition priming...")
            rep_vals, sem_vals, unrel_vals = [], [], []
            for target in REPETITION_TARGETS:
                if target not in lexicon:
                    continue
                # Find semantic pair for this target
                sem_prime = None
                unrel_prime = None
                for t, r, u in SEMANTIC_TESTS:
                    if t == target:
                        sem_prime, unrel_prime = r, u
                        break
                if sem_prime is None or sem_prime not in lexicon:
                    continue
                if unrel_prime not in lexicon:
                    continue

                m_rep = measure_pre_kwta_activation(
                    parser, target, target, lexicon, cfg.rounds)
                m_sem = measure_pre_kwta_activation(
                    parser, sem_prime, target, lexicon, cfg.rounds)
                m_unrel = measure_pre_kwta_activation(
                    parser, unrel_prime, target, lexicon, cfg.rounds)
                rep_vals.append(m_rep["global_energy"])
                sem_vals.append(m_sem["global_energy"])
                unrel_vals.append(m_unrel["global_energy"])

            if rep_vals:
                b_rep_seeds.append(float(np.mean(rep_vals)))
                b_sem_seeds.append(float(np.mean(sem_vals)))
                b_unrel_seeds.append(float(np.mean(unrel_vals)))
                self.log(f"    repetition={np.mean(rep_vals):.1f}  "
                         f"semantic={np.mean(sem_vals):.1f}  "
                         f"unrelated={np.mean(unrel_vals):.1f}")

            # -- Condition C: Shuffled priming --
            self.log("  Condition C: Shuffled control...")
            res_c = self._measure_condition(
                parser, SHUFFLED_TESTS, lexicon, cfg.rounds, "shuffled")
            if res_c["a"]:
                c_shuf_seeds.append(float(np.mean(res_c["a"])))
                c_unrel_seeds.append(float(np.mean(res_c["b"])))
                self.log(f"    shuffled={np.mean(res_c['a']):.1f}  "
                         f"unrelated={np.mean(res_c['b']):.1f}")

            # -- Condition D: Cross-category --
            self.log("  Condition D: Cross-category association...")
            res_d = self._measure_condition(
                parser, CROSS_CATEGORY_TESTS, lexicon, cfg.rounds, "cross-cat")
            if res_d["a"]:
                d_assoc_seeds.append(float(np.mean(res_d["a"])))
                d_unrel_seeds.append(float(np.mean(res_d["b"])))
                self.log(f"    associated={np.mean(res_d['a']):.1f}  "
                         f"unrelated={np.mean(res_d['b']):.1f}")

        # -- Statistical analysis --
        self.log("\n" + "=" * 70)
        self.log("RESULTS")
        self.log("=" * 70)

        metrics = {}

        # Condition A: semantic priming (baseline replication)
        self.log("\nCondition A: Semantic Priming (related < unrelated)")
        if len(a_rel_seeds) >= 2:
            stats_a = paired_ttest(a_rel_seeds, a_unrel_seeds)
            rel_s = summarize(a_rel_seeds)
            unrel_s = summarize(a_unrel_seeds)
            direction = "CORRECT" if rel_s["mean"] < unrel_s["mean"] else "REVERSED"
            self.log(f"  Related:   {rel_s['mean']:.1f} +/- {rel_s['sem']:.1f}")
            self.log(f"  Unrelated: {unrel_s['mean']:.1f} +/- {unrel_s['sem']:.1f}")
            self.log(f"  d={stats_a['d']:.3f}  p={stats_a['p']:.4f}  {direction}")
            metrics["A_semantic"] = {
                "related": rel_s, "unrelated": unrel_s,
                "test": stats_a, "direction": direction,
            }

        # Condition B: repetition priming
        self.log("\nCondition B: Repetition Priming (rep < semantic < unrelated)")
        if len(b_rep_seeds) >= 2:
            # Repetition vs unrelated
            stats_rep_unrel = paired_ttest(b_rep_seeds, b_unrel_seeds)
            # Repetition vs semantic
            stats_rep_sem = paired_ttest(b_rep_seeds, b_sem_seeds)
            # Semantic vs unrelated
            stats_sem_unrel = paired_ttest(b_sem_seeds, b_unrel_seeds)

            rep_s = summarize(b_rep_seeds)
            sem_s = summarize(b_sem_seeds)
            unrel_s = summarize(b_unrel_seeds)

            ordering_correct = (rep_s["mean"] < sem_s["mean"] < unrel_s["mean"])
            self.log(f"  Repetition: {rep_s['mean']:.1f} +/- {rep_s['sem']:.1f}")
            self.log(f"  Semantic:   {sem_s['mean']:.1f} +/- {sem_s['sem']:.1f}")
            self.log(f"  Unrelated:  {unrel_s['mean']:.1f} +/- {unrel_s['sem']:.1f}")
            self.log(f"  Ordering rep < sem < unrel: "
                     f"{'CORRECT' if ordering_correct else 'INCORRECT'}")
            self.log(f"  Rep vs Unrel:  d={stats_rep_unrel['d']:.3f}  "
                     f"p={stats_rep_unrel['p']:.4f}")
            self.log(f"  Rep vs Sem:    d={stats_rep_sem['d']:.3f}  "
                     f"p={stats_rep_sem['p']:.4f}")
            self.log(f"  Sem vs Unrel:  d={stats_sem_unrel['d']:.3f}  "
                     f"p={stats_sem_unrel['p']:.4f}")
            metrics["B_repetition"] = {
                "repetition": rep_s, "semantic": sem_s, "unrelated": unrel_s,
                "ordering_correct": ordering_correct,
                "rep_vs_unrel": stats_rep_unrel,
                "rep_vs_sem": stats_rep_sem,
                "sem_vs_unrel": stats_sem_unrel,
            }

        # Condition C: shuffled control
        self.log("\nCondition C: Shuffled Control (expect null effect)")
        if len(c_shuf_seeds) >= 2:
            stats_c = paired_ttest(c_shuf_seeds, c_unrel_seeds)
            shuf_s = summarize(c_shuf_seeds)
            unrel_s = summarize(c_unrel_seeds)
            # For shuffled, we expect NO significant difference
            is_null = stats_c["p"] > 0.05
            self.log(f"  Shuffled:  {shuf_s['mean']:.1f} +/- {shuf_s['sem']:.1f}")
            self.log(f"  Unrelated: {unrel_s['mean']:.1f} +/- {unrel_s['sem']:.1f}")
            self.log(f"  d={stats_c['d']:.3f}  p={stats_c['p']:.4f}  "
                     f"{'NULL (expected)' if is_null else 'SIGNIFICANT (unexpected)'}")
            metrics["C_shuffled"] = {
                "shuffled": shuf_s, "unrelated": unrel_s,
                "test": stats_c, "is_null_as_expected": is_null,
            }

        # Condition D: cross-category
        self.log("\nCondition D: Cross-Category Association (associated < unrelated)")
        if len(d_assoc_seeds) >= 2:
            stats_d = paired_ttest(d_assoc_seeds, d_unrel_seeds)
            assoc_s = summarize(d_assoc_seeds)
            unrel_s = summarize(d_unrel_seeds)
            direction = "CORRECT" if assoc_s["mean"] < unrel_s["mean"] else "REVERSED"
            self.log(f"  Associated: {assoc_s['mean']:.1f} +/- {assoc_s['sem']:.1f}")
            self.log(f"  Unrelated:  {unrel_s['mean']:.1f} +/- {unrel_s['sem']:.1f}")
            self.log(f"  d={stats_d['d']:.3f}  p={stats_d['p']:.4f}  {direction}")
            metrics["D_cross_category"] = {
                "associated": assoc_s, "unrelated": unrel_s,
                "test": stats_d, "direction": direction,
            }

        # Compare effect sizes across conditions
        self.log("\n" + "-" * 70)
        self.log("EFFECT SIZE COMPARISON")
        self.log("-" * 70)
        self.log(f"{'Condition':<30} {'d':<9} {'p':<9} {'Result'}")
        for key, label, expected in [
            ("A_semantic", "A: Semantic priming", "d < 0"),
            ("C_shuffled", "C: Shuffled control", "p > 0.05"),
            ("D_cross_category", "D: Cross-category", "d < 0"),
        ]:
            if key in metrics:
                m = metrics[key]
                t = m.get("test", {})
                d_val = t.get("d", float("nan"))
                p_val = t.get("p", float("nan"))
                result = m.get("direction", m.get("is_null_as_expected", "?"))
                self.log(f"  {label:<28} {d_val:<9.3f} {p_val:<9.4f} {result}")

        if "B_repetition" in metrics:
            m = metrics["B_repetition"]
            self.log(f"  {'B: Rep vs Unrel':<28} "
                     f"{m['rep_vs_unrel']['d']:<9.3f} "
                     f"{m['rep_vs_unrel']['p']:<9.4f} "
                     f"{'CORRECT' if m['ordering_correct'] else 'INCORRECT'}")
            self.log(f"  {'B: Sem vs Unrel':<28} "
                     f"{m['sem_vs_unrel']['d']:<9.3f} "
                     f"{m['sem_vs_unrel']['p']:<9.4f}")

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p,
                "beta": cfg.beta, "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
            },
            metrics=metrics,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="N400 Control Conditions Experiment")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = N400ControlsExperiment()
    exp.run(quick=args.quick)
