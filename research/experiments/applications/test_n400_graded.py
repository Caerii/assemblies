"""
N400 Graded Relatedness — Does Global Energy Scale with Semantic Distance?

Tests whether global pre-k-WTA energy decreases monotonically with
increasing semantic relatedness, as predicted by the mathematical analysis:

Level 0 — Identity:       prime == target (maximum relatedness)
Level 1 — Same category:  prime and target share category feature
                          (e.g., dog/cat both ANIMAL)
Level 2 — Distant category: prime shares no features with target
                          (e.g., dog/table — ANIMAL vs FURNITURE)

Prediction: E(identity) < E(same_category) < E(distant)

This tests the core mechanism: more shared features → more stimulus-
recurrence redundancy → lower global energy → smaller N400.

Additionally, we test within-category grading by comparing primes that
share MORE features with the target vs fewer:

Level 1a — High overlap: prime and target co-occurred frequently in
           training (multiple shared sentences)
Level 1b — Low overlap:  prime and target share category but rarely
           co-occurred (fewer shared sentences)

The N400 literature shows that associative strength (co-occurrence) and
categorical relatedness both modulate the N400 independently.
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
    build_core_lexicon, measure_pre_kwta_activation,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


@dataclass
class GradedConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5


def _build_vocab():
    """Vocabulary with clear categorical structure."""
    return {
        # Animals — all share ANIMAL
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL"]),
        # Furniture — all share FURNITURE
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "lamp":   GroundingContext(visual=["LAMP", "FURNITURE"]),
        "shelf":  GroundingContext(visual=["SHELF", "FURNITURE"]),
        # Vehicles — all share VEHICLE
        "car":    GroundingContext(visual=["CAR", "VEHICLE"]),
        "truck":  GroundingContext(visual=["TRUCK", "VEHICLE"]),
        "bike":   GroundingContext(visual=["BIKE", "VEHICLE"]),
        "boat":   GroundingContext(visual=["BOAT", "VEHICLE"]),
        # Verbs
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION"]),
        # Function
        "the":    GroundingContext(),
    }


def _build_training(vocab):
    """Training with varied co-occurrence frequencies.

    Some animal pairs co-occur frequently (dog-cat: 3 sentences),
    others rarely (dog-mouse: 1 sentence). This creates graded
    associative strength within the ANIMAL category.
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    # High co-occurrence animal pairs (3 sentences each)
    for subj, verb, obj in [
        ("dog", "chases", "cat"),
        ("cat", "chases", "dog"),
        ("dog", "sees", "cat"),
        ("cat", "sees", "bird"),
        ("bird", "chases", "cat"),
        ("bird", "sees", "dog"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Low co-occurrence animal pairs (1 sentence each)
    for subj, verb, obj in [
        ("horse", "chases", "fish"),
        ("fish", "sees", "horse"),
        ("mouse", "finds", "bird"),
        ("horse", "finds", "mouse"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Furniture pairs
    for subj, verb, obj in [
        ("table", "sees", "chair"),
        ("chair", "sees", "lamp"),
        ("lamp", "finds", "shelf"),
        ("shelf", "finds", "table"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Vehicle pairs
    for subj, verb, obj in [
        ("car", "chases", "truck"),
        ("truck", "sees", "bike"),
        ("bike", "finds", "boat"),
        ("boat", "chases", "car"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Cross-category sentences (creates some co-occurrence structure)
    for subj, verb, obj in [
        ("dog", "finds", "table"),
        ("cat", "likes", "chair"),
        ("bird", "sees", "car"),
        ("horse", "chases", "truck"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    return sentences


# Graded test conditions:
# For each target, we test primes at 4 levels of relatedness
# (target, identity, high_related, low_related, unrelated)
GRADED_TESTS = [
    # target     identity  high_related  low_related  unrelated
    ("cat",      "cat",    "dog",        "horse",     "table"),
    ("dog",      "dog",    "cat",        "fish",      "chair"),
    ("bird",     "bird",   "cat",        "mouse",     "car"),
    ("horse",    "horse",  "fish",       "cat",       "lamp"),
    ("table",    "table",  "chair",      "shelf",     "dog"),
    ("chair",    "chair",  "table",      "lamp",      "bird"),
    ("car",      "car",    "truck",      "boat",      "table"),
    ("truck",    "truck",  "car",        "bike",      "chair"),
]


class N400GradedExperiment(ExperimentBase):
    """Test N400 graded relatedness."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="n400_graded",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = GradedConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)

        test_words = set()
        for t, ident, hi, lo, unrel in GRADED_TESTS:
            test_words.update([t, ident, hi, lo, unrel])

        seeds = list(range(cfg.n_seeds))

        # Per-seed accumulators for each level
        identity_seeds = []
        high_rel_seeds = []
        low_rel_seeds = []
        unrelated_seeds = []

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            lexicon = build_core_lexicon(
                parser, list(test_words), cfg.rounds)

            e_ident, e_high, e_low, e_unrel = [], [], [], []

            for target, ident, hi_rel, lo_rel, unrel in GRADED_TESTS:
                if target not in lexicon:
                    continue
                if any(w not in lexicon for w in [ident, hi_rel, lo_rel, unrel]):
                    continue

                m_ident = measure_pre_kwta_activation(
                    parser, ident, target, lexicon, cfg.rounds)
                m_high = measure_pre_kwta_activation(
                    parser, hi_rel, target, lexicon, cfg.rounds)
                m_low = measure_pre_kwta_activation(
                    parser, lo_rel, target, lexicon, cfg.rounds)
                m_unrel = measure_pre_kwta_activation(
                    parser, unrel, target, lexicon, cfg.rounds)

                e_ident.append(m_ident["global_energy"])
                e_high.append(m_high["global_energy"])
                e_low.append(m_low["global_energy"])
                e_unrel.append(m_unrel["global_energy"])

            if e_ident:
                identity_seeds.append(float(np.mean(e_ident)))
                high_rel_seeds.append(float(np.mean(e_high)))
                low_rel_seeds.append(float(np.mean(e_low)))
                unrelated_seeds.append(float(np.mean(e_unrel)))
                self.log(f"  identity={np.mean(e_ident):.1f}  "
                         f"high_rel={np.mean(e_high):.1f}  "
                         f"low_rel={np.mean(e_low):.1f}  "
                         f"unrelated={np.mean(e_unrel):.1f}")

        # Analysis
        self.log(f"\n{'='*60}")
        self.log("GRADED RELATEDNESS RESULTS")
        self.log(f"{'='*60}")
        self.log("Prediction: identity < high_related < low_related < unrelated")

        metrics = {}

        if len(identity_seeds) >= 2:
            for label, vals in [
                ("identity", identity_seeds),
                ("high_related", high_rel_seeds),
                ("low_related", low_rel_seeds),
                ("unrelated", unrelated_seeds),
            ]:
                s = summarize(vals)
                self.log(f"  {label:<15}: {s['mean']:.1f} +/- {s['sem']:.1f}")
                metrics[label] = s

            # Check monotonicity
            means = [
                summarize(identity_seeds)["mean"],
                summarize(high_rel_seeds)["mean"],
                summarize(low_rel_seeds)["mean"],
                summarize(unrelated_seeds)["mean"],
            ]
            monotonic = all(means[i] <= means[i+1] for i in range(3))
            self.log(f"\n  Monotonic ordering: "
                     f"{'YES' if monotonic else 'NO'}")
            self.log(f"  Means: {' < '.join(f'{m:.1f}' for m in means)}")
            metrics["monotonic"] = monotonic

            # Pairwise comparisons
            self.log(f"\n  Pairwise comparisons:")
            comparisons = {}
            for label, a, b in [
                ("identity_vs_unrelated", identity_seeds, unrelated_seeds),
                ("high_vs_unrelated", high_rel_seeds, unrelated_seeds),
                ("low_vs_unrelated", low_rel_seeds, unrelated_seeds),
                ("identity_vs_high", identity_seeds, high_rel_seeds),
                ("high_vs_low", high_rel_seeds, low_rel_seeds),
            ]:
                stats = paired_ttest(a, b)
                direction = "CORRECT" if summarize(a)["mean"] < summarize(b)["mean"] else "REVERSED"
                self.log(f"    {label:<25}: d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}  {direction}")
                comparisons[label] = {
                    "test": stats, "direction": direction,
                }
            metrics["comparisons"] = comparisons

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
        description="N400 Graded Relatedness Experiment")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = N400GradedExperiment()
    exp.run(quick=args.quick)
