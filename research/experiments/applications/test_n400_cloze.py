"""
N400 Cloze Probability — The Canonical N400 Manipulation

Tests whether global pre-k-WTA energy correlates with cloze probability
(the probability that a word completes a sentence frame).

Kutas & Hillyard (1984) demonstrated that N400 amplitude is inversely
proportional to cloze probability: highly predictable sentence completions
produce small N400s, while unlikely completions produce large N400s.

Design:
- Train parser on sentences establishing predictive contexts
- Present sentence frames word-by-word through the parser
- At the final word position, measure global energy with record_activation
- Compare across cloze conditions:
  - High cloze:   "The dog chases the cat"   (cat is highly predictable)
  - Medium cloze: "The dog chases the bird"  (bird is possible but less common)
  - Low cloze:    "The dog chases the table" (table is anomalous)

Prediction: E(high_cloze) < E(medium_cloze) < E(low_cloze)

This goes beyond word-pair priming to test sentence-level context effects,
which is what the N400 is primarily known for in the literature.
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
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.ops import project


@dataclass
class ClozeConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5


def _build_vocab():
    """Vocabulary with clear predictive contexts."""
    return {
        # Animals
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL"]),
        # Objects
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "book":   GroundingContext(visual=["BOOK", "OBJECT"]),
        "ball":   GroundingContext(visual=["BALL", "TOY"]),
        # Verbs
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION"]),
        # Function
        "the":    GroundingContext(),
    }


def _build_training(vocab):
    """Training with clear co-occurrence patterns.

    "dog chases cat" appears 3 times — cat is HIGH cloze after "dog chases".
    "dog chases bird" appears 1 time — bird is MEDIUM cloze.
    "dog chases table" never appears — table is LOW cloze (anomalous).
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    # High-frequency patterns (3x repetition)
    for _ in range(3):
        for subj, verb, obj in [
            ("dog", "chases", "cat"),
            ("cat", "chases", "bird"),
            ("bird", "sees", "fish"),
            ("horse", "chases", "dog"),
        ]:
            sentences.append(GroundedSentence(
                words=["the", subj, verb, "the", obj],
                contexts=[ctx("the"), ctx(subj), ctx(verb),
                          ctx("the"), ctx(obj)],
                roles=[None, "agent", "action", None, "patient"],
            ))

    # Medium-frequency patterns (1x)
    for subj, verb, obj in [
        ("dog", "chases", "bird"),
        ("cat", "chases", "fish"),
        ("bird", "sees", "mouse"),
        ("horse", "chases", "cat"),
        ("dog", "sees", "horse"),
        ("cat", "sees", "dog"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Cross-category for variety
    for subj, verb, obj in [
        ("dog", "finds", "ball"),
        ("cat", "likes", "book"),
        ("bird", "finds", "table"),
        ("horse", "sees", "chair"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb),
                      ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    return sentences


# Cloze test frames:
# (context_words, high_cloze_target, medium_cloze_target, low_cloze_target)
# Context = words preceding the target
CLOZE_TESTS = [
    # "The dog chases the ___" → cat(high), bird(medium), table(low)
    (["the", "dog", "chases", "the"], "cat", "bird", "table"),
    # "The cat chases the ___" → bird(high), fish(medium), chair(low)
    (["the", "cat", "chases", "the"], "bird", "fish", "chair"),
    # "The bird sees the ___" → fish(high), mouse(medium), ball(low)
    (["the", "bird", "sees", "the"], "fish", "mouse", "ball"),
    # "The horse chases the ___" → dog(high), cat(medium), book(low)
    (["the", "horse", "chases", "the"], "dog", "cat", "book"),
]


def _process_context(parser, context_words, rounds):
    """Process context words through the parser incrementally.

    This establishes the sentence context by projecting each word
    through the parser's incremental processing pipeline.
    """
    for word in context_words:
        if word in parser.stim_map:
            core = parser._word_core_area(word)
            project(parser.brain, parser.stim_map[word], core, rounds=rounds)


def _measure_target_energy(parser, target, rounds):
    """Measure global pre-k-WTA energy for a target word.

    Assumes context has already been processed. Projects the target
    with record_activation=True and returns the global energy.
    """
    engine = parser.brain._engine
    core = parser._word_core_area(target)
    phon = parser.stim_map.get(target)
    if phon is None:
        return float("nan")

    # Project target with activation recording
    # Include self-recurrence to capture context effects
    result = engine.project_into(
        core,
        from_stimuli=[phon],
        from_areas=[core],
        plasticity_enabled=False,
        record_activation=True,
    )

    return result.pre_kwta_total


class N400ClozeExperiment(ExperimentBase):
    """Test N400 cloze probability effect."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="n400_cloze",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = ClozeConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)
        seeds = list(range(cfg.n_seeds))

        high_seeds = []
        medium_seeds = []
        low_seeds = []

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            e_high, e_med, e_low = [], [], []

            for context, hi, med, lo in CLOZE_TESTS:
                # Process context for high-cloze target
                from src.assembly_calculus.emergent.areas import CORE_AREAS
                parser.brain.inhibit_areas(CORE_AREAS)
                _process_context(parser, context, cfg.rounds)
                energy_hi = _measure_target_energy(parser, hi, cfg.rounds)

                # Process context for medium-cloze target
                parser.brain.inhibit_areas(CORE_AREAS)
                _process_context(parser, context, cfg.rounds)
                energy_med = _measure_target_energy(parser, med, cfg.rounds)

                # Process context for low-cloze target
                parser.brain.inhibit_areas(CORE_AREAS)
                _process_context(parser, context, cfg.rounds)
                energy_lo = _measure_target_energy(parser, lo, cfg.rounds)

                if not any(np.isnan(x) for x in [energy_hi, energy_med, energy_lo]):
                    e_high.append(energy_hi)
                    e_med.append(energy_med)
                    e_low.append(energy_lo)

                self.log(f"  '{' '.join(context)} ___': "
                         f"hi({hi})={energy_hi:.1f}  "
                         f"med({med})={energy_med:.1f}  "
                         f"lo({lo})={energy_lo:.1f}")

            if e_high:
                high_seeds.append(float(np.mean(e_high)))
                medium_seeds.append(float(np.mean(e_med)))
                low_seeds.append(float(np.mean(e_low)))

        # Analysis
        self.log(f"\n{'='*60}")
        self.log("CLOZE PROBABILITY RESULTS")
        self.log(f"{'='*60}")
        self.log("Prediction: E(high_cloze) < E(medium_cloze) < E(low_cloze)")

        metrics = {}

        if len(high_seeds) >= 2:
            hi_s = summarize(high_seeds)
            med_s = summarize(medium_seeds)
            lo_s = summarize(low_seeds)

            self.log(f"  High cloze:   {hi_s['mean']:.1f} +/- {hi_s['sem']:.1f}")
            self.log(f"  Medium cloze: {med_s['mean']:.1f} +/- {med_s['sem']:.1f}")
            self.log(f"  Low cloze:    {lo_s['mean']:.1f} +/- {lo_s['sem']:.1f}")

            metrics["high_cloze"] = hi_s
            metrics["medium_cloze"] = med_s
            metrics["low_cloze"] = lo_s

            # Check monotonicity
            means = [hi_s["mean"], med_s["mean"], lo_s["mean"]]
            monotonic = means[0] <= means[1] <= means[2]
            self.log(f"\n  Monotonic ordering: "
                     f"{'YES' if monotonic else 'NO'}")
            self.log(f"  Means: {means[0]:.1f} {'<' if means[0] <= means[1] else '>'} "
                     f"{means[1]:.1f} {'<' if means[1] <= means[2] else '>'} "
                     f"{means[2]:.1f}")
            metrics["monotonic"] = monotonic

            # Pairwise comparisons
            comparisons = {}
            for label, a, b in [
                ("high_vs_low", high_seeds, low_seeds),
                ("high_vs_medium", high_seeds, medium_seeds),
                ("medium_vs_low", medium_seeds, low_seeds),
            ]:
                stats = paired_ttest(a, b)
                direction = "CORRECT" if summarize(a)["mean"] < summarize(b)["mean"] else "REVERSED"
                self.log(f"  {label:<20}: d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}  {direction}")
                comparisons[label] = {"test": stats, "direction": direction}
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
        description="N400 Cloze Probability Experiment")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = N400ClozeExperiment()
    exp.run(quick=args.quick)
