"""
Grammaticality Judgment from Prediction Error

Tests whether accumulated prediction error (N400) can distinguish grammatical
from ungrammatical word sequences without explicit grammar knowledge.

Setup:
  - Train brain on standard SVO and SVO+PP sentences
  - Test with three types of novel sequences:
    - Grammatical novel: correct structure, unseen word combinations
    - Word-order violation: scrambled structure (e.g., "chases dog cat")
    - Category violation: wrong category in position (e.g., "dog dog cat")
  - Measure cumulative N400 across all positions in each sequence

Hypotheses:
  H1: Grammatical < scrambled in cumulative N400 (d > 1.0)
  H2: Category violations have intermediate N400
  H3: Rank-ordering by cumulative N400 correlates with grammaticality

Usage:
    uv run python research/experiments/primitives/test_grammaticality_judgment.py
    uv run python research/experiments/primitives/test_grammaticality_judgment.py --quick
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    paired_ttest,
    measure_overlap,
)
from research.experiments.lib.vocabulary import DEFAULT_VOCAB
from research.experiments.lib.grammar import SimpleCFG
from research.experiments.lib.brain_setup import (
    BrainConfig,
    create_language_brain,
    build_lexicon,
    activate_word,
)
from research.experiments.lib.training import train_sentence
from research.experiments.lib.measurement import measure_n400


@dataclass
class GrammaticalityConfig:
    # Brain
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.10
    w_max: float = 20.0
    lexicon_rounds: int = 20
    # Training
    n_train_sentences: int = 200
    prediction_rounds: int = 5
    binding_rounds: int = 10
    # Test
    n_test_items: int = 10
    lexicon_readout_rounds: int = 5


def measure_cumulative_n400(
    brain, words: List[str], vocab, lexicon: Dict[str, np.ndarray],
) -> float:
    """Measure cumulative N400 across all positions in a word sequence.

    For each position i > 0, activate words[0..i-1] as context, then
    forward-project into PREDICTION and measure N400 for words[i].
    Sum the N400 values across all positions.
    """
    total_n400 = 0.0
    n_measured = 0

    for i in range(1, len(words)):
        # Activate context words
        context_area = None
        for j in range(i):
            area = vocab.core_area_for(words[j])
            activate_word(brain, words[j], area, 3)
            context_area = area

        # Forward predict
        brain.inhibit_areas(["PREDICTION"])
        brain.project({}, {context_area: ["PREDICTION"]})
        predicted = np.array(
            brain.areas["PREDICTION"].winners, dtype=np.uint32)

        # Measure N400 for actual word at position i
        target = words[i]
        if target in lexicon:
            total_n400 += measure_n400(predicted, lexicon[target])
            n_measured += 1

    return total_n400 / n_measured if n_measured > 0 else 1.0


def generate_test_stimuli(
    vocab, rng: np.random.Generator, n_items: int,
) -> Dict[str, List[List[str]]]:
    """Generate grammatical, scrambled, and category-violation sequences."""
    nouns = vocab.words_for_category("NOUN")
    verbs = vocab.words_for_category("VERB")

    grammatical = []
    scrambled = []
    cat_violation = []

    for i in range(n_items):
        # Grammatical: SVO with novel combination
        agent = nouns[i % len(nouns)]
        verb = verbs[i % len(verbs)]
        patient = nouns[(i + 2) % len(nouns)]
        if patient == agent:
            patient = nouns[(i + 3) % len(nouns)]

        gram = [agent, verb, patient]
        grammatical.append(gram)

        # Scrambled: V N N (verb first)
        scram = [verb, agent, patient]
        scrambled.append(scram)

        # Category violation: N N N (verb replaced by noun)
        bad_noun = nouns[(i + 1) % len(nouns)]
        if bad_noun == agent:
            bad_noun = nouns[(i + 3) % len(nouns)]
        catv = [agent, bad_noun, patient]
        cat_violation.append(catv)

    return {
        "grammatical": grammatical,
        "scrambled": scrambled,
        "cat_violation": cat_violation,
    }


def run_trial(cfg: GrammaticalityConfig, seed: int) -> Dict[str, Any]:
    """Run one trial of grammaticality judgment."""
    vocab = DEFAULT_VOCAB
    rng = np.random.default_rng(seed)

    # Create and train brain
    bcfg = BrainConfig(
        n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
        w_max=cfg.w_max, lexicon_rounds=cfg.lexicon_rounds)
    brain = create_language_brain(bcfg, vocab, seed)

    grammar = SimpleCFG(pp_prob=0.4, vocab=vocab, rng=rng)
    train_sents = grammar.generate_batch(cfg.n_train_sentences)

    for sent in train_sents:
        train_sentence(brain, sent, vocab,
                       cfg.prediction_rounds, cfg.binding_rounds)

    # Build lexicon
    brain.disable_plasticity = True
    lexicon = build_lexicon(brain, vocab, cfg.lexicon_readout_rounds)

    # Generate test stimuli
    stimuli = generate_test_stimuli(vocab, rng, cfg.n_test_items)

    # Measure cumulative N400 for each condition
    gram_n400s = []
    scram_n400s = []
    catv_n400s = []

    for words in stimuli["grammatical"]:
        gram_n400s.append(
            measure_cumulative_n400(brain, words, vocab, lexicon))

    for words in stimuli["scrambled"]:
        scram_n400s.append(
            measure_cumulative_n400(brain, words, vocab, lexicon))

    for words in stimuli["cat_violation"]:
        catv_n400s.append(
            measure_cumulative_n400(brain, words, vocab, lexicon))

    brain.disable_plasticity = False

    return {
        "gram_n400": gram_n400s,
        "scram_n400": scram_n400s,
        "catv_n400": catv_n400s,
    }


class GrammaticalityJudgmentExperiment(ExperimentBase):
    """Grammaticality judgment from prediction error."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="grammaticality_judgment",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "primitives"),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = 10,
        config: Optional[GrammaticalityConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        self._start_timer()

        cfg = config or GrammaticalityConfig(
            **{k: v for k, v in kwargs.items()
               if k in GrammaticalityConfig.__dataclass_fields__})

        self.log("=" * 70)
        self.log("Grammaticality Judgment from Prediction Error")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  n_train={cfg.n_train_sentences}, "
                 f"n_test_items={cfg.n_test_items}")
        self.log(f"  n_seeds={n_seeds}")
        self.log("=" * 70)

        # Accumulators
        gram_means = []
        scram_means = []
        catv_means = []

        for s in range(n_seeds):
            self.log(f"  Seed {s + 1}/{n_seeds} ...")
            trial = run_trial(cfg, self.seed + s)

            gram_means.append(float(np.mean(trial["gram_n400"])))
            scram_means.append(float(np.mean(trial["scram_n400"])))
            catv_means.append(float(np.mean(trial["catv_n400"])))

        # Effect sizes
        gram_vs_scram = paired_ttest(scram_means, gram_means)
        gram_vs_catv = paired_ttest(catv_means, gram_means)

        gram_summary = summarize(gram_means)
        scram_summary = summarize(scram_means)
        catv_summary = summarize(catv_means)

        # H1: grammatical < scrambled (d > 1.0)
        h1 = gram_vs_scram["d"] > 1.0

        # H2: category violations intermediate
        h2 = (catv_summary["mean"] > gram_summary["mean"] and
               catv_summary["mean"] < scram_summary["mean"])

        # H3: rank ordering (gram < catv < scram)
        h3 = (gram_summary["mean"] < catv_summary["mean"] <
               scram_summary["mean"])

        self.log(f"\n  === Cumulative N400 ===")
        self.log(f"    Grammatical:      {gram_summary['mean']:.3f} "
                 f"+/- {gram_summary.get('sem', 0):.3f}")
        self.log(f"    Cat violation:    {catv_summary['mean']:.3f} "
                 f"+/- {catv_summary.get('sem', 0):.3f}")
        self.log(f"    Scrambled:        {scram_summary['mean']:.3f} "
                 f"+/- {scram_summary.get('sem', 0):.3f}")
        self.log(f"    Gram vs scram d:  {gram_vs_scram['d']:.2f}")
        self.log(f"    Gram vs catv d:   {gram_vs_catv['d']:.2f}")

        self.log(f"\n  === Hypotheses ===")
        self.log(f"    H1 (Gram < Scram, d > 1.0):       "
                 f"{'PASS' if h1 else 'FAIL'} (d={gram_vs_scram['d']:.2f})")
        self.log(f"    H2 (CatViol intermediate):         "
                 f"{'PASS' if h2 else 'FAIL'}")
        self.log(f"    H3 (Gram < CatViol < Scram):       "
                 f"{'PASS' if h3 else 'FAIL'}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        metrics = {
            "grammatical_n400": gram_summary,
            "scrambled_n400": scram_summary,
            "cat_violation_n400": catv_summary,
            "gram_vs_scram_d": gram_vs_scram["d"],
            "gram_vs_catv_d": gram_vs_catv["d"],
            "hypotheses": {
                "H1_gram_less_than_scram": h1,
                "H2_catv_intermediate": h2,
                "H3_rank_ordering": h3,
            },
        }

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "n_train_sentences": cfg.n_train_sentences,
                "n_test_items": cfg.n_test_items,
                "n_seeds": n_seeds,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Grammaticality Judgment Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    exp = GrammaticalityJudgmentExperiment(verbose=True)

    if args.quick:
        cfg = GrammaticalityConfig(
            n=5000, k=50,
            n_train_sentences=100,
            n_test_items=5)
        n_seeds = args.seeds or 3
    else:
        cfg = GrammaticalityConfig()
        n_seeds = args.seeds or 10

    result = exp.run(n_seeds=n_seeds, config=cfg)
    suffix = "_quick" if args.quick else ""
    exp.save_result(result, suffix)

    h = result.metrics["hypotheses"]
    print("\n" + "=" * 70)
    print("GRAMMATICALITY JUDGMENT SUMMARY")
    print("=" * 70)
    print(f"\nH1 Gram < Scram (d > 1.0): {'PASS' if h['H1_gram_less_than_scram'] else 'FAIL'}")
    print(f"H2 CatViol intermediate:   {'PASS' if h['H2_catv_intermediate'] else 'FAIL'}")
    print(f"H3 Full rank ordering:     {'PASS' if h['H3_rank_ordering'] else 'FAIL'}")
    print(f"\nDuration: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
