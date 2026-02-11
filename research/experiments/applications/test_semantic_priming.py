"""
Semantic Priming Experiment

Tests whether assembly calculus exhibits semantic priming: faster/stronger
assembly activation for semantically related words compared to unrelated
words, after exposure to a prime context.

This is one of the most robust findings in psycholinguistics:
- Meyer & Schvaneveldt (1971): Faster lexical decisions for related words.
- N400 ERP component: Reduced amplitude for expected/related continuations
  (Kutas & Hillyard, 1980; Kutas & Federmeier, 2011).
- Interpreted as pre-activation of semantically related representations
  via associative spreading activation.

In the Assembly Calculus, semantic priming should emerge from Hebbian
plasticity: words that co-occur in training strengthen their inter-area
connections, so activating one partially pre-activates the other.

We measure priming as the difference in assembly overlap between:
- Related condition: prime and target share grounding features or
  co-occurred in training sentences.
- Unrelated condition: prime and target have no feature overlap and
  never co-occurred.

If assemblies show priming, the architecture naturally produces the
computational substrate for the N400 effect.

Hypotheses:

H1: Sentential priming — After processing "The dog chased the ...",
    the NOUN_CORE activation overlaps more with "cat" (trained patient
    in similar sentences) than with an unrelated noun ("table").

H2: Feature-based priming — Words sharing grounding features (ANIMAL)
    show higher cross-activation than words with different features
    (ANIMAL vs FURNITURE), even without direct co-occurrence.

H3: Cumulative priming — Processing multiple related words in sequence
    produces stronger facilitation than a single prime.

H4: Priming decays — The facilitation effect decreases as intervening
    words are inserted between prime and target.

Statistical methodology:
- N_SEEDS independent random seeds per condition.
- Paired t-test: related vs unrelated overlap.
- Cohen's d effect sizes. Mean +/- SEM.

References:
- Meyer & Schvaneveldt (1971). Facilitation in recognizing pairs of words.
- Kutas & Hillyard (1980). Reading senseless sentences: Brain potentials.
- Kutas & Federmeier (2011). Thirty years and counting: N400.
- Mitropolsky & Papadimitriou (2025). Simulated Language Acquisition.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, ttest_vs_null, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.ops import project, _snap
from src.assembly_calculus.assembly import overlap as asm_overlap


@dataclass
class PrimingConfig:
    """Configuration for semantic priming experiment."""
    n: int = 10000
    k: int = 100
    n_seeds: int = 5
    p: float = 0.05
    beta: float = 0.1
    rounds: int = 10


# -- Vocabulary ----------------------------------------------------------------

def _build_priming_vocab() -> Dict[str, GroundingContext]:
    """Vocabulary with clear semantic clusters for priming tests.

    Animals (ANIMAL cluster), Objects (OBJECT cluster),
    Food (FOOD cluster) — priming should be strongest within-cluster.
    """
    return {
        # Animal cluster
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        # Object cluster
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        "book":   GroundingContext(visual=["BOOK", "OBJECT"]),
        "ball":   GroundingContext(visual=["BALL", "OBJECT"]),
        "car":    GroundingContext(visual=["CAR", "VEHICLE"]),
        # Food cluster
        "bread":  GroundingContext(visual=["BREAD", "FOOD"]),
        "cake":   GroundingContext(visual=["CAKE", "FOOD"]),
        # Verbs
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "eats":   GroundingContext(motor=["EATING", "CONSUMPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION"]),
        # Function words
        "the":    GroundingContext(),
        "a":      GroundingContext(),
    }


# -- Training sentences --------------------------------------------------------

def _build_priming_training(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Training corpus with animal-verb co-occurrences.

    Animals appear as agents/patients; objects rarely appear with
    animal verbs. This creates sentential co-occurrence bias.
    """
    def ctx(w):
        return vocab[w]

    sentences = []
    animal_triples = [
        ("dog", "chases", "cat"), ("cat", "sees", "bird"),
        ("bird", "finds", "fish"), ("horse", "chases", "dog"),
        ("fish", "sees", "horse"), ("dog", "finds", "bird"),
        ("cat", "chases", "fish"), ("horse", "sees", "cat"),
        ("bird", "chases", "horse"), ("fish", "finds", "cat"),
        ("dog", "sees", "horse"), ("cat", "finds", "dog"),
    ]
    for subj, verb, obj in animal_triples:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb), ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # A few object sentences to avoid total lack of coverage
    object_triples = [
        ("dog", "finds", "ball"), ("cat", "sees", "book"),
        ("bird", "finds", "car"),
    ]
    for subj, verb, obj in object_triples:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb), ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    return sentences


# -- Measurement ---------------------------------------------------------------

def measure_target_overlap(
    parser: EmergentParser,
    prime_words: List[str],
    target_word: str,
) -> float:
    """Activate prime context, then measure target activation.

    1. Project prime words sequentially into their core areas with
       recurrence (building context).
    2. Project target word into NOUN_CORE.
    3. Measure overlap between target assembly and the NOUN_CORE
       lexicon entry for that word.

    Higher overlap = stronger activation = more priming.
    """
    from src.assembly_calculus.emergent.areas import NOUN_CORE

    # Phase 1: Activate prime context
    for w in prime_words:
        phon = parser.stim_map.get(w)
        if phon is None:
            continue
        core = parser._word_core_area(w)
        project(parser.brain, phon, core, rounds=parser.rounds)

    # Phase 2: Project target
    target_phon = parser.stim_map.get(target_word)
    if target_phon is None:
        return 0.0

    target_core = parser._word_core_area(target_word)
    target_asm = project(parser.brain, target_phon, target_core,
                         rounds=parser.rounds)

    # Phase 3: Measure overlap with lexicon entry
    lex_asm = parser.core_lexicons.get(target_core, {}).get(target_word)
    if lex_asm is None:
        return 0.0

    return float(asm_overlap(target_asm, lex_asm))


# -- Experiment ----------------------------------------------------------------

class SemanticPrimingExperiment(ExperimentBase):
    """Test semantic priming in assembly calculus."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="semantic_priming",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = PrimingConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_priming_vocab()
        training = _build_priming_training(vocab)
        seeds = list(range(cfg.n_seeds))

        self.log(f"Training sentences: {len(training)}")
        self.log(f"Seeds: {cfg.n_seeds}")

        # ================================================================
        # H1: Sentential priming — co-occurrence context
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1: Sentential priming (co-occurrence)")
        self.log("=" * 60)

        # Prime: "the dog chases the ..."
        # Related target: "cat" (frequently co-occurs with dog)
        # Unrelated target: "table" (never co-occurs with dog in training)
        h1_related = []
        h1_unrelated = []

        prime_contexts = [
            (["the", "dog", "chases", "the"], "cat", "table"),
            (["the", "cat", "sees", "the"], "bird", "chair"),
            (["the", "bird", "finds", "the"], "fish", "book"),
        ]

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_related = []
            seed_unrelated = []
            for prime, related, unrelated in prime_contexts:
                r_ov = measure_target_overlap(parser, prime, related)
                u_ov = measure_target_overlap(parser, prime, unrelated)
                seed_related.append(r_ov)
                seed_unrelated.append(u_ov)

            h1_related.append(float(np.mean(seed_related)))
            h1_unrelated.append(float(np.mean(seed_unrelated)))

        h1_r_stats = summarize(h1_related)
        h1_u_stats = summarize(h1_unrelated)
        h1_test = paired_ttest(h1_related, h1_unrelated)

        self.log(f"  Related:   {h1_r_stats['mean']:.3f} +/- {h1_r_stats['sem']:.3f}")
        self.log(f"  Unrelated: {h1_u_stats['mean']:.3f} +/- {h1_u_stats['sem']:.3f}")
        self.log(f"  Delta:     {h1_r_stats['mean'] - h1_u_stats['mean']:+.3f}")
        self.log(f"  Test: t={h1_test['t']:.2f} p={h1_test['p']:.4f} "
                 f"d={h1_test['d']:.2f} {'*' if h1_test['significant'] else ''}")

        # ================================================================
        # H2: Feature-based priming — shared grounding
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H2: Feature-based priming (shared grounding features)")
        self.log("=" * 60)

        # No sentential context — just project one word and test activation
        # of a same-category vs different-category word
        h2_same = []
        h2_diff = []

        feature_tests = [
            ("dog", "horse", "table"),    # ANIMAL, ANIMAL, FURNITURE
            ("cat", "fish", "book"),      # ANIMAL, ANIMAL, OBJECT
            ("table", "chair", "cat"),    # FURNITURE, FURNITURE, ANIMAL
        ]

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_same = []
            seed_diff = []
            for prime, same_cat, diff_cat in feature_tests:
                s_ov = measure_target_overlap(parser, [prime], same_cat)
                d_ov = measure_target_overlap(parser, [prime], diff_cat)
                seed_same.append(s_ov)
                seed_diff.append(d_ov)

            h2_same.append(float(np.mean(seed_same)))
            h2_diff.append(float(np.mean(seed_diff)))

        h2_s_stats = summarize(h2_same)
        h2_d_stats = summarize(h2_diff)
        h2_test = paired_ttest(h2_same, h2_diff)

        self.log(f"  Same-category:  {h2_s_stats['mean']:.3f} +/- {h2_s_stats['sem']:.3f}")
        self.log(f"  Diff-category:  {h2_d_stats['mean']:.3f} +/- {h2_d_stats['sem']:.3f}")
        self.log(f"  Delta:          {h2_s_stats['mean'] - h2_d_stats['mean']:+.3f}")
        self.log(f"  Test: t={h2_test['t']:.2f} p={h2_test['p']:.4f} "
                 f"d={h2_test['d']:.2f} {'*' if h2_test['significant'] else ''}")

        # ================================================================
        # H3: Cumulative priming
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H3: Cumulative priming (1 vs 2 vs 3 primes)")
        self.log("=" * 60)

        h3_results = {}
        for n_primes in [1, 2, 3]:
            prime_seqs = {
                1: [["dog"]],
                2: [["dog", "cat"]],
                3: [["dog", "cat", "bird"]],
            }
            target = "horse"  # Same category, not in prime set

            accs = []
            for s in seeds:
                parser = EmergentParser(
                    n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                    seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
                )
                parser.train(sentences=training)

                seed_vals = []
                for seq in prime_seqs[n_primes]:
                    ov = measure_target_overlap(parser, seq, target)
                    seed_vals.append(ov)
                accs.append(float(np.mean(seed_vals)))

            stats = summarize(accs)
            h3_results[n_primes] = {"stats": stats, "values": accs}
            self.log(f"  {n_primes} prime(s): {stats['mean']:.3f} +/- {stats['sem']:.3f}")

        # Test: 3 primes > 1 prime
        if len(h3_results[3]["values"]) > 1 and len(h3_results[1]["values"]) > 1:
            h3_test = paired_ttest(h3_results[3]["values"], h3_results[1]["values"])
            self.log(f"  3 vs 1 test: t={h3_test['t']:.2f} p={h3_test['p']:.4f} "
                     f"d={h3_test['d']:.2f} {'*' if h3_test['significant'] else ''}")
        else:
            h3_test = {"t": 0.0, "p": 1.0, "d": 0.0, "significant": False}

        # ================================================================
        # H4: Priming decay with intervening words
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H4: Priming decay (intervening words)")
        self.log("=" * 60)

        target = "cat"
        # 0 intervening: ["dog"] -> cat
        # 1 intervening: ["dog", "sees"] -> cat
        # 3 intervening: ["dog", "sees", "the", "ball"] -> cat
        gap_conditions = {
            0: ["dog"],
            1: ["dog", "sees"],
            3: ["dog", "sees", "the", "ball"],
        }

        h4_results = {}
        for gap, context in gap_conditions.items():
            accs = []
            for s in seeds:
                parser = EmergentParser(
                    n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                    seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
                )
                parser.train(sentences=training)
                ov = measure_target_overlap(parser, context, target)
                accs.append(ov)

            stats = summarize(accs)
            h4_results[gap] = {"stats": stats, "values": accs}
            self.log(f"  Gap {gap}: {stats['mean']:.3f} +/- {stats['sem']:.3f}")

        # Test: gap 0 > gap 3
        if len(h4_results[0]["values"]) > 1 and len(h4_results[3]["values"]) > 1:
            h4_test = paired_ttest(h4_results[0]["values"], h4_results[3]["values"])
            self.log(f"  Gap 0 vs 3: t={h4_test['t']:.2f} p={h4_test['p']:.4f} "
                     f"d={h4_test['d']:.2f} {'*' if h4_test['significant'] else ''}")
        else:
            h4_test = {"t": 0.0, "p": 1.0, "d": 0.0, "significant": False}

        # ================================================================
        # Summary
        # ================================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("SEMANTIC PRIMING SUMMARY")
        self.log(f"  H1 (sentential priming):  "
                 f"{'SUPPORTED' if h1_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={h1_test['d']:.2f})")
        self.log(f"  H2 (feature priming):     "
                 f"{'SUPPORTED' if h2_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={h2_test['d']:.2f})")
        self.log(f"  H3 (cumulative):          "
                 f"{'SUPPORTED' if h3_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={h3_test['d']:.2f})")
        self.log(f"  H4 (decay):               "
                 f"{'SUPPORTED' if h4_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={h4_test['d']:.2f})")
        self.log(f"  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "n_training": len(training),
            },
            metrics={
                "h1_sentential": {
                    "related": h1_r_stats, "unrelated": h1_u_stats,
                    "test": h1_test,
                },
                "h2_feature": {
                    "same_category": h2_s_stats,
                    "diff_category": h2_d_stats,
                    "test": h2_test,
                },
                "h3_cumulative": {
                    str(k): v["stats"] for k, v in h3_results.items()
                } | {"test": h3_test},
                "h4_decay": {
                    str(k): v["stats"] for k, v in h4_results.items()
                } | {"test": h4_test},
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Semantic priming experiment")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    exp = SemanticPrimingExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
