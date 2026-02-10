"""
Morphological Agreement Experiment (Tier 2)

Tests whether the Assembly Calculus architecture can learn subject-verb
number/person agreement, detecting violations in sentences where the
subject and verb disagree in morphological form.

Scientific Questions:
1. Can assemblies trained on agreeing subject-verb pairs (e.g., "dog runs",
   "dogs run") detect agreement violations?
2. Does agreement detection work for both singular and plural subjects?
3. Do agreement patterns generalize to novel noun-verb pairs not seen
   during training?

Hypotheses:
H1: After training on agreeing pairs, the parser produces VP assemblies
    with higher internal overlap for agreeing sentences than for
    disagreeing sentences (e.g., "dog run", "dogs runs").
H2: Agreement detection works symmetrically: singular-subject violations
    and plural-subject violations are both detected (lower overlap).
H3: Agreement patterns generalize to held-out noun-verb combinations.

Protocol:
1. Build a vocabulary of singular/plural noun pairs and verb forms with
   explicit grounding features marking number (SG/PL).
2. Train the parser on ONLY agreeing sentences.
3. At test time, parse agreeing and disagreeing sentences and compare
   VP-area assembly overlap with training exemplars.
4. The VP assembly for agreeing sentences should overlap more with
   trained VP assemblies than the VP assembly for disagreeing sentences.

Statistical methodology:
- Each condition replicated across n_seeds independent random seeds.
- Agreeing vs disagreeing overlap compared via paired t-test.
- Effect sizes reported as Cohen's d.
- 95% confidence intervals on all point estimates.

References:
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
- Mitropolsky & Papadimitriou (2025), "Simulated Language Acquisition."
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from research.experiments.base import ExperimentBase, ExperimentResult, summarize, ttest_vs_null
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


@dataclass
class AgreementConfig:
    """Configuration for the morphological agreement experiment."""
    n: int = 10000           # neurons per area
    k: int = 100             # assembly size
    n_seeds: int = 5         # independent replications
    p: float = 0.05          # connection probability
    beta: float = 0.1        # Hebbian plasticity rate
    rounds: int = 10         # projection rounds


# -- Vocabulary with explicit number grounding ---------------------------------

def _build_agreement_vocab() -> Dict[str, GroundingContext]:
    """Build vocabulary with singular/plural noun pairs and verb forms.

    Number information is encoded as an additional grounding feature
    (SG or PL) so that the parser can learn number-sensitive assemblies.
    """
    vocab = {
        # Singular nouns — visual grounding + SG feature
        "dog":    GroundingContext(visual=["DOG", "ANIMAL", "SG"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL", "SG"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL", "SG"]),
        "boy":    GroundingContext(visual=["BOY", "PERSON", "SG"]),
        "girl":   GroundingContext(visual=["GIRL", "PERSON", "SG"]),

        # Plural nouns — visual grounding + PL feature
        "dogs":   GroundingContext(visual=["DOG", "ANIMAL", "PL"]),
        "cats":   GroundingContext(visual=["CAT", "ANIMAL", "PL"]),
        "birds":  GroundingContext(visual=["BIRD", "ANIMAL", "PL"]),
        "boys":   GroundingContext(visual=["BOY", "PERSON", "PL"]),
        "girls":  GroundingContext(visual=["GIRL", "PERSON", "PL"]),

        # Singular verb forms (3sg present) — motor grounding + SG
        "runs":   GroundingContext(motor=["RUNNING", "MOTION", "SG"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION", "SG"]),
        "eats":   GroundingContext(motor=["EATING", "CONSUMPTION", "SG"]),
        "chases": GroundingContext(motor=["CHASING", "PURSUIT", "SG"]),
        "sleeps": GroundingContext(motor=["SLEEPING", "REST", "SG"]),

        # Plural/bare verb forms — motor grounding + PL
        "run":    GroundingContext(motor=["RUNNING", "MOTION", "PL"]),
        "see":    GroundingContext(motor=["SEEING", "PERCEPTION", "PL"]),
        "eat":    GroundingContext(motor=["EATING", "CONSUMPTION", "PL"]),
        "chase":  GroundingContext(motor=["CHASING", "PURSUIT", "PL"]),
        "sleep":  GroundingContext(motor=["SLEEPING", "REST", "PL"]),

        # Determiners (function words — no grounding)
        "the":    GroundingContext(),
        "a":      GroundingContext(),
    }
    return vocab


def _ctx(word: str, vocab: Dict[str, GroundingContext]) -> GroundingContext:
    """Look up grounding context for a word in the agreement vocabulary."""
    return vocab[word]


def _build_agreeing_sentences(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Build training set of grammatically correct (agreeing) sentences only.

    Singular nouns pair with singular verbs; plural nouns pair with plural
    verbs. Each sentence is annotated with thematic roles.
    """
    sentences = []

    sg_pairs = [
        ("dog", "runs"), ("cat", "sleeps"), ("bird", "sees"),
        ("boy", "eats"), ("girl", "chases"),
        ("dog", "eats"), ("cat", "runs"), ("bird", "sleeps"),
        ("boy", "sees"), ("girl", "runs"),
    ]
    for noun, verb in sg_pairs:
        sentences.append(GroundedSentence(
            words=["the", noun, verb],
            contexts=[_ctx("the", vocab), _ctx(noun, vocab), _ctx(verb, vocab)],
            roles=[None, "agent", "action"],
        ))

    pl_pairs = [
        ("dogs", "run"), ("cats", "sleep"), ("birds", "see"),
        ("boys", "eat"), ("girls", "chase"),
        ("dogs", "eat"), ("cats", "run"), ("birds", "sleep"),
        ("boys", "see"), ("girls", "run"),
    ]
    for noun, verb in pl_pairs:
        sentences.append(GroundedSentence(
            words=["the", noun, verb],
            contexts=[_ctx("the", vocab), _ctx(noun, vocab), _ctx(verb, vocab)],
            roles=[None, "agent", "action"],
        ))

    return sentences


# -- Test sentence builders ----------------------------------------------------

def _build_test_pairs() -> List[Dict[str, Any]]:
    """Build matched pairs of agreeing and disagreeing test sentences.

    Each pair shares the same noun and verb root, differing only in
    number agreement. This controls for lexical content effects.
    """
    pairs = []

    sg_tests = [
        ("dog", "runs", "run"),
        ("cat", "sleeps", "sleep"),
        ("bird", "sees", "see"),
        ("boy", "eats", "eat"),
        ("girl", "chases", "chase"),
    ]
    for noun, agree_verb, disagree_verb in sg_tests:
        pairs.append({
            "label": f"SG: {noun}",
            "agree_words": ["the", noun, agree_verb],
            "disagree_words": ["the", noun, disagree_verb],
            "number": "singular",
        })

    pl_tests = [
        ("dogs", "run", "runs"),
        ("cats", "sleep", "sleeps"),
        ("birds", "see", "sees"),
        ("boys", "eat", "eats"),
        ("girls", "chase", "chases"),
    ]
    for noun, agree_verb, disagree_verb in pl_tests:
        pairs.append({
            "label": f"PL: {noun}",
            "agree_words": ["the", noun, agree_verb],
            "disagree_words": ["the", noun, disagree_verb],
            "number": "plural",
        })

    return pairs


def _build_generalization_pairs() -> List[Dict[str, Any]]:
    """Build test pairs using noun-verb combinations NOT in the training set.

    These probe whether the parser has learned abstract number agreement
    rather than memorizing specific noun-verb co-occurrences.
    """
    novel_pairs = [
        ("dog", "sleeps", "sleep", "singular"),
        ("cat", "chases", "chase", "singular"),
        ("bird", "eats", "eat", "singular"),
        ("dogs", "chase", "chases", "plural"),
        ("cats", "eat", "eats", "plural"),
        ("birds", "run", "runs", "plural"),
    ]
    pairs = []
    for noun, agree_verb, disagree_verb, number in novel_pairs:
        pairs.append({
            "label": f"NOVEL {number[:2].upper()}: {noun}+{agree_verb}",
            "agree_words": ["the", noun, agree_verb],
            "disagree_words": ["the", noun, disagree_verb],
            "number": number,
        })
    return pairs


# -- Core measurement ---------------------------------------------------------

def measure_agreement_overlap(
    parser: EmergentParser,
    words: List[str],
    training_vp_keys: List[str],
) -> float:
    """Parse a sentence and measure mean VP overlap with training exemplars.

    Projects the sentence through the parser and compares the resulting
    VP assembly against all training VP assemblies. High overlap indicates
    the sentence pattern is consistent with training; low overlap signals
    a violation.

    Args:
        parser: Trained EmergentParser instance.
        words: Sentence token list (e.g., ["the", "dog", "runs"]).
        training_vp_keys: Keys into parser.vp_assemblies to compare against.

    Returns:
        Mean overlap between the test VP assembly and training VP assemblies.
    """
    from src.assembly_calculus.ops import project, merge, _snap
    from src.assembly_calculus.assembly import overlap as asm_overlap
    from src.assembly_calculus.emergent.areas import VERB_CORE, VP

    noun_word = words[1]
    verb_word = words[2]

    # Activate noun in its core area
    noun_core = parser._word_core_area(noun_word)
    noun_phon = parser.stim_map.get(noun_word)
    verb_phon = parser.stim_map.get(verb_word)

    if noun_phon is None or verb_phon is None:
        return 0.0

    project(parser.brain, noun_phon, noun_core, rounds=parser.rounds)
    project(parser.brain, verb_phon, VERB_CORE, rounds=parser.rounds)

    test_vp = merge(parser.brain, noun_core, VERB_CORE, VP, rounds=parser.rounds)

    # Compare against training VP assemblies
    overlaps = []
    for vp_key in training_vp_keys:
        train_asm = parser.vp_assemblies.get(vp_key)
        if train_asm is not None:
            overlaps.append(asm_overlap(test_vp, train_asm))

    parser.brain._engine.reset_area_connections(VP)

    return float(np.mean(overlaps)) if overlaps else 0.0


class MorphologicalAgreementExperiment(ExperimentBase):
    """Test whether assembly calculus learns subject-verb number agreement."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="morphological_agreement",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "applications",
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        """Run the full morphological agreement experiment.

        Args:
            quick: If True, use reduced seeds (2) for fast validation.
        """
        self._start_timer()

        cfg = AgreementConfig()
        if quick:
            cfg.n_seeds = 2

        seeds = list(range(cfg.n_seeds))
        vocab = _build_agreement_vocab()
        training_sentences = _build_agreeing_sentences(vocab)
        test_pairs = _build_test_pairs()
        gen_pairs = _build_generalization_pairs()

        null_overlap = cfg.k / cfg.n
        self.log(f"Null (chance) overlap: {null_overlap:.4f}")
        self.log(f"Training sentences: {len(training_sentences)}")
        self.log(f"Test pairs: {len(test_pairs)} seen, {len(gen_pairs)} novel")

        # ================================================================
        # H1: Agreeing > disagreeing VP overlap (seen pairs)
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1: Agreement detection on seen noun-verb pairs")
        self.log("=" * 60)

        h1_agree_scores: List[List[float]] = [[] for _ in test_pairs]
        h1_disagree_scores: List[List[float]] = [[] for _ in test_pairs]

        for s in seeds:
            self.log(f"  Seed {s}/{cfg.n_seeds - 1}")
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training_sentences)

            # Collect VP keys from training for comparison
            vp_keys = list(parser.vp_assemblies.keys())

            for i, pair in enumerate(test_pairs):
                agree_ov = measure_agreement_overlap(
                    parser, pair["agree_words"], vp_keys)
                disagree_ov = measure_agreement_overlap(
                    parser, pair["disagree_words"], vp_keys)
                h1_agree_scores[i].append(agree_ov)
                h1_disagree_scores[i].append(disagree_ov)

        h1_results = []
        all_agree_means = []
        all_disagree_means = []
        for i, pair in enumerate(test_pairs):
            agree_stats = summarize(h1_agree_scores[i])
            disagree_stats = summarize(h1_disagree_scores[i])
            all_agree_means.append(agree_stats["mean"])
            all_disagree_means.append(disagree_stats["mean"])

            row = {
                "label": pair["label"], "number": pair["number"],
                "agree_overlap": agree_stats, "disagree_overlap": disagree_stats,
                "delta": agree_stats["mean"] - disagree_stats["mean"],
            }
            h1_results.append(row)
            self.log(f"  {pair['label']:20s}: agree={agree_stats['mean']:.3f}  "
                     f"disagree={disagree_stats['mean']:.3f}  "
                     f"delta={row['delta']:+.3f}")

        h1_test = ttest_vs_null(
            [a - d for a, d in zip(all_agree_means, all_disagree_means)], 0.0)
        self.log(f"  H1 paired delta: t={h1_test['t']:.2f} p={h1_test['p']:.4f} "
                 f"d={h1_test['d']:.2f} {'*' if h1_test['significant'] else ''}")

        # ================================================================
        # H2: Detection works for both singular and plural
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H2: Symmetric detection (singular vs plural violations)")
        self.log("=" * 60)

        sg_deltas = [r["delta"] for r in h1_results if r["number"] == "singular"]
        pl_deltas = [r["delta"] for r in h1_results if r["number"] == "plural"]

        h2_sg_stats = summarize(sg_deltas)
        h2_pl_stats = summarize(pl_deltas)
        h2_sg_test = ttest_vs_null(sg_deltas, 0.0)
        h2_pl_test = ttest_vs_null(pl_deltas, 0.0)

        self.log(f"  Singular delta: {h2_sg_stats['mean']:.3f} +/- {h2_sg_stats['sem']:.3f} "
                 f"t={h2_sg_test['t']:.2f} p={h2_sg_test['p']:.4f} "
                 f"{'*' if h2_sg_test['significant'] else ''}")
        self.log(f"  Plural delta:   {h2_pl_stats['mean']:.3f} +/- {h2_pl_stats['sem']:.3f} "
                 f"t={h2_pl_test['t']:.2f} p={h2_pl_test['p']:.4f} "
                 f"{'*' if h2_pl_test['significant'] else ''}")

        h2_results = {
            "singular": {"stats": h2_sg_stats, "test": h2_sg_test},
            "plural": {"stats": h2_pl_stats, "test": h2_pl_test},
        }

        # ================================================================
        # H3: Generalization to novel noun-verb pairs
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H3: Generalization to novel noun-verb combinations")
        self.log("=" * 60)

        h3_agree_scores: List[List[float]] = [[] for _ in gen_pairs]
        h3_disagree_scores: List[List[float]] = [[] for _ in gen_pairs]

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training_sentences)
            vp_keys = list(parser.vp_assemblies.keys())

            for i, pair in enumerate(gen_pairs):
                agree_ov = measure_agreement_overlap(
                    parser, pair["agree_words"], vp_keys)
                disagree_ov = measure_agreement_overlap(
                    parser, pair["disagree_words"], vp_keys)
                h3_agree_scores[i].append(agree_ov)
                h3_disagree_scores[i].append(disagree_ov)

        h3_results = []
        gen_deltas = []
        for i, pair in enumerate(gen_pairs):
            agree_stats = summarize(h3_agree_scores[i])
            disagree_stats = summarize(h3_disagree_scores[i])
            delta = agree_stats["mean"] - disagree_stats["mean"]
            gen_deltas.append(delta)

            row = {
                "label": pair["label"], "number": pair["number"],
                "agree_overlap": agree_stats, "disagree_overlap": disagree_stats,
                "delta": delta,
            }
            h3_results.append(row)
            self.log(f"  {pair['label']:30s}: agree={agree_stats['mean']:.3f}  "
                     f"disagree={disagree_stats['mean']:.3f}  "
                     f"delta={delta:+.3f}")

        h3_test = ttest_vs_null(gen_deltas, 0.0)
        self.log(f"  H3 generalization delta: t={h3_test['t']:.2f} "
                 f"p={h3_test['p']:.4f} d={h3_test['d']:.2f} "
                 f"{'*' if h3_test['significant'] else ''}")

        # ================================================================
        # Summary
        # ================================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("MORPHOLOGICAL AGREEMENT SUMMARY")
        self.log(f"  H1 (violation detection):     "
                 f"{'SUPPORTED' if h1_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={h1_test['d']:.2f})")
        self.log(f"  H2 (symmetric SG/PL):         "
                 f"SG={'YES' if h2_sg_test['significant'] else 'NO'} "
                 f"PL={'YES' if h2_pl_test['significant'] else 'NO'}")
        self.log(f"  H3 (generalization):           "
                 f"{'SUPPORTED' if h3_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={h3_test['d']:.2f})")
        self.log(f"  Duration: {duration:.1f}s ({cfg.n_seeds} seeds)")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "n_seeds": cfg.n_seeds,
                "p": cfg.p, "beta": cfg.beta, "rounds": cfg.rounds,
                "null_overlap": null_overlap,
                "n_training_sentences": len(training_sentences),
            },
            metrics={
                "h1_violation_detection": {
                    "pairs": h1_results,
                    "overall_test": h1_test,
                },
                "h2_symmetric_detection": h2_results,
                "h3_generalization": {
                    "pairs": h3_results,
                    "overall_test": h3_test,
                },
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    """Run morphological agreement experiment."""
    parser = argparse.ArgumentParser(
        description="Morphological agreement experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation run (2 seeds)")
    args = parser.parse_args()

    exp = MorphologicalAgreementExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nExperiment completed in {result.duration_seconds:.1f}s")
    print(f"H1 significant: {result.metrics['h1_violation_detection']['overall_test']['significant']}")
    print(f"H3 significant: {result.metrics['h3_generalization']['overall_test']['significant']}")


if __name__ == "__main__":
    main()
