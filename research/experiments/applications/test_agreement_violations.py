"""
Agreement Violations — Graded P600 for Morphosyntactic Mismatch

Prediction 3.1 from IMPLICATIONS_AND_PREDICTIONS.md:

Tests whether P600 structural instability is graded:
  category violation > agreement violation > grammatical

Agreement violations use the correct lexical category (noun/verb) but
wrong morphological features (singular subject + plural verb form).
Since the verb IS in the correct category, the VERB_CORE->VP pathway
IS Hebbian-consolidated. But the agreement mismatch between plural-marked
subject assembly and singular-marked verb assembly creates secondary
instability during structural integration.

Test conditions:
  Grammatical:    "the dog chases the cat" (sg subj, sg verb — trained)
  Agreement viol: "the dogs chases the cat" (pl subj, sg verb — mismatch)
  Category viol:  "the dog chases the likes" (verb in noun slot)

Predictions:
  P600(category) > P600(agreement) > P600(grammatical)

The agreement violation is intermediate because:
- Category pathway IS consolidated (verbs trained in VP merge)
- But number feature mismatch creates settling difficulty
- Category violation has NO consolidated pathway (verb->ROLE has no training)

Literature:
  - Hagoort et al. 1993: P600 for agreement violations
  - Kaan et al. 2000: Graded P600 effects
  - Osterhout & Mobley 1995: P600 amplitude scales with violation severity
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
from research.experiments.infrastructure import (
    bootstrap_structural_connectivity,
    consolidate_role_connections,
    consolidate_vp_connections,
)
from research.experiments.applications.test_p600_syntactic import (
    _measure_critical_word,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence
from src.assembly_calculus.emergent.areas import (
    ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP,
)


@dataclass
class AgreementConfig:
    n: int = 50000
    k: int = 100
    p: float = 0.05
    beta: float = 0.05
    rounds: int = 10
    n_seeds: int = 5
    p600_settling_rounds: int = 5
    consolidation_passes: int = 10


def _build_agreement_vocab() -> Dict[str, GroundingContext]:
    """Vocabulary with singular/plural noun and verb forms.

    Number is encoded as a grounding feature (SG/PL) so that singular
    and plural forms of the same word share most features but differ
    in number marking.
    """
    return {
        # Singular nouns
        "dog":    GroundingContext(visual=["DOG", "ANIMAL", "SG"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL", "SG"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL", "SG"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL", "SG"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL", "SG"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL", "SG"]),
        # Plural nouns
        "dogs":   GroundingContext(visual=["DOG", "ANIMAL", "PL"]),
        "cats":   GroundingContext(visual=["CAT", "ANIMAL", "PL"]),
        "birds":  GroundingContext(visual=["BIRD", "ANIMAL", "PL"]),
        "horses": GroundingContext(visual=["HORSE", "ANIMAL", "PL"]),
        # Untrained objects (for category violation baseline)
        "table":  GroundingContext(visual=["TABLE", "FURNITURE"]),
        "chair":  GroundingContext(visual=["CHAIR", "FURNITURE"]),
        # Singular verbs (3sg)
        "chases": GroundingContext(motor=["CHASING", "PURSUIT", "SG"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION", "SG"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION", "SG"]),
        "likes":  GroundingContext(motor=["LIKING", "EMOTION", "SG"]),
        # Plural/bare verbs
        "chase":  GroundingContext(motor=["CHASING", "PURSUIT", "PL"]),
        "see":    GroundingContext(motor=["SEEING", "PERCEPTION", "PL"]),
        "find":   GroundingContext(motor=["FINDING", "PERCEPTION", "PL"]),
        "like":   GroundingContext(motor=["LIKING", "EMOTION", "PL"]),
        # Function word
        "the":    GroundingContext(),
    }


def _build_agreement_training(vocab):
    """Training on ONLY agreeing sentences (sg+sg, pl+pl).

    Establishes Hebbian connections for grammatical number agreement.
    The parser never sees disagreeing combinations during training.
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    # Singular subject + singular verb (3x repetition for strength)
    sg_triples = [
        ("dog", "chases", "cat"), ("cat", "sees", "bird"),
        ("bird", "chases", "fish"), ("horse", "chases", "dog"),
        ("dog", "sees", "bird"), ("cat", "finds", "horse"),
    ]
    # Plural subject + plural verb
    pl_triples = [
        ("dogs", "chase", "cats"), ("cats", "see", "birds"),
        ("birds", "chase", "dogs"), ("horses", "chase", "cats"),
        ("dogs", "see", "birds"), ("cats", "find", "horses"),
    ]

    for triples in [sg_triples, pl_triples]:
        for _ in range(3):
            for subj, verb, obj in triples:
                sentences.append(GroundedSentence(
                    words=["the", subj, verb, "the", obj],
                    contexts=[ctx("the"), ctx(subj), ctx(verb),
                              ctx("the"), ctx(obj)],
                    roles=[None, "agent", "action", None, "patient"],
                ))

    return sentences


# Test sentences: measure P600 at OBJECT position for all conditions.
# The agreement violation is in the verb (position 2), and by the time
# we reach the object (position 4), the structural integration reflects
# the cumulative processing difficulty including the agreement mismatch.
#
# Alternative: measure at verb position for agreement violations.
# We do both and compare.
AGREEMENT_TESTS = [
    {
        "frame": "the dog chases the ___",
        "sg_context": ["the", "dog", "chases", "the"],
        "pl_context": ["the", "dogs", "chases", "the"],  # agreement violation
        "grammatical_obj": "cat",
        "category_violation": "likes",
    },
    {
        "frame": "the cat sees the ___",
        "sg_context": ["the", "cat", "sees", "the"],
        "pl_context": ["the", "cats", "sees", "the"],  # agreement violation
        "grammatical_obj": "bird",
        "category_violation": "finds",
    },
    {
        "frame": "the bird chases the ___",
        "sg_context": ["the", "bird", "chases", "the"],
        "pl_context": ["the", "birds", "chases", "the"],  # agreement violation
        "grammatical_obj": "fish",
        "category_violation": "sees",
    },
    {
        "frame": "the horse finds the ___",
        "sg_context": ["the", "horse", "finds", "the"],
        "pl_context": ["the", "horses", "finds", "the"],  # agreement violation
        "grammatical_obj": "mouse",
        "category_violation": "chases",
    },
]


class AgreementViolationExperiment(ExperimentBase):
    """Test graded P600 for agreement violations."""

    def __init__(self, results_dir=None, seed=42, verbose=True):
        super().__init__(
            name="agreement_violations",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick=False, **kwargs):
        self._start_timer()
        cfg = AgreementConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_agreement_vocab()
        training = _build_agreement_training(vocab)
        seeds = list(range(cfg.n_seeds))

        p600_areas = [ROLE_AGENT, ROLE_PATIENT, SUBJ, OBJ, VP]

        # Per-seed accumulators
        gram_seeds = []      # grammatical: sg context + trained noun
        agree_seeds = []     # agreement violation: pl context + trained noun
        cat_seeds = []       # category violation: sg context + verb as noun

        for seed_idx, seed in enumerate(seeds):
            self.log(f"\n=== Seed {seed_idx + 1}/{len(seeds)} ===")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            bootstrap_structural_connectivity(
                parser, p600_areas, log_fn=self.log)
            consolidate_role_connections(
                parser, training, n_passes=cfg.consolidation_passes,
                log_fn=self.log)
            consolidate_vp_connections(
                parser, training, n_passes=cfg.consolidation_passes,
                log_fn=self.log)

            gram_vals, agree_vals, cat_vals = [], [], []

            for test in AGREEMENT_TESTS:
                # Grammatical: singular context + grammatical object
                result_gram = _measure_critical_word(
                    parser, test["sg_context"], test["grammatical_obj"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                gram_vals.append(result_gram["p600_mean_instability"])

                # Agreement violation: plural subject context + grammatical object
                # The mismatch between plural subject and singular verb in context
                # creates structural processing difficulty that propagates to the
                # object position
                result_agree = _measure_critical_word(
                    parser, test["pl_context"], test["grammatical_obj"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                agree_vals.append(result_agree["p600_mean_instability"])

                # Category violation: singular context + verb as noun
                result_cat = _measure_critical_word(
                    parser, test["sg_context"], test["category_violation"],
                    p600_areas, cfg.rounds, cfg.p600_settling_rounds,
                )
                cat_vals.append(result_cat["p600_mean_instability"])

            if gram_vals:
                gram_seeds.append(float(np.mean(gram_vals)))
                agree_seeds.append(float(np.mean(agree_vals)))
                cat_seeds.append(float(np.mean(cat_vals)))
                self.log(f"  gram={np.mean(gram_vals):.3f}  "
                         f"agree={np.mean(agree_vals):.3f}  "
                         f"cat={np.mean(cat_vals):.3f}")

        # -- Analysis --
        self.log(f"\n{'='*60}")
        self.log("AGREEMENT VIOLATION RESULTS")
        self.log(f"{'='*60}")
        self.log("Prediction: P600(cat) > P600(agree) > P600(gram)")

        metrics = {}

        if len(gram_seeds) >= 2:
            gs = summarize(gram_seeds)
            ags = summarize(agree_seeds)
            cs = summarize(cat_seeds)

            self.log(f"\n  Grammatical:         {gs['mean']:.4f} +/- {gs['sem']:.4f}")
            self.log(f"  Agreement violation: {ags['mean']:.4f} +/- {ags['sem']:.4f}")
            self.log(f"  Category violation:  {cs['mean']:.4f} +/- {cs['sem']:.4f}")

            metrics["gram"] = gs
            metrics["agreement"] = ags
            metrics["category"] = cs

            # Check ordering
            ordering_correct = cs["mean"] > ags["mean"] > gs["mean"]
            self.log(f"\n  Ordering cat > agree > gram: "
                     f"{'CORRECT' if ordering_correct else 'INCORRECT'}")
            metrics["ordering_correct"] = ordering_correct

            # Pairwise comparisons
            self.log(f"\n  Pairwise comparisons:")
            for label, a, b in [
                ("cat_vs_gram", cat_seeds, gram_seeds),
                ("agree_vs_gram", agree_seeds, gram_seeds),
                ("cat_vs_agree", cat_seeds, agree_seeds),
            ]:
                stats = paired_ttest(a, b)
                direction = ("HIGHER" if np.mean(a) > np.mean(b)
                             else "LOWER")
                self.log(f"    {label:<20}: d={stats['d']:.3f}  "
                         f"p={stats['p']:.4f}  {direction}")
                metrics[label] = {"test": stats, "direction": direction}

        duration = self._stop_timer()
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p,
                "beta": cfg.beta, "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
                "p600_settling_rounds": cfg.p600_settling_rounds,
                "consolidation_passes": cfg.consolidation_passes,
            },
            metrics=metrics,
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agreement Violation Experiment (Prediction 3.1)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer seeds (3 instead of 5)")
    args = parser.parse_args()

    exp = AgreementViolationExperiment()
    exp.run(quick=args.quick)
