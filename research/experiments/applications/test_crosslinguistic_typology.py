"""
Cross-Linguistic Typology Experiment (Tier 1)

Tests whether the same assembly calculus neural substrate can acquire
fundamentally different word order typologies (SVO, SOV, VSO) from
exposure alone, using the emergent NEMO parser.

Scientific Question:
    Can a single neural architecture learn to distinguish and correctly
    produce SVO (English), SOV (Japanese/Turkish), and VSO (Arabic/Welsh)
    word orders purely from statistical exposure to ordered sentences?

Hypotheses:
    H1: SVO training produces SVO classification (infer_word_order
        returns "SVO" with confidence > 0.6).
    H2: SOV training produces SOV classification.
    H3: VSO training produces VSO classification.
    H4: Classification accuracy is comparable across typologies
        (pairwise differences within 10%).
    H5: Role assignment adapts to word order (agent before verb in SVO,
        agent after verb in VSO).

Protocol:
    For each typology (SVO, SOV, VSO), for each seed:
    1. Create EmergentParser(n=10000, k=100, seed=seed).
    2. Build a small grounded vocabulary (~20 words).
    3. Generate training sentences in the target word order.
    4. Train the parser using train_from_sentences().
    5. Test infer_word_order() for correct typology detection.
    6. Test parse() on held-out sentences for correct role assignments.
    7. Measure classification accuracy across all three typologies.

References:
    - Dryer, M. S. (2013). "Order of Subject, Object and Verb."
      In WALS Online. Max Planck Institute for Evolutionary Anthropology.
    - Mitropolsky, D. & Papadimitriou, C. H. (2025).
      "Simulated Language Acquisition with Neural Assemblies."
    - Greenberg, J. H. (1963). "Universals of Language." MIT Press.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from research.experiments.base import (
    ExperimentBase,
    ExperimentResult,
    summarize,
    ttest_vs_null,
    paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class TypologyConfig:
    """Configuration for cross-linguistic typology experiment."""
    n: int = 10000
    k: int = 100
    p: float = 0.05
    beta: float = 0.1
    rounds: int = 10
    n_seeds: int = 5
    n_training_sentences: int = 30


TYPOLOGIES = ["SVO", "SOV", "VSO"]

# Chance level for 7-category word classification
CHANCE_ACCURACY = 1.0 / 7.0  # ~0.143


# ======================================================================
# Vocabulary builder
# ======================================================================

def build_test_vocabulary() -> Dict[str, GroundingContext]:
    """Build a small grounded vocabulary for typology testing.

    Returns a dictionary of ~20 words across 4 POS categories:
        8 nouns  (visual grounding)
        5 verbs  (motor grounding)
        4 determiners (no grounding -- function words)
        3 adjectives  (property grounding)

    Returns:
        {word: GroundingContext} mapping.
    """
    vocab: Dict[str, GroundingContext] = {}

    # Nouns -- visual grounding
    vocab["dog"] = GroundingContext(visual=["DOG", "ANIMAL"])
    vocab["cat"] = GroundingContext(visual=["CAT", "ANIMAL"])
    vocab["bird"] = GroundingContext(visual=["BIRD", "ANIMAL"])
    vocab["boy"] = GroundingContext(visual=["BOY", "PERSON"])
    vocab["girl"] = GroundingContext(visual=["GIRL", "PERSON"])
    vocab["ball"] = GroundingContext(visual=["BALL", "OBJECT"])
    vocab["book"] = GroundingContext(visual=["BOOK", "OBJECT"])
    vocab["fish"] = GroundingContext(visual=["FISH", "ANIMAL"])

    # Verbs -- motor grounding
    vocab["chases"] = GroundingContext(motor=["CHASING", "PURSUIT"])
    vocab["sees"] = GroundingContext(motor=["SEEING", "PERCEPTION"])
    vocab["eats"] = GroundingContext(motor=["EATING", "CONSUMPTION"])
    vocab["finds"] = GroundingContext(motor=["FINDING", "PERCEPTION"])
    vocab["plays"] = GroundingContext(motor=["PLAYING", "ACTION"])

    # Determiners -- no grounding (function words)
    vocab["the"] = GroundingContext()
    vocab["a"] = GroundingContext()
    vocab["this"] = GroundingContext()
    vocab["that"] = GroundingContext()

    # Adjectives -- property grounding
    vocab["big"] = GroundingContext(properties=["SIZE", "BIG"])
    vocab["small"] = GroundingContext(properties=["SIZE", "SMALL"])
    vocab["red"] = GroundingContext(properties=["COLOR", "RED"])

    return vocab


# ======================================================================
# Sentence generators
# ======================================================================

def _make_transitive(subj_det: str, subj_noun: str, verb: str,
                     obj_det: str, obj_noun: str,
                     order: str) -> List[str]:
    """Arrange a transitive sentence in the given word order.

    Args:
        subj_det: Subject determiner.
        subj_noun: Subject noun.
        verb: Verb.
        obj_det: Object determiner.
        obj_noun: Object noun.
        order: One of "SVO", "SOV", "VSO".

    Returns:
        Ordered list of word tokens.
    """
    subj = [subj_det, subj_noun]
    obj_ = [obj_det, obj_noun]
    v = [verb]

    if order == "SVO":
        return subj + v + obj_
    elif order == "SOV":
        return subj + obj_ + v
    elif order == "VSO":
        return v + subj + obj_
    else:
        raise ValueError(f"Unknown word order: {order}")


def _make_intransitive(det: str, noun: str, verb: str,
                       order: str) -> List[str]:
    """Arrange an intransitive sentence in the given word order.

    Args:
        det: Determiner.
        noun: Subject noun.
        verb: Verb.
        order: One of "SVO", "SOV", "VSO".

    Returns:
        Ordered list of word tokens.
    """
    subj = [det, noun]
    v = [verb]

    if order in ("SVO", "SOV"):
        # Both SVO and SOV place subject before verb for intransitives
        return subj + v
    elif order == "VSO":
        return v + subj
    else:
        raise ValueError(f"Unknown word order: {order}")


def generate_typology_sentences(
    vocab: Dict[str, GroundingContext],
    order: str,
    n_sents: int,
    seed: int = 42,
) -> List[List[str]]:
    """Generate sentences in the specified word order from the vocabulary.

    Produces a mix of transitive and intransitive sentences using
    random selections from the vocabulary, arranged according to the
    given typological word order.

    Args:
        vocab: Word -> GroundingContext mapping.
        order: Target word order ("SVO", "SOV", "VSO").
        n_sents: Number of sentences to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of token lists (one per sentence).
    """
    rng = random.Random(seed)

    nouns = [w for w, c in vocab.items() if c.dominant_modality == "visual"]
    verbs = [w for w, c in vocab.items() if c.dominant_modality == "motor"]
    dets = [w for w, c in vocab.items() if c.dominant_modality == "none"]

    if not dets:
        dets = ["the"]

    sentences: List[List[str]] = []

    for i in range(n_sents):
        if i % 3 == 0 and len(nouns) >= 2:
            # Intransitive: DET NOUN VERB (arranged per typology)
            det = rng.choice(dets)
            noun = rng.choice(nouns)
            verb = rng.choice(verbs)
            sent = _make_intransitive(det, noun, verb, order)
        else:
            # Transitive: DET NOUN VERB DET NOUN (arranged per typology)
            d1 = rng.choice(dets)
            n1 = rng.choice(nouns)
            verb = rng.choice(verbs)
            d2 = rng.choice(dets)
            n2 = rng.choice(nouns)
            # Avoid same noun as subject and object
            for _ in range(5):
                if n2 != n1:
                    break
                n2 = rng.choice(nouns)
            sent = _make_transitive(d1, n1, verb, d2, n2, order)

        sentences.append(sent)

    return sentences


def _generate_test_sentences(
    vocab: Dict[str, GroundingContext],
    order: str,
    seed: int = 999,
) -> List[Tuple[List[str], str, str, str]]:
    """Generate a fixed set of held-out test sentences with known roles.

    Returns tuples of (sentence_tokens, agent_noun, verb, patient_noun).
    Uses a different seed from training to ensure novelty.

    Args:
        vocab: Word -> GroundingContext mapping.
        order: Target word order.
        seed: Random seed (different from training seeds).

    Returns:
        List of (tokens, agent, verb, patient) tuples.
    """
    rng = random.Random(seed)

    nouns = [w for w, c in vocab.items() if c.dominant_modality == "visual"]
    verbs = [w for w, c in vocab.items() if c.dominant_modality == "motor"]
    dets = [w for w, c in vocab.items() if c.dominant_modality == "none"]

    test_cases: List[Tuple[List[str], str, str, str]] = []

    for _ in range(10):
        d1, d2 = rng.choice(dets), rng.choice(dets)
        n1 = rng.choice(nouns)
        n2 = rng.choice(nouns)
        for _retry in range(5):
            if n2 != n1:
                break
            n2 = rng.choice(nouns)
        v = rng.choice(verbs)
        tokens = _make_transitive(d1, n1, v, d2, n2, order)
        test_cases.append((tokens, n1, v, n2))

    return test_cases


# ======================================================================
# Experiment
# ======================================================================

class CrossLinguisticTypologyExperiment(ExperimentBase):
    """Test whether assembly calculus acquires different word order typologies."""

    def __init__(
        self,
        results_dir: Path = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        super().__init__(
            name="crosslinguistic_typology",
            seed=seed,
            results_dir=(
                results_dir
                or Path(__file__).parent.parent.parent / "results" / "applications"
            ),
            verbose=verbose,
        )

    def run(
        self,
        n_seeds: int = None,
        n_training_sentences: int = None,
        quick: bool = False,
        **kwargs,
    ) -> ExperimentResult:
        """Run the cross-linguistic typology experiment.

        For each typology (SVO, SOV, VSO) and each seed:
        1. Create a fresh EmergentParser with the test vocabulary.
        2. Generate training sentences in the target word order.
        3. Train the parser via train_from_sentences().
        4. Evaluate: word order inference, classification accuracy,
           and role assignment accuracy on held-out test sentences.
        5. Aggregate statistics across seeds and compare typologies.

        Args:
            n_seeds: Number of random seeds per typology.
            n_training_sentences: Number of training sentences per condition.
            quick: If True, reduce parameters for fast validation.

        Returns:
            ExperimentResult with all metrics and raw data.
        """
        self._start_timer()

        cfg = TypologyConfig()
        if quick:
            cfg.n_seeds = 3
            cfg.n_training_sentences = 20
        if n_seeds is not None:
            cfg.n_seeds = n_seeds
        if n_training_sentences is not None:
            cfg.n_training_sentences = n_training_sentences

        vocab = build_test_vocabulary()
        seeds = [self.seed + i * 100 for i in range(cfg.n_seeds)]

        self.log("=" * 60)
        self.log("Cross-Linguistic Typology Experiment")
        self.log(f"  n={cfg.n}, k={cfg.k}, p={cfg.p}, beta={cfg.beta}")
        self.log(f"  rounds={cfg.rounds}")
        self.log(f"  n_seeds={cfg.n_seeds}")
        self.log(f"  n_training_sentences={cfg.n_training_sentences}")
        self.log(f"  vocabulary size={len(vocab)}")
        self.log(f"  typologies={TYPOLOGIES}")
        self.log("=" * 60)

        # Collect per-typology results
        all_raw: Dict[str, Any] = {}
        order_accuracies: Dict[str, List[float]] = {t: [] for t in TYPOLOGIES}
        order_confidences: Dict[str, List[float]] = {t: [] for t in TYPOLOGIES}
        role_accuracies: Dict[str, List[float]] = {t: [] for t in TYPOLOGIES}

        for typology in TYPOLOGIES:
            self.log(f"\n{'='*60}")
            self.log(f"Typology: {typology}")
            self.log(f"{'='*60}")

            typology_raw: List[Dict[str, Any]] = []

            for seed_idx, seed in enumerate(seeds):
                self.log(f"  Seed {seed_idx + 1}/{cfg.n_seeds} (seed={seed})")

                # -- Build parser with custom vocabulary --
                parser = EmergentParser(
                    n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                    seed=seed, rounds=cfg.rounds, vocabulary=vocab,
                )

                # -- Generate training sentences in target word order --
                train_sents = generate_typology_sentences(
                    vocab, typology, cfg.n_training_sentences,
                    seed=seed,
                )

                # -- Train --
                parser.train_from_sentences(train_sents, use_grounding=True)

                # -- Also train word order typological classifier --
                parser.train_word_order_typological(train_sents)

                # -- Evaluate: word order inference --
                inferred_order, confidence = parser.infer_word_order()
                order_correct = 1.0 if inferred_order == typology else 0.0
                order_accuracies[typology].append(order_correct)
                order_confidences[typology].append(confidence)

                self.log(
                    f"    Word order: inferred={inferred_order} "
                    f"(conf={confidence:.3f}), "
                    f"correct={order_correct == 1.0}"
                )

                # -- Evaluate: role assignment on held-out test sentences --
                test_cases = _generate_test_sentences(
                    vocab, typology, seed=seed + 7777,
                )
                n_role_correct = 0
                n_role_total = 0

                for tokens, agent_noun, verb, patient_noun in test_cases:
                    parse_result = parser.parse(tokens)
                    roles = parse_result.get("roles", {})

                    # Check agent assignment
                    agent_role = roles.get(agent_noun)
                    if agent_role == "AGENT":
                        n_role_correct += 1
                    n_role_total += 1

                    # Check patient assignment
                    patient_role = roles.get(patient_noun)
                    if patient_role == "PATIENT":
                        n_role_correct += 1
                    n_role_total += 1

                role_acc = n_role_correct / max(n_role_total, 1)
                role_accuracies[typology].append(role_acc)

                self.log(
                    f"    Role accuracy: {n_role_correct}/{n_role_total} "
                    f"= {role_acc:.3f}"
                )

                typology_raw.append({
                    "seed": seed,
                    "inferred_order": inferred_order,
                    "order_confidence": confidence,
                    "order_correct": order_correct,
                    "role_accuracy": role_acc,
                    "n_role_correct": n_role_correct,
                    "n_role_total": n_role_total,
                })

            all_raw[typology] = typology_raw

        # ==============================================================
        # Statistical analysis
        # ==============================================================
        self.log(f"\n{'='*60}")
        self.log("Statistical Analysis")
        self.log(f"{'='*60}")

        metrics: Dict[str, Any] = {}

        # -- Per-typology summaries --
        for typology in TYPOLOGIES:
            order_stats = summarize(order_accuracies[typology])
            conf_stats = summarize(order_confidences[typology])
            role_stats = summarize(role_accuracies[typology])

            # H1/H2/H3: Classification accuracy > chance (1/7)
            order_test = ttest_vs_null(order_accuracies[typology], CHANCE_ACCURACY)
            role_test = ttest_vs_null(role_accuracies[typology], 0.5)

            metrics[f"{typology}_order_accuracy"] = order_stats
            metrics[f"{typology}_order_confidence"] = conf_stats
            metrics[f"{typology}_role_accuracy"] = role_stats
            metrics[f"{typology}_order_vs_chance"] = order_test
            metrics[f"{typology}_role_vs_chance"] = role_test

            self.log(f"\n  {typology}:")
            self.log(
                f"    Order accuracy: "
                f"{order_stats['mean']:.3f} +/- {order_stats['sem']:.3f} "
                f"[{order_stats['ci95_lo']:.3f}, {order_stats['ci95_hi']:.3f}]"
            )
            self.log(
                f"    Order confidence: "
                f"{conf_stats['mean']:.3f} +/- {conf_stats['sem']:.3f}"
            )
            self.log(
                f"    Role accuracy: "
                f"{role_stats['mean']:.3f} +/- {role_stats['sem']:.3f}"
            )
            self.log(
                f"    Order vs chance: t={order_test['t']:.2f}, "
                f"p={order_test['p']:.4f}, d={order_test['d']:.2f}, "
                f"sig={order_test['significant']}"
            )

        # -- H4: Pairwise comparisons across typologies --
        self.log(f"\n  H4: Pairwise typology comparisons (order accuracy):")
        pairs = [("SVO", "SOV"), ("SVO", "VSO"), ("SOV", "VSO")]
        for t1, t2 in pairs:
            ptest = paired_ttest(
                order_accuracies[t1], order_accuracies[t2],
            )
            metrics[f"paired_{t1}_vs_{t2}_order"] = ptest
            self.log(
                f"    {t1} vs {t2}: t={ptest['t']:.2f}, "
                f"p={ptest['p']:.4f}, d={ptest['d']:.2f}, "
                f"sig={ptest['significant']}"
            )

        # -- H5: Role accuracy pairwise comparisons --
        self.log(f"\n  H5: Pairwise typology comparisons (role accuracy):")
        for t1, t2 in pairs:
            ptest = paired_ttest(
                role_accuracies[t1], role_accuracies[t2],
            )
            metrics[f"paired_{t1}_vs_{t2}_role"] = ptest
            self.log(
                f"    {t1} vs {t2}: t={ptest['t']:.2f}, "
                f"p={ptest['p']:.4f}, d={ptest['d']:.2f}, "
                f"sig={ptest['significant']}"
            )

        # -- Overall summary --
        all_order_accs = []
        all_role_accs = []
        for typology in TYPOLOGIES:
            all_order_accs.extend(order_accuracies[typology])
            all_role_accs.extend(role_accuracies[typology])

        metrics["overall_order_accuracy"] = summarize(all_order_accs)
        metrics["overall_role_accuracy"] = summarize(all_role_accs)

        self.log(f"\n  Overall:")
        self.log(
            f"    Order accuracy: "
            f"{metrics['overall_order_accuracy']['mean']:.3f} "
            f"+/- {metrics['overall_order_accuracy']['sem']:.3f}"
        )
        self.log(
            f"    Role accuracy: "
            f"{metrics['overall_role_accuracy']['mean']:.3f} "
            f"+/- {metrics['overall_role_accuracy']['sem']:.3f}"
        )

        duration = self._stop_timer()
        self.log(f"\nTotal duration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n,
                "k": cfg.k,
                "p": cfg.p,
                "beta": cfg.beta,
                "rounds": cfg.rounds,
                "n_seeds": cfg.n_seeds,
                "n_training_sentences": cfg.n_training_sentences,
                "typologies": TYPOLOGIES,
                "vocabulary_size": len(vocab),
            },
            metrics=metrics,
            raw_data=all_raw,
            duration_seconds=duration,
        )


# ======================================================================
# Main entry point
# ======================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-Linguistic Typology Experiment (Tier 1)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with fewer seeds and training sentences",
    )
    args = parser.parse_args()

    exp = CrossLinguisticTypologyExperiment(verbose=True)

    if args.quick:
        result = exp.run(quick=True)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("CROSS-LINGUISTIC TYPOLOGY SUMMARY")
    print("=" * 70)

    for typology in TYPOLOGIES:
        order_acc = result.metrics[f"{typology}_order_accuracy"]
        role_acc = result.metrics[f"{typology}_role_accuracy"]
        conf = result.metrics[f"{typology}_order_confidence"]
        test = result.metrics[f"{typology}_order_vs_chance"]
        print(
            f"\n  {typology}:"
            f"\n    Order accuracy:  {order_acc['mean']:.3f} "
            f"+/- {order_acc['sem']:.3f}"
            f"\n    Confidence:      {conf['mean']:.3f} "
            f"+/- {conf['sem']:.3f}"
            f"\n    Role accuracy:   {role_acc['mean']:.3f} "
            f"+/- {role_acc['sem']:.3f}"
            f"\n    vs chance:       t={test['t']:.2f}, p={test['p']:.4f}, "
            f"sig={test['significant']}"
        )

    overall_order = result.metrics["overall_order_accuracy"]
    overall_role = result.metrics["overall_role_accuracy"]
    print(
        f"\n  Overall:"
        f"\n    Order accuracy:  {overall_order['mean']:.3f}"
        f"\n    Role accuracy:   {overall_role['mean']:.3f}"
    )

    print(f"\n  Pairwise order comparisons:")
    for t1, t2 in [("SVO", "SOV"), ("SVO", "VSO"), ("SOV", "VSO")]:
        pt = result.metrics[f"paired_{t1}_vs_{t2}_order"]
        print(f"    {t1} vs {t2}: t={pt['t']:.2f}, p={pt['p']:.4f}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
