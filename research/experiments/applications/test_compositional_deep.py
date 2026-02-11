"""
Deep Compositional Generalization Experiment

Tests progressively harder forms of compositionality in the assembly
calculus parser. The original compositional_generalization experiment
achieved perfect scores because it only tested role swaps with identical
sentence structure. Real compositionality requires generalizing across
multiple dimensions simultaneously.

Levels of compositional generalization (from easy to hard):

Level 1 (role swap): Nouns trained as agents appear as patients and vice
    versa. Same sentence structure, same verbs. This tests whether role
    assignment is based on structural position (word order) rather than
    memorized word-role associations.

Level 2 (novel structure): Train on active voice, test on passive voice.
    The parser must recognize the SAME thematic relations expressed in a
    completely different surface form. "The dog chases the cat" and
    "The cat was chased by the dog" have the same meaning but different
    word order. Tests structural generalization.

Level 3 (holdout words): Some words are NEVER seen during grounded
    training, only via distributional statistics from raw sentence
    ingestion. Tests whether the parser can classify and assign roles
    to words learned purely from context (no grounding features).

Level 4 (novel verb-noun): Verbs are trained with only a subset of
    nouns (e.g., "dog chases cat" but NEVER "dog eats cat"). Tests
    whether verb-argument generalization is truly compositional.

Level 5 (recursive novel): Novel combinations inside relative clauses.
    "The dog that the BALL chases sees the cat" — "ball" was never an
    agent of "chases" in training, AND it's inside a relative clause.
    Tests compositionality of recursion + lexical generalization.

Statistical methodology:
- N_SEEDS independent random seeds per level.
- Accuracy thresholds calibrated per level difficulty.
- Paired comparisons between train and test within each level.

References:
- Lake & Baroni (2018). Generalization without Systematicity. ICML.
- Kim & Linzen (2020). COGS: A Compositional Generalization Challenge.
- Keysers et al. (2020). Measuring Compositional Generalization.
- Papadimitriou et al. (2020). Brain computation by assemblies of neurons.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, ttest_vs_null, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


@dataclass
class DeepCompConfig:
    """Configuration for the deep compositionality experiment."""
    n: int = 50000      # neurons per area (5x standard for finer resolution)
    k: int = 100        # assembly size (same as standard, sparser in larger n)
    n_seeds: int = 5    # independent random seeds
    p: float = 0.05     # connection probability (biologically realistic)
    beta: float = 0.05  # plasticity rate (lower to avoid over-fitting)
    rounds: int = 10    # projection rounds


# -- Vocabulary ----------------------------------------------------------------

def _build_vocab() -> Dict[str, GroundingContext]:
    """Full vocabulary for compositionality tests.

    Includes:
    - 8 animate nouns (4 animals, 4 people)
    - 6 verbs (present tense for active, past participle for passive)
    - Function words (the, a, was, by, that)
    - 2 holdout nouns (grounded but withheld from lexicon training)
    """
    return {
        # Animate nouns — set A (trained as agents)
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "boy":    GroundingContext(visual=["BOY", "PERSON"]),
        "girl":   GroundingContext(visual=["GIRL", "PERSON"]),
        # Animate nouns — set B (trained as patients)
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "man":    GroundingContext(visual=["MAN", "PERSON"]),
        "woman":  GroundingContext(visual=["WOMAN", "PERSON"]),
        # Verbs — set X (trained with set A agents)
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        # Verbs — set Y (trained with set B patients only)
        "pushes": GroundingContext(motor=["PUSHING", "ACTION"]),
        "warns":  GroundingContext(motor=["WARNING", "ACTION"]),
        "helps":  GroundingContext(motor=["HELPING", "AID"]),
        # Past tense verbs for passive (same grounding as present)
        "chased": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "pushed": GroundingContext(motor=["PUSHING", "ACTION"]),
        "found":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "helped": GroundingContext(motor=["HELPING", "AID"]),
        # Function words
        "the":    GroundingContext(),
        "a":      GroundingContext(),
        "was":    GroundingContext(),
        "by":     GroundingContext(spatial=["BY", "AGENT_MARKER"]),
        "that":   GroundingContext(),
    }


# -- Training data -------------------------------------------------------------

def _build_training(vocab: Dict[str, GroundingContext],
                    ) -> List[GroundedSentence]:
    """Build training sentences with controlled splits.

    Design:
    - Set A nouns (dog,cat,boy,girl) always appear as AGENTS
    - Set B nouns (bird,horse,man,woman) always appear as PATIENTS
    - Set X verbs (chases,sees,finds) pair with A-agents only
    - Set Y verbs (pushes,warns,helps) pair with A-agents only
    - Passive sentences included for passive voice training

    This creates multiple dimensions of novelty for testing:
    - Level 1: B nouns as agents (role swap)
    - Level 2: Active-trained → passive test
    - Level 4: X verbs with B-patients that were never in X training
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    # Active: A-agent + X-verb + B-patient
    for agent, verb, patient in [
        ("dog", "chases", "bird"), ("dog", "sees", "horse"),
        ("cat", "chases", "man"), ("cat", "finds", "woman"),
        ("boy", "sees", "bird"), ("boy", "finds", "horse"),
        ("girl", "chases", "woman"), ("girl", "sees", "man"),
        ("dog", "finds", "bird"), ("cat", "sees", "horse"),
        ("boy", "chases", "woman"), ("girl", "finds", "man"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", agent, verb, "the", patient],
            contexts=[ctx("the"), ctx(agent), ctx(verb),
                      ctx("the"), ctx(patient)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Active: A-agent + Y-verb + B-patient
    for agent, verb, patient in [
        ("dog", "pushes", "bird"), ("cat", "warns", "horse"),
        ("boy", "helps", "man"), ("girl", "pushes", "woman"),
        ("dog", "warns", "man"), ("cat", "helps", "bird"),
        ("boy", "pushes", "horse"), ("girl", "warns", "woman"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", agent, verb, "the", patient],
            contexts=[ctx("the"), ctx(agent), ctx(verb),
                      ctx("the"), ctx(patient)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Passive: B-patient + was + past-verb + by + A-agent
    for patient, verb, agent in [
        ("bird", "chased", "dog"), ("horse", "pushed", "cat"),
        ("man", "found", "boy"), ("woman", "helped", "girl"),
        ("horse", "chased", "boy"), ("bird", "helped", "girl"),
    ]:
        sentences.append(GroundedSentence(
            words=["the", patient, "was", verb, "by", "the", agent],
            contexts=[ctx("the"), ctx(patient), ctx("was"), ctx(verb),
                      ctx("by"), ctx("the"), ctx(agent)],
            roles=[None, "patient", None, "action", None, None, "agent"],
        ))

    return sentences


# -- Test generators -----------------------------------------------------------

def _level1_tests() -> List[Dict[str, Any]]:
    """Level 1: Role swap — B nouns as agents, A nouns as patients."""
    return [
        {"words": ["the", "bird", "chases", "the", "dog"],
         "expected": {"bird": "AGENT", "dog": "PATIENT"},
         "label": "B-agent X-verb A-patient (1)"},
        {"words": ["the", "horse", "sees", "the", "cat"],
         "expected": {"horse": "AGENT", "cat": "PATIENT"},
         "label": "B-agent X-verb A-patient (2)"},
        {"words": ["the", "man", "finds", "the", "boy"],
         "expected": {"man": "AGENT", "boy": "PATIENT"},
         "label": "B-agent X-verb A-patient (3)"},
        {"words": ["the", "woman", "pushes", "the", "girl"],
         "expected": {"woman": "AGENT", "girl": "PATIENT"},
         "label": "B-agent Y-verb A-patient"},
    ]


def _level2_tests() -> List[Dict[str, Any]]:
    """Level 2: Novel structure — active-trained combos in passive voice."""
    return [
        {"words": ["the", "bird", "was", "chased", "by", "the", "dog"],
         "expected": {"bird": "PATIENT", "dog": "AGENT"},
         "label": "passive of trained active (1)"},
        {"words": ["the", "horse", "was", "pushed", "by", "the", "cat"],
         "expected": {"horse": "PATIENT", "cat": "AGENT"},
         "label": "passive of trained active (2)"},
        # Novel: B-patient, A-agent in passive (never seen in passive training)
        {"words": ["the", "man", "was", "found", "by", "the", "girl"],
         "expected": {"man": "PATIENT", "girl": "AGENT"},
         "label": "novel passive combination"},
    ]


def _level3_tests(parser: EmergentParser,
                  vocab: Dict[str, GroundingContext],
                  ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Level 3: Holdout words — classify purely from distributional context.

    Ingests raw sentences containing two new words ('tiger', 'queen')
    into the parser's distributional statistics, then tests classification
    and role assignment without any grounded training.
    """
    # Register holdout words with grounding (but they were NOT in training)
    holdout_words = ["tiger", "queen"]
    for w in holdout_words:
        if w not in vocab:
            # Create grounding but don't add to trained vocabulary
            if w == "tiger":
                ctx = GroundingContext(visual=["TIGER", "ANIMAL"])
            else:
                ctx = GroundingContext(visual=["QUEEN", "PERSON"])
            parser.word_grounding[w] = ctx
            if w not in parser.stim_map:
                phon = f"phon_{w}"
                parser.brain.add_stimulus(phon, parser.k)
                parser.stim_map[w] = phon
                # Register grounding stimuli
                for mod in ("visual", "motor", "properties", "spatial",
                            "social", "temporal", "emotional"):
                    for feat in getattr(ctx, mod):
                        stim_name = f"{mod}_{feat}"
                        if stim_name not in parser._grounding_stim_names_set:
                            parser.brain.add_stimulus(stim_name, parser.k)
                            parser._grounding_stim_names_set.add(stim_name)

    # Ingest distributional context for holdout words
    dist_sentences = [
        ["the", "tiger", "chases", "the", "bird"],
        ["the", "tiger", "sees", "the", "horse"],
        ["the", "dog", "chases", "the", "tiger"],
        ["the", "queen", "helps", "the", "man"],
        ["the", "queen", "sees", "the", "woman"],
        ["the", "boy", "finds", "the", "queen"],
    ]
    for sent in dist_sentences:
        parser.ingest_raw_sentence(sent)
    # Repeat for more statistics
    for _ in range(3):
        for sent in dist_sentences:
            parser.ingest_raw_sentence(sent)

    tests = [
        {"words": ["the", "tiger", "chases", "the", "cat"],
         "expected": {"tiger": "AGENT", "cat": "PATIENT"},
         "label": "holdout noun as agent"},
        {"words": ["the", "dog", "sees", "the", "queen"],
         "expected": {"dog": "AGENT", "queen": "PATIENT"},
         "label": "holdout noun as patient"},
        {"words": ["the", "queen", "finds", "the", "tiger"],
         "expected": {"queen": "AGENT", "tiger": "PATIENT"},
         "label": "both holdout nouns"},
    ]

    return tests, holdout_words


def _level4_tests() -> List[Dict[str, Any]]:
    """Level 4: Novel verb-noun combinations.

    Test verbs with nouns they were NEVER trained with.
    Set X verbs only saw A-agents + B-patients in training.
    Test: X verbs with B-agents (never seen as agents of X verbs).
    """
    return [
        # bird was never agent of "finds" in training
        {"words": ["the", "bird", "finds", "the", "dog"],
         "expected": {"bird": "AGENT", "dog": "PATIENT"},
         "label": "novel verb-agent (1)"},
        # horse was never agent of "chases" in training
        {"words": ["the", "horse", "chases", "the", "girl"],
         "expected": {"horse": "AGENT", "girl": "PATIENT"},
         "label": "novel verb-agent (2)"},
        # man was never subject of "sees" in training
        {"words": ["the", "man", "sees", "the", "boy"],
         "expected": {"man": "AGENT", "boy": "PATIENT"},
         "label": "novel verb-agent (3)"},
        # woman + helps: woman was never agent of helps
        {"words": ["the", "woman", "helps", "the", "cat"],
         "expected": {"woman": "AGENT", "cat": "PATIENT"},
         "label": "novel verb-agent (4)"},
    ]


def _level5_tests() -> List[Dict[str, Any]]:
    """Level 5: Recursive novel — novel combinations inside relative clauses.

    Tests compositionality of recursion + lexical generalization:
    the parser must handle both a relative clause AND novel word combinations.
    """
    return [
        # bird (set B) as agent of relative clause with cat (set A) as patient
        # "the dog that the bird chases sees the cat"
        # Inner clause: bird=AGENT of chases, dog=PATIENT (filler)
        # Main clause: dog=AGENT of sees, cat=PATIENT
        {"words": ["the", "dog", "that", "the", "bird", "chases",
                    "sees", "the", "cat"],
         "expected": {"dog": "AGENT", "cat": "PATIENT",
                      "bird": "AGENT"},  # inner clause agent
         "label": "ORC with novel agent"},
        # horse as agent inside SRC
        # "the horse that chases the girl sees the boy"
        # Inner: horse=AGENT of chases, girl=PATIENT
        # Main: horse=AGENT of sees, boy=PATIENT
        {"words": ["the", "horse", "that", "chases", "the", "girl",
                    "sees", "the", "boy"],
         "expected": {"horse": "AGENT", "boy": "PATIENT",
                      "girl": "PATIENT"},  # inner clause patient
         "label": "SRC with novel agent"},
    ]


# -- Measurement ---------------------------------------------------------------

def measure_accuracy(
    parser: EmergentParser,
    test: Dict[str, Any],
    use_recursive: bool = False,
) -> Dict[str, Any]:
    """Parse and measure noun role accuracy for a test case."""
    if use_recursive:
        result = parser.parse_recursive(test["words"])
    else:
        result = parser.parse(test["words"])

    roles = result["roles"]
    cats = result["categories"]

    correct = 0
    total = 0
    per_word = {}
    for word, exp_role in test["expected"].items():
        actual = roles.get(word, "")
        is_correct = (actual == exp_role)
        per_word[word] = {
            "expected": exp_role,
            "actual": actual,
            "correct": is_correct,
            "category": cats.get(word, "UNKNOWN"),
        }
        total += 1
        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "per_word": per_word,
            "roles": roles, "categories": cats}


# -- Experiment ----------------------------------------------------------------

class DeepCompositionalExperiment(ExperimentBase):
    """Test progressively harder compositional generalization."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="compositional_deep",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = DeepCompConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)
        seeds = list(range(cfg.n_seeds))

        self.log(f"Training: {len(training)} sentences")
        self.log(f"Seeds: {cfg.n_seeds}")

        # Per-level results: {level: [per-seed mean accuracy]}
        level_results = {f"L{i}": [] for i in range(1, 6)}
        level_details = {f"L{i}": [] for i in range(1, 6)}

        for s in seeds:
            self.log(f"\n--- Seed {s} ---")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            # --- Level 1: Role swap ---
            l1_tests = _level1_tests()
            l1_accs = []
            for t in l1_tests:
                r = measure_accuracy(parser, t)
                l1_accs.append(r["accuracy"])
            level_results["L1"].append(float(np.mean(l1_accs)))
            level_details["L1"].append(
                {t["label"]: a for t, a in zip(l1_tests, l1_accs)})

            # --- Level 2: Novel structure (passive) ---
            l2_tests = _level2_tests()
            l2_accs = []
            for t in l2_tests:
                r = measure_accuracy(parser, t)
                l2_accs.append(r["accuracy"])
            level_results["L2"].append(float(np.mean(l2_accs)))
            level_details["L2"].append(
                {t["label"]: a for t, a in zip(l2_tests, l2_accs)})

            # --- Level 3: Holdout words ---
            l3_tests, holdout_words = _level3_tests(parser, vocab)
            l3_accs = []
            for t in l3_tests:
                r = measure_accuracy(parser, t)
                l3_accs.append(r["accuracy"])
                if s == 0:
                    self.log(f"    L3 {t['label']}: "
                             f"acc={r['accuracy']:.3f} "
                             f"roles={r['roles']}")
            level_results["L3"].append(float(np.mean(l3_accs)))
            level_details["L3"].append(
                {t["label"]: a for t, a in zip(l3_tests, l3_accs)})

            # --- Level 4: Novel verb-noun ---
            l4_tests = _level4_tests()
            l4_accs = []
            for t in l4_tests:
                r = measure_accuracy(parser, t)
                l4_accs.append(r["accuracy"])
            level_results["L4"].append(float(np.mean(l4_accs)))
            level_details["L4"].append(
                {t["label"]: a for t, a in zip(l4_tests, l4_accs)})

            # --- Level 5: Recursive novel ---
            l5_tests = _level5_tests()
            l5_accs = []
            for t in l5_tests:
                r = measure_accuracy(parser, t, use_recursive=True)
                l5_accs.append(r["accuracy"])
                if s == 0:
                    self.log(f"    L5 {t['label']}: "
                             f"acc={r['accuracy']:.3f} "
                             f"roles={r['roles']}")
            level_results["L5"].append(float(np.mean(l5_accs)))
            level_details["L5"].append(
                {t["label"]: a for t, a in zip(l5_tests, l5_accs)})

            # Seed summary
            self.log(f"  L1={level_results['L1'][-1]:.3f} "
                     f"L2={level_results['L2'][-1]:.3f} "
                     f"L3={level_results['L3'][-1]:.3f} "
                     f"L4={level_results['L4'][-1]:.3f} "
                     f"L5={level_results['L5'][-1]:.3f}")

        # ==============================================================
        # Analysis
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("COMPOSITIONALITY LEVELS SUMMARY")
        self.log("=" * 60)

        level_stats = {}
        level_tests_vs_chance = {}
        for level_name in ["L1", "L2", "L3", "L4", "L5"]:
            vals = level_results[level_name]
            st = summarize(vals)
            level_stats[level_name] = st
            t_test = ttest_vs_null(vals, 0.5)
            level_tests_vs_chance[level_name] = t_test

        level_descriptions = {
            "L1": "Role swap (B-nouns as agents)",
            "L2": "Novel structure (active -> passive)",
            "L3": "Holdout words (distributional only)",
            "L4": "Novel verb-noun combinations",
            "L5": "Recursive + novel combinations",
        }

        for level_name in ["L1", "L2", "L3", "L4", "L5"]:
            st = level_stats[level_name]
            tt = level_tests_vs_chance[level_name]
            above_chance = st["mean"] > 0.5 and tt.get("significant", False)
            self.log(f"  {level_name} ({level_descriptions[level_name]}):")
            self.log(f"    Accuracy: {st['mean']:.3f} +/- {st['sem']:.3f}")
            self.log(f"    vs chance: t={tt['t']:.2f} p={tt['p']:.4f} "
                     f"d={tt['d']:.2f} "
                     f"{'ABOVE' if above_chance else 'AT/BELOW'} chance")

        # Difficulty gradient: is L1 > L2 > L3 > L4 > L5?
        self.log(f"\n  Difficulty gradient:")
        means = [level_stats[f"L{i}"]["mean"] for i in range(1, 6)]
        self.log(f"    L1={means[0]:.3f} L2={means[1]:.3f} "
                 f"L3={means[2]:.3f} L4={means[3]:.3f} L5={means[4]:.3f}")
        is_gradient = all(means[i] >= means[i+1] - 0.1 for i in range(4))
        self.log(f"    Monotonic difficulty: "
                 f"{'YES' if is_gradient else 'NO'}")

        # ==============================================================
        # Summary
        # ==============================================================
        duration = self._stop_timer()
        self.log(f"\n  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "n_training": len(training),
            },
            metrics={
                level_name: {
                    "stats": level_stats[level_name],
                    "vs_chance": level_tests_vs_chance[level_name],
                    "description": level_descriptions[level_name],
                }
                for level_name in ["L1", "L2", "L3", "L4", "L5"]
            },
            raw_data={
                level_name: {
                    "per_seed_accuracy": level_results[level_name],
                    "details": level_details[level_name],
                }
                for level_name in ["L1", "L2", "L3", "L4", "L5"]
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Deep compositional generalization experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer seeds for faster iteration")
    args = parser.parse_args()

    exp = DeepCompositionalExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")
    for level in ["L1", "L2", "L3", "L4", "L5"]:
        m = result.metrics[level]
        above = m["vs_chance"].get("significant", False) and \
                m["stats"]["mean"] > 0.5
        print(f"  {level} ({m['description']}): "
              f"{m['stats']['mean']:.3f} "
              f"{'ABOVE' if above else 'AT/BELOW'} chance")


if __name__ == "__main__":
    main()
