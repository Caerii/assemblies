"""
Relative Clause Processing Experiment

Tests whether the Assembly Calculus parser correctly assigns thematic
roles in sentences containing subject- and object-relative clauses,
where word order alone is insufficient and structural position determines
interpretation.

This is among the most-studied phenomena in psycholinguistics and
neurolinguistics:
- Object-relative clauses ("The dog that the cat chased runs") are
  harder than subject-relative clauses ("The dog that chased the cat
  runs") across all human languages tested (King & Just, 1991).
- fMRI and ERP studies show increased left inferior frontal activation
  (Broca's area) for object-relatives, interpreted as working memory
  load for maintaining the displaced filler.
- Assembly calculus should show the same asymmetry if it faithfully
  models cortical computation.

The key challenge: in "The dog that the cat chased runs", both "dog"
and "cat" are NOUNS with identical grounding features, but "cat" is
the AGENT of "chased" while "dog" is the PATIENT of "chased" AND the
AGENT of "runs". This requires structural position sensitivity that
grounding alone cannot resolve.

Hypotheses:

H1: Subject-relative clauses (SRC) — "The dog that chased the cat runs"
    The parser correctly assigns AGENT(dog, runs) and AGENT(dog, chased)
    with accuracy significantly above chance (>50%).

H2: Object-relative clauses (ORC) — "The dog that the cat chased runs"
    The parser correctly assigns PATIENT(dog, chased), AGENT(cat, chased),
    and AGENT(dog, runs) above chance.

H3: SRC > ORC asymmetry — Subject-relatives produce higher role
    assignment accuracy than object-relatives, matching the universal
    human processing difficulty asymmetry.

H4: Embedding depth — Role accuracy degrades with nesting depth
    (single RC > double RC), paralleling human sentence processing
    limits.

Statistical methodology:
- N_SEEDS independent random seeds per condition.
- One-sample t-test against chance (1/3 for 3-way role assignment).
- Paired t-test for SRC vs ORC comparison.
- Cohen's d effect sizes. Mean +/- SEM.

References:
- King & Just (1991). Individual differences in syntactic processing.
- Gibson (1998). Dependency Locality Theory.
- Mitropolsky & Papadimitriou (2025). Simulated Language Acquisition.
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
class RelClauseConfig:
    """Configuration for relative clause experiment."""
    n: int = 10000
    k: int = 100
    n_seeds: int = 5
    p: float = 0.05
    beta: float = 0.1
    rounds: int = 10


# -- Vocabulary ----------------------------------------------------------------

def _build_rc_vocab() -> Dict[str, GroundingContext]:
    """Vocabulary for relative clause experiments.

    All nouns share the ANIMAL category so grounding cannot distinguish
    agent from patient — only structural position can.
    """
    return {
        # Animate nouns (all share grounding — forces structural role assignment)
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "fish":   GroundingContext(visual=["FISH", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "mouse":  GroundingContext(visual=["MOUSE", "ANIMAL"]),
        # Transitive verbs
        "chased": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "saw":    GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "bit":    GroundingContext(motor=["BITING", "ACTION"]),
        "found":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        # Intransitive verbs (for main clause)
        "runs":   GroundingContext(motor=["RUNNING", "MOTION"]),
        "sleeps": GroundingContext(motor=["SLEEPING", "REST"]),
        "falls":  GroundingContext(motor=["FALLING", "MOTION"]),
        # Function words
        "the":    GroundingContext(),
        "that":   GroundingContext(),
    }


# -- Training sentences --------------------------------------------------------

def _build_training_sentences(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Training corpus: simple transitive and intransitive sentences.

    No relative clauses in training — the parser must generalize its
    learned role-assignment mechanism to the RC test structures.
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    # Simple transitive: DET NOUN VERB DET NOUN
    transitive_triples = [
        ("dog", "chased", "cat"), ("cat", "saw", "bird"),
        ("bird", "bit", "mouse"), ("horse", "found", "fish"),
        ("mouse", "chased", "bird"), ("fish", "saw", "dog"),
        ("dog", "found", "horse"), ("cat", "bit", "fish"),
        ("bird", "chased", "dog"), ("horse", "saw", "cat"),
        ("mouse", "found", "bird"), ("fish", "bit", "horse"),
    ]
    for subj, verb, obj in transitive_triples:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb), ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Simple intransitive: DET NOUN VERB
    intransitive_pairs = [
        ("dog", "runs"), ("cat", "sleeps"), ("bird", "falls"),
        ("horse", "runs"), ("mouse", "sleeps"), ("fish", "falls"),
    ]
    for subj, verb in intransitive_pairs:
        sentences.append(GroundedSentence(
            words=["the", subj, verb],
            contexts=[ctx("the"), ctx(subj), ctx(verb)],
            roles=[None, "agent", "action"],
        ))

    return sentences


# -- Test sentence structures --------------------------------------------------

def _build_src_tests() -> List[Dict[str, Any]]:
    """Subject-relative clause test sentences.

    Pattern: "The NOUN1 that VERB_RC the NOUN2 VERB_MAIN"
    Example: "The dog that chased the cat runs"
    Roles: NOUN1=AGENT(VERB_RC), NOUN1=AGENT(VERB_MAIN), NOUN2=PATIENT(VERB_RC)
    """
    tests = [
        {"words": ["the", "dog", "that", "chased", "the", "cat", "runs"],
         "expected": {"dog": "agent", "cat": "patient", "chased": "action",
                      "runs": "action"},
         "label": "SRC: dog that chased cat runs"},
        {"words": ["the", "cat", "that", "saw", "the", "bird", "sleeps"],
         "expected": {"cat": "agent", "bird": "patient", "saw": "action",
                      "sleeps": "action"},
         "label": "SRC: cat that saw bird sleeps"},
        {"words": ["the", "horse", "that", "found", "the", "mouse", "falls"],
         "expected": {"horse": "agent", "mouse": "patient", "found": "action",
                      "falls": "action"},
         "label": "SRC: horse that found mouse falls"},
        {"words": ["the", "bird", "that", "bit", "the", "fish", "runs"],
         "expected": {"bird": "agent", "fish": "patient", "bit": "action",
                      "runs": "action"},
         "label": "SRC: bird that bit fish runs"},
    ]
    return tests


def _build_orc_tests() -> List[Dict[str, Any]]:
    """Object-relative clause test sentences.

    Pattern: "The NOUN1 that the NOUN2 VERB_RC VERB_MAIN"
    Example: "The dog that the cat chased runs"
    Roles: NOUN1=PATIENT(VERB_RC), NOUN2=AGENT(VERB_RC), NOUN1=AGENT(VERB_MAIN)

    This is the hard case: NOUN1 appears first but is the PATIENT
    of the embedded verb, not the agent. Position-based heuristics fail.
    """
    tests = [
        {"words": ["the", "dog", "that", "the", "cat", "chased", "runs"],
         "expected": {"dog": "patient", "cat": "agent", "chased": "action",
                      "runs": "action"},
         "label": "ORC: dog that cat chased runs"},
        {"words": ["the", "bird", "that", "the", "cat", "saw", "falls"],
         "expected": {"bird": "patient", "cat": "agent", "saw": "action",
                      "falls": "action"},
         "label": "ORC: bird that cat saw falls"},
        {"words": ["the", "mouse", "that", "the", "horse", "found", "sleeps"],
         "expected": {"mouse": "patient", "horse": "agent", "found": "action",
                      "sleeps": "action"},
         "label": "ORC: mouse that horse found sleeps"},
        {"words": ["the", "fish", "that", "the", "bird", "bit", "runs"],
         "expected": {"fish": "patient", "bird": "agent", "bit": "action",
                      "runs": "action"},
         "label": "ORC: fish that bird bit runs"},
    ]
    return tests


# -- Measurement ---------------------------------------------------------------

def measure_role_accuracy(
    parser: EmergentParser, test: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse a test sentence and measure role assignment accuracy.

    Returns per-word correctness and overall accuracy.
    """
    result = parser.parse_recursive(test["words"])
    roles = result["roles"]
    expected = test["expected"]

    correct = 0
    total = 0
    per_word = {}
    for word, exp_role in expected.items():
        actual_role = roles.get(word, "").lower() if roles.get(word) else ""
        is_correct = actual_role == exp_role
        per_word[word] = {"expected": exp_role, "actual": actual_role,
                          "correct": is_correct}
        if exp_role in ("agent", "patient"):
            total += 1
            if is_correct:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "per_word": per_word,
            "categories": result["categories"]}


# -- Experiment ----------------------------------------------------------------

class RelativeClauseExperiment(ExperimentBase):
    """Test relative clause processing in assembly calculus."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="relative_clauses",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = RelClauseConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_rc_vocab()
        training = _build_training_sentences(vocab)
        src_tests = _build_src_tests()
        orc_tests = _build_orc_tests()

        chance_role = 1.0 / 3.0  # 3-way: agent, patient, action
        seeds = list(range(cfg.n_seeds))

        self.log(f"Training sentences: {len(training)}")
        self.log(f"SRC tests: {len(src_tests)}, ORC tests: {len(orc_tests)}")
        self.log(f"Chance accuracy: {chance_role:.3f}")
        self.log(f"Seeds: {cfg.n_seeds}")

        # ================================================================
        # H1: Subject-relative clause accuracy
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1: Subject-relative clause role assignment")
        self.log("=" * 60)

        src_accs_by_seed = []
        src_details = []
        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_accs = []
            for test in src_tests:
                result = measure_role_accuracy(parser, test)
                seed_accs.append(result["accuracy"])
                src_details.append(result)
            src_accs_by_seed.append(float(np.mean(seed_accs)))

        src_stats = summarize(src_accs_by_seed)
        src_test_stat = ttest_vs_null(src_accs_by_seed, chance_role)

        self.log(f"  SRC accuracy: {src_stats['mean']:.3f} +/- "
                 f"{src_stats['sem']:.3f}")
        self.log(f"  vs chance ({chance_role:.3f}): t={src_test_stat['t']:.2f} "
                 f"p={src_test_stat['p']:.4f} d={src_test_stat['d']:.2f} "
                 f"{'*' if src_test_stat['significant'] else ''}")

        # Log per-sentence details
        for i, test in enumerate(src_tests):
            accs = [src_details[j * len(src_tests) + i]["accuracy"]
                    for j in range(cfg.n_seeds)]
            self.log(f"    {test['label']}: {np.mean(accs):.3f}")

        # ================================================================
        # H2: Object-relative clause accuracy
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H2: Object-relative clause role assignment")
        self.log("=" * 60)

        orc_accs_by_seed = []
        orc_details = []
        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_accs = []
            for test in orc_tests:
                result = measure_role_accuracy(parser, test)
                seed_accs.append(result["accuracy"])
                orc_details.append(result)
            orc_accs_by_seed.append(float(np.mean(seed_accs)))

        orc_stats = summarize(orc_accs_by_seed)
        orc_test_stat = ttest_vs_null(orc_accs_by_seed, chance_role)

        self.log(f"  ORC accuracy: {orc_stats['mean']:.3f} +/- "
                 f"{orc_stats['sem']:.3f}")
        self.log(f"  vs chance ({chance_role:.3f}): t={orc_test_stat['t']:.2f} "
                 f"p={orc_test_stat['p']:.4f} d={orc_test_stat['d']:.2f} "
                 f"{'*' if orc_test_stat['significant'] else ''}")

        for i, test in enumerate(orc_tests):
            accs = [orc_details[j * len(orc_tests) + i]["accuracy"]
                    for j in range(cfg.n_seeds)]
            self.log(f"    {test['label']}: {np.mean(accs):.3f}")

        # ================================================================
        # H3: SRC > ORC asymmetry
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H3: SRC vs ORC asymmetry")
        self.log("=" * 60)

        asymmetry = paired_ttest(src_accs_by_seed, orc_accs_by_seed)
        self.log(f"  SRC mean: {src_stats['mean']:.3f}")
        self.log(f"  ORC mean: {orc_stats['mean']:.3f}")
        self.log(f"  SRC - ORC: {src_stats['mean'] - orc_stats['mean']:.3f}")
        self.log(f"  Paired test: t={asymmetry['t']:.2f} "
                 f"p={asymmetry['p']:.4f} d={asymmetry['d']:.2f} "
                 f"{'*' if asymmetry['significant'] else ''}")

        # ================================================================
        # H4: Embedding depth
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H4: Effect of embedding depth")
        self.log("=" * 60)

        # Single RC (already measured above, use SRC as baseline)
        # Double RC: "The dog that the cat that the bird saw chased runs"
        double_rc_tests = [
            {"words": ["the", "dog", "that", "the", "cat", "that",
                       "the", "bird", "saw", "chased", "runs"],
             "expected": {"dog": "patient", "cat": "agent",
                          "bird": "agent", "chased": "action",
                          "saw": "action", "runs": "action"},
             "label": "2RC: dog that cat that bird saw chased runs"},
        ]

        depth_1_accs = src_accs_by_seed  # already computed
        depth_2_accs = []
        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)
            seed_accs = []
            for test in double_rc_tests:
                result = measure_role_accuracy(parser, test)
                seed_accs.append(result["accuracy"])
            depth_2_accs.append(float(np.mean(seed_accs)))

        d2_stats = summarize(depth_2_accs)
        depth_test = paired_ttest(depth_1_accs, depth_2_accs)

        self.log(f"  Depth 1 (SRC): {src_stats['mean']:.3f}")
        self.log(f"  Depth 2 (2RC): {d2_stats['mean']:.3f}")
        self.log(f"  Degradation: {src_stats['mean'] - d2_stats['mean']:.3f}")
        self.log(f"  Paired test: t={depth_test['t']:.2f} "
                 f"p={depth_test['p']:.4f} d={depth_test['d']:.2f}")

        # ================================================================
        # Summary
        # ================================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("RELATIVE CLAUSE SUMMARY")
        self.log(f"  H1 (SRC > chance):    "
                 f"{'SUPPORTED' if src_test_stat['significant'] else 'NOT SUPPORTED'} "
                 f"(acc={src_stats['mean']:.3f}, d={src_test_stat['d']:.2f})")
        self.log(f"  H2 (ORC > chance):    "
                 f"{'SUPPORTED' if orc_test_stat['significant'] else 'NOT SUPPORTED'} "
                 f"(acc={orc_stats['mean']:.3f}, d={orc_test_stat['d']:.2f})")
        self.log(f"  H3 (SRC > ORC):       "
                 f"{'SUPPORTED' if asymmetry['significant'] else 'NOT SUPPORTED'} "
                 f"(d={asymmetry['d']:.2f})")
        self.log(f"  H4 (depth degrades):  "
                 f"{'SUPPORTED' if depth_test['significant'] else 'NOT SUPPORTED'} "
                 f"(d={depth_test['d']:.2f})")
        self.log(f"  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "chance_role": chance_role,
                "n_training": len(training),
            },
            metrics={
                "h1_src": {"stats": src_stats, "test": src_test_stat},
                "h2_orc": {"stats": orc_stats, "test": orc_test_stat},
                "h3_asymmetry": asymmetry,
                "h4_depth": {
                    "depth_1": src_stats,
                    "depth_2": d2_stats,
                    "test": depth_test,
                },
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Relative clause processing experiment")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    exp = RelativeClauseExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")
    for h in ["h1_src", "h2_orc"]:
        sig = result.metrics[h]["test"]["significant"]
        acc = result.metrics[h]["stats"]["mean"]
        print(f"  {h}: acc={acc:.3f} sig={sig}")


if __name__ == "__main__":
    main()
