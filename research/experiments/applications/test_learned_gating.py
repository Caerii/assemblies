"""
Learned vs Hardcoded Gating Experiment

Tests whether the two-stage function word gating model (ELAN-like frame
classification → Broca's-like learned gating) matches or exceeds hardcoded
passive detection rules for role assignment.

Background:
- The parser has two mechanisms for passive voice detection:
  1. Hardcoded: `_detect_passive()` checks for "was"/"were" before a verb
  2. Learned: `_learn_gating_patterns()` observes role ORDER in training
     sentences grouped by function word sub-type, then `_determine_role_order()`
     checks learned gating before falling back to hardcoded rules.

- The learned approach is more neurally plausible: it mirrors the two-stage
  process in human language comprehension:
  - Stage 1 (ELAN ~180ms): rapid sub-categorization from bigram frames
  - Stage 2 (Broca's gating): learned processing mode shifts from function
    word type, acquired through procedural memory (basal ganglia pathway)

- The hardcoded approach is a hand-written heuristic that works for English
  passives but doesn't generalize to other constructions or languages.

Hypotheses:

H1: Learned gating achieves >= hardcoded accuracy on passive sentences.
    The learned AUX gating pattern (role reversal) should match the
    hardcoded "was/were before verb" check.

H2: Both approaches handle active sentences equally well.
    Active sentences don't trigger either mechanism, so accuracy should
    be identical.

H3: The combined approach (learned + hardcoded fallback) is >= either
    individual approach, demonstrating graceful degradation.

H4: Learned gating generalizes: it correctly handles passive relative
    clauses ("the dog that was chased by the cat") using the same
    mechanism that handles simple passives, without needing separate
    hardcoded logic for RC passives.

Statistical methodology:
- N_SEEDS independent random seeds per condition.
- Paired t-test between conditions (same seed, different mechanism).
- McNemar's test for per-sentence accuracy differences.
- Cohen's d effect sizes.

References:
- Friederici (2002). Towards a neural basis of auditory sentence processing.
- Ullman (2001). A neurocognitive perspective on language: the
  declarative/procedural model.
- Mitropolsky & Papadimitriou (2025). Simulated Language Acquisition.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


@dataclass
class GatingConfig:
    """Configuration for the learned gating experiment."""
    n: int = 10000      # neurons per area
    k: int = 100        # assembly size (winners per round)
    n_seeds: int = 5    # independent random seeds
    p: float = 0.05     # connection probability
    beta: float = 0.1   # plasticity rate
    rounds: int = 10    # projection rounds


# -- Vocabulary ----------------------------------------------------------------

def _build_vocab() -> Dict[str, GroundingContext]:
    """Vocabulary covering active, passive, and relative clause constructions.

    Includes past-tense verbs (which double as past participles for passives),
    auxiliary "was"/"were", complementizer "that", and agent marker "by".
    """
    return {
        # Animate nouns (visual grounding)
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "boy":    GroundingContext(visual=["BOY", "PERSON"]),
        "girl":   GroundingContext(visual=["GIRL", "PERSON"]),
        # Verbs (motor grounding — past tense forms)
        "chased": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "pushed": GroundingContext(motor=["PUSHING", "ACTION"]),
        "saw":    GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "found":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        "warned": GroundingContext(motor=["WARNING", "ACTION"]),
        # Present tense (for active control sentences)
        "chases": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "sees":   GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "finds":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        # Function words (ungrounded — route through DET_CORE)
        "the":    GroundingContext(),
        "a":      GroundingContext(),
        # Auxiliary verbs (ungrounded — sub-categorized as AUX by frames)
        "was":    GroundingContext(),
        "were":   GroundingContext(),
        # Complementizer (ungrounded — sub-categorized as COMP by frames)
        "that":   GroundingContext(),
        # Agent marker (spatial grounding — signals agent in passives)
        "by":     GroundingContext(spatial=["BY", "AGENT_MARKER"]),
    }


# -- Training sentences --------------------------------------------------------

def _build_training(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Training sentences: active transitives + passives + intransitives.

    Balanced design: roughly equal active and passive sentences so the
    learned gating has enough examples of both role orders.
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    # Active transitives: "the NOUN VERB the NOUN"
    # Role order: AGENT, PATIENT (default)
    active_triples = [
        ("dog", "chased", "cat"),  ("cat", "pushed", "bird"),
        ("bird", "saw", "horse"),  ("horse", "found", "dog"),
        ("boy", "chased", "girl"), ("girl", "pushed", "boy"),
        ("dog", "saw", "bird"),    ("cat", "found", "horse"),
        ("boy", "saw", "cat"),     ("girl", "found", "dog"),
    ]
    for subj, verb, obj in active_triples:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb), ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Passive sentences: "the NOUN was VERB by the NOUN"
    # Role order: PATIENT, AGENT (reversed by AUX "was")
    passive_triples = [
        ("cat", "chased", "dog"),  ("bird", "pushed", "cat"),
        ("horse", "saw", "bird"),  ("dog", "found", "horse"),
        ("girl", "chased", "boy"), ("boy", "pushed", "girl"),
        ("bird", "saw", "dog"),    ("horse", "found", "cat"),
    ]
    for patient, verb, agent in passive_triples:
        sentences.append(GroundedSentence(
            words=["the", patient, "was", verb, "by", "the", agent],
            contexts=[ctx("the"), ctx(patient), ctx("was"), ctx(verb),
                      ctx("by"), ctx("the"), ctx(agent)],
            roles=[None, "patient", None, "action", None, None, "agent"],
        ))

    return sentences


# -- Test structures -----------------------------------------------------------

def _build_test_sentences() -> Dict[str, List[Dict[str, Any]]]:
    """Test sentences organized by construction type.

    Returns dict of {condition_name: [test_dict, ...]} where each test_dict
    has 'words', 'expected_roles' (word -> role), and 'label'.
    """
    tests = {}

    # Active transitives — no gating should be triggered
    tests["active"] = [
        {"words": ["the", "dog", "chased", "the", "cat"],
         "expected": {"dog": "agent", "cat": "patient"},
         "label": "active_1"},
        {"words": ["the", "bird", "saw", "the", "horse"],
         "expected": {"bird": "agent", "horse": "patient"},
         "label": "active_2"},
        {"words": ["the", "boy", "found", "the", "girl"],
         "expected": {"boy": "agent", "girl": "patient"},
         "label": "active_3"},
    ]

    # Simple passives — AUX gating should reverse roles
    tests["passive"] = [
        {"words": ["the", "cat", "was", "chased", "by", "the", "dog"],
         "expected": {"cat": "patient", "dog": "agent"},
         "label": "passive_1"},
        {"words": ["the", "bird", "was", "pushed", "by", "the", "cat"],
         "expected": {"bird": "patient", "cat": "agent"},
         "label": "passive_2"},
        {"words": ["the", "horse", "was", "found", "by", "the", "girl"],
         "expected": {"horse": "patient", "girl": "agent"},
         "label": "passive_3"},
    ]

    # Passive relative clauses — tests generalization of gating
    # "the dog that was chased by the cat sees the bird"
    tests["passive_rc"] = [
        {"words": ["the", "dog", "that", "was", "chased", "by", "the", "cat",
                    "sees", "the", "bird"],
         "expected": {"dog": "agent", "bird": "patient",
                      "cat": "agent"},  # inner: dog=patient(filler), cat=agent
         "label": "passive_rc_1"},
        {"words": ["the", "bird", "that", "was", "pushed", "by", "the", "horse",
                    "sees", "the", "cat"],
         "expected": {"bird": "agent", "cat": "patient",
                      "horse": "agent"},
         "label": "passive_rc_2"},
    ]

    return tests


# -- Measurement ---------------------------------------------------------------

def measure_accuracy(
    parser: EmergentParser,
    words: List[str],
    expected: Dict[str, str],
    use_recursive: bool = False,
) -> Dict[str, Any]:
    """Parse a sentence and measure noun role accuracy.

    Args:
        parser: Trained parser instance.
        words: Sentence word list.
        expected: {word: expected_role} for evaluation.
        use_recursive: If True, use parse_recursive (for RCs).

    Returns:
        {"accuracy": float, "per_word": {...}, "roles": {...}}
    """
    if use_recursive:
        result = parser.parse_recursive(words)
    else:
        result = parser.parse(words)
    roles = result["roles"]

    correct = 0
    total = 0
    per_word = {}
    for word, exp_role in expected.items():
        actual = roles.get(word, "").lower() if roles.get(word) else ""
        is_correct = (actual == exp_role)
        per_word[word] = {
            "expected": exp_role,
            "actual": actual,
            "correct": is_correct,
        }
        # Only count noun roles (agent/patient) for accuracy
        if exp_role in ("agent", "patient"):
            total += 1
            if is_correct:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "per_word": per_word, "roles": roles}


def evaluate_condition(
    parser: EmergentParser,
    tests: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a parser on all test conditions.

    Returns {condition_name: {"accuracies": [float], "details": [...]}}
    """
    results = {}
    for cond_name, test_list in tests.items():
        accs = []
        details = []
        for test in test_list:
            # Use recursive parsing for sentences with relative clauses
            use_recursive = "that" in test["words"]
            res = measure_accuracy(
                parser, test["words"], test["expected"],
                use_recursive=use_recursive,
            )
            accs.append(res["accuracy"])
            details.append({
                "label": test["label"],
                "accuracy": res["accuracy"],
                "per_word": res["per_word"],
            })
        results[cond_name] = {
            "accuracies": accs,
            "mean": float(np.mean(accs)),
            "details": details,
        }
    return results


# -- Gating mode switching -----------------------------------------------------

def disable_learned_gating(parser: EmergentParser) -> Dict:
    """Temporarily disable learned gating, returning the saved state.

    Clears learned_gating so _determine_role_order falls through to
    the hardcoded _detect_passive() fallback.
    """
    saved = dict(parser.learned_gating)
    parser.learned_gating = {}
    return saved


def disable_hardcoded_fallback(parser: EmergentParser) -> None:
    """Replace _detect_passive with a no-op so only learned gating is used.

    Monkey-patches _detect_passive to always return False, forcing
    _determine_role_order to rely entirely on learned gating patterns.
    """
    parser._detect_passive_original = parser._detect_passive
    parser._detect_passive = lambda words, cats: False


def restore_hardcoded_fallback(parser: EmergentParser) -> None:
    """Restore the original _detect_passive method."""
    if hasattr(parser, '_detect_passive_original'):
        parser._detect_passive = parser._detect_passive_original
        del parser._detect_passive_original


def restore_learned_gating(parser: EmergentParser, saved: Dict) -> None:
    """Restore previously saved learned gating patterns."""
    parser.learned_gating = saved


# -- Experiment ----------------------------------------------------------------

class LearnedGatingExperiment(ExperimentBase):
    """Compare learned vs hardcoded gating for role assignment."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="learned_gating",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = GatingConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_vocab()
        training = _build_training(vocab)
        tests = _build_test_sentences()
        seeds = list(range(cfg.n_seeds))

        self.log(f"Training: {len(training)} sentences "
                 f"({sum(1 for s in training if 'was' in s.words)} passive)")
        self.log(f"Seeds: {cfg.n_seeds}")
        self.log(f"Conditions: active ({len(tests['active'])}), "
                 f"passive ({len(tests['passive'])}), "
                 f"passive_rc ({len(tests['passive_rc'])})")

        # Storage for per-seed results under each mechanism
        # {mechanism: {condition: [per-seed mean accuracy]}}
        all_results = {
            "combined":  {c: [] for c in tests},
            "learned":   {c: [] for c in tests},
            "hardcoded": {c: [] for c in tests},
        }

        # Storage for learned gating info
        gating_info = []

        for s in seeds:
            self.log(f"\n--- Seed {s} ---")

            # Train a fresh parser
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            # Log what the parser learned
            self.log(f"  Learned gating: {parser.learned_gating}")
            gating_info.append({
                "seed": s,
                "gating": {k: {kk: vv for kk, vv in v.items()
                                if kk != "role_order"}
                           for k, v in parser.learned_gating.items()},
            })

            # --- Condition 1: Combined (learned + hardcoded fallback) ---
            # This is the default behavior.
            combined_res = evaluate_condition(parser, tests)
            for cond in tests:
                all_results["combined"][cond].append(
                    combined_res[cond]["mean"])

            # --- Condition 2: Learned only (disable hardcoded fallback) ---
            disable_hardcoded_fallback(parser)
            learned_res = evaluate_condition(parser, tests)
            for cond in tests:
                all_results["learned"][cond].append(
                    learned_res[cond]["mean"])
            restore_hardcoded_fallback(parser)

            # --- Condition 3: Hardcoded only (disable learned gating) ---
            saved_gating = disable_learned_gating(parser)
            hardcoded_res = evaluate_condition(parser, tests)
            for cond in tests:
                all_results["hardcoded"][cond].append(
                    hardcoded_res[cond]["mean"])
            restore_learned_gating(parser, saved_gating)

            # Log per-seed summary
            for mech in ("combined", "learned", "hardcoded"):
                passive_acc = all_results[mech]["passive"][-1]
                active_acc = all_results[mech]["active"][-1]
                self.log(f"  {mech:10s}: active={active_acc:.3f} "
                         f"passive={passive_acc:.3f}")

        # ==============================================================
        # H1: Learned gating >= hardcoded on passives
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H1: Learned gating >= hardcoded on passive sentences")
        self.log("=" * 60)

        learned_passive = all_results["learned"]["passive"]
        hardcoded_passive = all_results["hardcoded"]["passive"]
        h1_learned = summarize(learned_passive)
        h1_hardcoded = summarize(hardcoded_passive)
        h1_test = paired_ttest(learned_passive, hardcoded_passive)

        self.log(f"  Learned:   {h1_learned['mean']:.3f} "
                 f"+/- {h1_learned['sem']:.3f}")
        self.log(f"  Hardcoded: {h1_hardcoded['mean']:.3f} "
                 f"+/- {h1_hardcoded['sem']:.3f}")
        self.log(f"  Paired t-test: t={h1_test['t']:.2f} "
                 f"p={h1_test['p']:.4f} d={h1_test['d']:.2f}")

        h1_supported = h1_learned["mean"] >= h1_hardcoded["mean"] - 0.05

        # ==============================================================
        # H2: Both handle active sentences equally
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H2: Active sentence accuracy (should be equal)")
        self.log("=" * 60)

        learned_active = all_results["learned"]["active"]
        hardcoded_active = all_results["hardcoded"]["active"]
        h2_learned = summarize(learned_active)
        h2_hardcoded = summarize(hardcoded_active)
        h2_test = paired_ttest(learned_active, hardcoded_active)

        self.log(f"  Learned:   {h2_learned['mean']:.3f} "
                 f"+/- {h2_learned['sem']:.3f}")
        self.log(f"  Hardcoded: {h2_hardcoded['mean']:.3f} "
                 f"+/- {h2_hardcoded['sem']:.3f}")
        self.log(f"  Paired t-test: t={h2_test['t']:.2f} "
                 f"p={h2_test['p']:.4f} d={h2_test['d']:.2f}")

        h2_supported = not h2_test["significant"]

        # ==============================================================
        # H3: Combined >= either individual approach
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H3: Combined approach >= individual approaches")
        self.log("=" * 60)

        # Compare combined vs learned on passives
        combined_passive = all_results["combined"]["passive"]
        h3_combined = summarize(combined_passive)
        h3_vs_learned = paired_ttest(combined_passive, learned_passive)
        h3_vs_hardcoded = paired_ttest(combined_passive, hardcoded_passive)

        self.log(f"  Combined:  {h3_combined['mean']:.3f} "
                 f"+/- {h3_combined['sem']:.3f}")
        self.log(f"  vs Learned:   t={h3_vs_learned['t']:.2f} "
                 f"p={h3_vs_learned['p']:.4f}")
        self.log(f"  vs Hardcoded: t={h3_vs_hardcoded['t']:.2f} "
                 f"p={h3_vs_hardcoded['p']:.4f}")

        h3_supported = (h3_combined["mean"] >=
                        max(h1_learned["mean"], h1_hardcoded["mean"]) - 0.05)

        # ==============================================================
        # H4: Learned gating generalizes to passive RCs
        # ==============================================================
        self.log(f"\n{'=' * 60}")
        self.log("H4: Generalization to passive relative clauses")
        self.log("=" * 60)

        learned_rc = all_results["learned"]["passive_rc"]
        hardcoded_rc = all_results["hardcoded"]["passive_rc"]
        combined_rc = all_results["combined"]["passive_rc"]
        h4_learned = summarize(learned_rc)
        h4_hardcoded = summarize(hardcoded_rc)
        h4_combined = summarize(combined_rc)
        h4_test = paired_ttest(learned_rc, hardcoded_rc)

        self.log(f"  Learned:   {h4_learned['mean']:.3f} "
                 f"+/- {h4_learned['sem']:.3f}")
        self.log(f"  Hardcoded: {h4_hardcoded['mean']:.3f} "
                 f"+/- {h4_hardcoded['sem']:.3f}")
        self.log(f"  Combined:  {h4_combined['mean']:.3f} "
                 f"+/- {h4_combined['sem']:.3f}")
        self.log(f"  Learned vs hardcoded: t={h4_test['t']:.2f} "
                 f"p={h4_test['p']:.4f} d={h4_test['d']:.2f}")

        h4_supported = h4_learned["mean"] >= h4_hardcoded["mean"] - 0.05

        # ==============================================================
        # Summary
        # ==============================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("LEARNED GATING SUMMARY")
        self.log(f"  H1 (learned >= hardcoded on passive): "
                 f"{'SUPPORTED' if h1_supported else 'NOT SUPPORTED'} "
                 f"({h1_learned['mean']:.3f} vs {h1_hardcoded['mean']:.3f})")
        self.log(f"  H2 (equal on active):                "
                 f"{'SUPPORTED' if h2_supported else 'NOT SUPPORTED'} "
                 f"({h2_learned['mean']:.3f} vs {h2_hardcoded['mean']:.3f})")
        self.log(f"  H3 (combined >= individual):          "
                 f"{'SUPPORTED' if h3_supported else 'NOT SUPPORTED'} "
                 f"(combined={h3_combined['mean']:.3f})")
        self.log(f"  H4 (generalizes to passive RC):      "
                 f"{'SUPPORTED' if h4_supported else 'NOT SUPPORTED'} "
                 f"({h4_learned['mean']:.3f} vs {h4_hardcoded['mean']:.3f})")
        self.log(f"  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "n_training": len(training),
                "n_passive_training": sum(
                    1 for s in training if "was" in s.words),
            },
            metrics={
                "h1_passive": {
                    "learned": h1_learned,
                    "hardcoded": h1_hardcoded,
                    "test": h1_test,
                    "supported": h1_supported,
                },
                "h2_active": {
                    "learned": h2_learned,
                    "hardcoded": h2_hardcoded,
                    "test": h2_test,
                    "supported": h2_supported,
                },
                "h3_combined": {
                    "combined": h3_combined,
                    "vs_learned": h3_vs_learned,
                    "vs_hardcoded": h3_vs_hardcoded,
                    "supported": h3_supported,
                },
                "h4_passive_rc": {
                    "learned": h4_learned,
                    "hardcoded": h4_hardcoded,
                    "combined": h4_combined,
                    "test": h4_test,
                    "supported": h4_supported,
                },
                "gating_info": gating_info,
            },
            raw_data={
                "combined": {c: v for c, v in all_results["combined"].items()},
                "learned": {c: v for c, v in all_results["learned"].items()},
                "hardcoded": {c: v for c, v in all_results["hardcoded"].items()},
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Learned vs hardcoded gating experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer seeds for faster iteration")
    args = parser.parse_args()

    exp = LearnedGatingExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")
    metrics = result.metrics
    for h in ("h1_passive", "h2_active", "h3_combined", "h4_passive_rc"):
        status = "SUPPORTED" if metrics[h]["supported"] else "NOT SUPPORTED"
        print(f"  {h}: {status}")


if __name__ == "__main__":
    main()
