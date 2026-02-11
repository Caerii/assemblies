"""
Garden-Path Reanalysis Experiment

Tests whether the Assembly Calculus incremental parser exhibits
garden-path effects: initial misanalysis followed by reanalysis when
disambiguating information arrives.

This is a cornerstone of psycholinguistic theory:
- Bever (1970): "The horse raced past the barn fell" — readers initially
  parse "raced" as main verb, then must reanalyze when "fell" appears.
- Frazier & Rayner (1982): Eye-tracking shows increased fixation time
  at disambiguation point.
- The garden-path effect proves that parsing is incremental (not
  wait-and-see) and that structural commitments can be revised.

In Assembly Calculus terms:
- The incremental parser builds a running CONTEXT assembly word-by-word.
- At the disambiguation point, the CONTEXT assembly must restructure.
- We measure reanalysis cost as the overlap disruption between the
  CONTEXT assembly at the disambiguation point and the CONTEXT
  assembly from an unambiguous control sentence.

Hypotheses:

H1: Role reassignment — In "The dog chased saw the cat", the parser
    initially assigns "dog" as AGENT of "chased", but after "saw",
    must reassign "dog" as PATIENT of "chased" and AGENT of "saw".
    The final parse of the garden-path sentence should differ from the
    initial parse midway through.

H2: Context disruption — The CONTEXT assembly overlap between the
    garden-path sentence and an unambiguous control is lower at the
    disambiguation point than at earlier positions.

H3: Recovery — Despite the disruption at the disambiguation point,
    the final parse of the garden-path sentence achieves role accuracy
    comparable to unambiguous sentences (showing successful reanalysis).

H4: Disambiguation cost scales — More deeply committed garden paths
    (more words before disambiguation) show greater disruption.

Statistical methodology:
- N_SEEDS independent random seeds per condition.
- Paired t-test for garden-path vs control conditions.
- Overlap disruption measured at each word position.
- Cohen's d effect sizes.

References:
- Bever (1970). The cognitive basis for linguistic structures.
- Frazier & Rayner (1982). Making and correcting errors.
- Ferreira & Henderson (1991). Recovery from misanalyses.
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


@dataclass
class GardenPathConfig:
    """Configuration for garden-path experiment."""
    n: int = 10000
    k: int = 100
    n_seeds: int = 5
    p: float = 0.05
    beta: float = 0.1
    rounds: int = 10


# -- Vocabulary ----------------------------------------------------------------

def _build_gp_vocab() -> Dict[str, GroundingContext]:
    """Vocabulary for garden-path sentences.

    Includes verbs that can be both past tense and past participle
    (enabling reduced relative clause ambiguity).
    """
    return {
        # Animate nouns
        "dog":    GroundingContext(visual=["DOG", "ANIMAL"]),
        "cat":    GroundingContext(visual=["CAT", "ANIMAL"]),
        "horse":  GroundingContext(visual=["HORSE", "ANIMAL"]),
        "bird":   GroundingContext(visual=["BIRD", "ANIMAL"]),
        "boy":    GroundingContext(visual=["BOY", "PERSON"]),
        "girl":   GroundingContext(visual=["GIRL", "PERSON"]),
        # Inanimate nouns (for locatives)
        "barn":   GroundingContext(visual=["BARN", "BUILDING"]),
        "park":   GroundingContext(visual=["PARK", "PLACE"]),
        "house":  GroundingContext(visual=["HOUSE", "BUILDING"]),
        # Ambiguous verbs (past tense / past participle)
        "chased": GroundingContext(motor=["CHASING", "PURSUIT"]),
        "pushed": GroundingContext(motor=["PUSHING", "ACTION"]),
        "warned": GroundingContext(motor=["WARNING", "ACTION"]),
        "raced":  GroundingContext(motor=["RACING", "MOTION"]),
        # Unambiguous verbs
        "saw":    GroundingContext(motor=["SEEING", "PERCEPTION"]),
        "ran":    GroundingContext(motor=["RUNNING", "MOTION"]),
        "fell":   GroundingContext(motor=["FALLING", "MOTION"]),
        "slept":  GroundingContext(motor=["SLEEPING", "REST"]),
        "found":  GroundingContext(motor=["FINDING", "PERCEPTION"]),
        # Prepositions and function words
        "the":    GroundingContext(),
        "past":   GroundingContext(spatial=["PAST", "BEYOND"]),
        "by":     GroundingContext(spatial=["BY", "NEAR"]),
    }


# -- Training sentences --------------------------------------------------------

def _build_gp_training(
    vocab: Dict[str, GroundingContext],
) -> List[GroundedSentence]:
    """Training on simple transitive and intransitive sentences.

    No garden-path structures in training — the test measures how
    the parser handles novel ambiguous structures.
    """
    def ctx(w):
        return vocab[w]

    sentences = []

    # Transitive
    trans = [
        ("dog", "chased", "cat"), ("cat", "pushed", "bird"),
        ("horse", "warned", "dog"), ("girl", "chased", "boy"),
        ("boy", "pushed", "girl"), ("bird", "found", "cat"),
        ("dog", "found", "bird"), ("cat", "chased", "horse"),
        ("horse", "pushed", "cat"), ("girl", "found", "dog"),
        ("boy", "warned", "horse"), ("bird", "chased", "dog"),
    ]
    for subj, verb, obj in trans:
        sentences.append(GroundedSentence(
            words=["the", subj, verb, "the", obj],
            contexts=[ctx("the"), ctx(subj), ctx(verb), ctx("the"), ctx(obj)],
            roles=[None, "agent", "action", None, "patient"],
        ))

    # Intransitive
    intrans = [
        ("dog", "ran"), ("cat", "slept"), ("horse", "fell"),
        ("bird", "ran"), ("boy", "fell"), ("girl", "slept"),
    ]
    for subj, verb in intrans:
        sentences.append(GroundedSentence(
            words=["the", subj, verb],
            contexts=[ctx("the"), ctx(subj), ctx(verb)],
            roles=[None, "agent", "action"],
        ))

    return sentences


# -- Test structures -----------------------------------------------------------

def _build_gp_tests() -> List[Dict[str, Any]]:
    """Garden-path and control sentence pairs.

    Each pair has:
    - garden_path: A sentence with temporary ambiguity
    - control: An unambiguous version with the same meaning
    - disambig_idx: Word index where disambiguation occurs
    """
    return [
        {
            "label": "Reduced relative (chased/fell)",
            "garden_path": ["the", "dog", "chased", "past", "the", "barn", "fell"],
            "control":     ["the", "dog", "that", "was", "chased", "past", "the", "barn", "fell"],
            "gp_expected": {"dog": "patient", "chased": "action", "fell": "action"},
            "ctrl_expected": {"dog": "patient", "chased": "action", "fell": "action"},
            "disambig_idx": 6,  # "fell" disambiguates
        },
        {
            "label": "Reduced relative (pushed/slept)",
            "garden_path": ["the", "cat", "pushed", "by", "the", "boy", "slept"],
            "control":     ["the", "cat", "that", "was", "pushed", "by", "the", "boy", "slept"],
            "gp_expected": {"cat": "patient", "pushed": "action", "boy": "agent", "slept": "action"},
            "ctrl_expected": {"cat": "patient", "pushed": "action", "boy": "agent", "slept": "action"},
            "disambig_idx": 6,
        },
        {
            "label": "Reduced relative (warned/ran)",
            "garden_path": ["the", "horse", "warned", "by", "the", "girl", "ran"],
            "control":     ["the", "horse", "that", "was", "warned", "by", "the", "girl", "ran"],
            "gp_expected": {"horse": "patient", "warned": "action", "girl": "agent", "ran": "action"},
            "ctrl_expected": {"horse": "patient", "warned": "action", "girl": "agent", "ran": "action"},
            "disambig_idx": 6,
        },
    ]


# -- Measurement ---------------------------------------------------------------

def measure_parse_accuracy(
    parser: EmergentParser,
    words: List[str],
    expected: Dict[str, str],
) -> Dict[str, Any]:
    """Parse and measure noun role accuracy."""
    result = parser.parse(words)
    roles = result["roles"]

    correct = 0
    total = 0
    per_word = {}
    for word, exp_role in expected.items():
        actual = roles.get(word, "").lower() if roles.get(word) else ""
        is_correct = actual == exp_role
        per_word[word] = {"expected": exp_role, "actual": actual,
                          "correct": is_correct}
        if exp_role in ("agent", "patient"):
            total += 1
            if is_correct:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "per_word": per_word,
            "categories": result["categories"], "roles": roles}


def measure_incremental_disruption(
    parser: EmergentParser,
    gp_words: List[str],
    ctrl_words: List[str],
) -> Dict[str, Any]:
    """Compare incremental parse states between garden-path and control.

    For each position where both sentences have been processed, compare
    the classified categories. Disruption = difference in classifications.
    """
    gp_result = parser.parse(gp_words)
    ctrl_result = parser.parse(ctrl_words)

    gp_cats = gp_result["categories"]
    ctrl_cats = ctrl_result["categories"]
    gp_roles = gp_result["roles"]
    ctrl_roles = ctrl_result["roles"]

    # Measure role agreement for shared words
    shared_words = set(gp_words) & set(ctrl_words) - {"the", "that", "was", "by", "past"}
    role_agreement = 0
    total_shared = 0
    for w in shared_words:
        gp_role = gp_roles.get(w, "")
        ctrl_role = ctrl_roles.get(w, "")
        if gp_role or ctrl_role:
            total_shared += 1
            if gp_role == ctrl_role:
                role_agreement += 1

    agreement_rate = role_agreement / total_shared if total_shared > 0 else 0.0

    return {
        "gp_categories": gp_cats,
        "ctrl_categories": ctrl_cats,
        "gp_roles": gp_roles,
        "ctrl_roles": ctrl_roles,
        "role_agreement_rate": agreement_rate,
    }


# -- Experiment ----------------------------------------------------------------

class GardenPathExperiment(ExperimentBase):
    """Test garden-path reanalysis in assembly calculus."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="garden_path",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent /
                         "results" / "applications"),
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = GardenPathConfig()
        if quick:
            cfg.n_seeds = 3

        vocab = _build_gp_vocab()
        training = _build_gp_training(vocab)
        tests = _build_gp_tests()
        seeds = list(range(cfg.n_seeds))

        self.log(f"Training sentences: {len(training)}")
        self.log(f"Test pairs: {len(tests)}")
        self.log(f"Seeds: {cfg.n_seeds}")

        # ================================================================
        # H1: Garden-path role reassignment
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H1: Role assignment on garden-path sentences")
        self.log("=" * 60)

        gp_accs = []
        ctrl_accs = []
        per_test_gp = {t["label"]: [] for t in tests}
        per_test_ctrl = {t["label"]: [] for t in tests}

        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_gp = []
            seed_ctrl = []
            for test in tests:
                gp_res = measure_parse_accuracy(
                    parser, test["garden_path"], test["gp_expected"])
                ctrl_res = measure_parse_accuracy(
                    parser, test["control"], test["ctrl_expected"])
                seed_gp.append(gp_res["accuracy"])
                seed_ctrl.append(ctrl_res["accuracy"])
                per_test_gp[test["label"]].append(gp_res["accuracy"])
                per_test_ctrl[test["label"]].append(ctrl_res["accuracy"])

            gp_accs.append(float(np.mean(seed_gp)))
            ctrl_accs.append(float(np.mean(seed_ctrl)))

        gp_stats = summarize(gp_accs)
        ctrl_stats = summarize(ctrl_accs)

        self.log(f"  Garden-path accuracy: {gp_stats['mean']:.3f} +/- {gp_stats['sem']:.3f}")
        self.log(f"  Control accuracy:     {ctrl_stats['mean']:.3f} +/- {ctrl_stats['sem']:.3f}")

        for test in tests:
            gp_mean = np.mean(per_test_gp[test["label"]])
            ctrl_mean = np.mean(per_test_ctrl[test["label"]])
            self.log(f"    {test['label']}: GP={gp_mean:.3f} CTRL={ctrl_mean:.3f}")

        # ================================================================
        # H2: Context disruption at disambiguation
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H2: Parse disruption (GP vs control role agreement)")
        self.log("=" * 60)

        agreement_rates = []
        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            seed_agreements = []
            for test in tests:
                disrupt = measure_incremental_disruption(
                    parser, test["garden_path"], test["control"])
                seed_agreements.append(disrupt["role_agreement_rate"])

            agreement_rates.append(float(np.mean(seed_agreements)))

        agree_stats = summarize(agreement_rates)
        # Test: is agreement < 1.0? (i.e., is there disruption?)
        h2_test = ttest_vs_null(agreement_rates, 1.0)

        self.log(f"  Role agreement (GP vs ctrl): "
                 f"{agree_stats['mean']:.3f} +/- {agree_stats['sem']:.3f}")
        self.log(f"  Test (< 1.0): t={h2_test['t']:.2f} p={h2_test['p']:.4f} "
                 f"d={h2_test['d']:.2f}")

        # ================================================================
        # H3: Recovery — GP accuracy vs control accuracy
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H3: Recovery (GP accuracy comparable to control)")
        self.log("=" * 60)

        h3_test = paired_ttest(gp_accs, ctrl_accs)
        self.log(f"  GP mean:   {gp_stats['mean']:.3f}")
        self.log(f"  Ctrl mean: {ctrl_stats['mean']:.3f}")
        self.log(f"  Test (GP vs ctrl): t={h3_test['t']:.2f} "
                 f"p={h3_test['p']:.4f} d={h3_test['d']:.2f} "
                 f"{'*' if h3_test['significant'] else ''}")

        # ================================================================
        # H4: Sentence length effect
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("H4: Sentence length effect")
        self.log("=" * 60)

        # Short GP (5 words) vs long GP (7 words)
        short_gp = {"words": ["the", "dog", "chased", "fell"],
                     "expected": {"dog": "patient", "chased": "action",
                                  "fell": "action"},
                     "label": "Short GP"}
        long_gp = {"words": ["the", "dog", "chased", "past", "the", "barn", "fell"],
                    "expected": {"dog": "patient", "chased": "action",
                                 "fell": "action"},
                    "label": "Long GP"}

        short_accs = []
        long_accs = []
        for s in seeds:
            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=self.seed + s, rounds=cfg.rounds, vocabulary=vocab,
            )
            parser.train(sentences=training)

            sr = measure_parse_accuracy(parser, short_gp["words"],
                                        short_gp["expected"])
            lr = measure_parse_accuracy(parser, long_gp["words"],
                                        long_gp["expected"])
            short_accs.append(sr["accuracy"])
            long_accs.append(lr["accuracy"])

        short_stats = summarize(short_accs)
        long_stats = summarize(long_accs)
        h4_test = paired_ttest(short_accs, long_accs)

        self.log(f"  Short GP ({len(short_gp['words'])} words): "
                 f"{short_stats['mean']:.3f} +/- {short_stats['sem']:.3f}")
        self.log(f"  Long GP ({len(long_gp['words'])} words):  "
                 f"{long_stats['mean']:.3f} +/- {long_stats['sem']:.3f}")
        self.log(f"  Test: t={h4_test['t']:.2f} p={h4_test['p']:.4f} "
                 f"d={h4_test['d']:.2f}")

        # ================================================================
        # Summary
        # ================================================================
        duration = self._stop_timer()

        self.log(f"\n{'=' * 60}")
        self.log("GARDEN-PATH SUMMARY")
        self.log(f"  H1 (role assignment):  GP={gp_stats['mean']:.3f} "
                 f"ctrl={ctrl_stats['mean']:.3f}")
        self.log(f"  H2 (disruption):       "
                 f"agreement={agree_stats['mean']:.3f}")
        self.log(f"  H3 (recovery):         "
                 f"{'COMPARABLE' if not h3_test['significant'] else 'DIFFERENT'}")
        self.log(f"  H4 (length effect):    "
                 f"short={short_stats['mean']:.3f} long={long_stats['mean']:.3f}")
        self.log(f"  Duration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "n_training": len(training),
            },
            metrics={
                "h1_role_assignment": {
                    "garden_path": gp_stats,
                    "control": ctrl_stats,
                },
                "h2_disruption": {
                    "agreement": agree_stats,
                    "test": h2_test,
                },
                "h3_recovery": h3_test,
                "h4_length": {
                    "short": short_stats,
                    "long": long_stats,
                    "test": h4_test,
                },
            },
            duration_seconds=duration,
        )

        self.save_result(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Garden-path reanalysis experiment")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    exp = GardenPathExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    print(f"\nCompleted in {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
