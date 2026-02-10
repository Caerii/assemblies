"""
Compositional Generalization Experiment

Tests whether the assembly calculus parser can generalize compositionally --
correctly parsing novel agent-verb-patient combinations never seen during
training.

Hypotheses:
    H1: Classification of novel combinations matches training accuracy (within 15%).
    H2: Role assignment on novel combinations exceeds chance (>50%).
    H3: Novel nouns in trained syntactic positions are classified correctly.
    H4: The generalization gap (train - test accuracy) is smaller than for
        memorization-based systems.

Protocol:
    1. Build vocabulary: 10 nouns, 6 verbs, 4 determiners.
    2. Training: nouns 1-5 as agents, nouns 6-10 as patients.
    3. Test: nouns 6-10 as agents, nouns 1-5 as patients (NEVER in training).
    4. Per seed: create parser, train, measure train/test accuracy, gap.
    5. Statistical analysis: paired_ttest(train, test), ttest_vs_null(role, 0.5).

References:
    - Papadimitriou et al., PNAS 117(25):14464-14472, 2020
    - Mitropolsky & Papadimitriou, "Simulated Language Acquisition", 2025
    - Lake & Baroni, "Generalization without Systematicity", ICML 2018
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from research.experiments.base import (
    ExperimentBase, ExperimentResult, summarize, ttest_vs_null, paired_ttest,
)
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class CompositionalConfig:
    """Configuration for the compositional generalization experiment."""
    n: int = 10000          # neurons per area
    k: int = 100            # assembly size (winners per area)
    p: float = 0.05         # connection probability
    beta: float = 0.1       # Hebbian plasticity rate
    rounds: int = 10        # projection rounds per operation
    n_seeds: int = 5        # independent random seeds
    n_training_sents: int = 30   # training sentences
    n_test_sents: int = 15       # test sentences (novel combinations)


# ======================================================================
# Vocabulary
# ======================================================================

_NOUNS = [
    ("dog",   GroundingContext(visual=["DOG", "ANIMAL"])),
    ("cat",   GroundingContext(visual=["CAT", "ANIMAL"])),
    ("bird",  GroundingContext(visual=["BIRD", "ANIMAL"])),
    ("boy",   GroundingContext(visual=["BOY", "PERSON"])),
    ("girl",  GroundingContext(visual=["GIRL", "PERSON"])),
    ("ball",  GroundingContext(visual=["BALL", "OBJECT"])),
    ("book",  GroundingContext(visual=["BOOK", "OBJECT"])),
    ("food",  GroundingContext(visual=["FOOD", "OBJECT"])),
    ("table", GroundingContext(visual=["TABLE", "FURNITURE"])),
    ("car",   GroundingContext(visual=["CAR", "OBJECT"])),
]
_VERBS = [
    ("chases", GroundingContext(motor=["CHASING", "PURSUIT"])),
    ("sees",   GroundingContext(motor=["SEEING", "PERCEPTION"])),
    ("finds",  GroundingContext(motor=["FINDING", "PERCEPTION"])),
    ("eats",   GroundingContext(motor=["EATING", "CONSUMPTION"])),
    ("plays",  GroundingContext(motor=["PLAYING", "ACTION"])),
    ("reads",  GroundingContext(motor=["READING", "COGNITION"])),
]
_DETS = [
    ("the",  GroundingContext()),
    ("a",    GroundingContext()),
    ("one",  GroundingContext()),
    ("this", GroundingContext()),
]

# Expected POS for evaluation
_EXPECTED_CATEGORY: Dict[str, str] = {}
for _w, _ in _NOUNS:
    _EXPECTED_CATEGORY[_w] = "NOUN"
for _w, _ in _VERBS:
    _EXPECTED_CATEGORY[_w] = "VERB"
for _w, _ in _DETS:
    _EXPECTED_CATEGORY[_w] = "DET"


def _build_experiment_vocab() -> Dict[str, GroundingContext]:
    """Build the complete vocabulary for the experiment."""
    return {w: ctx for w, ctx in _NOUNS + _VERBS + _DETS}


# ======================================================================
# Sentence Generation
# ======================================================================

def _make_transitive(det1: str, agent: str, verb: str,
                     det2: str, patient: str,
                     vocab: Dict[str, GroundingContext]) -> GroundedSentence:
    """Create a grounded transitive sentence: DET NOUN VERB DET NOUN."""
    words = [det1, agent, verb, det2, patient]
    return GroundedSentence(
        words=words,
        contexts=[vocab[w] for w in words],
        roles=[None, "agent", "action", None, "patient"],
    )


def generate_split_sentences(
    vocab: Dict[str, GroundingContext], n_train: int, n_test: int, seed: int,
) -> Tuple[List[GroundedSentence], List[GroundedSentence]]:
    """Generate compositionally split training and test sentences.

    Training: nouns 0-4 as agents, nouns 5-9 as patients.
    Test:     nouns 5-9 as agents, nouns 0-4 as patients (novel combos).
    """
    rng = np.random.default_rng(seed)
    noun_words = [w for w, _ in _NOUNS]
    verb_words = [w for w, _ in _VERBS]
    det_words = [w for w, _ in _DETS]

    train_agents, train_patients = noun_words[:5], noun_words[5:]
    test_agents, test_patients = noun_words[5:], noun_words[:5]

    def _gen(agents, patients, n_sents):
        sents = []
        for _ in range(n_sents):
            ag = agents[rng.integers(len(agents))]
            vb = verb_words[rng.integers(len(verb_words))]
            pt = patients[rng.integers(len(patients))]
            while pt == ag:
                pt = patients[rng.integers(len(patients))]
            d1 = det_words[rng.integers(len(det_words))]
            d2 = det_words[rng.integers(len(det_words))]
            sents.append(_make_transitive(d1, ag, vb, d2, pt, vocab))
        return sents

    return _gen(train_agents, train_patients, n_train), \
           _gen(test_agents, test_patients, n_test)


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_parse(parse_result: dict, sent: GroundedSentence) -> Dict[str, Any]:
    """Evaluate a single parse against ground truth roles and categories."""
    cats = parse_result.get("categories", {})
    roles = parse_result.get("roles", {})
    n_cls, n_cls_ok, n_role, n_role_ok = 0, 0, 0, 0

    for word in sent.words:
        exp = _EXPECTED_CATEGORY.get(word)
        if exp is not None:
            n_cls += 1
            if cats.get(word, "UNKNOWN") == exp:
                n_cls_ok += 1

    for word, gold in zip(sent.words, sent.roles):
        if gold in ("agent", "patient"):
            n_role += 1
            expected = "AGENT" if gold == "agent" else "PATIENT"
            if roles.get(word) == expected:
                n_role_ok += 1

    return {
        "classification_accuracy": n_cls_ok / n_cls if n_cls else 0.0,
        "role_accuracy": n_role_ok / n_role if n_role else 0.0,
        "n_classified": n_cls, "n_roles": n_role,
    }


def evaluate_sentence_set(parser: EmergentParser,
                          sentences: List[GroundedSentence]) -> Dict[str, float]:
    """Evaluate parser on a set of sentences, returning aggregate metrics."""
    cls_ok = cls_tot = role_ok = role_tot = 0
    for sent in sentences:
        ev = evaluate_parse(parser.parse(sent.words), sent)
        cls_ok += int(ev["classification_accuracy"] * ev["n_classified"])
        cls_tot += ev["n_classified"]
        role_ok += int(ev["role_accuracy"] * ev["n_roles"])
        role_tot += ev["n_roles"]
    return {
        "classification_accuracy": cls_ok / cls_tot if cls_tot else 0.0,
        "role_accuracy": role_ok / role_tot if role_tot else 0.0,
        "n_sentences": len(sentences),
    }


# ======================================================================
# Experiment
# ======================================================================

class CompositionalGeneralizationExperiment(ExperimentBase):
    """Test compositional generalization in the emergent NEMO parser."""

    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="compositional_generalization",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "applications",
            verbose=verbose,
        )

    def run(self, quick: bool = False, **kwargs) -> ExperimentResult:
        """Run the full compositional generalization experiment."""
        self._start_timer()
        cfg = CompositionalConfig()
        if quick:
            cfg.n, cfg.k, cfg.n_seeds = 5000, 50, 2
            cfg.n_training_sents, cfg.n_test_sents, cfg.rounds = 15, 8, 5

        vocab = _build_experiment_vocab()
        self.log("=" * 60)
        self.log("Compositional Generalization Experiment")
        self.log("=" * 60)
        self.log(f"  Vocab: {len(vocab)} words | n={cfg.n} k={cfg.k} | "
                 f"seeds={cfg.n_seeds} | quick={quick}")

        seed_results: List[Dict[str, Any]] = []
        train_cls, test_cls = [], []
        train_role, test_role = [], []
        gaps: List[float] = []

        for i in range(cfg.n_seeds):
            seed = self.seed + i * 1000
            self.log(f"\n--- Seed {i+1}/{cfg.n_seeds} (seed={seed}) ---")

            train_sents, test_sents = generate_split_sentences(
                vocab, cfg.n_training_sents, cfg.n_test_sents, seed=seed)
            self.log(f"  {len(train_sents)} train, {len(test_sents)} test sents")
            if train_sents:
                self.log(f"  Train ex: {' '.join(train_sents[0].words)}")
            if test_sents:
                self.log(f"  Test  ex: {' '.join(test_sents[0].words)}")

            parser = EmergentParser(
                n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
                seed=seed, rounds=cfg.rounds, vocabulary=vocab)
            parser.train(sentences=train_sents)
            self.log("  Trained")

            tr = evaluate_sentence_set(parser, train_sents)
            te = evaluate_sentence_set(parser, test_sents)
            gap = tr["classification_accuracy"] - te["classification_accuracy"]
            self.log(f"  Train cls={tr['classification_accuracy']:.3f} "
                     f"role={tr['role_accuracy']:.3f}")
            self.log(f"  Test  cls={te['classification_accuracy']:.3f} "
                     f"role={te['role_accuracy']:.3f}  gap={gap:+.3f}")

            train_cls.append(tr["classification_accuracy"])
            test_cls.append(te["classification_accuracy"])
            train_role.append(tr["role_accuracy"])
            test_role.append(te["role_accuracy"])
            gaps.append(gap)
            seed_results.append({
                "seed": seed, "train": tr, "test": te, "gap": gap,
                "train_ex": [" ".join(s.words) for s in train_sents[:3]],
                "test_ex": [" ".join(s.words) for s in test_sents[:3]],
            })

        # -- Statistical analysis --
        self.log("\n" + "=" * 60)
        self.log("STATISTICAL ANALYSIS")
        self.log("=" * 60)

        tr_cls_s = summarize(train_cls)
        te_cls_s = summarize(test_cls)
        tr_role_s = summarize(train_role)
        te_role_s = summarize(test_role)
        gap_s = summarize(gaps)

        # H1: gap < 15%
        h1 = paired_ttest(train_cls, test_cls)
        h1_ok = abs(gap_s["mean"]) < 0.15
        self.log(f"\nH1: Classification gap < 15%")
        self.log(f"  Train={tr_cls_s['mean']:.3f}+/-{tr_cls_s['sem']:.3f}  "
                 f"Test={te_cls_s['mean']:.3f}+/-{te_cls_s['sem']:.3f}  "
                 f"Gap={gap_s['mean']:+.3f}")
        self.log(f"  t={h1['t']:.2f} p={h1['p']:.4f} d={h1['d']:.2f}  "
                 f"-> {'SUPPORTED' if h1_ok else 'REJECTED'}")

        # H2: role > chance (50%)
        h2 = ttest_vs_null(test_role, 0.5)
        h2_ok = te_role_s["mean"] > 0.5 and h2["significant"]
        self.log(f"\nH2: Role accuracy > chance (50%)")
        self.log(f"  Test role={te_role_s['mean']:.3f}+/-{te_role_s['sem']:.3f}  "
                 f"t={h2['t']:.2f} p={h2['p']:.4f} d={h2['d']:.2f}  "
                 f"-> {'SUPPORTED' if h2_ok else 'NOT SUPPORTED'}")

        # H3: novel classification > chance
        h3 = ttest_vs_null(test_cls, 0.5)
        h3_ok = te_cls_s["mean"] > 0.5 and h3["significant"]
        self.log(f"\nH3: Novel noun classification > chance")
        self.log(f"  Test cls={te_cls_s['mean']:.3f}+/-{te_cls_s['sem']:.3f}  "
                 f"t={h3['t']:.2f} p={h3['p']:.4f} d={h3['d']:.2f}  "
                 f"-> {'SUPPORTED' if h3_ok else 'NOT SUPPORTED'}")

        # H4: gap < memorization baseline (50%)
        h4 = ttest_vs_null(gaps, 0.5)
        h4_ok = gap_s["mean"] < 0.5
        self.log(f"\nH4: Gap < memorization baseline (50%)")
        self.log(f"  Gap={gap_s['mean']:+.3f}+/-{gap_s['sem']:.3f}  "
                 f"t={h4['t']:.2f} p={h4['p']:.4f} d={h4['d']:.2f}  "
                 f"-> {'SUPPORTED' if h4_ok else 'REJECTED'}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "p": cfg.p, "beta": cfg.beta,
                "rounds": cfg.rounds, "n_seeds": cfg.n_seeds,
                "n_training_sents": cfg.n_training_sents,
                "n_test_sents": cfg.n_test_sents,
                "vocab_size": len(vocab), "quick": quick,
            },
            metrics={
                "train_classification": tr_cls_s,
                "test_classification": te_cls_s,
                "train_role_accuracy": tr_role_s,
                "test_role_accuracy": te_role_s,
                "generalization_gap": gap_s,
                "h1_classification_gap": {"paired_ttest": h1, "within_15pct": h1_ok},
                "h2_role_vs_chance": {"ttest": h2, "above_chance": h2_ok},
                "h3_novel_classification": {"ttest": h3, "above_chance": h3_ok},
                "h4_generalization_gap": {"ttest": h4, "smaller_than_memorization": h4_ok},
            },
            raw_data={
                "seed_results": seed_results,
                "train_cls": train_cls, "test_cls": test_cls,
                "train_role": train_role, "test_role": test_role, "gaps": gaps,
            },
            duration_seconds=duration,
        )
        self.save_result(result)
        return result


# ======================================================================
# Main
# ======================================================================

def main():
    """Run compositional generalization experiment."""
    ap = argparse.ArgumentParser(
        description="Compositional generalization in assembly calculus parser")
    ap.add_argument("--quick", action="store_true",
                    help="Run with reduced parameters for fast validation")
    args = ap.parse_args()

    exp = CompositionalGeneralizationExperiment(verbose=True)
    result = exp.run(quick=args.quick)

    m = result.metrics
    print("\n" + "=" * 70)
    print("FINDINGS SUMMARY")
    print("=" * 70)
    print(f"Train cls: {m['train_classification']['mean']:.3f} +/- "
          f"{m['train_classification']['sem']:.3f}")
    print(f"Test  cls: {m['test_classification']['mean']:.3f} +/- "
          f"{m['test_classification']['sem']:.3f}")
    print(f"Gap:       {m['generalization_gap']['mean']:+.3f} +/- "
          f"{m['generalization_gap']['sem']:.3f}")
    print(f"Train role: {m['train_role_accuracy']['mean']:.3f}  "
          f"Test role: {m['test_role_accuracy']['mean']:.3f}")
    print(f"\nH1 (gap<15%): {'PASS' if m['h1_classification_gap']['within_15pct'] else 'FAIL'}")
    print(f"H2 (role>50%): {'PASS' if m['h2_role_vs_chance']['above_chance'] else 'FAIL'}")
    print(f"H3 (cls>50%):  {'PASS' if m['h3_novel_classification']['above_chance'] else 'FAIL'}")
    print(f"H4 (gap<mem):  {'PASS' if m['h4_generalization_gap']['smaller_than_memorization'] else 'FAIL'}")
    print(f"\nTime: {result.duration_seconds:.1f}s ({result.parameters['n_seeds']} seeds)")


if __name__ == "__main__":
    main()
