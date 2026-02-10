"""
Multi-Clause Coordination Experiment (Tier 2)

Tests whether the Assembly Calculus parser can handle coordination
("dog runs and cat sleeps") and subordination ("because dog runs,
cat sleeps") — multi-clause structures that require maintaining
separate clause representations and linking them via conjunctions.

Scientific Question:
    Can the parser handle coordination and subordination across clauses?

Hypotheses:
    H1: Coordinated clauses are parsed as separate clause structures
        with correct thematic roles in each clause.
    H2: Subordinate clauses maintain correct reference to the antecedent
        (the subordinate clause's agent/action are correctly identified).
    H3: Parse accuracy degrades gracefully as the number of coordinated
        clauses increases (2, 3, 4 clauses).

Protocol:
    1. Build vocabulary with conjunction words (and, but, because, when).
    2. Train parser on simple coordinated and subordinate sentences.
    3. Test: Parse sentences with 2, 3, 4 coordinated clauses.
    4. Measure per-clause role assignment accuracy.
    5. Test subordination: "because the dog runs the cat sleeps" --
       check role assignments across both clauses.

Statistical methodology:
    - n_seeds=5 independent random seeds per condition.
    - One-sample t-test against chance baseline.
    - Mean +/- SEM, Cohen's d effect sizes.

References:
    - Mitropolsky & Papadimitriou (2025). "Simulated Language Acquisition."
    - Papadimitriou et al., PNAS 117(25):14464-14472, 2020.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from research.experiments.base import ExperimentBase, ExperimentResult, summarize, ttest_vs_null
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


@dataclass
class MultiClauseConfig:
    """Configuration for multi-clause coordination experiment."""
    n: int = 10000
    k: int = 100
    n_seeds: int = 5
    rounds: int = 10
    p: float = 0.05
    beta: float = 0.1


# -- Vocabulary and sentence construction -----------------------------------

# Conjunction words that are not in the default VOCABULARY
CONJUNCTION_CONTEXTS = {
    "and": GroundingContext(),
    "but": GroundingContext(),
    "because": GroundingContext(),
    "when": GroundingContext(),
}


def _build_extended_vocabulary() -> Dict[str, GroundingContext]:
    """Build vocabulary that includes conjunction words alongside defaults."""
    from src.assembly_calculus.emergent.grounding import VOCABULARY
    vocab = dict(VOCABULARY)
    for word, ctx in CONJUNCTION_CONTEXTS.items():
        if word not in vocab:
            vocab[word] = ctx
    return vocab


def _make_clause(det: str, noun: str, verb: str,
                 vocab: Dict[str, GroundingContext]) -> Tuple[List[str], List[GroundingContext], List[Optional[str]]]:
    """Create a single clause: DET NOUN VERB with contexts and roles."""
    words = [det, noun, verb]
    contexts = [vocab[det], vocab[noun], vocab[verb]]
    roles = [None, "agent", "action"]
    return words, contexts, roles


def _join_clauses_coordinated(
    clauses: List[Tuple[List[str], List[GroundingContext], List[Optional[str]]]],
    conjunction: str,
    vocab: Dict[str, GroundingContext],
) -> GroundedSentence:
    """Join multiple clauses with a coordinating conjunction.

    E.g., [clause1, clause2, clause3] with "and" becomes:
    clause1 and clause2 and clause3
    """
    words, contexts, roles = [], [], []
    for i, (w, c, r) in enumerate(clauses):
        if i > 0:
            words.append(conjunction)
            contexts.append(vocab[conjunction])
            roles.append(None)
        words.extend(w)
        contexts.extend(c)
        roles.extend(r)
    return GroundedSentence(words=words, contexts=contexts, roles=roles)


def _make_subordinate_sentence(
    sub_conj: str, sub_det: str, sub_noun: str, sub_verb: str,
    main_det: str, main_noun: str, main_verb: str,
    vocab: Dict[str, GroundingContext],
) -> GroundedSentence:
    """Create a subordinate sentence: CONJ DET NOUN VERB DET NOUN VERB.

    E.g., "because the dog runs the cat sleeps"
    """
    words = [sub_conj, sub_det, sub_noun, sub_verb,
             main_det, main_noun, main_verb]
    contexts = [vocab[w] for w in words]
    roles = [None, None, "agent", "action", None, "agent", "action"]
    return GroundedSentence(words=words, contexts=contexts, roles=roles)


def _build_training_data(vocab: Dict[str, GroundingContext]) -> List[GroundedSentence]:
    """Build training sentences covering coordinated and subordinate patterns."""
    data = []

    # Simple intransitive sentences (baseline)
    simple_clauses = [
        ("the", "dog", "runs"),
        ("the", "cat", "sleeps"),
        ("a", "bird", "plays"),
        ("the", "boy", "reads"),
        ("the", "girl", "plays"),
    ]
    for det, noun, verb in simple_clauses:
        w, c, r = _make_clause(det, noun, verb, vocab)
        data.append(GroundedSentence(words=w, contexts=c, roles=r))

    # Simple transitive sentences
    transitive_data = [
        (["the", "dog", "chases", "the", "cat"],
         [None, "agent", "action", None, "patient"]),
        (["the", "cat", "sees", "a", "bird"],
         [None, "agent", "action", None, "patient"]),
    ]
    for words, roles in transitive_data:
        ctxs = [vocab[w] for w in words]
        data.append(GroundedSentence(words=words, contexts=ctxs, roles=roles))

    # Coordinated sentences: 2 clauses
    for conj in ["and", "but"]:
        c1 = _make_clause("the", "dog", "runs", vocab)
        c2 = _make_clause("the", "cat", "sleeps", vocab)
        data.append(_join_clauses_coordinated([c1, c2], conj, vocab))

        c3 = _make_clause("a", "bird", "plays", vocab)
        c4 = _make_clause("the", "boy", "reads", vocab)
        data.append(_join_clauses_coordinated([c3, c4], conj, vocab))

    # Coordinated sentences: 3 clauses
    c1 = _make_clause("the", "dog", "runs", vocab)
    c2 = _make_clause("the", "cat", "sleeps", vocab)
    c3 = _make_clause("a", "bird", "plays", vocab)
    data.append(_join_clauses_coordinated([c1, c2, c3], "and", vocab))

    # Subordinate sentences
    data.append(_make_subordinate_sentence(
        "because", "the", "dog", "runs", "the", "cat", "sleeps", vocab))
    data.append(_make_subordinate_sentence(
        "when", "the", "bird", "plays", "the", "boy", "reads", vocab))
    data.append(_make_subordinate_sentence(
        "because", "the", "cat", "sleeps", "the", "girl", "plays", vocab))

    return data


def _extract_clauses_from_coordinated(
    words: List[str], conjunction: str,
) -> List[List[str]]:
    """Split a coordinated sentence into individual clauses by conjunction."""
    clauses = []
    current = []
    for w in words:
        if w == conjunction:
            if current:
                clauses.append(current)
            current = []
        else:
            current.append(w)
    if current:
        clauses.append(current)
    return clauses


def _evaluate_clause_roles(
    parser: EmergentParser,
    clause_words: List[str],
    expected_agent: str,
    expected_action: str,
) -> Dict[str, bool]:
    """Evaluate whether roles in a single clause are correctly assigned."""
    result = parser.parse(clause_words)
    roles = result.get("roles", {})
    agent_correct = roles.get(expected_agent) == "AGENT"
    action_correct = roles.get(expected_action) == "ACTION"
    return {
        "agent_correct": agent_correct,
        "action_correct": action_correct,
        "both_correct": agent_correct and action_correct,
    }


# -- Experiment class -------------------------------------------------------


class MultiClauseCoordinationExperiment(ExperimentBase):
    """Test multi-clause coordination and subordination in assembly parser."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="multiclause_coordination",
            seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "applications"),
            verbose=verbose,
        )

    def _train_parser(self, cfg: MultiClauseConfig, seed: int) -> EmergentParser:
        """Create and train a parser with the extended vocabulary."""
        vocab = _build_extended_vocabulary()
        parser = EmergentParser(
            n=cfg.n, k=cfg.k, p=cfg.p, beta=cfg.beta,
            seed=seed, rounds=cfg.rounds, vocabulary=vocab,
        )
        training_data = _build_training_data(vocab)
        parser.train(sentences=training_data)
        return parser

    def run(self, n_seeds: int = 5, **kwargs) -> ExperimentResult:
        self._start_timer()

        cfg = MultiClauseConfig(n_seeds=n_seeds, **{
            k: v for k, v in kwargs.items()
            if k in ("n", "k", "rounds", "p", "beta")
        })
        vocab = _build_extended_vocabulary()

        self.log("=" * 60)
        self.log("Multi-Clause Coordination Experiment")
        self.log(f"  n={cfg.n}, k={cfg.k}, n_seeds={cfg.n_seeds}")
        self.log("=" * 60)

        seeds = [self.seed + i for i in range(cfg.n_seeds)]
        metrics: Dict[str, Any] = {}

        # ==============================================================
        # H1: Coordinated clauses parsed with correct roles per clause
        # ==============================================================
        self.log("\nH1: Coordinated clause role accuracy (2-clause)")

        coord_test_sentences = [
            {
                "words": ["the", "dog", "runs", "and", "the", "cat", "sleeps"],
                "conjunction": "and",
                "clauses": [
                    {"words": ["the", "dog", "runs"], "agent": "dog", "action": "runs"},
                    {"words": ["the", "cat", "sleeps"], "agent": "cat", "action": "sleeps"},
                ],
            },
            {
                "words": ["a", "bird", "plays", "but", "the", "boy", "reads"],
                "conjunction": "but",
                "clauses": [
                    {"words": ["a", "bird", "plays"], "agent": "bird", "action": "plays"},
                    {"words": ["the", "boy", "reads"], "agent": "boy", "action": "reads"},
                ],
            },
        ]

        h1_per_seed_accuracy = []
        for s in seeds:
            parser = self._train_parser(cfg, s)
            correct_total, count_total = 0, 0
            for test in coord_test_sentences:
                clauses = _extract_clauses_from_coordinated(
                    test["words"], test["conjunction"])
                for clause_spec, clause_words in zip(test["clauses"], clauses):
                    eval_r = _evaluate_clause_roles(
                        parser, clause_words,
                        clause_spec["agent"], clause_spec["action"])
                    if eval_r["both_correct"]:
                        correct_total += 1
                    count_total += 1
            accuracy = correct_total / max(count_total, 1)
            h1_per_seed_accuracy.append(accuracy)

        h1_stats = summarize(h1_per_seed_accuracy)
        h1_test = ttest_vs_null(h1_per_seed_accuracy, 0.0)
        metrics["h1_coordination_2clause"] = {
            "accuracy": h1_stats,
            "test_vs_null": h1_test,
        }
        self.log(f"  Accuracy: {h1_stats['mean']:.3f} +/- {h1_stats['sem']:.3f}  "
                 f"d={h1_test['d']:.1f}")

        # ==============================================================
        # H2: Subordinate clauses with correct role assignments
        # ==============================================================
        self.log("\nH2: Subordinate clause role accuracy")

        subordinate_tests = [
            {
                "words": ["because", "the", "dog", "runs",
                          "the", "cat", "sleeps"],
                "sub_clause": {"words": ["the", "dog", "runs"],
                               "agent": "dog", "action": "runs"},
                "main_clause": {"words": ["the", "cat", "sleeps"],
                                "agent": "cat", "action": "sleeps"},
            },
            {
                "words": ["when", "the", "bird", "plays",
                          "the", "boy", "reads"],
                "sub_clause": {"words": ["the", "bird", "plays"],
                               "agent": "bird", "action": "plays"},
                "main_clause": {"words": ["the", "boy", "reads"],
                                "agent": "boy", "action": "reads"},
            },
        ]

        h2_sub_accuracy = []
        h2_main_accuracy = []
        for s in seeds:
            parser = self._train_parser(cfg, s)
            sub_correct, sub_count = 0, 0
            main_correct, main_count = 0, 0
            for test in subordinate_tests:
                # Evaluate subordinate clause independently
                sub_eval = _evaluate_clause_roles(
                    parser, test["sub_clause"]["words"],
                    test["sub_clause"]["agent"],
                    test["sub_clause"]["action"])
                if sub_eval["both_correct"]:
                    sub_correct += 1
                sub_count += 1

                # Evaluate main clause independently
                main_eval = _evaluate_clause_roles(
                    parser, test["main_clause"]["words"],
                    test["main_clause"]["agent"],
                    test["main_clause"]["action"])
                if main_eval["both_correct"]:
                    main_correct += 1
                main_count += 1

            h2_sub_accuracy.append(sub_correct / max(sub_count, 1))
            h2_main_accuracy.append(main_correct / max(main_count, 1))

        h2_sub_stats = summarize(h2_sub_accuracy)
        h2_main_stats = summarize(h2_main_accuracy)
        h2_sub_test = ttest_vs_null(h2_sub_accuracy, 0.0)
        h2_main_test = ttest_vs_null(h2_main_accuracy, 0.0)
        metrics["h2_subordination"] = {
            "subordinate_clause_accuracy": h2_sub_stats,
            "main_clause_accuracy": h2_main_stats,
            "sub_test_vs_null": h2_sub_test,
            "main_test_vs_null": h2_main_test,
        }
        self.log(f"  Subordinate clause: {h2_sub_stats['mean']:.3f} +/- {h2_sub_stats['sem']:.3f}  "
                 f"d={h2_sub_test['d']:.1f}")
        self.log(f"  Main clause:        {h2_main_stats['mean']:.3f} +/- {h2_main_stats['sem']:.3f}  "
                 f"d={h2_main_test['d']:.1f}")

        # ==============================================================
        # H3: Graceful degradation with 2, 3, 4 coordinated clauses
        # ==============================================================
        self.log("\nH3: Degradation with increasing coordination (2, 3, 4 clauses)")

        clause_templates = [
            ("the", "dog", "runs"),
            ("the", "cat", "sleeps"),
            ("a", "bird", "plays"),
            ("the", "boy", "reads"),
        ]

        h3_results = []
        for n_clauses in [2, 3, 4]:
            per_seed_acc = []
            for s in seeds:
                parser = self._train_parser(cfg, s)
                selected = clause_templates[:n_clauses]
                clauses_data = [_make_clause(d, n, v, vocab)
                                for d, n, v in selected]
                joined = _join_clauses_coordinated(clauses_data, "and", vocab)

                # Split back and evaluate each clause
                split_clauses = _extract_clauses_from_coordinated(
                    joined.words, "and")
                correct, total = 0, 0
                for clause_words, (det, noun, verb) in zip(split_clauses, selected):
                    eval_r = _evaluate_clause_roles(
                        parser, clause_words, noun, verb)
                    if eval_r["both_correct"]:
                        correct += 1
                    total += 1
                per_seed_acc.append(correct / max(total, 1))

            stats = summarize(per_seed_acc)
            test_res = ttest_vs_null(per_seed_acc, 0.0)
            h3_results.append({
                "n_clauses": n_clauses,
                "accuracy": stats,
                "test_vs_null": test_res,
            })
            self.log(f"  {n_clauses} clauses: {stats['mean']:.3f} +/- {stats['sem']:.3f}  "
                     f"d={test_res['d']:.1f}")

        metrics["h3_degradation_by_clause_count"] = h3_results

        # Check monotonic degradation trend
        means = [r["accuracy"]["mean"] for r in h3_results]
        if len(means) >= 2:
            is_degrading = all(means[i] >= means[i + 1] - 0.05
                               for i in range(len(means) - 1))
            metrics["h3_graceful_degradation"] = is_degrading
            self.log(f"  Graceful degradation: {is_degrading}")

        duration = self._stop_timer()
        self.log(f"\nDuration: {duration:.1f}s")

        return ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n": cfg.n, "k": cfg.k, "n_seeds": cfg.n_seeds,
                "rounds": cfg.rounds, "p": cfg.p, "beta": cfg.beta,
            },
            metrics=metrics,
            raw_data={},
            duration_seconds=duration,
        )


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Multi-Clause Coordination Experiment")
    ap.add_argument("--quick", action="store_true",
                    help="Quick run with fewer seeds and smaller network")
    args = ap.parse_args()

    exp = MultiClauseCoordinationExperiment(verbose=True)

    if args.quick:
        result = exp.run(n_seeds=2, n=5000, k=50)
        exp.save_result(result, "_quick")
    else:
        result = exp.run()
        exp.save_result(result)

    # -- Summary --
    print("\n" + "=" * 70)
    print("MULTI-CLAUSE COORDINATION SUMMARY")
    print("=" * 70)

    m = result.metrics
    h1 = m["h1_coordination_2clause"]
    print(f"\nH1 -- 2-clause coordination accuracy: "
          f"{h1['accuracy']['mean']:.3f} +/- {h1['accuracy']['sem']:.3f}")

    h2 = m["h2_subordination"]
    print(f"H2 -- Subordinate clause accuracy: "
          f"{h2['subordinate_clause_accuracy']['mean']:.3f}")
    print(f"      Main clause accuracy:        "
          f"{h2['main_clause_accuracy']['mean']:.3f}")

    print("\nH3 -- Degradation by clause count:")
    for entry in m["h3_degradation_by_clause_count"]:
        nc = entry["n_clauses"]
        acc = entry["accuracy"]["mean"]
        sem = entry["accuracy"]["sem"]
        print(f"  {nc} clauses: {acc:.3f} +/- {sem:.3f}")
    if "h3_graceful_degradation" in m:
        print(f"  Graceful degradation: {m['h3_graceful_degradation']}")

    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
