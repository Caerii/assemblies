"""Evaluation framework for EmergentParser.

Provides classification accuracy (per-category P/R/F1, confusion matrix),
role assignment accuracy, word order correctness, tense/mood/polarity
accuracy, generalization metrics, and generation quality.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from .areas import CORE_TO_CATEGORY


class EvaluationSuite:
    """Comprehensive evaluation metrics for an EmergentParser.

    Provides classification accuracy (per-category P/R/F1, confusion matrix),
    role assignment accuracy, word order correctness, tense/mood/polarity
    accuracy, generalization metrics, and generation quality.
    """

    def __init__(self, parser):
        self.parser = parser

    def evaluate_classification(
        self, test_vocab: Dict[str, str],
    ) -> dict:
        """Evaluate classification accuracy on a labeled vocabulary.

        Args:
            test_vocab: {word: expected_category} dict.

        Returns:
            {"accuracy": float, "per_category": {cat: {"precision", "recall",
            "f1"}}, "confusion_matrix": {expected: {predicted: count}}}
        """
        correct = 0
        total = 0
        confusion: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

        for word, expected in test_vocab.items():
            grounding = self.parser.word_grounding.get(word)
            predicted, _ = self.parser.classify_word(
                word, grounding=grounding)
            confusion[expected][predicted] += 1
            if predicted == expected:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)

        # Per-category precision, recall, F1
        all_cats = set(test_vocab.values()) | {
            pred for row in confusion.values()
            for pred in row
        }
        per_category: Dict[str, dict] = {}
        for cat in sorted(all_cats):
            tp = confusion[cat].get(cat, 0)
            fn = sum(v for k, v in confusion[cat].items() if k != cat)
            fp = sum(
                row.get(cat, 0) for exp, row in confusion.items()
                if exp != cat
            )
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (2 * precision * recall / max(precision + recall, 1e-9))
            per_category[cat] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        # Convert confusion matrix to plain dict
        cm = {k: dict(v) for k, v in confusion.items()}

        return {
            "accuracy": accuracy,
            "per_category": per_category,
            "confusion_matrix": cm,
        }

    def evaluate_roles(
        self, test_sentences: List[dict],
    ) -> dict:
        """Evaluate thematic role assignment accuracy.

        Args:
            test_sentences: List of dicts with "words" and "expected_roles"
                (a dict mapping word -> expected role label).

        Returns:
            {"accuracy": float, "per_role": {role: {"precision", "recall",
            "f1"}}}
        """
        correct = 0
        total = 0
        role_tp: Dict[str, int] = defaultdict(int)
        role_fp: Dict[str, int] = defaultdict(int)
        role_fn: Dict[str, int] = defaultdict(int)

        for item in test_sentences:
            words = item["words"]
            expected_roles = item["expected_roles"]
            result = self.parser.parse(words)
            assigned = result.get("roles", {})

            for word, expected in expected_roles.items():
                if expected is None:
                    continue
                predicted = assigned.get(word)
                if predicted == expected:
                    correct += 1
                    role_tp[expected] += 1
                else:
                    role_fn[expected] += 1
                    if predicted is not None:
                        role_fp[predicted] += 1
                total += 1

        accuracy = correct / max(total, 1)

        all_roles = set(role_tp) | set(role_fp) | set(role_fn)
        per_role: Dict[str, dict] = {}
        for role in sorted(all_roles):
            tp = role_tp[role]
            fp = role_fp[role]
            fn = role_fn[role]
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            per_role[role] = {"precision": p, "recall": r, "f1": f1}

        return {"accuracy": accuracy, "per_role": per_role}

    def evaluate_word_order(self, target: str = "SVO") -> dict:
        """Evaluate whether the parser's inferred word order matches target.

        Args:
            target: Expected typology ("SVO", "SOV", "VSO").

        Returns:
            {"inferred": str, "confidence": float, "correct": bool}
        """
        inferred, confidence = self.parser.infer_word_order()
        return {
            "inferred": inferred,
            "confidence": confidence,
            "correct": inferred == target,
        }

    def evaluate_tense_mood_polarity(
        self, test_sentences: List[dict],
    ) -> dict:
        """Evaluate tense, mood, polarity detection accuracy.

        Args:
            test_sentences: List of dicts with "words" and optional
                "expected_tense", "expected_mood", "expected_polarity".

        Returns:
            {"tense_accuracy": float, "mood_accuracy": float,
             "polarity_accuracy": float}
        """
        tense_correct = tense_total = 0
        mood_correct = mood_total = 0
        pol_correct = pol_total = 0

        for item in test_sentences:
            words = item["words"]

            if "expected_tense" in item:
                detected = self.parser.detect_tense(words)
                if detected == item["expected_tense"]:
                    tense_correct += 1
                tense_total += 1

            if "expected_mood" in item:
                detected = self.parser.detect_mood(words)
                if detected == item["expected_mood"]:
                    mood_correct += 1
                mood_total += 1

            if "expected_polarity" in item:
                detected = self.parser.detect_polarity(words)
                if detected == item["expected_polarity"]:
                    pol_correct += 1
                pol_total += 1

        return {
            "tense_accuracy": tense_correct / max(tense_total, 1),
            "mood_accuracy": mood_correct / max(mood_total, 1),
            "polarity_accuracy": pol_correct / max(pol_total, 1),
        }

    def evaluate_generalization(
        self, holdout_words: Dict[str, str],
    ) -> dict:
        """Evaluate classification on held-out words.

        Args:
            holdout_words: {word: expected_category} for words NOT in
                the training set.

        Returns:
            {"accuracy": float, "total": int, "correct": int}
        """
        correct = 0
        total = 0
        for word, expected in holdout_words.items():
            grounding = self.parser.word_grounding.get(word)
            predicted, _ = self.parser.classify_word(
                word, grounding=grounding)
            if predicted == expected:
                correct += 1
            total += 1

        return {
            "accuracy": correct / max(total, 1),
            "total": total,
            "correct": correct,
        }

    def evaluate_generation_quality(
        self, semantics_list: List[dict],
    ) -> dict:
        """Evaluate generation quality via roundtrip metrics.

        For each semantics dict ({"agent": ..., "action": ..., ...}):
        1. Generate a sentence
        2. Parse it back
        3. Check if roles are recovered

        Args:
            semantics_list: List of semantic dicts for generate().

        Returns:
            {"roundtrip_accuracy": float, "content_recall": float,
             "word_order_correct": float}
        """
        role_correct = 0
        role_total = 0
        content_found = 0
        content_total = 0
        order_correct = 0
        order_total = 0

        expected_order = self.parser.word_order_type or "SVO"

        for semantics in semantics_list:
            output = self.parser.generate(semantics)
            if not output:
                continue

            # Content recall: check if key content words appear
            for key in ("agent", "action", "patient"):
                if key in semantics:
                    content_total += 1
                    if semantics[key] in output:
                        content_found += 1

            # Parse back
            result = self.parser.parse(output)
            roles = result.get("roles", {})

            # Check agent roundtrip
            if "agent" in semantics:
                role_total += 1
                agent_word = semantics["agent"]
                if roles.get(agent_word) == "AGENT":
                    role_correct += 1

            if "patient" in semantics:
                role_total += 1
                patient_word = semantics["patient"]
                if roles.get(patient_word) == "PATIENT":
                    role_correct += 1

            # Word order check
            if "agent" in semantics and "action" in semantics:
                order_total += 1
                agent_word = semantics["agent"]
                action_word = semantics["action"]
                if agent_word in output and action_word in output:
                    ai = output.index(agent_word)
                    vi = output.index(action_word)
                    if expected_order == "SVO" and ai < vi:
                        order_correct += 1
                    elif expected_order == "SOV" and ai < vi:
                        order_correct += 1
                    elif expected_order == "VSO" and vi < ai:
                        order_correct += 1

        return {
            "roundtrip_accuracy": role_correct / max(role_total, 1),
            "content_recall": content_found / max(content_total, 1),
            "word_order_correct": order_correct / max(order_total, 1),
        }

    def full_evaluation(self) -> dict:
        """Run all available evaluations with default test data.

        Returns a comprehensive metrics dictionary.
        """
        results: dict = {}

        # Classification: test on known vocabulary
        test_vocab = {}
        for core_area, lex in self.parser.core_lexicons.items():
            cat = CORE_TO_CATEGORY.get(core_area)
            if cat:
                for word in lex:
                    test_vocab[word] = cat
        if test_vocab:
            results["classification"] = self.evaluate_classification(
                test_vocab)

        # Word order
        results["word_order"] = self.evaluate_word_order()

        # Generation (if vocabulary supports it)
        nouns = [w for w, c in test_vocab.items() if c == "NOUN"]
        verbs = [w for w, c in test_vocab.items() if c == "VERB"]
        if nouns and verbs:
            semantics_list = []
            for i in range(min(5, len(nouns), len(verbs))):
                sem = {"agent": nouns[i], "action": verbs[i % len(verbs)]}
                if len(nouns) > i + 1:
                    sem["patient"] = nouns[
                        (i + 1) % len(nouns)]
                semantics_list.append(sem)
            results["generation"] = self.evaluate_generation_quality(
                semantics_list)

        return results

    def generate_report(self, results: Optional[dict] = None) -> str:
        """Generate a human-readable evaluation report.

        Args:
            results: Pre-computed results dict, or None to run
                full_evaluation().

        Returns:
            Multi-line report string.
        """
        if results is None:
            results = self.full_evaluation()

        lines = ["=" * 50, "EmergentParser Evaluation Report", "=" * 50]

        if "classification" in results:
            cls = results["classification"]
            lines.append(f"\nClassification Accuracy: "
                         f"{cls['accuracy']:.1%}")
            lines.append("  Per-category F1:")
            for cat, m in sorted(cls["per_category"].items()):
                lines.append(f"    {cat:8s}: P={m['precision']:.2f} "
                             f"R={m['recall']:.2f} F1={m['f1']:.2f}")

        if "word_order" in results:
            wo = results["word_order"]
            lines.append(f"\nWord Order: {wo['inferred']} "
                         f"(confidence={wo['confidence']:.2f}, "
                         f"correct={wo['correct']})")

        if "generation" in results:
            gen = results["generation"]
            lines.append(f"\nGeneration:")
            lines.append(f"  Roundtrip role accuracy: "
                         f"{gen['roundtrip_accuracy']:.1%}")
            lines.append(f"  Content recall: "
                         f"{gen['content_recall']:.1%}")
            lines.append(f"  Word order correct: "
                         f"{gen['word_order_correct']:.1%}")

        if "tense_mood_polarity" in results:
            tmp = results["tense_mood_polarity"]
            lines.append(f"\nTense/Mood/Polarity:")
            lines.append(f"  Tense accuracy: "
                         f"{tmp['tense_accuracy']:.1%}")
            lines.append(f"  Mood accuracy: "
                         f"{tmp['mood_accuracy']:.1%}")
            lines.append(f"  Polarity accuracy: "
                         f"{tmp['polarity_accuracy']:.1%}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
