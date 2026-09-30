"""Evaluation framework for EmergentParser.

Provides classification accuracy (per-category P/R/F1, confusion matrix),
role assignment accuracy, word order correctness, tense/mood/polarity
accuracy, generalization metrics, and generation quality.

TWO THINGS TO KNOW BEFORE QUOTING ANY NUMBER FROM THIS FILE.

First, evaluation is not free of side effects.  Classification projects into
core areas and resets their recurrent connections; role probes drive role
areas.  Unless plasticity is explicitly disabled, measuring a parser also
changes it, and measuring twice does not give the same answer twice.  Take
metrics on a parser you are finished training, or on a clone.

Second, and more important, the metrics are not all measuring the same kind
of thing.  Some -- classification accuracy, generalization on held-out
words -- score a genuinely neural decision, where the answer came out of
projecting and reading out.  Others score a pipeline that includes
hand-written steps: word-order correctness on generated sentences partly
scores the ordering ladder in ``GenerationMixin``, and tense/mood/polarity
accuracy partly scores the word-list detectors in ``MorphosyntaxMixin``.  A
high score on the second kind is evidence that the pipeline runs end to end;
it is not evidence about what the assemblies learned.  Each metric's
docstring should be read before it is cited.
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional

from ..core.areas import CORE_TO_CATEGORY


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
            target: Expected typology, one of the six basic orders
                (SVO, SOV, VSO, OSV, OVS, VOS).

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
            from ..acquisition.pos_inference import classify_word_bootstrapped

            predicted, _ = classify_word_bootstrapped(
                self.parser, word, grounding=grounding,
            )
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
                    # Does the agent precede the verb in this typology?
                    # Definitional, from the slot sequence -- covers all six
                    # orders rather than the three subject-initial ones.
                    from ..core.word_order import WORD_ORDERS, order_slots

                    label = (expected_order
                             if expected_order in WORD_ORDERS else "SVO")
                    slots = order_slots(label)
                    s_before_v = slots.index("S") < slots.index("V")
                    if (ai < vi) == s_before_v:
                        order_correct += 1

        return {
            "roundtrip_accuracy": role_correct / max(role_total, 1),
            "content_recall": content_found / max(content_total, 1),
            "word_order_correct": order_correct / max(order_total, 1),
        }

    def evaluate_instruction_following(
        self,
        test_cases: List[dict],
    ) -> dict:
        """Evaluate ``parse_instruction`` against expected frame fields.

        Each case: ``{"words": [...], "expected": {"mood", "action", "patient", ...}}``
        """
        correct = 0
        total = 0
        per_field: Dict[str, List[bool]] = defaultdict(list)

        for case in test_cases:
            words = case["words"]
            expected = case.get("expected", {})
            frame = self.parser.parse_instruction(words)
            total += 1
            case_ok = True
            for field, exp_val in expected.items():
                got = getattr(frame, field, None)
                ok = got == exp_val
                per_field[field].append(ok)
                if not ok:
                    case_ok = False
            if case_ok:
                correct += 1

        return {
            "accuracy": correct / max(total, 1),
            "total": total,
            "correct": correct,
            "per_field_accuracy": {
                f: sum(v) / max(len(v), 1) for f, v in per_field.items()
            },
        }

    def evaluate_next_token(
        self,
        prefixes: List[List[str]],
        expected_any: Optional[List[List[str]]] = None,
    ) -> dict:
        """Score next-token predictions on prefix list.

        If ``expected_any`` is provided, success = top-5 contains any expected word.
        """
        if not hasattr(self.parser, "predict_next"):
            return {"accuracy": 0.0, "total": 0}

        hits = 0
        total = len(prefixes)
        for i, prefix in enumerate(prefixes):
            preds = self.parser.predict_next(prefix)
            if not preds:
                continue
            top5 = {w for w, _ in preds[:5]}
            if expected_any and i < len(expected_any):
                if top5 & set(expected_any[i]):
                    hits += 1
            elif preds[0][1] > 0.0:
                hits += 1

        return {
            "accuracy": hits / max(total, 1),
            "total": total,
            "hits": hits,
        }

    def evaluate_dialogue(
        self,
        qa_pairs: List[dict],
        *,
        online_learn: bool = False,
    ) -> dict:
        """Evaluate multi-turn Q-A accuracy via ``EmergentSession``.

        Each pair: ``{"question": "who chases the cat", "acceptable": ["dog", "the dog"]}``
        """
        from ..session.interactive import EmergentSession

        session = EmergentSession(
            parser=self.parser,
            online_learn=online_learn,
        )
        correct = 0
        per_pattern: Dict[str, List[bool]] = defaultdict(list)

        for pair in qa_pairs:
            qtext = pair["question"]
            acceptable = pair.get("acceptable", [])
            if isinstance(acceptable, str):
                acceptable = [acceptable]
            pattern = pair.get("pattern_type", "default")

            reply = session.interact(qtext).lower()
            reply_tokens = set(session.tokenize(reply))
            hit = any(
                ans.lower() in reply or ans.lower() in reply_tokens
                for ans in acceptable
            )
            per_pattern[pattern].append(hit)
            if hit:
                correct += 1

        return {
            "accuracy": correct / max(len(qa_pairs), 1),
            "total": len(qa_pairs),
            "correct": correct,
            "per_pattern": {
                p: sum(v) / max(len(v), 1) for p, v in per_pattern.items()
            },
        }

    def evaluate_tool_compliance(
        self,
        test_cases: Optional[List[dict]] = None,
    ) -> dict:
        """Evaluate language → tool call argument accuracy (blocks + default tools)."""
        from ..curriculum.blocks import blocks_compliance_test_cases
        from ..tools import ToolRegistry

        if test_cases is None:
            test_cases = blocks_compliance_test_cases()

        registry = ToolRegistry()
        correct = 0
        arg_hits = 0
        arg_total = 0

        for case in test_cases:
            words = case["words"]
            frame = self.parser.parse_instruction(words)
            call = registry.dispatch(frame)
            exp_tool = case.get("expected_tool")
            exp_args = case.get("expected_args", {})

            if call is not None and call.name == exp_tool:
                correct += 1
                for key, val in exp_args.items():
                    arg_total += 1
                    if str(call.arguments.get(key)) == str(val):
                        arg_hits += 1

        return {
            "tool_match_accuracy": correct / max(len(test_cases), 1),
            "arg_accuracy": arg_hits / max(arg_total, 1),
            "total": len(test_cases),
            "correct": correct,
        }

    def evaluate_blocks_execution(
        self,
        commands: List[dict],
    ) -> dict:
        """Parse commands and apply via ``BlocksLanguageExecutor``; check final state."""
        from ..blocks_bridge import BlocksLanguageExecutor, frame_to_blocks_action

        executor = BlocksLanguageExecutor(blocks=("A", "B", "C"))
        success = 0
        for case in commands:
            words = case["words"]
            frame = self.parser.parse_instruction(words)
            action = frame_to_blocks_action(frame)
            if action is None:
                continue
            try:
                executor.apply_action(action)
                exp_on = case.get("expected_on")
                if exp_on is not None and executor.state.on == exp_on:
                    success += 1
                elif exp_on is None:
                    success += 1
            except ValueError:
                pass

        return {
            "accuracy": success / max(len(commands), 1),
            "total": len(commands),
            "success": success,
        }

    def evaluate_schema_compliance(
        self,
        json_samples: List[str],
    ) -> dict:
        """Validate JSON strings against tool_call schema."""
        from ..structured_json import StructuredRecord, validate_record

        valid = 0
        for text in json_samples:
            try:
                record = StructuredRecord.from_json(text)
                ok, _ = validate_record(record)
                if ok:
                    valid += 1
            except (json.JSONDecodeError, ValueError):
                pass

        return {
            "accuracy": valid / max(len(json_samples), 1),
            "total": len(json_samples),
            "valid": valid,
        }

    def evaluate_json_roundtrip(
        self,
        word_commands: Optional[List[List[str]]] = None,
    ) -> dict:
        """Language → JSON → language → tool call roundtrip accuracy."""
        from ..curriculum.blocks import blocks_compliance_test_cases

        if word_commands is None:
            word_commands = [c["words"] for c in blocks_compliance_test_cases()]
            word_commands.append(["chases", "the", "cat"])

        hits = 0
        schema_ok = 0
        for words in word_commands:
            result = self.parser.structured_roundtrip(words)
            if result.get("schema_valid"):
                schema_ok += 1
            if result.get("tool_match"):
                hits += 1

        n = len(word_commands)
        return {
            "roundtrip_accuracy": hits / max(n, 1),
            "schema_valid_rate": schema_ok / max(n, 1),
            "total": n,
            "hits": hits,
        }

    def evaluate_multi_tool_plan(
        self,
        cases: Optional[List[dict]] = None,
    ) -> dict:
        """Evaluate multi-step plans (explicit + BFS goal-directed)."""
        from ..blocks_bridge import BlocksLanguageExecutor

        if cases is None:
            cases = [
                {
                    "text": "move a to table then move a to b",
                    "blocks": ("A", "B"),
                    "start_on": {"A": "B", "B": None},
                    "start_clear": {"A": True, "B": False},
                    "expected_final": {"A": "B"},
                    "min_steps": 2,
                },
                {
                    "text": "stack a on b",
                    "start_on": {"A": None, "B": None, "C": None},
                    "start_clear": {"A": True, "B": True, "C": True},
                    "expected_final": {"A": "B"},
                    "min_steps": 1,
                    "source": "bfs",
                },
            ]

        hits = 0
        for case in cases:
            blocks = tuple(case.get("blocks", ("A", "B", "C")))
            ex = BlocksLanguageExecutor(blocks=blocks)
            start_on = dict(case["start_on"])
            start_clear = dict(case["start_clear"])
            for blk in blocks:
                start_on.setdefault(blk, None)
                start_clear.setdefault(blk, True)
            ex.state = ex.state.__class__(on=start_on, clear=start_clear)
            plan = self.parser.text_to_tool_plan(case["text"], ex)
            if plan is None or plan.length < case.get("min_steps", 1):
                continue
            if case.get("source") and plan.source != case["source"]:
                continue
            result = self.parser.execute_plan(plan, ex)
            if not result.success:
                continue
            expected = case.get("expected_final", {})
            if all(ex.state.on.get(k) == v for k, v in expected.items()):
                hits += 1

        return {
            "accuracy": hits / max(len(cases), 1),
            "total": len(cases),
            "hits": hits,
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
            lines.append("\nGeneration:")
            lines.append(f"  Roundtrip role accuracy: "
                         f"{gen['roundtrip_accuracy']:.1%}")
            lines.append(f"  Content recall: "
                         f"{gen['content_recall']:.1%}")
            lines.append(f"  Word order correct: "
                         f"{gen['word_order_correct']:.1%}")

        if "tense_mood_polarity" in results:
            tmp = results["tense_mood_polarity"]
            lines.append("\nTense/Mood/Polarity:")
            lines.append(f"  Tense accuracy: "
                         f"{tmp['tense_accuracy']:.1%}")
            lines.append(f"  Mood accuracy: "
                         f"{tmp['mood_accuracy']:.1%}")
            lines.append(f"  Polarity accuracy: "
                         f"{tmp['polarity_accuracy']:.1%}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
