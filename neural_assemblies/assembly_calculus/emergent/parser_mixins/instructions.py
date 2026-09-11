"""InstructionMixin — imperative parsing and agent-oriented training."""

from __future__ import annotations

from typing import List

from ..structured_io import InstructionFrame


class InstructionMixin:
    """Instruction following: semantic frames and agent training pipeline."""

    def parse_instruction(self, words: List[str]) -> InstructionFrame:
        """Parse text into an actionable semantic frame.

        Wraps ``parse()`` and normalizes roles for imperatives (implicit
        addressee) and interrogatives.
        """
        parsed = self.parse(words)
        roles = parsed.get("roles", {})
        categories = parsed.get("categories", {})
        mood = parsed.get("mood", "DECLARATIVE")

        if mood == "IMPERATIVE":
            return self._instruction_frame_imperative(
                words, parsed, categories, roles,
            )

        agent = patient = action = None
        for word, role in roles.items():
            if role == "AGENT":
                agent = word
            elif role == "PATIENT":
                patient = word
            elif role == "ACTION":
                action = word

        if action is None:
            for word, cat in categories.items():
                if cat == "VERB":
                    action = word
                    break

        intent = action
        if intent is None and words:
            intent = words[0].lower()

        return InstructionFrame(
            intent=intent,
            mood=mood,
            agent=agent,
            action=action,
            patient=patient,
            polarity=parsed.get("polarity", "AFFIRMATIVE"),
            raw_words=list(words),
            categories=dict(categories),
            roles=dict(roles),
        )

    def _instruction_frame_imperative(
        self,
        words: List[str],
        parsed: dict,
        categories: dict,
        roles: dict,
    ) -> InstructionFrame:
        """Extract verb + patient from verb-initial commands."""
        action = patient = None
        seen_verb = False
        for word in words:
            cat = categories.get(word)
            if cat == "VERB" and action is None:
                action = word
                seen_verb = True
            elif seen_verb and cat in ("NOUN", "PRON") and patient is None:
                patient = word

        if action is None and words:
            g = self.word_grounding.get(words[0])
            if g and g.dominant_modality == "motor":
                action = words[0]

        return InstructionFrame(
            intent=action,
            mood="IMPERATIVE",
            agent="you",
            action=action,
            patient=patient,
            polarity=parsed.get("polarity", "AFFIRMATIVE"),
            raw_words=list(words),
            categories=dict(categories),
            roles=dict(roles),
        )

    def generate_confirmation(self, frame: InstructionFrame) -> List[str]:
        """Surface acknowledgment for an executed instruction (closed vocab)."""
        if frame.action and frame.patient:
            return ["ok", frame.action, frame.patient]
        if frame.action:
            return ["ok", frame.action]
        return ["ok"]
