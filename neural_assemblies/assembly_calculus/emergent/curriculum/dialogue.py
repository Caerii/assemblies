"""Dialogue Q-A curriculum on closed ``VOCABULARY`` (numpy port of nemo patterns)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from ..core.grounding import VOCABULARY
from .data import GroundedSentence, _ctx


@dataclass
class DialoguePair:
    """Question–answer pair for chat tuning."""

    question: GroundedSentence
    answer: GroundedSentence
    pattern_type: str


_NOUNS = [w for w, c in VOCABULARY.items() if c.dominant_modality == "visual"]
_INTRANS = [w for w, c in VOCABULARY.items() if c.dominant_modality == "motor"]


def _gs(words: List[str], roles: List, mood: str = "declarative") -> GroundedSentence:
    return GroundedSentence(
        words=words,
        contexts=[_ctx(w) for w in words],
        roles=roles,
        mood=mood,
    )


def create_dialogue_pairs(seed: int = 42) -> List[DialoguePair]:
    """Pinned + sampled Q-A pairs using only package vocabulary."""
    rng = random.Random(seed)
    pairs: List[DialoguePair] = []

    # Pinned who-questions (transitive)
    pairs.append(DialoguePair(
        question=_gs(
            ["who", "chases", "the", "cat"],
            [None, "action", None, "patient"],
            mood="interrogative",
        ),
        answer=_gs(
            ["the", "dog", "chases", "the", "cat"],
            [None, "agent", "action", None, "patient"],
        ),
        pattern_type="who_query",
    ))
    pairs.append(DialoguePair(
        question=_gs(
            ["who", "sees", "the", "bird"],
            [None, "action", None, "patient"],
            mood="interrogative",
        ),
        answer=_gs(
            ["the", "cat", "sees", "the", "bird"],
            [None, "agent", "action", None, "patient"],
        ),
        pattern_type="who_query",
    ))

    # Pinned yes/no
    pairs.append(DialoguePair(
        question=_gs(
            ["does", "the", "dog", "runs"],
            [None, None, "agent", "action"],
            mood="interrogative",
        ),
        answer=_gs(
            ["yes", "the", "dog", "runs"],
            [None, None, "agent", "action"],
        ),
        pattern_type="yesno_affirm",
    ))

    for _ in range(8):
        noun = rng.choice(_NOUNS)
        verb = rng.choice(_INTRANS)
        pairs.append(DialoguePair(
            question=_gs(
                ["who", verb],
                [None, "action"],
                mood="interrogative",
            ),
            answer=_gs(
                ["the", noun, verb],
                [None, "agent", "action"],
            ),
            pattern_type="who_query",
        ))

    return pairs


def get_dialogue_curriculum(seed: int = 42) -> List[GroundedSentence]:
    """All Q and A sentences for grammar training."""
    sents: List[GroundedSentence] = []
    for pair in create_dialogue_pairs(seed):
        sents.append(pair.question)
        sents.append(pair.answer)
    return sents


def get_dialogue_pairs(seed: int = 42) -> List[DialoguePair]:
    return create_dialogue_pairs(seed)
