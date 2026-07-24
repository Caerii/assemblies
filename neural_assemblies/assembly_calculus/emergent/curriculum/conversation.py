"""Scaled conversation curriculum for naturalistic multi-turn chat.

Builds Q-A pairs and multi-turn scripts from any ``GroundingContext``
vocabulary (typically ``build_vocabulary_preset``), progressing from
pinned patterns to sampled open-vocab dialogue.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..curriculum.dialogue import DialoguePair
from ..core.grounding import GroundingContext, VOCABULARY
from .data import GroundedSentence
from ..vocabulary_builder import verb_surface_form, words_by_modality


def _gs_vocab(
    words: List[str],
    roles: List,
    vocab: Dict[str, GroundingContext],
    mood: str = "declarative",
) -> GroundedSentence:
    return GroundedSentence(
        words=words,
        contexts=[vocab.get(w, GroundingContext()) for w in words],
        roles=roles,
        mood=mood,
    )


def _verb_for_vocab(lemma: str, vocab: Dict[str, GroundingContext]) -> Optional[str]:
    surface = verb_surface_form(lemma)
    if surface in vocab:
        return surface
    if lemma in vocab:
        return lemma
    return None


@dataclass
class ConversationScript:
    """Multi-turn conversation for context / prediction training."""

    turns: List[GroundedSentence]
    script_type: str


def _pick_det(vocab: Dict[str, GroundingContext], rng: random.Random) -> str:
    if "the" in vocab:
        return "the"
    dets = [w for w, c in vocab.items() if c.dominant_modality == "none"]
    return rng.choice(dets) if dets else "the"


def _transitive_pairs(
    vocab: Dict[str, GroundingContext],
    rng: random.Random,
    n: int,
) -> List[DialoguePair]:
    """Sample who-questions over transitive verb + patient patterns."""
    mods = words_by_modality(vocab)
    nouns = mods["visual"]
    verb_lemmas = mods["motor"]
    verbs = [v for vl in verb_lemmas if (v := _verb_for_vocab(vl, vocab))]
    if len(nouns) < 2 or not verbs:
        return []

    det = _pick_det(vocab, rng)
    pairs: List[DialoguePair] = []
    for _ in range(n):
        agent, patient = rng.sample(nouns, 2)
        verb = rng.choice(verbs)

        pairs.append(DialoguePair(
            question=_gs_vocab(
                ["who", verb, det, patient],
                [None, "action", None, "patient"],
                vocab,
                mood="interrogative",
            ),
            answer=_gs_vocab(
                [det, agent, verb, det, patient],
                [None, "agent", "action", None, "patient"],
                vocab,
            ),
            pattern_type="who_query",
        ))
    return pairs


def _what_questions(
    vocab: Dict[str, GroundingContext],
    rng: random.Random,
    n: int,
) -> List[DialoguePair]:
    mods = words_by_modality(vocab)
    nouns = mods["visual"]
    verb_lemmas = mods["motor"]
    verbs = [v for vl in verb_lemmas if (v := _verb_for_vocab(vl, vocab))]
    if not nouns or not verbs:
        return []

    det = _pick_det(vocab, rng)
    pairs: List[DialoguePair] = []
    for _ in range(n):
        agent = rng.choice(nouns)
        patient = rng.choice(nouns)
        verb = rng.choice(verbs)

        pairs.append(DialoguePair(
            question=_gs_vocab(
                ["what", "does", det, agent, verb],
                [None, None, None, "agent", "action"],
                vocab,
                mood="interrogative",
            ),
            answer=_gs_vocab(
                [det, patient],
                [None, "patient"],
                vocab,
            ),
            pattern_type="what_query",
        ))
    return pairs


def _yesno_pairs(
    vocab: Dict[str, GroundingContext],
    rng: random.Random,
    n: int,
) -> List[DialoguePair]:
    mods = words_by_modality(vocab)
    nouns = mods["visual"]
    verb_lemmas = mods["motor"]
    verbs = [v for vl in verb_lemmas if (v := _verb_for_vocab(vl, vocab))]
    if not nouns or not verbs or "yes" not in vocab:
        return []

    det = _pick_det(vocab, rng)
    pairs: List[DialoguePair] = []
    for _ in range(n):
        agent = rng.choice(nouns)
        verb = rng.choice(verbs)

        pairs.append(DialoguePair(
            question=_gs_vocab(
                ["does", det, agent, verb],
                [None, None, "agent", "action"],
                vocab,
                mood="interrogative",
            ),
            answer=_gs_vocab(
                ["yes", det, agent, verb],
                [None, None, "agent", "action"],
                vocab,
            ),
            pattern_type="yesno_affirm",
        ))
    return pairs


def _greeting_pairs(vocab: Dict[str, GroundingContext]) -> List[DialoguePair]:
    pairs: List[DialoguePair] = []
    if "hello" in vocab or "hi" in vocab:
        greet = "hello" if "hello" in vocab else "hi"
        pairs.append(DialoguePair(
            question=_gs_vocab([greet], [None], vocab, mood="declarative"),
            answer=_gs_vocab([greet], [None], vocab),
            pattern_type="greeting",
        ))
    if "yes" in vocab and "thank" in vocab and "you" in vocab:
        pairs.append(DialoguePair(
            question=_gs_vocab(
                ["thank", "you"], [None, None], vocab, mood="declarative",
            ),
            answer=_gs_vocab(["yes"], [None], vocab),
            pattern_type="acknowledgment",
        ))
    return pairs


def create_conversation_pairs(
    vocab: Optional[Dict[str, GroundingContext]] = None,
    *,
    seed: int = 42,
    max_transitive: int = 12,
    max_what: int = 6,
    max_yesno: int = 6,
) -> List[DialoguePair]:
    """Build scaled dialogue pairs from a vocabulary."""
    vocab = vocab or VOCABULARY
    rng = random.Random(seed)
    pairs: List[DialoguePair] = []

    pairs.extend(_greeting_pairs(vocab))
    pairs.extend(_transitive_pairs(vocab, rng, max_transitive))
    pairs.extend(_what_questions(vocab, rng, max_what))
    pairs.extend(_yesno_pairs(vocab, rng, max_yesno))
    return pairs


def create_conversation_scripts(
    vocab: Optional[Dict[str, GroundingContext]] = None,
    *,
    seed: int = 42,
    n_scripts: int = 8,
) -> List[ConversationScript]:
    """Multi-turn scripts: statement → follow-up question → answer."""
    vocab = vocab or VOCABULARY
    rng = random.Random(seed)
    mods = words_by_modality(vocab)
    nouns = mods["visual"]
    verb_lemmas = mods["motor"]
    verbs = [v for vl in verb_lemmas if (v := _verb_for_vocab(vl, vocab))]
    if len(nouns) < 2 or not verbs:
        return []

    det = _pick_det(vocab, rng)
    scripts: List[ConversationScript] = []

    for _ in range(n_scripts):
        agent, patient = rng.sample(nouns, 2)
        verb = rng.choice(verbs)

        turn1 = _gs_vocab(
            [det, agent, verb, det, patient],
            [None, "agent", "action", None, "patient"],
            vocab,
        )
        turn2 = _gs_vocab(
            ["who", verb, det, patient],
            [None, "action", None, "patient"],
            vocab,
            mood="interrogative",
        )
        turn3 = _gs_vocab(
            [det, agent, verb, det, patient],
            [None, "agent", "action", None, "patient"],
            vocab,
        )
        scripts.append(ConversationScript(
            turns=[turn1, turn2, turn3],
            script_type="narrative_followup",
        ))

    return scripts


def get_conversation_curriculum(
    vocab: Optional[Dict[str, GroundingContext]] = None,
    **kwargs,
) -> List[GroundedSentence]:
    """All Q/A sentences plus multi-turn script lines for grammar training."""
    sents: List[GroundedSentence] = []
    for pair in create_conversation_pairs(vocab, **kwargs):
        sents.append(pair.question)
        sents.append(pair.answer)
    for script in create_conversation_scripts(vocab, seed=kwargs.get("seed", 42)):
        sents.extend(script.turns)
    return sents


def get_conversation_pairs(
    vocab: Optional[Dict[str, GroundingContext]] = None,
    **kwargs,
) -> List[DialoguePair]:
    return create_conversation_pairs(vocab, **kwargs)


def train_conversation_exposure(parser, scripts: Optional[Sequence[ConversationScript]] = None) -> int:
    """Present multi-turn scripts through ``present_turn`` (lightweight online learning)."""
    if scripts is None:
        scripts = create_conversation_scripts(parser.word_grounding)

    count = 0
    for script in scripts:
        for turn in script.turns:
            known = [w for w in turn.words if w in parser.stim_map]
            if known:
                parser.present_turn(known, speaker="user", learn=True)
                count += 1
    return count
