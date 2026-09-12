"""Build grounded sentences for holdout bridge / prediction training.

A HOLDOUT IS HELD OUT OF LEXICON TRAINING, NOT OUT OF THE LANGUAGE.  This
distinction is the whole reason this module exists and is easy to get wrong.

The generalization claim under test is that a word never learned as a lexical
item can still be categorised and used, because its GROUNDING FEATURES are
shared with words that were learned.  Testing that requires the holdout to
appear in context -- it has to occur in sentences for a prediction bridge to
have anything to predict around it.  What must not happen is the holdout
getting its own trained assembly, which would make the test vacuous.

So the sentences built here deliberately CONTAIN holdout words while lexicon
training deliberately SKIPS them.  A holdout is unlearned as a word and
present as an event.  Anyone tightening this by excluding holdouts from these
corpora too would not be making the test stricter; they would be removing the
signal the test measures.

The corresponding failure in the other direction is subtler: if these
sentences are fed through a path that does form lexical assemblies for their
words, the holdout silently stops being held out and every generalization
number becomes a training number.  Check which phase consumes a corpus before
adding holdout sentences to it.
"""

from __future__ import annotations

from typing import List, Optional, Set, TYPE_CHECKING

from ..core.grounding import GroundingContext
from ..core.sentence import GroundedSentence

if TYPE_CHECKING:
    from ..parser import EmergentParser


def holdout_bridge_token_lists(
    holdout_words: Optional[Set[str]] = None,
) -> List[List[str]]:
    """Token lists targeting default or custom holdout lemmas."""
    from neural_assemblies.lexicon.curriculum.holdout_bridge_corpus import (
        HOLDOUT_BRIDGE_CORPUS,
    )

    lines = list(HOLDOUT_BRIDGE_CORPUS)
    if holdout_words:
        holdout = set(holdout_words)
        lines = [
            line for line in lines
            if any(t in holdout for t in line.split())
        ]
    return [line.split() for line in lines]


def compile_holdout_bridge_probe_index(
    parser,
    holdout_words: Optional[Set[str]] = None,
):
    """Corpus index for bridge OOV probes — matches training corpus, no register_word."""
    from ..core.corpus_index import compile_corpus
    from ..core.grounding import GroundingContext
    from ..core.sentence import GroundedSentence

    sentences: List[GroundedSentence] = []
    for tokens in holdout_bridge_token_lists(holdout_words):
        contexts = [
            parser.word_grounding.get(w, GroundingContext())
            for w in tokens
        ]
        sentences.append(GroundedSentence(
            words=tokens,
            contexts=contexts,
            roles=[None] * len(tokens),
        ))
    return compile_corpus(parser, sentences)


def build_holdout_bridge_sentences(
    parser,
    holdout_words: Optional[Set[str]] = None,
) -> List[GroundedSentence]:
    """Grounded holdout-bridge sentences; registers words for stim only."""
    sentences: List[GroundedSentence] = []
    for tokens in holdout_bridge_token_lists(holdout_words):
        for w in tokens:
            if hasattr(parser, "register_word"):
                parser.register_word(w)
        contexts = [
            parser.word_grounding.get(w, GroundingContext())
            for w in tokens
        ]
        sentences.append(GroundedSentence(
            words=tokens,
            contexts=contexts,
            roles=[None] * len(tokens),
        ))
    return sentences


def merge_prediction_corpus(
    parser,
    holdout_words: Optional[Set[str]] = None,
    *,
    stage_name: str = "SENTENCES",
) -> List[GroundedSentence]:
    """Instruction sentences + repeated holdout bridge frames."""
    from ..training.perf import stage_holdout_bridge_reps
    from .data import create_instruction_sentences

    seen: set = set()
    merged: List[GroundedSentence] = []

    def _add_unique(sent: GroundedSentence) -> None:
        key = tuple(sent.words)
        if key in seen:
            return
        seen.add(key)
        merged.append(sent)

    for sent in create_instruction_sentences():
        _add_unique(sent)

    holdout_sents = build_holdout_bridge_sentences(parser, holdout_words)
    reps = stage_holdout_bridge_reps(
        stage_name, fast=getattr(parser, "fast_training", False),
    )
    for _ in range(reps):
        merged.extend(holdout_sents)
    return merged


def train_holdout_bridge_boost(
    parser: "EmergentParser",
    holdout_words: Set[str],
    *,
    stage_name: str = "SENTENCES",
) -> int:
    """Focused prediction pass on holdout-bridge transitions only."""
    from ..core.corpus_index import compile_corpus
    from ..training.perf import stage_holdout_bridge_reps

    if not holdout_words:
        return 0

    reps = stage_holdout_bridge_reps(
        stage_name, fast=parser.fast_training,
    )
    grounded: List[GroundedSentence] = []
    for _ in range(reps):
        grounded.extend(build_holdout_bridge_sentences(parser, holdout_words))

    if not grounded:
        return 0

    idx = compile_corpus(parser, grounded)
    targets = sorted(set(holdout_words) & set(parser.stim_map.keys()))
    if targets:
        parser._ensure_prediction_lexicon(targets)

    from ..evaluation.parity import exact_training_mode

    old_bridge = parser.bridge_rounds
    parser.bridge_rounds = max(old_bridge * 2, 6)
    try:
        with exact_training_mode():
            parser.train_next_token(
                idx.grounded,
                corpus_index=idx,
                transition_cache=None,
                dedupe_sentences=False,
                force_link=True,
            )
    finally:
        parser.bridge_rounds = old_bridge
    return len(idx.transitions)
