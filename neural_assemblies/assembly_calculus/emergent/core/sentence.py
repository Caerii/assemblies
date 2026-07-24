"""Grounded sentence substrate shared by corpus index and curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .grounding import GroundingContext


@dataclass
class GroundedSentence:
    """A sentence with grounding context for each word.

    This is the model's unit of EXPERIENCE, not of text.  A learner does not
    receive word strings; it receives words uttered while something is going
    on, and the parallel ``contexts`` list is that "something" -- the sensory
    features active as each word arrives.  Pairing them positionally is what
    makes grounded word learning possible at all (see
    ``parser_mixins.core.train_lexicon``, where word and context are projected
    in the same time step).

    ``roles`` is a different kind of thing and should not be read as part of
    the experience.  It is a per-word thematic annotation used by SUPERVISED
    role training, and it is ``None`` throughout in the unsupervised path,
    where roles are instead derived from position relative to the verb.  A
    corpus with roles filled in is testing a different claim than one without.

    The length assertions below are the invariant that keeps the three lists
    interpretable as a single aligned record; a mismatch would silently pair
    each word with another word's grounding.
    """
    words: List[str]
    contexts: List[GroundingContext]
    roles: List[Optional[str]] = None
    mood: str = "declarative"

    def __post_init__(self):
        if self.roles is None:
            self.roles = [None] * len(self.words)
        assert len(self.words) == len(self.contexts), (
            f"words ({len(self.words)}) != contexts ({len(self.contexts)})"
        )
        assert len(self.words) == len(self.roles), (
            f"words ({len(self.words)}) != roles ({len(self.roles)})"
        )
