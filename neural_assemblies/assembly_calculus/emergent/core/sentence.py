"""Grounded sentence substrate shared by corpus index and curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .grounding import GroundingContext
from .scene import SceneEvent, roles_from_scene


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
    roles: List[Optional[str]] = None  # pyright: ignore[reportAssignmentType]
    mood: str = "declarative"
    #: PERCEIVED event structure, when the corpus supplies it. Unlike `roles`,
    #: this IS part of the experience: it says who acted on whom, with
    #: participants identified by perceptual FEATURES rather than by words, and
    #: ordered by CAUSAL role rather than by word order. `scene.roles_from_scene`
    #: turns it into per-word roles without consulting position, which is what
    #: makes word-order induction non-circular. See `core/scene.py`.
    event: Optional["SceneEvent"] = None

    def __post_init__(self):
        if self.roles is None:
            # ``None`` is the constructor shorthand for an unannotated
            # sentence; after normalization every instance carries the full
            # positional role vector required by the aligned-record contract.
            self.roles = [None] * len(self.words)  # pyright: ignore[reportAssignmentType]
        if len(self.words) != len(self.contexts):
            raise ValueError(
                f"words ({len(self.words)}) != contexts ({len(self.contexts)})"
            )
        if len(self.words) != len(self.roles):
            raise ValueError(
                f"words ({len(self.words)}) != roles ({len(self.roles)})"
            )


@dataclass
class SentencePlan:
    """A sentence BEFORE grounding: tokens plus the event they describe.

    WHY THIS EXISTS RATHER THAN A BARE TOKEN LIST. A generator that returns
    ``List[str]`` can only say WHAT WAS SAID, so every downstream consumer has
    to recover who-did-what from word order -- which is exactly the mapping
    role induction is trying to learn, and is simply WRONG for a passive
    ("the cat was chased by the dog" reads cat=agent positionally). The event
    is what the speaker SAW; it is the same for both voices, and it is the only
    non-circular place a role label can come from. See ``core/scene.py``.

    Grounding is deliberately NOT resolved here. Contexts must be looked up
    after inflected surface forms are registered (``_register_surface_forms``),
    so a plan that grounded itself eagerly would capture empty contexts and
    every derived role would be ``None``. ``ground_plans`` is the one place
    that conversion happens.
    """

    tokens: List[str]
    event: Optional[SceneEvent] = None
    mood: str = "declarative"

    def __len__(self) -> int:
        return len(self.tokens)


def ground_plans(parser, plans: List[SentencePlan]) -> List[GroundedSentence]:
    """Resolve plans against a parser's grounding — the ONE conversion.

    Roles come from PERCEPTION when the plan carries an event, and are left
    ``None`` otherwise (raw text, e.g. the CDS corpora, genuinely has no role
    information and must not be given fabricated labels). Deriving them here
    rather than at each call site is what keeps the two role paths -- the
    positional inducer in ``corpus_index`` and the gating learner in
    ``parser_mixins.gating`` -- reading the same answer.
    """
    out: List[GroundedSentence] = []
    for plan in plans:
        sentence = GroundedSentence(
            words=list(plan.tokens),
            contexts=[
                parser.word_grounding.get(w, GroundingContext())
                for w in plan.tokens
            ],
            mood=plan.mood,
            event=plan.event,
        )
        if plan.event is not None:
            sentence.roles = roles_from_scene(sentence)
        out.append(sentence)
    return out
