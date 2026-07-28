"""Perceived event structure: who acted on whom, independent of how it is said.

WHY THIS EXISTS
---------------
`GroundedSentence.roles` is a LINGUISTIC ANNOTATION, and its own docstring says
so: "a corpus with roles filled in is testing a different claim than one
without". The unsupervised fallback derives roles from POSITION RELATIVE TO THE
VERB -- which is fine for parsing but CIRCULAR for word-order induction, since
it assumes the very mapping being induced.

The grounding cannot rescue that either, because as it stands it is role-free.
Measured on the lesion corpus, "dog chases ball" grounds as

    dog     visual [DOG, ANIMAL]
    chases  motor  [CHASING, PURSUIT]
    ball    visual [BALL, OBJECT]

A dog exists, a chasing exists, a ball exists. NOTHING says the dog is the
chaser. Delete `roles` and the two participants are perceptually symmetric, so
there is nothing left to learn a role mapping from.

A `SceneEvent` supplies what a learner actually perceives: an event with its
participants, identified by their PERCEPTUAL FEATURES rather than by the words
used for them.

WHAT MAKES THIS GROUNDING RATHER THAN A RELABELLED ANNOTATION
--------------------------------------------------------------
Two properties, and both matter:

1. Participants are named by FEATURES ([DOG, ANIMAL]), never by words. The
   word -> participant link must be established through the same grounding the
   lexicon is learned from; nothing here says which TOKEN is the agent.

2. `participants` is ordered by CAUSAL ROLE -- actor first, undergoer second --
   and that order is a property of the WORLD, not of the sentence. The
   linguistic order varies across languages; the causal order does not. So the
   mapping between them is exactly what a learner must acquire, and supplying
   the causal side is not supplying the answer.

This is the semantic-bootstrapping setting: the child sees who chased whom, and
has to work out how their language encodes it. It is NOT unsupervised induction
from raw text, which for thematic roles is circular in principle -- text does
not contain who-did-what except through the syntax being learned.

STILL GIVEN, and worth naming rather than hiding: word -> referent alignment is
easy here, because each word's own grounding lists its features. Genuine
cross-situational learning would present an UNALIGNED scene and force the
learner to solve reference and structure together. This module is one rung
below that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .sentence import GroundedSentence

#: Abstract roles, in the CAUSAL order participants are listed in.
#: Index 0 is the actor, index 1 the undergoer. Generalising event-specific
#: participants (chaser, finder, eater) into these abstract roles is a real
#: acquisition problem in its own right; ordering by causal structure is the
#: minimal assumption that lets it be deferred rather than solved by fiat.
CAUSAL_ROLE_ORDER = ("agent", "patient")


@dataclass
class SceneEvent:
    """An event and its participants, as PERCEIVED.

    `action` holds the event's own features (CHASING, PURSUIT).
    `participants[i]` holds the feature bundle of the i-th participant in
    CAUSAL order -- actor first. Feature bundles, never words.
    """

    action: List[str] = field(default_factory=list)
    participants: List[List[str]] = field(default_factory=list)

    def role_of_features(self, features: Sequence[str]) -> Optional[str]:
        """Which causal role do these perceived features fill, if any?

        BEST overlap, not first overlap, and the difference is not cosmetic.
        Participants routinely share a SUPERORDINATE feature while differing in
        identity: `boy` is [BOY, PERSON] and `girl` is [GIRL, PERSON]. Matching
        on any shared feature made both of them the actor, because both
        intersect [BOY, PERSON] on PERSON -- 36 of 198 sentences derived the
        wrong roles, every one of them a two-person event. Caught by
        `grounded_corpus.check()` against the annotations, not by inspection.

        Scoring by overlap SIZE separates them: `girl` shares one feature with
        [BOY, PERSON] and two with [GIRL, PERSON]. That is also the right
        perceptual story -- the shared superordinate is category information,
        the distinctive feature is identity.

        A tie returns None rather than guessing. Two participants a learner
        genuinely cannot tell apart should yield no role, not an arbitrary one.
        """
        wanted = set(features)
        if not wanted:
            return None
        scores = [len(wanted & set(bundle)) for bundle in self.participants]
        best = max(scores, default=0)
        if best == 0 or scores.count(best) != 1:
            return None
        idx = scores.index(best)
        return CAUSAL_ROLE_ORDER[idx] if idx < len(CAUSAL_ROLE_ORDER) else None

    def is_action(self, features: Sequence[str]) -> bool:
        return bool(set(features) & set(self.action))


def _word_features(context) -> List[str]:
    """Every perceptual feature active for a word, across modalities."""
    out: List[str] = []
    for name in ("visual", "motor", "properties", "spatial",
                 "social", "temporal", "emotional"):
        out.extend(getattr(context, name, ()) or ())
    return out


def roles_from_scene(sentence: "GroundedSentence") -> List[Optional[str]]:
    """Derive per-word roles from PERCEIVED event structure.

    Returns the same shape as `GroundedSentence.roles`, but computed from the
    scene rather than read off an annotation. Words whose features match no
    participant and no action get None -- determiners, for instance, which have
    no grounding at all and therefore no role, exactly as the annotation has it.

    NOTHING HERE CONSULTS WORD ORDER. That is the whole point: a positional
    derivation would assume the mapping that word-order induction is trying to
    recover.
    """
    event = getattr(sentence, "event", None)
    if event is None:
        return list(sentence.roles)
    roles: List[Optional[str]] = []
    for context in sentence.contexts:
        features = _word_features(context)
        if event.is_action(features):
            roles.append("action")
            continue
        roles.append(event.role_of_features(features))
    return roles
