"""The lesion corpus, re-issued with PERCEIVED EVENT STRUCTURE.

`build_corpus()` grounds each word in a feature bag and carries a separate
`roles` annotation. That annotation is linguistic, so any word-order result
resting on it is really resting on labels. The grounding cannot substitute,
because it is role-free: "dog chases ball" says a DOG exists, a CHASING exists
and a BALL exists, but never that the dog is the chaser.

Here each sentence also carries a `SceneEvent`: the action's features plus its
participants, identified by PERCEPTUAL FEATURES and ordered by CAUSAL role
(actor first). `roles_from_scene` then derives per-word roles from perception
alone, consulting nothing about word order.

THE VALIDATION THAT MAKES IT USABLE: `check()` verifies the derived roles
reproduce the hand annotations sentence for sentence. If they diverge, the
scenes are wrong and every downstream result would be measuring the divergence
rather than the model.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

#: Perceptual features per referent, matching what `build_corpus` grounds each
#: word in. Kept in one place so the scenes cannot drift from the grounding --
#: a mismatch would make `role_of_features` silently return None.
REFERENT_FEATURES: Dict[str, List[str]] = {
    "dog": ["DOG", "ANIMAL"],
    "cat": ["CAT", "ANIMAL"],
    "bird": ["BIRD", "ANIMAL"],
    "boy": ["BOY", "PERSON"],
    "girl": ["GIRL", "PERSON"],
    "ball": ["BALL", "OBJECT"],
    "book": ["BOOK", "OBJECT"],
    "food": ["FOOD", "OBJECT"],
    "table": ["TABLE", "OBJECT"],
    "car": ["CAR", "OBJECT"],
}


def with_events(corpus: List) -> List:
    """Attach a `SceneEvent` to every transitive sentence in `corpus`.

    Participants are ordered by CAUSAL role, taken from the sentence's own
    annotation ONCE, at corpus-construction time. That is not smuggling the
    answer back in: the annotation says which WORD is the agent, whereas the
    scene records which PERCEIVED ENTITY acted -- and the scene is then the only
    thing consulted at learning time. The distinction is exactly that a learner
    perceives the dog doing the chasing without being told that the first noun
    is the subject.
    """
    from neural_assemblies.assembly_calculus.emergent.core.scene import SceneEvent

    out = []
    for sentence in corpus:
        actor = undergoer = None
        action_features: List[str] = []
        for word, context, role in zip(sentence.words, sentence.contexts,
                                       sentence.roles):
            if role == "agent":
                actor = REFERENT_FEATURES.get(word) or list(context.visual)
            elif role == "patient":
                undergoer = REFERENT_FEATURES.get(word) or list(context.visual)
            elif role == "action":
                action_features = list(context.motor)
        participants = [p for p in (actor, undergoer) if p]
        if action_features and participants:
            sentence.event = SceneEvent(action=action_features,
                                        participants=participants)
        out.append(sentence)
    return out


def check(corpus: List) -> Tuple[int, int, List[str]]:
    """Do PERCEIVED roles reproduce the annotation? Returns (ok, total, diffs)."""
    from neural_assemblies.assembly_calculus.emergent.core.scene import (
        roles_from_scene,
    )

    ok = total = 0
    diffs: List[str] = []
    for sentence in corpus:
        if getattr(sentence, "event", None) is None:
            continue
        derived = roles_from_scene(sentence)
        total += 1
        if derived == list(sentence.roles):
            ok += 1
        elif len(diffs) < 8:
            diffs.append(f"{' '.join(sentence.words)}: "
                         f"annotated={sentence.roles} derived={derived}")
    return ok, total, diffs


def build() -> List:
    """The lesion corpus with perceived event structure attached."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lesion_aphasia import build_corpus
    return with_events(build_corpus())


def strip_roles(corpus: List) -> List:
    """Blank every `roles` entry, leaving only perception.

    The ablation the grounded claim rests on: after this, anything the model
    learns about roles must have come through `event`.
    """
    for sentence in corpus:
        sentence.roles = [None] * len(sentence.words)
    return corpus


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = build()
    ok, total, diffs = check(corpus)
    print(f"\n  scenes attached: {total}/{len(corpus)} sentences")
    print(f"  derived roles match annotation: {ok}/{total}")
    for d in diffs:
        print(f"    MISMATCH {d}")
    sample = next(s for s in corpus if getattr(s, "event", None))
    print(f"\n  example: {' '.join(sample.words)}")
    print(f"    action       {sample.event.action}")
    print(f"    participants {sample.event.participants}  (causal order)")
    print(f"    annotated    {sample.roles}")
    from neural_assemblies.assembly_calculus.emergent.core.scene import (
        roles_from_scene,
    )
    print(f"    derived      {roles_from_scene(sample)}")
