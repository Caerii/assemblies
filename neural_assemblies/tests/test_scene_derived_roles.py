"""Roles must reach the pipeline, and they must come from the SCENE.

WHY THIS IS A TEST. `TrainingScheduleExecutor.build_stage_schedule` used to
hardcode ``roles=[None] * len(sent)``, so every role-consuming mechanism on the
curriculum path -- including the voice gating that a passive needs -- received
no signal at all and silently learned nothing. Nothing in the suite noticed,
because a mechanism that is never fed produces no failure, only an absence.

WHY THE SCENE AND NOT THE STRING. Role labels cannot be read off word order
without circularity: the reversal a passive expresses is exactly the case where
position stops predicting the answer. The non-circular signal is the perceived
event, which is the same for both voices. See ``core/scene.py``.

THE CHECK THAT MAKES THIS MORE THAN PLUMBING is `test_scene_agrees_with_position`
below: on an unambiguous active clause the two derivations are computed by
completely unrelated means -- perceptual features and causal order on one side,
position relative to the verb on the other -- so agreement is real evidence and
a disagreement localises the defect immediately. It found three when written.
"""
from __future__ import annotations

import pytest

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.core.corpus_index import (
    _assign_noun_roles, category_oracle,
)
from neural_assemblies.assembly_calculus.emergent.core.scene import (
    SceneEvent, _denotes,
)
from neural_assemblies.assembly_calculus.emergent.core.sentence import (
    ground_plans,
)
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
)

STAGES = [("SENTENCES", 4), ("COMPLEX_GRAMMAR", 6)]
_ROLE_AREA = {"agent": ROLE_AGENT, "patient": ROLE_PATIENT,
              "action": ROLE_ACTION}


@pytest.fixture(scope="module")
def trainer():
    # The generator does not touch the substrate, so a small brain is enough.
    return CurriculumTrainer(EmergentParser(n=500, k=20, seed=42))


def _grounded(trainer, stage, complexity):
    words = trainer._get_stage_words(stage)
    for w in words:
        trainer.parser.register_word(w.lemma)
    plans = trainer._generate_sentences(words, complexity, stage_name=stage)
    trainer._register_surface_forms(plans, words)
    return plans, ground_plans(trainer.parser, plans)


@pytest.mark.parametrize("stage,complexity", STAGES)
def test_roles_reach_the_pipeline(trainer, stage, complexity):
    """THE GATE. Zero roles arriving voids every claim downstream of here."""
    plans, grounded = _grounded(trainer, stage, complexity)
    with_event = [p for p in plans if p.event is not None]
    assert with_event, f"{stage}: no plan carries a SceneEvent"

    filled = [g for g in grounded if any(r is not None for r in g.roles)]
    assert len(filled) == len(with_event), (
        f"{stage}: {len(with_event)} plans carry an event but only "
        f"{len(filled)} sentences came out with any role -- the event is "
        f"being attached and then not read"
    )
    roles = {r for g in grounded for r in g.roles if r is not None}
    assert {"agent", "action"} <= roles, f"{stage}: roles seen = {roles}"


def test_scene_agrees_with_position(trainer):
    """Two unrelated derivations of the same fact must not disagree.

    Restricted to clauses the POSITIONAL inducer can actually analyse: exactly
    one verb-classified token, and no word occurring twice. Outside that set
    the inducer is not a valid reference, and the exclusions are not
    hypothetical --

      * "the open beach meets the street off the couch" -- `open` classifies as
        a VERB, so the inducer splits the clause at the adjective and calls the
        subject a post-verb PATIENT.
      * "the library sleeps among the library" -- one referent occupies both a
        thematic role and the PP, so no single assignment of it can be right.
        The generator no longer emits these; the guard stays because the
        comparison must not depend on that.

    In both the scene is right and position is wrong, so demanding agreement
    there would pin the weaker method as the standard. Accumulated across
    stages rather than parametrized, because COMPLEX_GRAMMAR alone can yield
    no analysable clause -- adjective/verb ambiguity is common enough at
    complexity >= 5 to disqualify nearly all of it.
    """
    parser = trainer.parser
    compared = 0
    failures = []

    for stage, complexity in STAGES:
        _plans, grounded = _grounded(trainer, stage, complexity)
        for g in grounded:
            if g.event is None:
                continue
            toks = [w for w in g.words if w in parser.stim_map]
            if len(toks) != len(g.words) or len(set(toks)) != len(toks):
                continue
            cats = [category_oracle(parser, w, parser.word_grounding.get(w))
                    for w in toks]
            if cats.count("VERB") != 1:
                continue
            verb_pos = cats.index("VERB")
            positional = {i: area for i, _w, area
                          in _assign_noun_roles(toks, cats, verb_pos, "SVO")}
            for i, role in enumerate(g.roles):
                if role is None or i not in positional:
                    continue
                if role not in _ROLE_AREA:
                    # `goal` has NO positional counterpart -- the inducer maps
                    # nouns onto S/O slots only, so a recipient is precisely
                    # where the scene knows something position cannot express.
                    # Nothing to compare, not a disagreement.
                    continue
                compared += 1
                if _ROLE_AREA[role] != positional[i]:
                    failures.append(
                        f"{stage}: {' '.join(g.words)!r} word {g.words[i]!r} "
                        f"-- scene says {_ROLE_AREA[role]}, position says "
                        f"{positional[i]}"
                    )

    assert compared > 0, "no analysable clause in any stage to compare"
    assert not failures, (
        f"{len(failures)}/{compared} role assignments disagree:\n  "
        + "\n  ".join(failures[:5])
    )


def test_unscened_sentences_get_no_roles(trainer):
    """Raw text must NOT be given fabricated labels.

    CDS lines and holdout bridges are transcribed speech with no recorded
    scene. Inventing one would manufacture exactly the supervision this path
    exists to avoid, and the resulting numbers would measure the invention.

    Constructed directly rather than filtered out of a generated stage: whether
    a CDS corpus is available depends on the vocabulary preset, and a test that
    silently finds nothing to check is not a test.
    """
    from neural_assemblies.assembly_calculus.emergent.core.sentence import (
        SentencePlan,
    )

    plans = [SentencePlan(["the", "dog", "runs"]),
             SentencePlan(["more", "milk"])]
    grounded = ground_plans(trainer.parser, plans)
    assert all(g.event is None for g in grounded)
    assert all(r is None for g in grounded for r in g.roles), (
        "a sentence with no perceived event was given roles anyway"
    )


class TestReferenceNotResemblance:
    """A shared property is not identity -- the rule both matchers use."""

    def test_superordinate_overlap_is_not_reference(self):
        # `bear` and `friend` are both animate; that does not make a bear the
        # one who acted. This exact shape put PP-object nouns into thematic
        # roles in the generated corpus.
        event = SceneEvent(action=["MOVING"],
                           participants=[["FRIEND", "PERSON", "ANIMATE"]])
        assert event.role_of_features(["FRIEND", "PERSON", "ANIMATE"]) == "agent"
        assert event.role_of_features(["BEAR", "ANIMAL", "ANIMATE"]) is None

    def test_distinct_participants_still_separate(self):
        # The property that best-overlap was introduced for must survive:
        # boy and girl share PERSON and must not collapse onto each other.
        event = SceneEvent(action=["CHASING"],
                           participants=[["GIRL", "PERSON"], ["BOY", "PERSON"]])
        assert event.role_of_features(["GIRL", "PERSON"]) == "agent"
        assert event.role_of_features(["BOY", "PERSON"]) == "patient"

    def test_action_uses_the_same_rule(self):
        # A past participle shares motor features with the finite verb. Both
        # scoring as the action gives one clause two actions.
        event = SceneEvent(action=["SEEMING", "APPEARING"], participants=[])
        assert event.is_action(["SEEMING", "APPEARING"])
        assert not event.is_action(["CLOSING", "APPEARING"])

    def test_containment_holds_in_both_directions(self):
        # The scene may record fewer features than the word carries, or more.
        # Either way one describes the other; only a partial cross is rejected.
        assert _denotes(["A", "B", "C"], ["A", "B"])
        assert _denotes(["A", "B"], ["A", "B", "C"])
        assert not _denotes(["A", "B"], ["B", "C"])
        assert not _denotes([], ["A"])
