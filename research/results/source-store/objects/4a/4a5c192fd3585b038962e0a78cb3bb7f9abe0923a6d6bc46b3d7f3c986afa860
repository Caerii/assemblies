"""The paper's empty-project detector, firing on the REAL emergent parser.

WHY THIS FILE EXISTS SEPARATELY FROM test_parse_errors.py. That file tests the
detector primitives on synthetic inputs, which proves they compute what they
claim and nothing about whether anything calls them. This repo's dominant
failure is a mechanism that is implemented, wired, documented and never
executed -- mutual inhibition sat dormant through 1373 project() calls (#24),
and the P600 detector has a margin 11.9x above anything observable (#104). So
the load-bearing assertion here is that a violation the PAPER describes
produces an error on a parser built the normal way.

WHAT HAD TO CHANGE FOR IT TO BE POSSIBLE. The parser's rules were TOTAL:
`_get_syntactic_target` clamps with `min(noun_count, len(seq) - 1)` and always
returns SUBJ or OBJ, so every word always had somewhere to go and no violation
could ever surface. It also gated only FIBERS; the paper's mechanism is an
inhibited AREA ("has not disinhibited area OBJ"), which needed
`Brain.inhibit_area` to exist at all.

MEASURED while building this, and it is the reason for the PP tests below:
without a prepositional guard, "the dog sleeps on the table" reported TWO
errors -- exactly the count of the genuine violation "the dog sleeps the cat".
A detector that cannot tell a grammatical PP from an ungrammatical object is
worse than none, because it reports confidently.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
    infer_transitive_verbs,
)
from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser

N, K = 1000, 30


@pytest.fixture(scope="module")
def sentences():
    return create_training_sentences()


@pytest.fixture(scope="module")
def trained(sentences):
    p = EmergentParser(n=N, k=K, seed=1, fast_training=True)
    p.train(sentences)
    return p


def _errors(parser, words) -> int:
    parser.build_context_incremental(list(words), reset=True)
    return len(parser.parse_errors)


class TestTransitivityIsLearned:

    def test_transitive_verbs_come_from_the_corpus_not_a_hand_list(self, sentences):
        """Distributional, using role annotation `train_roles` already consumes.

        A hand-written list would make the detector's competence a property of
        the list rather than of the model -- the circularity
        [[categories-are-annotation-driven]] records for POS induction.
        """
        learned = infer_transitive_verbs(sentences)
        assert learned, "no transitive verbs inferred; the corpus has no patients?"
        assert "chases" in learned
        assert "sleeps" not in learned, (
            f"'sleeps' was inferred transitive from {sorted(learned)} -- the "
            f"corpus gives it a patient somewhere, or the inference is wrong")


class TestEmptyProjectFiresOnTheRealParser:

    def test_an_intransitive_verb_followed_by_an_object_is_DETECTED(self, trained,
                                                                    sentences):
        """The paper's example, in this corpus's vocabulary."""
        trained.transitive_verbs = infer_transitive_verbs(sentences)
        try:
            assert _errors(trained, ["the", "dog", "sleeps", "the", "cat"]) > 0
        finally:
            trained.transitive_verbs = None

    def test_and_the_SAME_frame_with_a_transitive_verb_is_CLEAN(self, trained,
                                                               sentences):
        """The positive control, and it is the whole test.

        Identical structure, one word different. Without this, the test above
        passes on a parser that flags every sentence.
        """
        trained.transitive_verbs = infer_transitive_verbs(sentences)
        try:
            assert _errors(trained, ["the", "dog", "chases", "the", "cat"]) == 0
        finally:
            trained.transitive_verbs = None

    @pytest.mark.parametrize("words", [
        ["the", "dog", "sleeps"],                                  # no object
        ["the", "dog", "chases", "the", "cat"],                    # transitive
        ["the", "dog", "chases", "the", "cat", "in", "the", "car"],  # + PP
        ["the", "dog", "sleeps", "on", "the", "table"],            # intrans + PP
    ])
    def test_grammatical_sentences_are_never_flagged(self, trained, sentences,
                                                     words):
        """FALSE POSITIVES ARE THE FAILURE MODE THAT MATTERS.

        The last case is the one that actually broke: an intransitive verb
        followed by a prepositional phrase reported 2 errors before the PP
        guard, the same as a real violation.
        """
        trained.transitive_verbs = infer_transitive_verbs(sentences)
        try:
            assert _errors(trained, words) == 0, (
                f"grammatical sentence {' '.join(words)!r} was flagged")
        finally:
            trained.transitive_verbs = None


class TestItStaysOffUntilTransitivityIsKnown:

    def test_no_transitivity_information_means_no_detections(self, trained):
        """Default behaviour is unchanged, and silence here is CORRECT.

        An untrained parser rejecting every object would turn a missing lexicon
        into a stream of confident violations -- an apparatus defect reading as
        a linguistic result.
        """
        trained.transitive_verbs = None
        assert _errors(trained, ["the", "dog", "sleeps", "the", "cat"]) == 0

    def test_no_gating_state_is_allocated_either(self, sentences):
        """A FRESH parser, deliberately -- the shared fixture has been gated by
        earlier tests, and asserting on it would only measure test order."""
        p = EmergentParser(n=N, k=K, seed=2, fast_training=True)
        p.train(sentences)
        p.build_context_incremental(["the", "dog", "sleeps"], reset=True)
        assert p.brain._inhibition is None, (
            "the parser allocated a gate with no transitivity information, so "
            "every Brain now pays the per-projection inhibition check")


class TestErrorsDoNotLeakAcrossSentences:

    def test_a_violation_does_not_contaminate_the_NEXT_sentence(self, trained,
                                                                sentences):
        """A gate left closed from the last sentence produces confident wrong
        detections, which is worse than detecting nothing."""
        trained.transitive_verbs = infer_transitive_verbs(sentences)
        try:
            assert _errors(trained, ["the", "dog", "sleeps", "the", "cat"]) > 0
            assert _errors(trained, ["the", "dog", "chases", "the", "cat"]) == 0
        finally:
            trained.transitive_verbs = None


class TestKnownGap:

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN GAP, pinned deliberately. The PP guard stops a noun after a "
        "preposition being FLAGGED, but the circuit declares no `core -> PP` "
        "fiber, so the noun cannot bind into PP either. Suppressing a false "
        "alarm is not the same as routing the word, and only the first is "
        "done. Fixing it means adding core->PP fibers and PP-aware targeting, "
        "which changes every prepositional parse."))
    def test_the_circuit_declares_a_route_from_a_noun_into_PP(self, trained):
        """ASSERTS THE STRUCTURE, not a proxy for it.

        The first version of this test asserted `len(brain.areas[PP].winners)
        > 0` and XPASSED -- PP does have winners, but they are the
        PREPOSITION's, put there by `PREP_CORE -> PP`. The assertion was true
        and measured nothing about the noun, which is the same mistake as
        scoring a contrast on an area that only one arm ever writes. The
        structural claim is crisp and cannot pass by accident: is there a
        declared fiber from a noun core into PP?
        """
        from neural_assemblies.assembly_calculus.emergent.core.areas import (
            NOUN_CORE, PP,
        )

        circuit = trained._get_incremental_circuit(reset=True)
        circuit.is_active(NOUN_CORE, PP)     # raises KeyError while undeclared
