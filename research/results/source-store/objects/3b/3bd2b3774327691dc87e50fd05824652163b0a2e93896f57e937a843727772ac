"""Regression tests for the bounded-state prediction path.

``StatePredictionMixin`` was wired into the parser but had NO tests, and was
silently non-functional: every state -> PREDICTION connectome stayed empty, so
retrieval overlap was exactly 0.0 for every prefix. Four earlier debugging
attempts failed to find the cause.

Root cause: an area->area connectome starts empty and is initialised LAZILY --
registered for deferred init during input accumulation and sampled at the END of
``project_into``. But an empty fiber delivers no drive, so a projection whose
only source is that fiber hits the "zero signal -> preserve current assembly"
early return and never reaches the init block. The fiber is stuck: it cannot
deliver drive until initialised, and is not initialised unless drive arrives.
Co-firing a stimulus into the target breaks the deadlock (the pattern
``_bootstrap_prediction_connectivity`` already used for the CONTEXT fiber).

These tests pin the observable consequences so the path cannot silently rot
again.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    CORE_AREAS, OBJ, PREDICTION, SUBJ,
)
from neural_assemblies.assembly_calculus.emergent.parser_mixins.state_prediction import (
    StatePredictionMixin,
)


class _RecordingBrain:
    def __init__(self):
        self.areas = {
            CORE_AREAS[0]: SimpleNamespace(active_count=0, w=10),
            CORE_AREAS[1]: SimpleNamespace(active_count=1, w=0),
            SUBJ: SimpleNamespace(active_count=0, w=10),
            OBJ: SimpleNamespace(active_count=0, w=10),
            PREDICTION: SimpleNamespace(active_count=1, w=1),
        }
        self.calls = []

    @contextmanager
    def frozen(self):
        yield self

    def project(self, stimuli, fibers):
        self.calls.append((stimuli, fibers))
        for targets in fibers.values():
            for target in targets:
                if target in (SUBJ, OBJ):
                    self.areas[target].active_count = 1

    def inhibit_areas(self, _areas):
        pass


def test_state_bootstrap_uses_active_sources_not_ambiguous_w():
    parser = StatePredictionMixin()
    parser.brain = _RecordingBrain()
    parser.stim_map = {"word": "phon"}
    parser._state_pred_bootstrapped = False

    parser._bootstrap_state_paths()

    seeded = [fibers for stimuli, fibers in parser.brain.calls if not stimuli]
    assert seeded == [
        {CORE_AREAS[1]: [SUBJ]},
        {CORE_AREAS[1]: [OBJ]},
    ]
    prediction_sources = {
        next(iter(fibers))
        for stimuli, fibers in parser.brain.calls
        if stimuli
    }
    assert prediction_sources == {CORE_AREAS[1], SUBJ, OBJ}
    assert CORE_AREAS[0] not in prediction_sources


def test_parser_rejects_unknown_sampled_recurrence_policy():
    with pytest.raises(ValueError, match="sampled_recurrence_policy"):
        EmergentParser(
            n=100, k=10, engine="numpy_sparse",
            sampled_recurrence_policy="silence-it",
        )


@pytest.fixture(scope="module")
def state_parser():
    parser = EmergentParser(
        n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10,
        sampled_recurrence_policy="acknowledged",
    )
    sentences = create_training_sentences()
    parser.train(sentences)
    parser.train_next_token_state(sentences)
    return parser


def _conn_nnz(parser, src):
    import numpy as np
    conn = parser.brain._engine._area_conns.get(src, {}).get(PREDICTION)
    if conn is None or conn.weights is None:
        return 0
    return int((np.asarray(conn.weights) != 0).sum())


@pytest.mark.slow
class TestStatePathMaterializes:
    """The bug: these connectomes were all shape (0,0), nnz=0, forever."""

    @pytest.mark.parametrize("src", ["DET_CORE", "NOUN_CORE", "SUBJ", "MOOD"])
    def test_state_fiber_is_materialized(self, state_parser, src):
        assert _conn_nnz(state_parser, src) > 0, (
            f"{src}->PREDICTION never materialized -- the lazy-init deadlock "
            "has regressed (see module docstring)")


@pytest.mark.slow
class TestStateRetrieval:
    def test_predictions_are_nonzero(self, state_parser):
        # Was exactly 0.0 for every word, for every prefix.
        preds = state_parser.predict_next_state(["the"], top_k=5)
        assert preds, "no predictions returned"
        assert preds[0][1] > 0.0, f"top score still zero: {preds[:3]}"

    def test_predictions_are_ranked(self, state_parser):
        preds = state_parser.predict_next_state(["the", "dog"], top_k=5)
        scores = [s for _, s in preds]
        assert scores == sorted(scores, reverse=True)

    def test_predictions_depend_on_prefix(self, state_parser):
        # A bounded state that ignored its input would rank identically for
        # every prefix -- which is what a saturated CONTEXT buffer does.
        a = state_parser.predict_next_state(["the", "dog"], top_k=5)
        b = state_parser.predict_next_state(["the", "cat", "chases"], top_k=5)
        assert [w for w, _ in a] != [w for w, _ in b] or \
               [s for _, s in a] != [s for _, s in b], (
            "state prediction is prefix-independent")

    def test_empty_prefix_is_safe(self, state_parser):
        assert state_parser.predict_next_state([]) == []
