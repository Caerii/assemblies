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
import pytest

from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
    create_training_sentences,
)
from neural_assemblies.assembly_calculus.emergent.core.areas import PREDICTION

pytestmark = pytest.mark.slow  # full parser training (~1 min)


@pytest.fixture(scope="module")
def state_parser():
    parser = EmergentParser(n=1000, k=50, p=0.05, beta=0.1, seed=42, rounds=10)
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


class TestStatePathMaterializes:
    """The bug: these connectomes were all shape (0,0), nnz=0, forever."""

    @pytest.mark.parametrize("src", ["DET_CORE", "NOUN_CORE", "SUBJ", "MOOD"])
    def test_state_fiber_is_materialized(self, state_parser, src):
        assert _conn_nnz(state_parser, src) > 0, (
            f"{src}->PREDICTION never materialized -- the lazy-init deadlock "
            "has regressed (see module docstring)")


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
