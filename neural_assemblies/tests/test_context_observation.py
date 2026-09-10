"""A prefix probe reuses neuron identity; sentence construction may reset it."""
import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.emergent.core.areas import CONTEXT, PREDICTION
from neural_assemblies.assembly_calculus.emergent.parser_mixins.incremental import IncrementalMixin
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import _settle_context_into_prediction


class ContextHarness(IncrementalMixin):
    """Actual brain and prefix loop, without unrelated lexicon training."""
    inference_rounds = 1

    def __init__(self):
        self.brain = Brain(engine="numpy_sparse", p=.1, seed=12, norm_init=False)
        for name in (CONTEXT, PREDICTION):
            self.brain.add_area(name, 300, 20, .1)
        self.brain.add_stimulus("word", 20)
        self.brain.project({"word": [CONTEXT, PREDICTION]}, {})
        self.brain.project({}, {CONTEXT: [PREDICTION]})

    def _bootstrap_prediction_connectivity(self):
        pass  # Both populations were explicitly initialized above.

    def _advance_context_direct(self, word):
        self.brain.project({word: [CONTEXT]}, {})
        return "NOUN"

    def _clear_prediction_activity(self):
        self.brain.inhibit_areas([PREDICTION])


@pytest.mark.parametrize("bridge", [False, True])
def test_legacy_context_reset_is_rejected_before_mutating_any_identity(bridge):
    parser = ContextHarness()
    state = parser.brain._engine._areas[CONTEXT]
    identities, count, cursor = list(state.compact_to_neuron_id), state.w, state.neuron_id_pool_ptr
    winners = state.winners.copy()
    with parser.brain.read_only():
        with pytest.raises(ValueError, match="cannot reset CONTEXT"):
            if bridge:
                parser._reset_context_for_bridge(preserve_topology=True)
            else:
                parser._reset_context_state()
        np.testing.assert_array_equal(state.winners, winners)
        assert state.compact_to_neuron_id == identities
    assert state.compact_to_neuron_id == identities
    assert (state.w, state.neuron_id_pool_ptr) == (count, cursor)


def test_prefix_settling_uses_existing_population_and_restores_activity():
    parser = ContextHarness()
    state = parser.brain._engine._areas[CONTEXT]
    identities, count, winners = list(state.compact_to_neuron_id), state.w, state.winners.copy()
    with parser.brain.read_only():
        _settle_context_into_prediction(parser, ("word", "word"))
        assert len(parser.brain.areas[PREDICTION].winners) == 20
        assert state.w == count
        assert state.compact_to_neuron_id == identities
    np.testing.assert_array_equal(state.winners, winners)
    assert state.compact_to_neuron_id == identities


def test_construction_reset_remains_explicitly_destructive_outside_observation():
    parser = ContextHarness()
    parser._reset_context_state()
    state = parser.brain._engine._areas[CONTEXT]
    assert state.w == 0
    assert state.compact_to_neuron_id == []
    assert state.neuron_id_pool_ptr == 0


@pytest.mark.parametrize("requested_capacity", [0, 1, 80])
def test_bridge_reset_preserves_actual_population_not_requested_capacity(requested_capacity):
    parser = ContextHarness()
    state = parser.brain._engine._areas[CONTEXT]
    identities = list(state.compact_to_neuron_id)
    count = state.w
    parser._context_ring_capacity_cols = requested_capacity
    parser._reset_context_for_bridge(preserve_topology=True)
    assert state.w == count
    assert parser.brain.areas[CONTEXT].w == count
    assert state.compact_to_neuron_id == identities
    assert not len(state.winners)
    parser._advance_context_direct("word")
    from neural_assemblies.assembly_calculus.ops import _snap
    snapshot = _snap(parser.brain, CONTEXT)
    assert len(snapshot.winners) == parser.brain.areas[CONTEXT].k
