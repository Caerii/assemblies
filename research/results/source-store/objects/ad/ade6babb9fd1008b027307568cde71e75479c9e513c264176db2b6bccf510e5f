"""C4: a probe must preserve the dynamical state consumed by the next probe."""
import copy

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain


def brain_for(engine):
    brain = Brain(engine=engine, p=.1, norm_init=False, seed=12, save_winners=True)
    brain.add_area("A", 300, 20, .1)
    brain.add_stimulus("s", 20)
    brain.add_stimulus("t", 20)
    brain.project({"s": ["A"]}, {})
    return brain


@pytest.mark.parametrize("engine", ["numpy_exact", "numpy_sparse", "numpy_explicit"])
def test_probe_preserves_recorded_activity_and_counts(engine):
    brain = brain_for(engine)
    area = brain.areas["A"]
    before_count = area.get_num_ever_fired()
    before_history = len(area.saved_winners), len(area.saved_w)
    backend_count = brain._engine.get_num_ever_fired("A")
    with brain.read_only():
        brain.project({"t": ["A"]}, {})
    assert area.get_num_ever_fired() == before_count
    assert brain._engine.get_num_ever_fired("A") == backend_count
    assert (len(area.saved_winners), len(area.saved_w)) == before_history


def test_probe_restores_refractory_history_even_on_exception():
    brain = brain_for("numpy_sparse")
    brain.set_lri("A", refractory_period=3, inhibition_strength=5.)
    state = brain._engine._areas["A"]
    before = copy.deepcopy(state._refractory_history)
    with pytest.raises(RuntimeError, match="stop"):
        with brain.read_only():
            brain.project({"t": ["A"]}, {"A": ["A"]})
            assert state._refractory_history != before
            raise RuntimeError("stop")
    assert state._refractory_history == before


def test_empty_sparse_probe_does_not_materialize_a_hidden_connectome():
    brain = Brain(engine="numpy_sparse", p=.1, norm_init=False)
    brain.add_area("A", 300, 20)
    brain.add_stimulus("s", 20)
    before = brain._engine._stim_conns["s"]["A"].weights.copy()
    # Reading an uninitialized sampled population is not an initialization API.
    with pytest.raises(ValueError, match="materializ"):
        with brain.read_only():
            brain.project({"s": ["A"]}, {})
    assert brain._engine._areas["A"].w == 0
    np.testing.assert_array_equal(before, brain._engine._stim_conns["s"]["A"].weights)


def test_exact_probe_records_the_population_its_drive_was_summed_over():
    brain = brain_for("numpy_exact")
    brain.record_activation = True
    with brain.read_only():
        brain.project({"t": ["A"]}, {})
        assert brain.last_pre_kwta_counts["A"] == 300


def test_nested_probe_restores_the_outer_state_and_buffer_references():
    brain = brain_for("numpy_exact")
    area = brain.areas["A"]
    backend = brain._engine._areas["A"]
    mask = backend.ever_fired
    saved_mask = mask.copy()
    history = area.saved_winners
    with brain.read_only():
        brain.project({"t": ["A"]}, {})
        outer = area.winners.copy()
        with brain.read_only():
            brain.project({"s": ["A"]}, {})
        np.testing.assert_array_equal(area.winners, outer)
    assert backend.ever_fired is mask
    assert area.saved_winners is history
    np.testing.assert_array_equal(mask, saved_mask)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("with_stimulus", [False, True])
def test_read_only_cannot_initialize_an_unused_fiber(eager, with_stimulus):
    brain = brain_for("numpy_sparse")
    brain.add_area("B", 300, 20, .1)
    brain.project({"s": ["B"]}, {})
    brain._engine.eager_fiber_init = eager
    unobserved = copy.deepcopy(brain)
    frozen_control = copy.deepcopy(brain)
    fiber = brain._engine._area_conns["A"]["B"]
    before = fiber.weights.copy()
    assert before.shape == (0, 0)
    stims = {"t": ["B"]} if with_stimulus else {}
    with brain.read_only():
        brain.project(stims, {"A": ["B"]})
        np.testing.assert_array_equal(fiber.weights, before)
    # Freezing plasticity alone still permits this construction.
    with frozen_control.frozen():
        frozen_control.project(stims, {"A": ["B"]})
    assert frozen_control._engine._area_conns["A"]["B"].weights.size > 0
    brain.project(stims, {"A": ["B"]})
    unobserved.project(stims, {"A": ["B"]})
    np.testing.assert_array_equal(brain.areas["B"].winners, unobserved.areas["B"].winners)
    np.testing.assert_array_equal(fiber.weights,
                                  unobserved._engine._area_conns["A"]["B"].weights)
