"""A fork must preserve the instrument, its next transition, and owned aliases."""
import copy

import numpy as np
import pytest

from neural_assemblies import Brain


@pytest.mark.parametrize("norm_init,recurrent", [(False, False), (True, True)])
def test_brain_clone_keeps_configuration_and_next_projection(norm_init, recurrent):
    brain = Brain(engine="numpy_sparse", p=0.2, seed=19, norm_init=norm_init,
                  recurrent_projection=recurrent, save_winners=True)
    brain.add_area("A", n=120, k=6, beta=0.1)
    brain.add_stimulus("cue", 20)
    brain.project({"cue": ["A"]}, {})
    brain.record_activation = True
    fork = brain.clone()
    assert vars(fork).keys() == vars(brain).keys()
    assert fork.norm_init == norm_init
    assert fork.recurrent_projection == recurrent
    parent_weights = brain.connectomes_by_stimulus["cue"]["A"].weights.copy()
    fork.project_rounds("A", {"cue": ["A"]}, {}, 2)
    np.testing.assert_array_equal(brain.connectomes_by_stimulus["cue"]["A"].weights, parent_weights)
    brain.project_rounds("A", {"cue": ["A"]}, {}, 2)
    np.testing.assert_array_equal(fork.areas["A"].winners, brain.areas["A"].winners)
    np.testing.assert_array_equal(fork.connectomes_by_stimulus["cue"]["A"].weights,
                                  brain.connectomes_by_stimulus["cue"]["A"].weights)
    assert fork.last_pre_kwta_counts == brain.last_pre_kwta_counts
    assert fork._engine._rng.bit_generator.state == brain._engine._rng.bit_generator.state


def test_brain_clone_keeps_internal_aliases_without_sharing_with_parent():
    brain = Brain(engine="numpy_sparse", seed=12)
    brain.add_area("A", n=120, k=6, beta=0.1)
    brain.add_stimulus("cue", 20)
    brain.project({"cue": ["A"]}, {})
    fork = brain.clone()
    connection = fork.connectomes_by_stimulus["cue"]["A"]
    assert connection is fork._engine._stim_conns["cue"]["A"]
    assert connection is not brain.connectomes_by_stimulus["cue"]["A"]
    assert not np.shares_memory(connection.weights, brain.connectomes_by_stimulus["cue"]["A"].weights)
    # The independent mixed-connectome RNG was omitted by the old field list.
    assert fork._conn_rng_gen.bit_generator.state == brain._conn_rng_gen.bit_generator.state
    fork._conn_rng_gen.random()
    assert fork._conn_rng_gen.bit_generator.state != brain._conn_rng_gen.bit_generator.state


def test_engine_clone_preserves_deferred_scaling_and_observation_state():
    brain = Brain(engine="numpy_sparse", seed=13, synaptic_scaling={"A"},
                  synaptic_scaling_deferred=True)
    brain.add_area("A", n=120, k=6, beta=0.1)
    brain.add_stimulus("cue", 20)
    brain.project({"cue": ["A"]}, {})
    engine = brain._engine
    original = copy.deepcopy(engine.__dict__)
    fork = engine.clone()
    assert vars(fork).keys() == vars(engine).keys()
    assert fork.synaptic_scaling_deferred == engine.synaptic_scaling_deferred
    assert fork._rng.bit_generator.state == engine._rng.bit_generator.state
    # Construction of a clone must not consume randomness in the original.
    assert engine._rng.bit_generator.state == original["_rng"].bit_generator.state


def test_mixed_brain_clone_keeps_secondary_engine_and_connection_ownership():
    brain = Brain(engine="numpy_sparse", seed=21, norm_init=False)
    brain.add_area("S", n=120, k=6, beta=0.1)
    brain.add_area("E", n=120, k=6, beta=0.1, explicit=True)
    brain.add_stimulus("cue", 20)
    brain.project({"cue": ["E", "S"]}, {})
    assert brain._explicit_engine is not None
    fork = brain.clone()
    assert fork._explicit_engine is not None
    assert fork._explicit_engine is not brain._explicit_engine
    for name in ("E", "S"):
        connection = fork.connectomes_by_stimulus["cue"][name]
        engine = fork._explicit_engine if name == "E" else fork._engine
        assert connection is engine._stim_conns["cue"][name]
    fork.project({"cue": ["E", "S"]}, {})
    brain.project({"cue": ["E", "S"]}, {})
    for name in ("E", "S"):
        np.testing.assert_array_equal(fork.areas[name].winners, brain.areas[name].winners)
