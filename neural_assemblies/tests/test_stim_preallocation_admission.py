"""Stimulus preallocation is a capability, not an inherited receipt."""

import pytest

from neural_assemblies import Brain


def test_dense_engine_rejects_storage_preallocation():
    brain = Brain(engine="numpy_exact", norm_init=False)
    brain.add_stimulus("s", 5)
    brain.add_area("A", 20, 2, 0.1)
    assert not brain._engine.supports_stim_preallocation
    with pytest.raises(NotImplementedError, match="stimulus preallocation"):
        brain._engine.preallocate_stim_targets("A", 10)


def test_sparse_engine_preallocation_extends_the_live_vector():
    brain = Brain(engine="numpy_sparse", norm_init=False)
    brain.add_stimulus("s", 5)
    brain.add_area("A", 20, 2, 0.1)
    engine = brain._engine
    conn = engine._stim_conns["s"]["A"]
    assert engine.supports_stim_preallocation
    assert len(conn.weights) == 0
    engine.preallocate_stim_targets("A", 10)
    assert len(conn.weights) == 10
    assert conn.weights.tolist() == [0.0] * 10
