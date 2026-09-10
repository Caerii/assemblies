"""Reject malformed public inputs before conversion or partial synchronization."""
import numpy as np
import pytest

from neural_assemblies import Brain


@pytest.fixture(params=["numpy_sparse", "numpy_exact", "numpy_explicit"])
def brain(request):
    brain = Brain(engine=request.param, norm_init=False, seed=51)
    for name in ("A", "B"):
        brain.add_area(name, 8, 2, .1)
    return brain


@pytest.mark.parametrize("bad", [[.5], [-1], [2**32], [8], [[0]], [0, 0]])
def test_direct_injection_validates_all_inputs_before_mutation(brain, bad):
    with pytest.raises(ValueError):
        brain.project(external_inputs={"A": [0, 1], "B": bad}, projections={})
    for name in ("A", "B"):
        assert len(brain.areas[name].winners) == 0
        assert len(brain._engine.get_winners(name)) == 0


def test_invalid_route_does_not_leave_injected_winners(brain):
    with pytest.raises(IndexError):
        brain.project(external_inputs={"A": [0, 1]}, projections={"A": ["missing"]})
    assert len(brain.areas["A"].winners) == 0
    assert len(brain._engine.get_winners("A")) == 0


@pytest.mark.parametrize("bad", [[.5], [-1], [2**32], [8], [[0]], [0, 0]])
def test_source_sync_rechecks_all_public_buffers_before_sync(brain, bad):
    brain.areas["A"].winners = [0, 1]
    brain.areas["B"]._winners = np.array(bad)
    with pytest.raises(ValueError):
        brain.project({}, {"A": ["A"], "B": ["A"]})
    assert len(brain._engine.get_winners("A")) == 0
    assert len(brain._engine.get_winners("B")) == 0


@pytest.mark.parametrize("bad", [[.5], [-1], [2**32], [8], [[0]], [0, 0]])
def test_direct_engine_assignment_uses_same_index_validation(brain, bad):
    with pytest.raises(ValueError):
        brain._engine.set_winners("A", bad)
    assert len(brain._engine.get_winners("A")) == 0


def test_valid_injection_and_clear(brain):
    brain.project(external_inputs={"A": [0, 1]}, projections={})
    np.testing.assert_array_equal(brain._engine.get_winners("A"), [0, 1])
    brain.project(external_inputs={"A": []}, projections={})
    assert len(brain._engine.get_winners("A")) == 0
