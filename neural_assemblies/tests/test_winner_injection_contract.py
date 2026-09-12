"""Reject malformed public inputs before conversion or partial synchronization."""
import numpy as np
import pytest

from neural_assemblies import Brain
from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.core.index_spaces import CompactIdx


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


def test_torch_set_winners_validates_before_device_conversion():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("TorchSparseEngine requires CUDA")
    from neural_assemblies.core.torch_engine import TorchSparseEngine
    engine = TorchSparseEngine(p=.1)
    engine.add_area("A", 8, 2, .1)
    with pytest.raises(ValueError):
        engine.set_winners("A", np.array([-1, 0], dtype=np.int64))
    assert len(engine.get_winners("A")) == 0


def test_engine_winner_getters_preserve_compact_brand(brain):
    brain.project(external_inputs={"A": [0, 1]}, projections={})
    assert isinstance(brain._engine.get_winners("A"), CompactIdx)


def test_stable_ids_cannot_be_laundered_through_public_sparse_injection():
    brain = Brain(engine="numpy_sparse", p=.1, seed=51)
    brain.add_area("A", 8, 2, .1)
    brain.materialize_area("A")
    brain.project(external_inputs={"A": [0, 1]}, projections={})
    stable = NeuronIds(np.array([0, 1], dtype=np.uint32))
    with pytest.raises(TypeError, match="compact indices"):
        brain.project(external_inputs={"A": stable}, projections={})
