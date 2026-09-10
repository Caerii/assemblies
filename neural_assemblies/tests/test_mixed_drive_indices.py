"""Mixed drive uses stable neuron rows, with no invented or discarded IDs."""
import numpy as np
import pytest

from neural_assemblies import Brain


@pytest.fixture
def brain():
    brain = Brain(engine="numpy_sparse", p=0.2, seed=31, norm_init=False)
    brain.add_area("S", n=60, k=4, beta=0.1)
    brain.add_area("E", n=30, k=4, beta=0.1, explicit=True)
    state = brain._engine._areas["S"]
    state.compact_to_neuron_id = [8, 13, 29]
    state.w = 3
    brain.areas["S"].w = 3
    brain.areas["S"].winners = np.array([0, 2], dtype=np.uint32)
    weights = brain.connectomes["S"]["E"].weights
    weights[:] = 0
    weights[[0, 2]] = 99  # Wrong-space readout is unmistakable.
    weights[8] = 2
    weights[29] = 3
    return brain


def test_drive_sums_stable_neuron_rows(brain):
    np.testing.assert_array_equal(brain._sparse_sources_drive_to_explicit("E", ["S"]),
                                  np.full(30, 5, dtype=np.float32))


@pytest.mark.parametrize("winners", [[-1], [3], [0.5], [[0]], [2**32]])
def test_invalid_compact_winners_raise_without_changing_weights(brain, winners):
    before = brain.connectomes["S"]["E"].weights.copy()
    with pytest.raises(ValueError):
        brain.areas["S"].winners = np.asarray(winners)
        brain._sparse_sources_drive_to_explicit("E", ["S"])
    np.testing.assert_array_equal(brain.connectomes["S"]["E"].weights, before)


@pytest.mark.parametrize("mapping", [[], [-1, 13, 29], [60, 13, 29]])
def test_invalid_or_missing_sampled_mapping_cannot_become_identity(brain, mapping):
    brain._engine._areas["S"].compact_to_neuron_id = mapping
    with pytest.raises(ValueError):
        brain._sparse_sources_drive_to_explicit("E", ["S"])


def test_empty_source_contributes_zero(brain):
    brain.areas["S"].winners = np.array([], dtype=np.uint32)
    np.testing.assert_array_equal(brain._sparse_sources_drive_to_explicit("E", ["S"]),
                                  np.zeros(30, dtype=np.float32))


@pytest.mark.parametrize("value", [[-1], [0.5], [2**32], [[0]], [60]])
def test_area_rejects_invalid_assignment_before_mutation(brain, value):
    area = brain.areas["S"]
    before = area.winners.copy()
    count = area.w
    with pytest.raises(ValueError):
        area.winners = value
    np.testing.assert_array_equal(area.winners, before)
    assert area.w == count
