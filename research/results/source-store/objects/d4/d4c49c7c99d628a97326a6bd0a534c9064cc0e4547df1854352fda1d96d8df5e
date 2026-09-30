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



@pytest.mark.parametrize('materialize', [False, True])
def test_dense_bootstrap_reserves_selected_ids_before_later_recruitment(materialize):
    brain = Brain(engine='numpy_sparse', p=.2, seed=31, norm_init=False)
    brain.add_area('S', 20, 2)
    engine = brain._engine
    state = engine._areas['S']
    pool = state.neuron_id_pool.copy()
    selected = pool[-2:].tolist()
    drive = np.zeros(20, dtype=np.float32)
    drive[selected[0]], drive[selected[1]] = 100, 99
    engine._bootstrap_from_explicit_dense('S', drive, [], [], False)
    if materialize:
        engine.materialize_area('S')
        assert len(set(state.compact_to_neuron_id)) == 20
    else:
        assert state.compact_to_neuron_id == selected
        pending = state.neuron_id_pool[state.neuron_id_pool_ptr:].tolist()
        assert not set(selected).intersection(pending)
        assert pending == [int(i) for i in pool if i not in selected]



def test_initial_reservation_without_pool_uses_identity_remainder():
    from neural_assemblies.core.index_spaces import reserve_initial_neuron_ids
    np.testing.assert_array_equal(reserve_initial_neuron_ids(None, [3, 1], n=5),
                                  [3, 1, 0, 2, 4])


@pytest.mark.parametrize('pool,selected', [([0, 0, 2], [1]), ([0, 1], [0]),
                                         ([0, 1, 2], [1, 1]), ([0, 1, 2], [3])])
def test_invalid_initial_reservation_does_not_change_inputs(pool, selected):
    from neural_assemblies.core.index_spaces import reserve_initial_neuron_ids
    before = list(pool)
    with pytest.raises(ValueError):
        reserve_initial_neuron_ids(pool, selected, n=3)
    assert pool == before
