"""Supervised dense reinforcement has explicit coordinates and mutation extent."""
import numpy as np
import pytest

from neural_assemblies import Brain


@pytest.fixture
def brain():
    b = Brain(engine="numpy_sparse", seed=71, norm_init=False, w_max=5)
    b.add_area("S", 8, 2, .2, explicit=True)
    b.add_area("T", 6, 2, .2, explicit=True)
    b.areas["S"].winners = [1, 3]
    b.connectomes["S"]["T"].weights[:] = 0
    return b


def test_selected_block_only_and_supervised_zero_seed(brain):
    weights = brain.connectomes["S"]["T"].weights
    weights[0, 0] = 9
    weights[1, 2] = 4
    expected = weights.copy()
    expected[np.ix_([1, 3], [2, 4])] = [[5, 2], [2, 2]]
    brain.reinforce_connectome("S", "T", [2, 4], beta=1)
    np.testing.assert_array_equal(weights, expected)


@pytest.mark.parametrize("disabled", ["zero", "frozen", "mask"])
def test_learning_null_does_not_seed_zero_edges(brain, disabled):
    if disabled == "frozen":
        brain.disable_plasticity = True
    if disabled == "mask":
        brain.set_fiber_plasticity("S", "T", False)
    brain.reinforce_connectome("S", "T", [2], beta=0 if disabled == "zero" else 1)
    assert not brain.connectomes["S"]["T"].weights.any()


@pytest.mark.parametrize("post", [[-1], [.5], [6], [2**32], [[1]], [1, 1]])
@pytest.mark.parametrize("disabled", [False, True])
def test_bad_post_is_rejected_even_with_learning_disabled(brain, post, disabled):
    brain.disable_plasticity = disabled
    with pytest.raises(ValueError):
        brain.reinforce_connectome("S", "T", post, beta=0)
    assert not brain.connectomes["S"]["T"].weights.any()


@pytest.mark.parametrize("beta", [-1, float("nan"), float("inf")])
def test_invalid_beta_rejected_before_mutation(brain, beta):
    with pytest.raises(ValueError):
        brain.reinforce_connectome("S", "T", [2], beta=beta)
    assert not brain.connectomes["S"]["T"].weights.any()


def test_sampled_source_uses_stable_dense_rows():
    b = Brain(engine="numpy_sparse", seed=71, norm_init=False)
    b.add_area("S", 8, 2, .2)
    b.add_area("T", 6, 2, .2, explicit=True)
    b._engine._areas["S"].compact_to_neuron_id = [5, 7]
    b._engine._areas["S"].w = 2
    b.areas["S"].winners = [0, 1]
    weights = b.connectomes["S"]["T"].weights
    weights[:] = 0
    b.reinforce_connectome("S", "T", [2], beta=1)
    expected = np.zeros((8, 6), dtype=np.float32)
    expected[[5, 7], 2] = 2
    np.testing.assert_array_equal(weights, expected)


def test_unbounded_overflow_rejected_without_writing(brain):
    brain.w_max = None
    with pytest.raises(ValueError, match="representable"):
        brain.reinforce_connectome("S", "T", [2], beta=1e300)
    assert not brain.connectomes["S"]["T"].weights.any()


def test_compact_storage_is_not_treated_as_stable_slot_axes():
    b = Brain(engine="numpy_sparse", seed=71, norm_init=False)
    b.add_area("S", 8, 2, .2)
    b.add_area("T", 6, 2, .2)
    with pytest.raises(NotImplementedError, match="dense"):
        b.reinforce_connectome("S", "T", [2], beta=1)


def test_teacher_does_not_write_stable_ids_into_compact_source():
    from neural_assemblies.programs.patch_merge import _teacher_align_high_to_mid
    from neural_assemblies.programs.patch_merge import HIGH
    b = Brain(engine="numpy_sparse", seed=71, norm_init=False)
    b.add_area("S", 8, 2, .2)
    b.add_area(HIGH, 6, 2, .2, explicit=True)
    b._engine._areas["S"].compact_to_neuron_id = [5, 7]
    b._engine._areas["S"].w = 2
    b.areas["S"].winners = [0, 1]
    weights = b.connectomes["S"][HIGH].weights
    weights[:] = 0
    _teacher_align_high_to_mid(b, np.array([0, 0, 1, 0, 0, 0]), "S", teacher_beta=1)
    np.testing.assert_array_equal(b.areas["S"].winners, [0, 1])
    assert weights[5, 2] == weights[7, 2] == 2
