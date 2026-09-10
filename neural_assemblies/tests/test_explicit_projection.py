"""Explicit projection bookkeeping and observable Hebbian updates."""
import numpy as np
import pytest

from neural_assemblies import Brain


@pytest.fixture
def brain():
    return Brain(p=0.1, save_size=True, save_winners=True, seed=123,
                 engine="numpy_sparse", norm_init=False)


def test_explicit_target_marks_fired_winners(brain):
    brain.add_area("T", n=20, k=5, beta=0.05, explicit=True)
    brain.add_stimulus("S", size=50)
    brain.project({"S": ["T"]}, {})
    area = brain.areas["T"]
    assert area.num_first_winners == 0
    assert len(area.winners) == area.k
    assert np.all(area.ever_fired[area.winners])
    assert area.num_ever_fired == int(np.sum(area.ever_fired))


def test_invalid_source_assignment_fails_before_projection(brain):
    brain.add_area("SRC", n=10, k=3, beta=0.05, explicit=True)
    before = brain.areas["SRC"].winners.copy()
    with pytest.raises(ValueError):
        brain.areas["SRC"].winners = np.array([0, 5, 999], dtype=np.uint32)
    np.testing.assert_array_equal(brain.areas["SRC"].winners, before)


@pytest.mark.parametrize("beta", [0.0, 0.2])
def test_explicit_plasticity_updates_only_active_pairs(brain, beta):
    brain.add_area("SRC", n=10, k=3, beta=beta, explicit=True)
    brain.add_area("T", n=12, k=3, beta=beta, explicit=True)
    brain.areas["SRC"].winners = np.array([0, 1, 2], dtype=np.uint32)
    weights = brain.connectomes["SRC"]["T"].weights
    weights[:] = 0
    weights[:3] = 1
    before = weights.copy()
    brain.project({}, {"SRC": ["T"]})
    posts = brain.areas["T"].winners
    assert len(posts) == 3  # An empty probe must not satisfy the update check.
    expected = before.copy()
    expected[np.ix_(np.arange(3), posts)] *= 1 + beta
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-6)
    if beta:
        assert np.any(weights != before)
    else:
        np.testing.assert_array_equal(weights, before)
