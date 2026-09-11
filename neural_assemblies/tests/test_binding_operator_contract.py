"""The low-level binding operator admits only resolved, typed schedules."""

import pytest

from neural_assemblies.assembly_calculus.binding import bind
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=17, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    return brain


def test_bind_rejects_unknown_area_names():
    with pytest.raises(KeyError, match="area name"):
        bind(_brain(), sources=["TYPO"], target_area="DST")


@pytest.mark.parametrize("rounds", [0, -1, 1.5, True])
def test_bind_rejects_ambiguous_round_schedule(rounds):
    with pytest.raises(ValueError, match="rounds must be a positive integer"):
        bind(_brain(), sources=["SRC"], target_area="DST", rounds=rounds)
