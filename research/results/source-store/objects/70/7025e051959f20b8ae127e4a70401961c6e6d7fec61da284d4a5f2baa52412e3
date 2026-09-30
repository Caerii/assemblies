"""Separation validates the complete schedule before mutating the brain."""

import pytest

from neural_assemblies.assembly_calculus.ops import separate
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=19, engine="numpy_sparse")
    brain.add_stimulus("A", 10)
    brain.add_stimulus("B", 10)
    brain.add_area("X", 100, 10, 0.1)
    return brain


def test_separate_preflights_second_stimulus_before_first_projection():
    brain = _brain()
    with pytest.raises(KeyError, match="stimulus"):
        separate(brain, "A", "TYPO", "X")
    assert len(brain.areas["X"].winners) == 0


def test_separate_rejects_same_stimulus_and_invalid_schedule():
    brain = _brain()
    with pytest.raises(ValueError, match="distinct stimuli"):
        separate(brain, "A", "A", "X")
    with pytest.raises(ValueError, match="positive integer"):
        separate(brain, "A", "B", "X", rounds=0)
