"""Sequence memorization admits only complete, meaningful schedules."""

import pytest

from neural_assemblies.assembly_calculus.ops import sequence_memorize
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=31, engine="numpy_sparse")
    brain.add_stimulus("s0", 10)
    brain.add_stimulus("s1", 10)
    brain.add_area("A", 100, 10, 0.1)
    return brain


def test_sequence_memorize_preflights_all_stimuli_before_mutation():
    brain = _brain()
    with pytest.raises(KeyError, match="stimulus name"):
        sequence_memorize(brain, ["s0", "TYPO"], "A")
    assert len(brain.areas["A"].winners) == 0


@pytest.mark.parametrize("stimuli", ["s0", b"s0", 7])
def test_sequence_memorize_rejects_scalar_stimulus_input(stimuli):
    with pytest.raises(TypeError, match="ordered collection"):
        sequence_memorize(_brain(), stimuli, "A")


@pytest.mark.parametrize("kwargs", [
    {"rounds_per_step": 0}, {"repetitions": 0},
    {"phase_b_ratio": -0.1}, {"phase_b_ratio": 1.1},
    {"beta_boost": -0.1}, {"beta_boost": float("nan")},
])
def test_sequence_memorize_rejects_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        sequence_memorize(_brain(), ["s0", "s1"], "A", **kwargs)
