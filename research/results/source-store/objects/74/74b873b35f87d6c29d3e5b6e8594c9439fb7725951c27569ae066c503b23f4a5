"""The scaffold sequence wrapper must preflight before adding topology."""

import pytest

from neural_assemblies.assembly_calculus.scaffold import sequence_memorize_scaffold
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=73, engine="numpy_sparse")
    brain.add_stimulus("s0", 10)
    brain.add_area("MAIN", 100, 10, 0.1)
    return brain


def test_scaffold_preflights_stimuli_before_adding_auxiliary_area():
    brain = _brain()
    with pytest.raises(KeyError, match="stimulus name"):
        sequence_memorize_scaffold(brain, ["s0", "TYPO"], "MAIN", "AUX")
    assert "AUX" not in brain.areas


@pytest.mark.parametrize("stimuli", ["s0", b"s0", 4, []])
def test_scaffold_rejects_non_sequence_input(stimuli):
    with pytest.raises((TypeError, ValueError)):
        sequence_memorize_scaffold(_brain(), stimuli, "MAIN", "AUX")
