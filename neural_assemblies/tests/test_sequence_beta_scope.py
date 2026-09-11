"""Temporary sequence plasticity boosts must be exception-safe."""

import pytest

from neural_assemblies.assembly_calculus.ops import sequence_memorize
from neural_assemblies.core.brain import Brain


def test_beta_boost_is_restored_when_recurrent_projection_fails():
    brain = Brain(p=0.05, seed=41, engine="numpy_sparse")
    brain.add_stimulus("s", 10)
    brain.add_area("A", 100, 10, 0.1)
    original_project = brain.project
    calls = 0

    def fail_during_recurrence(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 9:  # after the eight legacy Phase-A rounds
            raise RuntimeError("injected backend failure")
        return original_project(*args, **kwargs)

    brain.project = fail_during_recurrence
    with pytest.raises(RuntimeError, match="injected backend failure"):
        sequence_memorize(
            brain, ["s"], "A", beta_boost=0.5,
        )
    assert brain.areas["A"].beta == pytest.approx(0.1)
