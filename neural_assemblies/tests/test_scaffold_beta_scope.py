"""Scaffold recurrent boosts must be exception-safe."""

import pytest

from neural_assemblies.assembly_calculus.scaffold import _train_scaffold_step
from neural_assemblies.core.brain import Brain


def test_scaffold_beta_boosts_restore_after_projection_failure():
    brain = Brain(p=0.05, seed=79, engine="numpy_sparse")
    brain.add_stimulus("s", 10)
    brain.add_area("MAIN", 100, 10, 0.1)
    brain.add_area("AUX", 100, 10, 0.2)
    original_project = brain.project
    calls = 0

    def fail_during_recurrence(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:  # five Phase-A rounds, then first Phase-B call
            raise RuntimeError("injected scaffold failure")
        return original_project(*args, **kwargs)

    brain.project = fail_during_recurrence
    with pytest.raises(RuntimeError, match="injected scaffold failure"):
        _train_scaffold_step(
            brain, "s", "MAIN", "AUX", rounds_per_step=10,
            phase_b_ratio=0.5, beta_boost=0.5,
        )
    assert brain.areas["MAIN"].beta == pytest.approx(0.1)
    assert brain.areas["AUX"].beta == pytest.approx(0.2)


def test_scaffold_step_rejects_unknown_stimulus_before_projection():
    brain = Brain(p=0.05, seed=79, engine="numpy_sparse")
    brain.add_area("MAIN", 100, 10, 0.1)
    brain.add_area("AUX", 100, 10, 0.2)
    with pytest.raises(KeyError, match="stimulus"):
        _train_scaffold_step(brain, "TYPO", "MAIN", "AUX")
