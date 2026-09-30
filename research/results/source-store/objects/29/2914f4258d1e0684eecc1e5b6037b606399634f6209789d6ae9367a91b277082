"""Sequence recall validates its complete protocol before mutation."""

import pytest

from neural_assemblies.assembly_calculus.ops import ordered_recall
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=29, engine="numpy_sparse")
    brain.add_stimulus("cue", 10)
    brain.add_area("A", 100, 10, 0.1, refractory_period=2)
    return brain


def test_ordered_recall_rejects_unknown_inputs_before_clearing_state():
    brain = _brain()
    with pytest.raises(KeyError, match="cue stimulus"):
        ordered_recall(brain, "A", "TYPO")
    assert len(brain.areas["A"].winners) == 0


@pytest.mark.parametrize("kwargs", [
    {"max_steps": 0}, {"rounds_per_step": 0},
    {"max_steps": 1.5}, {"rounds_per_step": True},
    {"convergence_threshold": -0.1}, {"novelty_threshold": 1.1},
])
def test_ordered_recall_rejects_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        ordered_recall(_brain(), "A", "cue", **kwargs)
