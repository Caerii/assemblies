"""Traced sequence recall must use the same validated plan as recall."""

import pytest

from neural_assemblies.assembly_calculus.tracing.operations import ordered_recall_trace
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=53, engine="numpy_sparse")
    brain.add_stimulus("cue", 10)
    brain.add_area("A", 100, 10, 0.1, refractory_period=2)
    return brain


@pytest.mark.parametrize("kwargs", [
    {"max_steps": 0}, {"rounds_per_step": 0},
    {"convergence_threshold": 1.1}, {"novelty_threshold": -0.1},
])
def test_trace_recall_rejects_invalid_shared_plan(kwargs):
    with pytest.raises(ValueError):
        ordered_recall_trace(_brain(), "A", "cue", **kwargs)


def test_trace_recall_rejects_unknown_cue_before_mutation():
    brain = _brain()
    with pytest.raises(KeyError, match="cue stimulus"):
        ordered_recall_trace(brain, "A", "TYPO")
    assert len(brain.areas["A"].winners) == 0
