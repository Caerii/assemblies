"""Traced projection must consume the executable projection schedule."""

import pytest

from neural_assemblies.assembly_calculus.tracing.operations import project_trace
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=59, engine="numpy_sparse")
    brain.add_stimulus("s", 10)
    brain.add_area("A", 100, 10, 0.1)
    return brain


def test_project_trace_rejects_unknown_stimulus_before_mutation():
    brain = _brain()
    with pytest.raises(IndexError, match="stimuli"):
        project_trace(brain, "TYPO", "A")
    assert len(brain.areas["A"].winners) == 0


def test_project_trace_exposes_recurrence_choice():
    brain = _brain()
    trace = project_trace(brain, "s", "A", rounds=2, recurrent=False)
    assert all("recurrence" not in step.drive for step in trace)


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_project_trace_rejects_invalid_rounds(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        project_trace(_brain(), "s", "A", rounds)
