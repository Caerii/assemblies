"""Reciprocal traces consume the executable reciprocal projection plan."""

import pytest

from neural_assemblies.assembly_calculus.tracing.operations import reciprocal_project_trace
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=61, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    brain.areas["SRC"].winners = list(range(10))
    return brain


def test_reciprocal_trace_rejects_unknown_topology_before_clamping():
    with pytest.raises(IndexError, match="Not in brain"):
        reciprocal_project_trace(_brain(), "SRC", "TYPO")


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_reciprocal_trace_rejects_invalid_rounds(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        reciprocal_project_trace(_brain(), "SRC", "DST", rounds)
