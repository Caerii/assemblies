"""Merge traces must use the executable merge schedule and admission rules."""

import pytest

from neural_assemblies.assembly_calculus.tracing.operations import merge_trace
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=67, engine="numpy_sparse")
    brain.add_stimulus("a", 10)
    brain.add_stimulus("b", 10)
    brain.add_area("A", 100, 10, 0.1)
    brain.add_area("B", 100, 10, 0.1)
    brain.add_area("T", 100, 10, 0.1)
    return brain


def test_merge_trace_rejects_unknown_topology_before_clamping():
    with pytest.raises(IndexError, match="Not in brain"):
        merge_trace(_brain(), "A", "TYPO", "T")


def test_merge_trace_rejects_partial_stimulus_without_mode():
    with pytest.raises(ValueError, match="partial-stimulus"):
        merge_trace(_brain(), "A", "B", "T", stim_a="a")
