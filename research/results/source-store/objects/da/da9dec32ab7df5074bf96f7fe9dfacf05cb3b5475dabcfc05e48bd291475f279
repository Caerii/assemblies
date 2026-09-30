"""Fiber materialization distinguishes topology errors from inactive sources."""

import pytest

from neural_assemblies.assembly_calculus.binding import materialize_fiber
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=43, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    return brain


def test_materialize_fiber_rejects_unknown_topology():
    with pytest.raises(KeyError, match="source area"):
        materialize_fiber(_brain(), "TYPO", "DST")
    with pytest.raises(KeyError, match="target area"):
        materialize_fiber(_brain(), "SRC", "TYPO")


def test_materialize_fiber_reports_inactive_source_explicitly():
    assert materialize_fiber(_brain(), "SRC", "DST") is False
