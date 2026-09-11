"""Binding strength must not turn invalid measurement domains into zero."""

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.binding import bind_strength
from neural_assemblies.core.brain import Brain


def _brain():
    brain = Brain(p=0.05, seed=47, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    return brain


def test_bind_strength_rejects_inactive_source():
    with pytest.raises(ValueError, match="active source"):
        bind_strength(
            _brain(), sources=["SRC"], target_area="DST",
            target_assembly=Assembly("DST", np.arange(10, dtype=np.uint32)),
        )


def test_bind_strength_rejects_snapshot_from_wrong_area():
    brain = _brain()
    brain.areas["SRC"].winners = np.arange(10, dtype=np.uint32)
    with pytest.raises(ValueError, match="belongs to"):
        bind_strength(
            brain, sources=["SRC"], target_area="DST",
            target_assembly=Assembly("SRC", np.arange(10, dtype=np.uint32)),
        )
