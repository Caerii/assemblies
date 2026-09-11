"""The binding operator must have an explicit source of activity."""

import pytest

from neural_assemblies.assembly_calculus.ops import bind
from neural_assemblies.core.brain import Brain


def test_bind_rejects_empty_implicit_source():
    brain = Brain(p=0.05, seed=7, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)

    with pytest.raises(ValueError, match="requires source_assembly"):
        bind(brain, "SRC", "DST")

    assert len(brain.areas["DST"].winners) == 0
