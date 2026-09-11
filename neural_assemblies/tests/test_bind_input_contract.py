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


@pytest.mark.parametrize("kwargs, message", [
    ({"project_rounds": 0}, "project_rounds must be a positive integer"),
    ({"project_rounds": True}, "project_rounds must be a positive integer"),
    ({"tail_rounds": -1}, "tail_rounds must be a nonnegative integer"),
    ({"tail_rounds": True}, "tail_rounds must be a nonnegative integer"),
    ({"fix_source": 1}, "fix_source must be an explicit boolean"),
])
def test_bind_rejects_ambiguous_schedule_before_source_resolution(kwargs, message):
    brain = Brain(p=0.05, seed=7, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)

    with pytest.raises(ValueError, match=message):
        bind(brain, "SRC", "DST", **kwargs)
