"""Input-drive measurements must name a supported quantity explicitly."""

import pytest

from neural_assemblies.assembly_calculus.binding import input_drive
from neural_assemblies.core.brain import Brain


def test_input_drive_rejects_unknown_metric_instead_of_using_winners_path():
    brain = Brain(p=0.05, seed=13, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    with pytest.raises(ValueError, match="metric must be"):
        input_drive(brain, sources=["SRC"], target_areas=["DST"], metric="energy")
