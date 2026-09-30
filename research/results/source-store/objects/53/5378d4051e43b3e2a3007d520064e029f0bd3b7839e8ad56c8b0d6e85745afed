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


def test_input_drive_rejects_empty_measurement_domains():
    brain = Brain(p=0.05, seed=13, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    with pytest.raises(ValueError, match="at least one"):
        input_drive(brain, sources=[], target_areas=["DST"])
    with pytest.raises(ValueError, match="at least one"):
        input_drive(brain, sources=["SRC"], target_areas=[])


def test_input_drive_rejects_sources_without_live_activity():
    brain = Brain(p=0.05, seed=13, engine="numpy_sparse")
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    with pytest.raises(ValueError, match="active source"):
        input_drive(brain, sources=["SRC"], target_areas=["DST"])


def test_input_drive_rejects_missing_engine_observation_instead_of_fabricating_zero(
    monkeypatch,
):
    brain = Brain(p=0.05, seed=13, engine="numpy_exact")
    brain.add_stimulus("S", 10)
    brain.add_area("SRC", 100, 10, 0.1)
    brain.add_area("DST", 100, 10, 0.1)
    brain.project({"S": ["SRC"]}, {})

    def drop_observation(*args, **kwargs):
        brain.last_activation_scores.clear()
        brain.last_pre_kwta_totals.clear()
        brain.last_pre_kwta_counts.clear()

    monkeypatch.setattr(brain, "project", drop_observation)
    with pytest.raises(RuntimeError, match="omitted target area"):
        input_drive(brain, sources=["SRC"], target_areas=["DST"])
