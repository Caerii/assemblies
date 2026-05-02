from __future__ import annotations

import copy

import pytest

from neural_assemblies.assembly_calculus import (
    AssemblyTrace,
    merge_trace,
    overlap,
    project,
    project_trace,
    reciprocal_project_trace,
    snapshot_area,
)
from neural_assemblies.core.brain import Brain


N = 2_000
K = 40
BETA = 0.08


def _brain(seed: int = 42) -> Brain:
    return Brain(p=0.05, save_winners=True, seed=seed, engine="numpy_sparse")


def test_project_trace_records_round_metrics() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_area("COLOR", N, K, BETA)

    trace = project_trace(brain, "red", "COLOR", rounds=6)

    assert isinstance(trace, AssemblyTrace)
    assert len(trace) == 6
    assert trace.final.area == "COLOR"
    assert len(trace.final) == K
    assert trace[0].overlap_with_previous is None
    assert all(step.num_winners == K for step in trace)
    assert trace[-1].overlap_with_previous is not None

    records = trace.to_records()
    assert records[0]["round"] == 1
    assert records[0]["new_winners"] == K


def test_merge_trace_records_target_and_unfixes_sources() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_stimulus("triangle", K)
    brain.add_area("COLOR", N, K, BETA)
    brain.add_area("SHAPE", N, K, BETA)
    brain.add_area("OBJECT", N, K, BETA)

    project(brain, "red", "COLOR", rounds=6)
    project(brain, "triangle", "SHAPE", rounds=6)

    trace = merge_trace(brain, "COLOR", "SHAPE", "OBJECT", rounds=4)

    assert len(trace) == 4
    assert trace.final.area == "OBJECT"
    assert len(trace.final) == K
    assert trace[0].sources == ("COLOR", "SHAPE")
    assert not brain.areas["COLOR"].fixed_assembly
    assert not brain.areas["SHAPE"].fixed_assembly


def test_reciprocal_project_trace_can_probe_merged_target() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_stimulus("triangle", K)
    brain.add_area("COLOR", N, K, BETA)
    brain.add_area("SHAPE", N, K, BETA)
    brain.add_area("OBJECT", N, K, BETA)

    project(brain, "red", "COLOR", rounds=6)
    project(brain, "triangle", "SHAPE", rounds=6)
    merged = merge_trace(brain, "COLOR", "SHAPE", "OBJECT", rounds=5).final

    probe_brain = copy.deepcopy(brain)
    response = reciprocal_project_trace(probe_brain, "COLOR", "OBJECT", rounds=5)

    assert len(response.final) == K
    assert overlap(merged, response.final) > 0.2
    assert not probe_brain.areas["COLOR"].fixed_assembly


def test_reciprocal_project_trace_preserves_existing_fixed_source() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_area("COLOR", N, K, BETA)
    brain.add_area("OBJECT", N, K, BETA)
    project(brain, "red", "COLOR", rounds=6)
    brain.areas["COLOR"].fix_assembly()

    reciprocal_project_trace(brain, "COLOR", "OBJECT", rounds=3)

    assert brain.areas["COLOR"].fixed_assembly


def test_snapshot_area_exposes_public_snapshot() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_area("COLOR", N, K, BETA)
    project(brain, "red", "COLOR", rounds=6)

    snapshot = snapshot_area(brain, "COLOR")

    assert snapshot.area == "COLOR"
    assert len(snapshot) == K


def test_trace_rounds_must_be_positive() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_area("COLOR", N, K, BETA)

    with pytest.raises(ValueError, match="rounds"):
        project_trace(brain, "red", "COLOR", rounds=0)
