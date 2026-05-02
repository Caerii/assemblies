from __future__ import annotations

import copy

import pytest

from neural_assemblies.assembly_calculus import (
    AssemblyTrace,
    PatternCompletionDiagnostic,
    ProjectionSweepConfig,
    RecallSweepConfig,
    ResponseDiagnostic,
    associate_trace,
    lri_recall_sweep,
    merge_trace,
    ordered_recall_trace,
    overlap,
    pattern_complete_trace,
    project,
    project_trace,
    projection_sweep,
    reciprocal_project_trace,
    sequence_memorize,
    snapshot_area,
    source_response_traces,
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
    assert trace.summary()["operation"] == "project"
    assert trace.stabilized_at(threshold=0.0) == 2


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


def test_associate_trace_records_three_phases() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_stimulus("triangle", K)
    brain.add_area("COLOR", N, K, BETA)
    brain.add_area("SHAPE", N, K, BETA)
    brain.add_area("ASSOC", N, K, BETA)

    project(brain, "red", "COLOR", rounds=6)
    project(brain, "triangle", "SHAPE", rounds=6)

    trace = associate_trace(brain, "COLOR", "SHAPE", "ASSOC", rounds=3)

    assert trace.operation == "associate"
    assert len(trace) == 9
    assert trace[0].sources == ("COLOR",)
    assert trace[3].sources == ("SHAPE",)
    assert trace[-1].sources == ("COLOR", "SHAPE")
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


def test_source_response_traces_compare_each_source_to_reference() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_stimulus("triangle", K)
    brain.add_area("COLOR", N, K, BETA)
    brain.add_area("SHAPE", N, K, BETA)
    brain.add_area("OBJECT", N, K, BETA)

    project(brain, "red", "COLOR", rounds=6)
    project(brain, "triangle", "SHAPE", rounds=6)
    merged = merge_trace(brain, "COLOR", "SHAPE", "OBJECT", rounds=5).final

    diagnostic = source_response_traces(
        brain,
        ["COLOR", "SHAPE"],
        "OBJECT",
        reference=merged,
        labels=["color only", "shape only"],
        rounds=4,
    )

    assert isinstance(diagnostic, ResponseDiagnostic)
    assert diagnostic.reference == merged
    assert len(diagnostic.responses) == 2
    assert diagnostic.responses[0].label == "color only"
    assert diagnostic.responses[0].overlap_with_reference >= 0.0
    assert diagnostic.to_records()[0]["chance_baseline"] == K / N


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


def test_pattern_complete_trace_reports_recovery() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_area("COLOR", N, K, BETA)
    project(brain, "red", "COLOR", rounds=8)

    diagnostic = pattern_complete_trace(brain, "COLOR", fraction=0.5, rounds=4, seed=1)

    assert isinstance(diagnostic, PatternCompletionDiagnostic)
    assert diagnostic.reference.area == "COLOR"
    assert len(diagnostic.partial) == K // 2
    assert diagnostic.trace.operation == "pattern_complete"
    assert diagnostic.trace[0].round_index == 0
    assert 0.0 <= diagnostic.recovery_overlap <= 1.0


def test_ordered_recall_trace_records_lri_steps() -> None:
    brain = _brain()
    for index in range(3):
        brain.add_stimulus(f"s{index}", K)
    brain.add_area("SEQ", N, K, BETA)

    memorized = sequence_memorize(
        brain,
        ["s0", "s1", "s2"],
        "SEQ",
        rounds_per_step=8,
        repetitions=2,
    )
    brain.set_lri("SEQ", refractory_period=3, inhibition_strength=100.0)

    trace = ordered_recall_trace(
        brain,
        "SEQ",
        "s0",
        max_steps=6,
        known_assemblies=list(memorized),
    )

    assert trace.operation == "ordered_recall"
    assert trace.target == "SEQ"
    assert len(trace) >= 1
    assert overlap(trace[0].assembly, memorized[0]) > 0.3


def test_projection_sweep_returns_compact_rows() -> None:
    rows = projection_sweep(
        [
            ProjectionSweepConfig(n=N, k=K, beta=BETA, rounds=4, seed=1),
            ProjectionSweepConfig(n=N, k=K, beta=0.12, rounds=4, seed=1),
        ]
    )

    assert len(rows) == 2
    assert rows[0]["chance_overlap"] == K / N
    assert rows[0]["final_overlap_prev"] is not None


def test_lri_recall_sweep_returns_recall_rows() -> None:
    rows = lri_recall_sweep(
        [
            RecallSweepConfig(
                refractory_period=2,
                inhibition_strength=50.0,
                n=N,
                k=K,
                rounds_per_step=6,
                repetitions=1,
                sequence=("s0", "s1", "s2"),
            )
        ]
    )

    assert len(rows) == 1
    assert rows[0]["recall_steps"] >= 1
    assert rows[0]["ordered_matches"] >= 0


def test_trace_rounds_must_be_positive() -> None:
    brain = _brain()
    brain.add_stimulus("red", K)
    brain.add_area("COLOR", N, K, BETA)

    with pytest.raises(ValueError, match="rounds"):
        project_trace(brain, "red", "COLOR", rounds=0)

    with pytest.raises(ValueError, match="max_steps"):
        ordered_recall_trace(brain, "COLOR", "red", max_steps=0)

    with pytest.raises(ValueError, match="fraction"):
        pattern_complete_trace(brain, "COLOR", fraction=0)
