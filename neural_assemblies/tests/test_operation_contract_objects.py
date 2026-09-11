"""Executable obligations for first-class Assembly Calculus contracts."""

import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.contracts import (
    OPERATION_CONTRACTS, PROJECTION_CONTRACT, ProjectionPlan,
)
from neural_assemblies.assembly_calculus.ops import project
from neural_assemblies.assembly_calculus.tracing import snapshot_area
from neural_assemblies.core.brain import Brain


class RecordingBrain:
    def __init__(self):
        self.stimuli = {"s": object()}
        self.areas = {"T": object()}
        self.calls = []

    def project(self, stimuli, fibers):
        self.calls.append((stimuli, fibers))


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_projection_plan_rejects_invalid_round_count(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        ProjectionPlan("s", "T", rounds)


@pytest.mark.parametrize("field,value", [
    ("stimulus", ""), ("target", None), ("recurrent", 1),
])
def test_projection_plan_rejects_ambiguous_fields(field, value):
    fields = {"stimulus": "s", "target": "T", "rounds": 2, "recurrent": False}
    fields[field] = value
    with pytest.raises(ValueError):
        ProjectionPlan(**fields)


def test_projection_plan_is_immutable_and_canonicalizes_numpy_integer():
    plan = ProjectionPlan("s", "T", np.int64(2), True)
    assert type(plan.rounds) is int
    with pytest.raises(FrozenInstanceError):
        plan.rounds = 3


@pytest.mark.parametrize("recurrent", [False, True])
def test_execution_is_exactly_the_inspectable_schedule(recurrent):
    brain = RecordingBrain()
    plan = ProjectionPlan("s", "T", 3, recurrent)
    expected = [(step.stimuli_dict(), step.fibers_dict()) for step in plan.steps]
    plan.execute(brain)
    assert brain.calls == expected
    assert brain.calls[0] == ({"s": ["T"]}, {})
    assert brain.calls[1][1] == ({"T": ["T"]} if recurrent else {})


@pytest.mark.parametrize("missing", ["stimulus", "target"])
def test_topology_rejects_before_the_first_mutation(missing):
    brain = RecordingBrain()
    getattr(brain, "stimuli" if missing == "stimulus" else "areas").clear()
    with pytest.raises(IndexError, match="Not in brain"):
        ProjectionPlan("s", "T", 2, True).execute(brain)
    assert brain.calls == []


def test_public_operation_carries_the_registered_contract():
    assert project.operation_contract is PROJECTION_CONTRACT
    assert OPERATION_CONTRACTS["projection"] is PROJECTION_CONTRACT
    assert PROJECTION_CONTRACT.plan_type is ProjectionPlan
    for surface in (
        PROJECTION_CONTRACT.inputs,
        PROJECTION_CONTRACT.reads,
        PROJECTION_CONTRACT.mutates,
        PROJECTION_CONTRACT.regime,
        PROJECTION_CONTRACT.observed_outcome,
        PROJECTION_CONTRACT.failure_conditions,
        PROJECTION_CONTRACT.constructed_controls,
    ):
        assert surface


def test_constructed_control_node_resolves():
    for node in PROJECTION_CONTRACT.constructed_controls:
        path, function = node.split("::")
        source = Path(path).read_text(encoding="utf-8")
        assert f"def {function}(" in source


@pytest.mark.parametrize("surface", [
    "inputs", "reads", "mutates", "regime", "observed_outcome",
    "failure_conditions", "constructed_controls",
])
def test_contract_rejects_an_empty_scientific_surface(surface):
    with pytest.raises(ValueError, match="empty surfaces"):
        replace(PROJECTION_CONTRACT, **{surface: ()})


@pytest.mark.parametrize("engine_name", [
    "numpy_sparse", "numpy_exact", "numpy_explicit",
])
def test_plan_execution_reproduces_the_former_projection_path(engine_name):
    """Migration means equal state, not merely an equal final return type."""
    brain = Brain(engine=engine_name, p=.2, seed=31, norm_init=False)
    brain.add_area("T", 60, 6, beta=.1)
    brain.add_stimulus("s", 6)
    previous = copy.deepcopy(brain)

    actual = project(brain, "s", "T", rounds=3, recurrent=False)
    previous.project({"s": ["T"]}, {})
    previous.project_rounds("T", {"s": ["T"]}, {}, rounds=2)

    expected = snapshot_area(previous, "T")
    np.testing.assert_array_equal(actual.winners, expected.winners)
    np.testing.assert_array_equal(
        brain._engine.get_winners("T"), previous._engine.get_winners("T")
    )
    assert brain._engine.get_num_ever_fired("T") == (
        previous._engine.get_num_ever_fired("T")
    )
    if hasattr(brain._engine, "_rng"):
        assert brain._engine._rng.bit_generator.state == (
            previous._engine._rng.bit_generator.state
        )
    # Probe through the public read-only boundary so unlike backend storage is
    # compared by the next observation it produces, not by private field names.
    future = []
    for candidate in (brain, previous):
        with candidate.read_only():
            candidate.project({"s": ["T"]}, {})
            future.append(snapshot_area(candidate, "T"))
    np.testing.assert_array_equal(future[0].winners, future[1].winners)
