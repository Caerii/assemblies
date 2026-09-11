"""Executable obligations for first-class Assembly Calculus contracts."""

import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.contracts import (
    ASSOCIATION_CONTRACT, OPERATION_CONTRACTS, PROJECTION_CONTRACT,
    RECIPROCAL_PROJECTION_CONTRACT, AssociationPlan, ProjectionPlan,
    ReciprocalProjectionPlan,
)
from neural_assemblies.assembly_calculus.ops import (
    associate, project, reciprocal_project,
)
from neural_assemblies.assembly_calculus.tracing import snapshot_area
from neural_assemblies.core.brain import Brain


class RecordingBrain:
    def __init__(self):
        self.stimuli = {"s": object()}
        self.areas = {"T": object()}
        self.calls = []

    def project(self, stimuli, fibers):
        self.calls.append((stimuli, fibers))


class RecordingReciprocalBrain:
    def __init__(self):
        self.areas = {
            "A": SimpleNamespace(winners=[4, 7]),
            "B": SimpleNamespace(winners=[]),
        }
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


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_reciprocal_plan_rejects_invalid_round_count(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        ReciprocalProjectionPlan("A", "B", rounds)


@pytest.mark.parametrize("field,value", [
    ("source", ""), ("target", None), ("fix_source", 1),
])
def test_reciprocal_plan_rejects_ambiguous_fields(field, value):
    fields = {"source": "A", "target": "B", "rounds": 2, "fix_source": True}
    fields[field] = value
    with pytest.raises(ValueError):
        ReciprocalProjectionPlan(**fields)


def test_reciprocal_plan_rejects_a_self_projection():
    with pytest.raises(ValueError, match="distinct"):
        ReciprocalProjectionPlan("A", "A")


def test_reciprocal_plan_is_immutable_and_canonicalizes_numpy_integer():
    plan = ReciprocalProjectionPlan("A", "B", np.int64(2), False)
    assert type(plan.rounds) is int
    with pytest.raises(FrozenInstanceError):
        plan.rounds = 3


def test_reciprocal_execution_is_exactly_the_inspectable_schedule():
    brain = RecordingReciprocalBrain()
    plan = ReciprocalProjectionPlan("A", "B", 3)
    expected = [(step.stimuli_dict(), step.fibers_dict()) for step in plan.steps]
    plan.execute_steps(brain)
    assert brain.calls == expected
    assert brain.calls[0] == ({}, {"A": ["B"]})
    assert brain.calls[1] == (
        {}, {"A": ["B"], "B": ["B", "A"]},
    )


@pytest.mark.parametrize("failure", ["source", "target", "empty_source"])
def test_reciprocal_preflight_rejects_before_the_first_mutation(failure):
    brain = RecordingReciprocalBrain()
    if failure == "empty_source":
        brain.areas["A"].winners = []
    else:
        del brain.areas["A" if failure == "source" else "B"]
    with pytest.raises((IndexError, ValueError)):
        ReciprocalProjectionPlan("A", "B", 2).execute_steps(brain)
    assert brain.calls == []


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_association_plan_rejects_invalid_pathway_rounds(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        AssociationPlan("A", "B", "T", rounds=rounds)


@pytest.mark.parametrize("cofire_rounds", [-1, True, 1.5, "2"])
def test_association_plan_rejects_invalid_cofire_rounds(cofire_rounds):
    with pytest.raises(ValueError, match="nonnegative integer"):
        AssociationPlan("A", "B", "T", cofire_rounds=cofire_rounds)


@pytest.mark.parametrize("stim_a,stim_b", [("sa", None), (None, "sb")])
def test_association_plan_rejects_partial_stimulus_protocol(stim_a, stim_b):
    with pytest.raises(ValueError, match="both present or both absent"):
        AssociationPlan("A", "B", "T", stim_a, stim_b)


def test_association_plan_rejects_an_aliased_stimulus_pair():
    with pytest.raises(ValueError, match="distinct source stimuli"):
        AssociationPlan("A", "B", "T", "same", "same")


@pytest.mark.parametrize("names", [
    ("A", "A", "T"), ("A", "B", "A"), ("A", "B", "B"),
])
def test_association_plan_rejects_aliased_areas(names):
    with pytest.raises(ValueError, match="three distinct"):
        AssociationPlan(*names)


def test_association_plan_canonicalizes_counts_and_is_immutable():
    plan = AssociationPlan("A", "B", "T", rounds=np.int64(2))
    assert type(plan.rounds) is int
    assert plan.cofire_rounds == 2
    with pytest.raises(FrozenInstanceError):
        plan.rounds = 3


@pytest.mark.parametrize("driven", [False, True])
def test_association_schedule_exposes_each_phase(driven):
    stimuli = ("sa", "sb") if driven else (None, None)
    plan = AssociationPlan("A", "B", "T", *stimuli, rounds=2, cofire_rounds=1)
    calls = [(step.stimuli_dict(), step.fibers_dict()) for step in plan.steps]
    assert len(calls) == 5
    expected_a_targets = ["A", "T"] if driven else ["T"]
    assert calls[0] == (
        ({"sa": ["A"]} if driven else {}),
        {"A": expected_a_targets},
    )
    assert calls[1][1]["T"] == ["T"]
    assert calls[2][1] == {"B": (["B", "T"] if driven else ["T"])}
    assert calls[4][1]["T"] == ["T"]
    assert set(calls[4][1]) == {"A", "B", "T"}


@pytest.mark.parametrize("failure", ["source_a", "source_b", "target", "stimulus", "empty"])
def test_association_preflight_rejects_before_the_first_mutation(failure):
    brain = RecordingReciprocalBrain()
    brain.areas["T"] = SimpleNamespace(winners=[])
    if failure in {"source_a", "source_b", "target"}:
        del brain.areas[{"source_a": "A", "source_b": "B", "target": "T"}[failure]]
        plan = AssociationPlan("A", "B", "T", rounds=2)
    elif failure == "stimulus":
        brain.stimuli = {"sa": object()}
        plan = AssociationPlan("A", "B", "T", "sa", "missing", rounds=2)
    else:
        brain.areas["B"].winners = []
        plan = AssociationPlan("A", "B", "T", rounds=2)
    with pytest.raises((IndexError, ValueError)):
        plan.execute_steps(brain)
    assert brain.calls == []


@pytest.mark.parametrize("name,operation,contract,plan_type", [
    ("projection", project, PROJECTION_CONTRACT, ProjectionPlan),
    (
        "reciprocal_projection", reciprocal_project,
        RECIPROCAL_PROJECTION_CONTRACT, ReciprocalProjectionPlan,
    ),
    ("association", associate, ASSOCIATION_CONTRACT, AssociationPlan),
])
def test_public_operation_carries_the_registered_contract(
    name, operation, contract, plan_type,
):
    assert operation.operation_contract is contract
    assert OPERATION_CONTRACTS[name] is contract
    assert contract.plan_type is plan_type
    for surface in (
        contract.inputs,
        contract.reads,
        contract.mutates,
        contract.regime,
        contract.observed_outcome,
        contract.failure_conditions,
        contract.constructed_controls,
    ):
        assert surface


def test_constructed_control_node_resolves():
    for contract in OPERATION_CONTRACTS.values():
        for node in contract.constructed_controls:
            path, function = node.split("::")
            source = Path(path).read_text(encoding="utf-8")
            assert f"def {function}(" in source


@pytest.mark.parametrize("surface", [
    "inputs", "reads", "mutates", "regime", "observed_outcome",
    "failure_conditions", "constructed_controls",
])
def test_contract_rejects_an_empty_scientific_surface(surface):
    with pytest.raises(ValueError, match="invalid surfaces"):
        replace(PROJECTION_CONTRACT, **{surface: ()})


@pytest.mark.parametrize("bad_terms", [["mutable"], ("",), ("same", "same")])
def test_contract_rejects_malformed_scientific_surfaces(bad_terms):
    with pytest.raises(ValueError, match="invalid surfaces"):
        replace(PROJECTION_CONTRACT, regime=bad_terms)


def test_contract_requires_an_immutable_plan_type():
    with pytest.raises(ValueError, match="frozen dataclass"):
        replace(PROJECTION_CONTRACT, plan_type=object)


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


@pytest.mark.parametrize("engine_name", [
    "numpy_sparse", "numpy_exact", "numpy_explicit",
])
def test_plan_execution_reproduces_the_former_reciprocal_path(engine_name):
    """The plan preserves the valid pre-migration schedule and state."""
    brain = Brain(engine=engine_name, p=.2, seed=37, norm_init=False)
    brain.add_area("A", 60, 6, beta=.1)
    brain.add_area("B", 60, 6, beta=.1)
    brain.add_stimulus("s", 6)
    project(brain, "s", "A", rounds=3, recurrent=True)
    previous = copy.deepcopy(brain)

    actual = reciprocal_project(brain, "A", "B", rounds=3, fix_source=True)

    previous.areas["A"].fix_assembly()
    previous.project({}, {"A": ["B"]})
    for _ in range(2):
        previous.project({}, {"A": ["B"], "B": ["B", "A"]})
    previous.areas["A"].unfix_assembly()
    previous._engine_for(previous.areas["A"]).unfix_assembly("A")

    expected = snapshot_area(previous, "B")
    np.testing.assert_array_equal(actual.winners, expected.winners)
    for area_name in ("A", "B"):
        np.testing.assert_array_equal(
            brain._engine_for(brain.areas[area_name]).get_winners(area_name),
            previous._engine_for(previous.areas[area_name]).get_winners(area_name),
        )
        assert brain._engine_for(brain.areas[area_name]).get_num_ever_fired(
            area_name,
        ) == previous._engine_for(previous.areas[area_name]).get_num_ever_fired(
            area_name,
        )
    if hasattr(brain._engine, "_rng"):
        assert brain._engine._rng.bit_generator.state == (
            previous._engine._rng.bit_generator.state
        )
    assert brain.areas["A"].fixed_assembly is False
    assert brain._engine_for(brain.areas["A"]).is_fixed("A") is False

    future = []
    for candidate in (brain, previous):
        with candidate.read_only():
            candidate.project({}, {"B": ["A"]})
            future.append(snapshot_area(candidate, "A"))
    np.testing.assert_array_equal(future[0].winners, future[1].winners)


@pytest.mark.parametrize("engine_name", [
    "numpy_sparse", "numpy_exact", "numpy_explicit",
])
@pytest.mark.parametrize("driven", [False, True])
def test_plan_execution_reproduces_the_former_association_path(
    engine_name, driven,
):
    brain = Brain(engine=engine_name, p=.2, seed=41, norm_init=False)
    for area in ("A", "B", "T"):
        brain.add_area(area, 60, 6, beta=.1)
    brain.add_stimulus("sa", 6)
    brain.add_stimulus("sb", 6)
    project(brain, "sa", "A", rounds=3, recurrent=True)
    project(brain, "sb", "B", rounds=3, recurrent=True)
    previous = copy.deepcopy(brain)
    stimuli = ("sa", "sb") if driven else (None, None)

    actual = associate(
        brain, "A", "B", "T", stim_a=stimuli[0], stim_b=stimuli[1],
        rounds=3, cofire_rounds=2,
    )

    if not driven:
        previous.areas["A"].fix_assembly()
        previous.areas["B"].fix_assembly()
    for source, stimulus in (("A", stimuli[0]), ("B", stimuli[1])):
        stim_dict = {stimulus: [source]} if stimulus is not None else {}
        source_targets = ["T"] if not driven else [source, "T"]
        for index in range(3):
            fibers = {source: source_targets}
            if index:
                fibers["T"] = ["T"]
            previous.project(stim_dict, fibers)
    joint_stimuli = (
        {"sa": ["A"], "sb": ["B"]} if driven else {}
    )
    joint_fibers = {
        "A": (["A", "T"] if driven else ["T"]),
        "B": (["B", "T"] if driven else ["T"]),
        "T": ["T"],
    }
    for _ in range(2):
        previous.project(joint_stimuli, joint_fibers)
    if not driven:
        for source in ("A", "B"):
            previous.areas[source].unfix_assembly()
            previous._engine_for(previous.areas[source]).unfix_assembly(source)

    expected = snapshot_area(previous, "T")
    np.testing.assert_array_equal(actual.winners, expected.winners)
    for area_name in ("A", "B", "T"):
        actual_engine = brain._engine_for(brain.areas[area_name])
        prior_engine = previous._engine_for(previous.areas[area_name])
        np.testing.assert_array_equal(
            actual_engine.get_winners(area_name), prior_engine.get_winners(area_name),
        )
        assert actual_engine.get_num_ever_fired(area_name) == (
            prior_engine.get_num_ever_fired(area_name)
        )
    if hasattr(brain._engine, "_rng"):
        assert brain._engine._rng.bit_generator.state == (
            previous._engine._rng.bit_generator.state
        )
    for source in ("A", "B"):
        assert brain.areas[source].fixed_assembly is False
        assert brain._engine_for(brain.areas[source]).is_fixed(source) is False

    future = []
    for candidate in (brain, previous):
        with candidate.read_only():
            candidate.project({}, {"A": ["T"], "T": ["T"]})
            future.append(snapshot_area(candidate, "T"))
    np.testing.assert_array_equal(future[0].winners, future[1].winners)
