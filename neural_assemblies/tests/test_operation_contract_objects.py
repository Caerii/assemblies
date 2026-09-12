"""Executable obligations for first-class Assembly Calculus contracts."""

import ast
import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.contracts import (
    ASSOCIATION_CONTRACT, COMPLETION_CONTRACT, MERGE_CONTRACT,
    ORDERED_RECALL_CONTRACT,
    OPERATION_CONTRACTS, PROJECTION_CONTRACT, RECIPROCAL_PROJECTION_CONTRACT,
    AssociationPlan, BindingReadPlan, CompletionPlan, ConsolidationPlan, InputDrivePlan, MergePlan, OrderedRecallPlan,
    PreparedCompletion, SEQUENCE_MEMORIZE_CONTRACT, SEPARATION_CONTRACT,
    SequenceMemorizePlan, SeparationPlan,
    ProjectionPlan, ProjectionStep, ReciprocalProjectionPlan,
)
from neural_assemblies.assembly_calculus.ops import (
    associate, merge, ordered_recall, pattern_complete, project,
    separate, sequence_memorize,
    reciprocal_project,
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
            "A": SimpleNamespace(winners=[4, 7], fixed_assembly=False),
            "B": SimpleNamespace(winners=[], fixed_assembly=False),
        }
        self.calls = []

    def project(self, stimuli, fibers):
        self.calls.append((stimuli, fibers))


def test_ordered_recall_plan_requires_lri():
    brain = SimpleNamespace(
        areas={"A": SimpleNamespace(refractory_period=0)},
        stimuli={"cue": object()},
    )
    with pytest.raises(ValueError, match="refractory_period > 0"):
        OrderedRecallPlan("A", "cue").preflight(brain)


def test_binding_read_plan_rejects_self_area_and_invalid_tail():
    with pytest.raises(ValueError, match="distinct"):
        BindingReadPlan("A", "A")
    with pytest.raises(ValueError, match="nonnegative integer"):
        BindingReadPlan("A", "B", tail_rounds=-1)


def test_input_drive_plan_rejects_empty_lists_and_unknown_metric():
    with pytest.raises(ValueError, match="source area"):
        InputDrivePlan((), ("A",))
    with pytest.raises(ValueError, match="metric"):
        InputDrivePlan(("A",), ("B",), metric="mean")


@pytest.mark.parametrize("stimuli", [(), ("",), ("s", 1)])
def test_sequence_memorize_plan_rejects_malformed_stimulus_tuple(stimuli):
    with pytest.raises(ValueError, match="stimuli"):
        SequenceMemorizePlan(stimuli, "A")


def test_sequence_memorize_plan_preflight_rejects_unknown_topology():
    brain = SimpleNamespace(areas={}, stimuli={"s": object()})
    plan = SequenceMemorizePlan(("s",), "A")
    with pytest.raises(KeyError, match="target area"):
        plan.preflight(brain)


def test_separation_plan_rejects_identical_stimuli():
    with pytest.raises(ValueError, match="distinct stimuli"):
        SeparationPlan("s", "s", "A")


def test_separation_plan_preflight_rejects_unknown_topology():
    brain = SimpleNamespace(areas={}, stimuli={"a": object(), "b": object()})
    with pytest.raises(KeyError, match="target area"):
        SeparationPlan("a", "b", "A").preflight(brain)


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


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_merge_plan_rejects_invalid_rounds(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        MergePlan("A", "B", "T", rounds=rounds)


@pytest.mark.parametrize("switch", ["parent_self", "target_self", "back_project"])
def test_merge_plan_requires_explicit_boolean_switches(switch):
    fields = {switch: 1}
    with pytest.raises(ValueError, match="explicit boolean"):
        MergePlan("A", "B", "T", **fields)


def test_partial_merge_requires_an_explicit_unstimulated_source_mode():
    with pytest.raises(ValueError, match="partial-stimulus merge requires"):
        MergePlan("A", "B", "T", stim_b="sb")


def test_static_partial_merge_calls_name_the_unstimulated_source_mode():
    failures = []
    for root in (Path("neural_assemblies"), Path("research"), Path("examples")):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                name = (
                    function.id if isinstance(function, ast.Name)
                    else function.attr if isinstance(function, ast.Attribute)
                    else ""
                )
                if name != "merge":
                    continue
                keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}

                def present(field, mapping=keywords):
                    value = mapping.get(field)
                    return value is not None and not (
                        isinstance(value, ast.Constant) and value.value is None
                    )

                if present("stim_a") != present("stim_b"):
                    if "unstimulated_source_mode" not in keywords:
                        failures.append(f"{path}:{node.lineno}")
    assert failures == []


def test_partial_merge_rejects_an_unknown_source_mode():
    with pytest.raises(ValueError, match="partial-stimulus merge requires"):
        MergePlan(
            "A", "B", "T", stim_b="sb", unstimulated_source_mode="guess",
        )


@pytest.mark.parametrize("names", [
    ("A", "A", "T"), ("A", "B", "A"), ("A", "B", "B"),
])
def test_merge_plan_rejects_aliased_areas(names):
    with pytest.raises(ValueError, match="three distinct"):
        MergePlan(*names)


def test_merge_plan_rejects_aliased_parent_stimuli():
    with pytest.raises(ValueError, match="distinct parent stimuli"):
        MergePlan("A", "B", "T", "same", "same")


@pytest.mark.parametrize("mode", ["require-fixed", "fix-current", "evolving"])
def test_partial_merge_accepts_each_explicit_source_mode(mode):
    assert MergePlan(
        "A", "B", "T", stim_b="sb", unstimulated_source_mode=mode,
    ).unstimulated_source == "A"


def test_nonpartial_merge_rejects_a_meaningless_source_mode():
    with pytest.raises(ValueError, match="only valid"):
        MergePlan("A", "B", "T", unstimulated_source_mode="evolving")


def test_merge_back_projection_switch_changes_the_declared_schedule():
    enabled = MergePlan("A", "B", "T", rounds=2, back_project=True)
    disabled = MergePlan("A", "B", "T", rounds=2, back_project=False)
    assert enabled.steps[1].fibers_dict()["T"] == ["T", "A", "B"]
    assert disabled.steps[1].fibers_dict()["T"] == ["T"]


@pytest.mark.parametrize("mode,fixed", [
    ("require-fixed", False), ("evolving", True),
])
def test_merge_preflight_rejects_a_source_state_that_contradicts_its_mode(
    mode, fixed,
):
    brain = RecordingReciprocalBrain()
    brain.areas["T"] = SimpleNamespace(winners=[], fixed_assembly=False)
    brain.areas["A"].fixed_assembly = fixed
    brain.stimuli = {"sb": object()}
    plan = MergePlan(
        "A", "B", "T", stim_b="sb", unstimulated_source_mode=mode,
    )
    with pytest.raises(ValueError, match="to be fixed|to be evolving"):
        plan.execute_steps(brain)
    assert brain.calls == []


@pytest.mark.parametrize("name,operation,contract,plan_type", [
    ("projection", project, PROJECTION_CONTRACT, ProjectionPlan),
    (
        "reciprocal_projection", reciprocal_project,
        RECIPROCAL_PROJECTION_CONTRACT, ReciprocalProjectionPlan,
    ),
    ("association", associate, ASSOCIATION_CONTRACT, AssociationPlan),
    ("merge", merge, MERGE_CONTRACT, MergePlan),
    (
        "pattern_completion", pattern_complete,
        COMPLETION_CONTRACT, CompletionPlan,
    ),
    (
        "ordered_recall", ordered_recall,
        ORDERED_RECALL_CONTRACT, OrderedRecallPlan,
    ),
    (
        "sequence_memorize", sequence_memorize,
        SEQUENCE_MEMORIZE_CONTRACT, SequenceMemorizePlan,
    ),
    (
        "separate", separate,
        SEPARATION_CONTRACT, SeparationPlan,
    ),
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
        contract.true_negative_controls,
    ):
        assert surface


def test_registry_and_public_callable_cannot_drift():
    """Every registry key must name a callable carrying that exact contract."""
    import neural_assemblies.assembly_calculus.ops as operations
    from neural_assemblies.assembly_calculus.attention import attend
    from neural_assemblies.assembly_calculus.readout import build_lexicon, fuzzy_readout
    from neural_assemblies.assembly_calculus.next_token import predict_next_token, score_corpus, train_on_corpus
    from neural_assemblies.assembly_calculus.recovery import observe_recovery
    from neural_assemblies.assembly_calculus.recovery import replace_neurons
    from neural_assemblies.assembly_calculus.binding import materialize_fiber
    from neural_assemblies.assembly_calculus.binding import bind as source_bind, binding_strength, input_drive, recall
    from neural_assemblies.assembly_calculus.consolidation import accumulate_context, accumulate_context_step, consolidate

    names = {
        "projection": "project",
        "reciprocal_projection": "reciprocal_project",
        "association": "associate",
        "pattern_completion": "pattern_complete",
    }
    for name, contract in OPERATION_CONTRACTS.items():
        operation = (replace_neurons if name == "replace_neurons" else
                     observe_recovery if name == "observe_recovery" else
                     score_corpus if name == "score_corpus" else
                     train_on_corpus if name == "train_on_corpus" else
                     predict_next_token if name == "predict_next_token" else
                     build_lexicon if name == "build_lexicon" else
                     materialize_fiber if name == "materialize_fiber" else
                     fuzzy_readout if name == "fuzzy_readout" else
                     attend if name == "attention" else
                     source_bind if name == "source_binding" else
                     recall if name == "binding_recall" else
                     input_drive if name == "input_drive" else
                     binding_strength if name == "binding_strength" else
                     accumulate_context_step if name == "accumulate_context_step" else
                     accumulate_context if name == "accumulate_context" else
                     consolidate if name == "consolidate" else getattr(
            operations, names.get(name, name),
        ))
        assert callable(operation), name
        assert getattr(operation, "operation_contract", None) is contract
        assert contract.specification in (operation.__doc__ or ""), (
            f"{name} must link its registered specification at the source"
        )


@pytest.mark.parametrize(
    "plan",
    [
        ProjectionPlan("s", "T", rounds=2, recurrent=True),
        ReciprocalProjectionPlan("A", "B", rounds=2),
        AssociationPlan("A", "B", "T", rounds=2, cofire_rounds=1),
        MergePlan("A", "B", "T", rounds=2),
        CompletionPlan("T", rounds=2, seed=7, observation_mode="frozen"),
    ],
)
def test_schedule_composition_is_a_closed_immutable_value(plan):
    """Pure schedule composition is checked by construction, before a brain runs."""
    steps = plan.steps
    assert isinstance(steps, tuple) and steps
    assert all(isinstance(step, ProjectionStep) for step in steps)
    assert steps == plan.steps

    # Public mapping views are fresh: callers cannot mutate a plan's schedule
    # accidentally while constructing a larger program from its steps.
    first = steps[0].stimuli_dict()
    first.clear()
    assert steps[0].stimuli_dict() != first or not steps[0].stimuli
    first_fibers = steps[0].fibers_dict()
    first_fibers.clear()
    assert steps[0].fibers_dict() != first_fibers or not steps[0].fibers


def test_next_token_model_alias_uses_lexicon_contract():
    from neural_assemblies.assembly_calculus.next_token import build_next_token_model
    from neural_assemblies.assembly_calculus.contracts import LEXICON_BUILD_CONTRACT

    assert build_next_token_model.operation_contract is LEXICON_BUILD_CONTRACT
    assert LEXICON_BUILD_CONTRACT.specification in (build_next_token_model.__doc__ or "")


def test_constructed_control_node_resolves():
    for contract in OPERATION_CONTRACTS.values():
        for node in contract.constructed_controls:
            path, function = node.split("::")
            source = Path(path).read_text(encoding="utf-8")
            assert f"def {function}(" in source


def test_every_contract_specification_resolves_to_a_real_anchor():
    """A contract cannot quietly point at a deleted or renamed specification."""
    root = Path(__file__).resolve().parents[2]
    for contract in OPERATION_CONTRACTS.values():
        relative, anchor = contract.specification.split("#", 1)
        specification = root / relative
        assert specification.is_file(), contract.specification
        source = specification.read_text(encoding="utf-8")
        assert f'id="{anchor}"' in source or f"id='{anchor}'" in source, (
            f"missing specification anchor: {contract.specification}"
        )


def test_every_operation_contract_names_a_true_negative_control():
    for contract in OPERATION_CONTRACTS.values():
        assert contract.true_negative_controls
        for node in contract.true_negative_controls:
            path, function = node.split("::")
            source = Path(path).read_text(encoding="utf-8")
            assert f"def {function}(" in source


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, True, float("inf"), float("nan"), "0.5"])
def test_completion_plan_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        CompletionPlan("A", fraction, seed=1, observation_mode="plastic")


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_completion_plan_rejects_invalid_rounds(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        CompletionPlan("A", rounds=rounds, seed=1, observation_mode="plastic")


@pytest.mark.parametrize("seed", [None, True, 1.5, "1"])
def test_completion_plan_requires_an_explicit_integer_seed(seed):
    with pytest.raises(ValueError, match="explicit integer"):
        CompletionPlan("A", seed=seed, observation_mode="plastic")


@pytest.mark.parametrize("mode", [None, "", "probe", 1])
def test_completion_plan_requires_an_explicit_observation_mode(mode):
    with pytest.raises(ValueError, match="observation_mode"):
        CompletionPlan("A", seed=1, observation_mode=mode)


def test_completion_plan_is_immutable_and_canonicalizes_numbers():
    plan = CompletionPlan(
        "A", np.float32(0.5), np.int64(2), np.int64(3), "plastic",
    )
    assert type(plan.fraction) is float
    assert type(plan.rounds) is int
    assert type(plan.seed) is int
    with pytest.raises(FrozenInstanceError):
        plan.seed = 4


def _trained_completion_brain(engine_name="numpy_explicit"):
    brain = Brain(
        engine=engine_name, p=.2, seed=47, norm_init=False,
        sampled_recurrence_policy="acknowledged",
    )
    brain.add_area("A", 60, 6, beta=.1)
    brain.add_stimulus("s", 6)
    project(brain, "s", "A", rounds=3, recurrent=True)
    return brain


def test_completion_prepare_names_both_index_spaces_and_exact_cue():
    brain = _trained_completion_brain()
    compact = tuple(int(value) for value in brain.areas["A"].winners)
    plan = CompletionPlan("A", .5, 2, 11, "plastic")
    prepared = plan.prepare(brain)

    assert isinstance(prepared, PreparedCompletion)
    assert prepared.plan is plan
    assert prepared.reference.area == "A"
    assert prepared.entry_compact == compact
    assert prepared.compact_cue == tuple(random.Random(11).sample(compact, 3))
    assert prepared.reference.neuron_ids.flags.writeable is False
    with pytest.raises(FrozenInstanceError):
        prepared.compact_cue = ()


@pytest.mark.parametrize("failure", ["different-brain", "changed-source"])
def test_prepared_completion_rejects_stale_state_before_injection(failure):
    brain = _trained_completion_brain()
    prepared = CompletionPlan("A", .5, 2, 11, "plastic").prepare(brain)
    target = copy.deepcopy(brain) if failure == "different-brain" else brain
    if failure == "changed-source":
        target.areas["A"].winners = target.areas["A"].winners[::-1].copy()
    before = target.areas["A"].winners.copy()
    with pytest.raises(ValueError, match="different brain|source changed"):
        prepared.inject_cue(target)
    np.testing.assert_array_equal(target.areas["A"].winners, before)


def test_completion_schedule_is_exactly_declared():
    plan = CompletionPlan("A", .5, 3, 1, "plastic")
    assert [
        (step.stimuli_dict(), step.fibers_dict()) for step in plan.steps
    ] == [({}, {"A": ["A"]})] * 3


@pytest.mark.parametrize("failure", ["unknown", "empty", "zero-cue"])
def test_completion_prepare_rejects_before_mutation(failure):
    brain = Brain(engine="numpy_explicit", p=.2, seed=3, norm_init=False)
    brain.add_area("A", 20, 2, beta=.1)
    before = copy.deepcopy(brain)
    plan = CompletionPlan(
        "missing" if failure == "unknown" else "A",
        .1 if failure == "zero-cue" else .5,
        1,
        1,
        "plastic",
    )
    if failure == "zero-cue":
        brain.areas["A"].winners = np.asarray([0, 1], dtype=np.uint32)
        before = copy.deepcopy(brain)
    with pytest.raises((IndexError, ValueError)):
        plan.prepare(brain)
    np.testing.assert_array_equal(
        brain.areas["A"].winners, before.areas["A"].winners,
    )
    assert brain._engine._rng.bit_generator.state == before._engine._rng.bit_generator.state


def test_static_completion_calls_name_seed_and_observation_policy():
    failures = []
    for root in (Path("neural_assemblies"), Path("research")):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                name = (
                    function.id if isinstance(function, ast.Name)
                    else function.attr if isinstance(function, ast.Attribute)
                    else ""
                )
                if name not in {"pattern_complete", "pattern_complete_trace"}:
                    continue
                keywords = {kw.arg for kw in node.keywords if kw.arg}
                missing = {"seed", "observation_mode"} - keywords
                if missing:
                    failures.append(f"{path}:{node.lineno}: {sorted(missing)}")
    assert failures == []


@pytest.mark.parametrize("surface", [
    "inputs", "reads", "mutates", "regime", "observed_outcome",
    "failure_conditions", "constructed_controls",
    "true_negative_controls",
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
    brain = Brain(
        engine=engine_name, p=.2, seed=31, norm_init=False,
        sampled_recurrence_policy="acknowledged",
    )
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
    brain = Brain(
        engine=engine_name, p=.2, seed=37, norm_init=False,
        sampled_recurrence_policy="acknowledged",
    )
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
    brain = Brain(
        engine=engine_name, p=.2, seed=41, norm_init=False,
        sampled_recurrence_policy="acknowledged",
    )
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


@pytest.mark.parametrize("engine_name", [
    "numpy_sparse", "numpy_exact", "numpy_explicit",
])
@pytest.mark.parametrize("mode", [
    "fixed-both", "driven-both", "require-fixed", "fix-current", "evolving",
])
def test_plan_execution_reproduces_the_former_merge_path(engine_name, mode):
    brain = Brain(
        engine=engine_name, p=.2, seed=43, norm_init=False,
        sampled_recurrence_policy="acknowledged",
    )
    for area in ("A", "B", "T"):
        brain.add_area(area, 60, 6, beta=.1)
    brain.add_stimulus("sa", 6)
    brain.add_stimulus("sb", 6)
    project(brain, "sa", "A", rounds=3, recurrent=True)
    project(brain, "sb", "B", rounds=3, recurrent=True)

    if mode == "fixed-both":
        stimuli = (None, None)
        source_mode = None
    elif mode == "driven-both":
        stimuli = ("sa", "sb")
        source_mode = None
    else:
        stimuli = (None, "sb")
        source_mode = mode
    if mode == "require-fixed":
        brain.areas["A"].fix_assembly()
    previous = copy.deepcopy(brain)

    actual = merge(
        brain, "A", "B", "T", stim_a=stimuli[0], stim_b=stimuli[1],
        rounds=3, parent_self=True, target_self=True, back_project=True,
        unstimulated_source_mode=source_mode,
    )

    reference_fixed = (
        ("A", "B") if mode == "fixed-both"
        else ("A",) if mode == "fix-current"
        else ()
    )
    for source in reference_fixed:
        previous.areas[source].fix_assembly()
    stim_dict = {}
    if stimuli[0] is not None:
        stim_dict[stimuli[0]] = ["A"]
    if stimuli[1] is not None:
        stim_dict[stimuli[1]] = ["B"]
    parents = {"A": ["A", "T"], "B": ["B", "T"]}
    previous.project(stim_dict, parents)
    for _ in range(2):
        previous.project(stim_dict, {**parents, "T": ["T", "A", "B"]})
    for source in reference_fixed:
        previous.areas[source].unfix_assembly()
        previous._engine_for(previous.areas[source]).unfix_assembly(source)

    np.testing.assert_array_equal(actual.winners, snapshot_area(previous, "T").winners)
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
    assert brain.areas["A"].fixed_assembly is (mode == "require-fixed")


@pytest.mark.parametrize("engine_name", [
    "numpy_sparse", "numpy_exact", "numpy_explicit",
])
def test_plan_execution_reproduces_the_former_completion_path(engine_name):
    """Plastic mode preserves the exact pre-contract protocol and state."""
    brain = _trained_completion_brain(engine_name)
    previous = copy.deepcopy(brain)

    actual, actual_score = pattern_complete(
        brain, "A", fraction=.5, rounds=3, seed=13,
        observation_mode="plastic",
    )

    reference = snapshot_area(previous, "A")
    compact = list(previous.areas["A"].winners)
    cue = random.Random(13).sample(compact, int(len(reference) * .5))
    previous.areas["A"].winners = np.asarray(cue, dtype=np.uint32)
    for _ in range(3):
        previous.project({}, {"A": ["A"]})
    expected = snapshot_area(previous, "A")

    np.testing.assert_array_equal(actual.neuron_ids, expected.neuron_ids)
    assert actual_score == expected.overlap(reference)
    np.testing.assert_array_equal(
        brain._engine.get_winners("A"), previous._engine.get_winners("A"),
    )
    assert brain._engine.get_num_ever_fired("A") == (
        previous._engine.get_num_ever_fired("A")
    )
    if hasattr(brain._engine, "_rng"):
        assert brain._engine._rng.bit_generator.state == (
            previous._engine._rng.bit_generator.state
        )

    future = []
    for candidate in (brain, previous):
        with candidate.read_only():
            candidate.project({}, {"A": ["A"]})
            future.append(snapshot_area(candidate, "A"))
    np.testing.assert_array_equal(future[0].neuron_ids, future[1].neuron_ids)


def test_completion_observation_modes_have_distinct_mutation_contracts():
    trained = _trained_completion_brain()

    plastic = copy.deepcopy(trained)
    plastic_before = plastic.connectomes["A"]["A"].weights.copy()
    pattern_complete(
        plastic, "A", fraction=.5, rounds=2, seed=7,
        observation_mode="plastic",
    )
    assert not np.array_equal(
        plastic.connectomes["A"]["A"].weights, plastic_before,
    )

    frozen = copy.deepcopy(trained)
    frozen_before = frozen.connectomes["A"]["A"].weights.copy()
    frozen.project = lambda *_args, **_kwargs: None
    frozen_result, _ = pattern_complete(
        frozen, "A", fraction=.5, rounds=2, seed=7,
        observation_mode="frozen",
    )
    np.testing.assert_array_equal(frozen.connectomes["A"]["A"].weights, frozen_before)
    assert frozen.disable_plasticity is False
    assert len(frozen_result) == 3
    assert len(frozen.areas["A"].winners) == 3

    read_only = copy.deepcopy(trained)
    read_only_entry = read_only.areas["A"].winners.copy()
    read_only_count = read_only._engine.get_num_ever_fired("A")
    read_only_rng = copy.deepcopy(read_only._engine._rng.bit_generator.state)
    read_only_weights = read_only.connectomes["A"]["A"].weights.copy()
    recovered, _ = pattern_complete(
        read_only, "A", fraction=.5, rounds=2, seed=7,
        observation_mode="read-only",
    )
    assert len(recovered) == read_only.areas["A"].k
    np.testing.assert_array_equal(read_only.areas["A"].winners, read_only_entry)
    np.testing.assert_array_equal(
        read_only.connectomes["A"]["A"].weights, read_only_weights,
    )
    assert read_only._engine.get_num_ever_fired("A") == read_only_count
    assert read_only._engine._rng.bit_generator.state == read_only_rng


@pytest.mark.parametrize("mode", ["frozen", "read-only"])
def test_completion_observation_scope_restores_policy_after_failure(monkeypatch, mode):
    brain = _trained_completion_brain()
    entry = brain.areas["A"].winners.copy()
    original = brain.project

    def fail_after_projection(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(brain, "project", fail_after_projection)
    with pytest.raises(RuntimeError, match="injected projection failure"):
        pattern_complete(
            brain, "A", fraction=.5, rounds=2, seed=5,
            observation_mode=mode,
        )
    assert brain.disable_plasticity is False
    if mode == "read-only":
        np.testing.assert_array_equal(brain.areas["A"].winners, entry)


def test_consolidation_plan_rejects_empty_direction_before_mutation():
    from neural_assemblies.assembly_calculus.assembly import Assembly

    with pytest.raises(ValueError, match="at least one replay direction"):
        ConsolidationPlan("A", Assembly("A", [1]), "B", Assembly("B", [2]),
                          a_to_b=False, b_to_a=False)
