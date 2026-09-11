"""Inspectable contracts and immutable schedules for calculus operations.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-operation-objects
"""

from dataclasses import dataclass, is_dataclass
from contextlib import nullcontext
import math
from numbers import Integral, Real
import random
from types import MappingProxyType
from typing import Callable

import numpy as np

from .assembly import Assembly


def _require_name(label: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty name")


def _positive_rounds(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("rounds must be a positive integer")
    return int(value)


def _explicit_bool(label: str, value: object) -> None:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an explicit boolean")


@dataclass(frozen=True)
class ProjectionStep:
    """One simultaneous projection step in immutable tuple form."""

    stimuli: tuple[tuple[str, tuple[str, ...]], ...]
    fibers: tuple[tuple[str, tuple[str, ...]], ...]

    def stimuli_dict(self) -> dict[str, list[str]]:
        return {source: list(targets) for source, targets in self.stimuli}

    def fibers_dict(self) -> dict[str, list[str]]:
        return {source: list(targets) for source, targets in self.fibers}


@dataclass(frozen=True)
class ProjectionPlan:
    """Validated schedule for the named stimulus-to-area operation."""

    stimulus: str
    target: str
    rounds: int = 10
    recurrent: bool = False

    def __post_init__(self) -> None:
        for label, value in (("stimulus", self.stimulus), ("target", self.target)):
            _require_name(label, value)
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        _explicit_bool("recurrent", self.recurrent)

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        stimulus = ((self.stimulus, (self.target,)),)
        first = ProjectionStep(stimuli=stimulus, fibers=())
        tail = ProjectionStep(
            stimuli=stimulus,
            fibers=((self.target, (self.target,)),) if self.recurrent else (),
        )
        return (first,) + (tail,) * (self.rounds - 1)

    def execute(self, brain) -> None:
        """Preflight the named topology, then execute the declared steps."""
        if self.stimulus not in brain.stimuli:
            raise IndexError(f"Not in brain.stimuli: {self.stimulus}")
        if self.target not in brain.areas:
            raise IndexError(f"Not in brain.areas: {self.target}")
        for step in self.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())


@dataclass(frozen=True)
class ReciprocalProjectionPlan:
    """Validated two-area copy and return-edge schedule."""

    source: str
    target: str
    rounds: int = 10
    fix_source: bool = True

    def __post_init__(self) -> None:
        for label, value in (("source", self.source), ("target", self.target)):
            _require_name(label, value)
        if self.source == self.target:
            raise ValueError("reciprocal projection requires distinct source and target")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        _explicit_bool("fix_source", self.fix_source)

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        first = ProjectionStep(stimuli=(), fibers=((self.source, (self.target,)),))
        tail = ProjectionStep(
            stimuli=(),
            fibers=(
                (self.source, (self.target,)),
                (self.target, (self.target, self.source)),
            ),
        )
        return (first,) + (tail,) * (self.rounds - 1)

    def preflight(self, brain) -> None:
        for name in (self.source, self.target):
            if name not in brain.areas:
                raise IndexError(f"Not in brain.areas: {name}")
        if len(brain.areas[self.source].winners) == 0:
            raise ValueError("reciprocal projection requires an active source assembly")

    def execute_steps(self, brain) -> None:
        self.preflight(brain)
        for step in self.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())


@dataclass(frozen=True)
class AssociationPlan:
    """Validated sequential-pathway and joint-coactivation schedule."""

    source_a: str
    source_b: str
    target: str
    stim_a: str | None = None
    stim_b: str | None = None
    rounds: int = 10
    cofire_rounds: int | None = None

    def __post_init__(self) -> None:
        names = (
            ("source_a", self.source_a),
            ("source_b", self.source_b),
            ("target", self.target),
        )
        for label, value in names:
            _require_name(label, value)
        if len({value for _, value in names}) != len(names):
            raise ValueError("association requires three distinct areas")
        if (self.stim_a is None) != (self.stim_b is None):
            raise ValueError("association stimuli must be both present or both absent")
        for label, value in (("stim_a", self.stim_a), ("stim_b", self.stim_b)):
            if value is not None:
                _require_name(label, value)
        if self.stim_a is not None and self.stim_a == self.stim_b:
            raise ValueError("association requires distinct source stimuli")
        rounds = _positive_rounds(self.rounds)
        cofire = rounds if self.cofire_rounds is None else self.cofire_rounds
        if (
            isinstance(cofire, bool)
            or not isinstance(cofire, Integral)
            or cofire < 0
        ):
            raise ValueError("cofire_rounds must be a nonnegative integer or None")
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "cofire_rounds", int(cofire))

    @property
    def fix_sources(self) -> bool:
        return self.stim_a is None

    def _single_source_steps(
        self, source: str, stimulus: str | None,
    ) -> tuple[ProjectionStep, ...]:
        stimuli = ((stimulus, (source,)),) if stimulus is not None else ()
        source_targets = (self.target,) if self.fix_sources else (source, self.target)
        steps = []
        for index in range(self.rounds):
            fibers = ((source, source_targets),)
            if index:
                fibers += ((self.target, (self.target,)),)
            steps.append(ProjectionStep(stimuli=stimuli, fibers=fibers))
        return tuple(steps)

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        joint_stimuli = () if self.fix_sources else (
            (self.stim_a, (self.source_a,)),
            (self.stim_b, (self.source_b,)),
        )
        source_a_targets = (
            (self.target,) if self.fix_sources else (self.source_a, self.target)
        )
        source_b_targets = (
            (self.target,) if self.fix_sources else (self.source_b, self.target)
        )
        joint = ProjectionStep(
            stimuli=joint_stimuli,
            fibers=(
                (self.source_a, source_a_targets),
                (self.source_b, source_b_targets),
                (self.target, (self.target,)),
            ),
        )
        return (
            self._single_source_steps(self.source_a, self.stim_a)
            + self._single_source_steps(self.source_b, self.stim_b)
            + (joint,) * self.cofire_rounds
        )

    def preflight(self, brain) -> None:
        for name in (self.source_a, self.source_b, self.target):
            if name not in brain.areas:
                raise IndexError(f"Not in brain.areas: {name}")
        for stimulus in (self.stim_a, self.stim_b):
            if stimulus is not None and stimulus not in brain.stimuli:
                raise IndexError(f"Not in brain.stimuli: {stimulus}")
        if self.fix_sources:
            empty = [
                name for name in (self.source_a, self.source_b)
                if len(brain.areas[name].winners) == 0
            ]
            if empty:
                raise ValueError(f"association requires active fixed sources: {empty}")

    def execute_steps(self, brain) -> None:
        self.preflight(brain)
        for step in self.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())


_UNSTIMULATED_SOURCE_MODES = frozenset({"require-fixed", "fix-current", "evolving"})


@dataclass(frozen=True)
class MergePlan:
    """Validated simultaneous two-parent merge schedule."""

    source_a: str
    source_b: str
    target: str
    stim_a: str | None = None
    stim_b: str | None = None
    rounds: int = 10
    parent_self: bool = True
    target_self: bool = True
    back_project: bool = True
    unstimulated_source_mode: str | None = None

    def __post_init__(self) -> None:
        names = (
            ("source_a", self.source_a),
            ("source_b", self.source_b),
            ("target", self.target),
        )
        for label, value in names:
            _require_name(label, value)
        if len({value for _, value in names}) != len(names):
            raise ValueError("merge requires three distinct areas")
        for label, value in (("stim_a", self.stim_a), ("stim_b", self.stim_b)):
            if value is not None:
                _require_name(label, value)
        if self.stim_a is not None and self.stim_a == self.stim_b:
            raise ValueError("merge requires distinct parent stimuli")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        for label in ("parent_self", "target_self", "back_project"):
            _explicit_bool(label, getattr(self, label))

        partial = (self.stim_a is None) != (self.stim_b is None)
        mode = self.unstimulated_source_mode
        if partial:
            if mode not in _UNSTIMULATED_SOURCE_MODES:
                raise ValueError(
                    "partial-stimulus merge requires unstimulated_source_mode "
                    "'require-fixed', 'fix-current', or 'evolving'"
                )
        elif mode is not None:
            raise ValueError(
                "unstimulated_source_mode is only valid for a partial-stimulus merge"
            )

    @property
    def unstimulated_source(self) -> str | None:
        if self.stim_a is None and self.stim_b is not None:
            return self.source_a
        if self.stim_b is None and self.stim_a is not None:
            return self.source_b
        return None

    @property
    def fixed_sources(self) -> tuple[str, ...]:
        if self.stim_a is None and self.stim_b is None:
            return (self.source_a, self.source_b)
        source = self.unstimulated_source
        if self.unstimulated_source_mode == "fix-current" and source is not None:
            return (source,)
        return ()

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        stimuli = tuple(
            (stimulus, (source,))
            for source, stimulus in (
                (self.source_a, self.stim_a),
                (self.source_b, self.stim_b),
            )
            if stimulus is not None
        )
        def parent_targets(source: str) -> tuple[str, ...]:
            return (source, self.target) if self.parent_self else (self.target,)
        parents = (
            (self.source_a, parent_targets(self.source_a)),
            (self.source_b, parent_targets(self.source_b)),
        )
        target_targets = (
            ((self.target,) if self.target_self else ())
            + ((self.source_a, self.source_b) if self.back_project else ())
        )
        first = ProjectionStep(stimuli=stimuli, fibers=parents)
        tail_fibers = parents + (
            ((self.target, target_targets),) if target_targets else ()
        )
        tail = ProjectionStep(stimuli=stimuli, fibers=tail_fibers)
        return (first,) + (tail,) * (self.rounds - 1)

    def preflight(self, brain) -> None:
        for name in (self.source_a, self.source_b, self.target):
            if name not in brain.areas:
                raise IndexError(f"Not in brain.areas: {name}")
        for stimulus in (self.stim_a, self.stim_b):
            if stimulus is not None and stimulus not in brain.stimuli:
                raise IndexError(f"Not in brain.stimuli: {stimulus}")
        unstimulated = [
            source for source, stimulus in (
                (self.source_a, self.stim_a),
                (self.source_b, self.stim_b),
            )
            if stimulus is None
        ]
        empty = [name for name in unstimulated if len(brain.areas[name].winners) == 0]
        if empty:
            raise ValueError(f"merge requires active unstimulated sources: {empty}")
        source = self.unstimulated_source
        if source is not None and self.unstimulated_source_mode == "require-fixed":
            if not brain.areas[source].fixed_assembly:
                raise ValueError(f"merge requires source {source!r} to be fixed")
        if source is not None and self.unstimulated_source_mode == "evolving":
            if brain.areas[source].fixed_assembly:
                raise ValueError(f"merge requires source {source!r} to be evolving")

    def execute_steps(self, brain) -> None:
        self.preflight(brain)
        for step in self.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())


_COMPLETION_OBSERVATION_MODES = frozenset({"plastic", "frozen", "read-only"})


@dataclass(frozen=True)
class PreparedCompletion:
    """A reference assembly and its exact sampled compact-index cue."""

    plan: "CompletionPlan"
    reference: Assembly
    entry_compact: tuple[int, ...]
    compact_cue: tuple[int, ...]
    brain_identity: int

    def inject_cue(self, brain) -> None:
        if id(brain) != self.brain_identity:
            raise ValueError("prepared completion belongs to a different brain")
        current = tuple(int(value) for value in brain.areas[self.plan.area].winners)
        if current != self.entry_compact:
            raise ValueError("completion source changed after cue preparation")
        brain.areas[self.plan.area].winners = np.asarray(
            self.compact_cue, dtype=np.uint32,
        )


@dataclass(frozen=True)
class CompletionPlan:
    """Validated partial-cue construction and recurrent recovery schedule."""

    area: str
    fraction: float = 0.5
    rounds: int = 5
    seed: int | None = None
    observation_mode: str | None = None

    def __post_init__(self) -> None:
        _require_name("area", self.area)
        if (
            isinstance(self.fraction, bool)
            or not isinstance(self.fraction, Real)
            or not math.isfinite(float(self.fraction))
            or not 0 < self.fraction <= 1
        ):
            raise ValueError("fraction must be finite and in (0, 1]")
        object.__setattr__(self, "fraction", float(self.fraction))
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        if (
            self.seed is None
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
        ):
            raise ValueError("seed must be an explicit integer")
        object.__setattr__(self, "seed", int(self.seed))
        if self.observation_mode not in _COMPLETION_OBSERVATION_MODES:
            raise ValueError(
                "observation_mode must be 'plastic', 'frozen', or 'read-only'"
            )

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        step = ProjectionStep(stimuli=(), fibers=((self.area, (self.area,)),))
        return (step,) * self.rounds

    def observation_scope(self, brain):
        """Return the one state policy named by this protocol."""
        if self.observation_mode == "plastic":
            return nullcontext(brain)
        if self.observation_mode == "frozen":
            return brain.frozen()
        return brain.read_only()

    def prepare(self, brain) -> PreparedCompletion:
        if self.area not in brain.areas:
            raise IndexError(f"Not in brain.areas: {self.area}")
        reference = Assembly.from_area(brain, self.area)
        compact = tuple(int(value) for value in brain.areas[self.area].winners)
        if not compact:
            raise ValueError(f"area {self.area!r} has no assembly to complete")
        if len(compact) != len(reference):
            raise ValueError("completion reference and compact cue source disagree")
        cue_size = int(len(reference) * self.fraction)
        if cue_size < 1:
            raise ValueError(
                "fraction retains no neurons at the current assembly size"
            )
        cue = tuple(random.Random(self.seed).sample(compact, cue_size))
        return PreparedCompletion(
            plan=self,
            reference=reference,
            entry_compact=compact,
            compact_cue=cue,
            brain_identity=id(brain),
        )


@dataclass(frozen=True)
class OperationContract:
    """Reviewable scientific surface attached to an executable operation."""

    operation_id: str
    specification: str
    plan_type: type
    inputs: tuple[str, ...]
    reads: tuple[str, ...]
    mutates: tuple[str, ...]
    regime: tuple[str, ...]
    observed_outcome: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    constructed_controls: tuple[str, ...]
    true_negative_controls: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation contract requires a nonempty string ID")
        if not isinstance(self.specification, str) or "#contract-" not in self.specification:
            raise ValueError("operation contract requires an ID and specification anchor")
        plan_params = getattr(self.plan_type, "__dataclass_params__", None)
        if not is_dataclass(self.plan_type) or not getattr(plan_params, "frozen", False):
            raise ValueError("operation contract plan_type must be a frozen dataclass")
        surfaces = {
            "inputs": self.inputs,
            "reads": self.reads,
            "mutates": self.mutates,
            "regime": self.regime,
            "observed outcome": self.observed_outcome,
            "failure conditions": self.failure_conditions,
            "constructed controls": self.constructed_controls,
            "true-negative controls": self.true_negative_controls,
        }
        invalid = []
        for name, values in surfaces.items():
            if (
                not isinstance(values, tuple)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)
            ):
                invalid.append(name)
        if invalid:
            raise ValueError(f"operation contract has invalid surfaces: {invalid}")


PROJECTION_CONTRACT = OperationContract(
    operation_id="projection-v1",
    specification=(
        "docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-projection"
    ),
    plan_type=ProjectionPlan,
    inputs=("brain", "stimulus", "target", "rounds", "recurrent"),
    reads=("stimulus", "target", "afferent weights", "plasticity state"),
    mutates=("target winners", "enabled incoming weights", "engine history"),
    regime=("registered stimulus and target", "backend-defined model regime"),
    observed_outcome=("final target neuron-ID snapshot",),
    failure_conditions=(
        "invalid schedule",
        "unknown stimulus",
        "unknown target",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_operation_semantic_cards.py::"
        "test_p3_operation_owns_recurrence_schedule",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_topology_rejects_before_the_first_mutation",
    ),
)


RECIPROCAL_PROJECTION_CONTRACT = OperationContract(
    operation_id="reciprocal-projection-v1",
    specification=(
        "docs/reviews/whole-codebase/SEMANTIC_CARDS.md#"
        "contract-reciprocal-projection"
    ),
    plan_type=ReciprocalProjectionPlan,
    inputs=("brain", "source", "target", "rounds", "fix_source"),
    reads=("source winners", "forward and return fibers", "plasticity state"),
    mutates=("target winners", "participating weights", "engine history", "clamps"),
    regime=("distinct registered areas", "active source assembly"),
    observed_outcome=("final target neuron-ID snapshot",),
    failure_conditions=(
        "invalid schedule",
        "unknown area",
        "empty source assembly",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_pnas_roundtrip_contract.py::"
        "test_roundtrip_responds_to_learning_disabled_control",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_reciprocal_preflight_rejects_before_the_first_mutation",
    ),
)


ASSOCIATION_CONTRACT = OperationContract(
    operation_id="association-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-association",
    plan_type=AssociationPlan,
    inputs=(
        "brain", "source_a", "source_b", "target", "stim_a", "stim_b",
        "rounds", "cofire_rounds",
    ),
    reads=("source winners", "optional source stimuli", "pathway weights"),
    mutates=("target winners", "participating weights", "engine history", "clamps"),
    regime=(
        "three distinct registered areas",
        "two distinct registered stimuli or two active fixed sources",
    ),
    observed_outcome=("final jointly-driven target neuron-ID snapshot",),
    failure_conditions=(
        "invalid phase schedule",
        "partial stimulus specification",
        "unknown topology",
        "empty fixed source",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_ac_conformance.py::"
        "test_association_grows_with_coactivation",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_association_preflight_rejects_before_the_first_mutation",
    ),
)


MERGE_CONTRACT = OperationContract(
    operation_id="merge-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-merge",
    plan_type=MergePlan,
    inputs=(
        "brain", "source_a", "source_b", "target", "stim_a", "stim_b",
        "rounds", "parent_self", "target_self", "back_project",
        "unstimulated_source_mode",
    ),
    reads=("parent winners", "optional parent stimuli", "participating weights"),
    mutates=("target winners", "participating weights", "engine history", "clamps"),
    regime=(
        "three distinct registered areas",
        "active unstimulated parents",
        "explicit mode for exactly one unstimulated parent",
    ),
    observed_outcome=("final jointly-driven target neuron-ID snapshot",),
    failure_conditions=(
        "invalid schedule switch",
        "ambiguous partial-stimulus protocol",
        "source clamp contradicts declared mode",
        "unknown topology",
        "empty unstimulated source",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_merge_back_projection_switch_changes_the_declared_schedule",
        "neural_assemblies/tests/test_ac_conformance.py::"
        "test_merge_creates_two_way_connectivity_with_bounded_support",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_merge_preflight_rejects_a_source_state_that_contradicts_its_mode",
    ),
)


COMPLETION_CONTRACT = OperationContract(
    operation_id="pattern-completion-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-completion",
    plan_type=CompletionPlan,
    inputs=("brain", "area", "fraction", "rounds", "seed", "observation_mode"),
    reads=("entry assembly", "compact winners", "recurrent weights"),
    mutates=(
        "area winners",
        "recurrent weights in plastic mode",
        "recruitment outside read-only mode",
        "engine history outside read-only mode",
    ),
    regime=("registered active area", "nonempty retained cue"),
    observed_outcome=(
        "final recovered neuron-ID snapshot",
        "min-normalized overlap with immutable entry reference",
    ),
    failure_conditions=(
        "invalid cue protocol",
        "missing seed or observation mode",
        "unknown or empty area",
        "cue rounds to zero neurons",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_public_model_boundaries.py::"
        "test_teaching_example_has_a_working_learning_disabled_control",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_completion_prepare_rejects_before_mutation",
    ),
)


OPERATION_CONTRACTS = MappingProxyType({
    "projection": PROJECTION_CONTRACT,
    "reciprocal_projection": RECIPROCAL_PROJECTION_CONTRACT,
    "association": ASSOCIATION_CONTRACT,
    "merge": MERGE_CONTRACT,
    "pattern_completion": COMPLETION_CONTRACT,
})


def implements(contract: OperationContract) -> Callable:
    """Attach the exact contract object to its public implementation."""
    def decorate(operation: Callable) -> Callable:
        operation.operation_contract = contract
        return operation
    return decorate
