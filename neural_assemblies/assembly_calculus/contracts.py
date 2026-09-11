"""Inspectable contracts and immutable schedules for calculus operations.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-operation-objects
"""

from dataclasses import dataclass, is_dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Callable


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
)


OPERATION_CONTRACTS = MappingProxyType({
    "projection": PROJECTION_CONTRACT,
    "reciprocal_projection": RECIPROCAL_PROJECTION_CONTRACT,
})


def implements(contract: OperationContract) -> Callable:
    """Attach the exact contract object to its public implementation."""
    def decorate(operation: Callable) -> Callable:
        operation.operation_contract = contract
        return operation
    return decorate
