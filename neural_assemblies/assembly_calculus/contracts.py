"""Inspectable contracts and immutable schedules for calculus operations.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-operation-objects
"""

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Callable


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
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty name")
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, Integral) or self.rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if type(self.recurrent) is not bool:
            raise ValueError("recurrent must be an explicit boolean")
        object.__setattr__(self, "rounds", int(self.rounds))

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
        if not self.operation_id or "#contract-" not in self.specification:
            raise ValueError("operation contract requires an ID and specification anchor")
        surfaces = {
            "inputs": self.inputs,
            "reads": self.reads,
            "mutates": self.mutates,
            "regime": self.regime,
            "observed outcome": self.observed_outcome,
            "failure conditions": self.failure_conditions,
            "constructed controls": self.constructed_controls,
        }
        missing = [name for name, values in surfaces.items() if not values]
        if missing:
            raise ValueError(f"operation contract has empty surfaces: {missing}")


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


OPERATION_CONTRACTS = MappingProxyType({"projection": PROJECTION_CONTRACT})


def implements(contract: OperationContract) -> Callable:
    """Attach the exact contract object to its public implementation."""
    def decorate(operation: Callable) -> Callable:
        operation.operation_contract = contract
        return operation
    return decorate
