"""Argument checks every plan shares, and a schedule: the ProjectionSteps an operation
runs, built from a first step and a repeated tail, and executed on a brain.

Part of neural_assemblies.assembly_calculus.contracts.
Specification: neural_assemblies/ir/VERIFICATION.md#contract-operation-objects"""
from dataclasses import dataclass
from numbers import Integral




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

    def __post_init__(self) -> None:
        for label, edges in (("stimuli", self.stimuli), ("fibers", self.fibers)):
            if not isinstance(edges, tuple):
                raise TypeError(f"{label} must be a tuple of edges")
            seen: set[str] = set()
            for edge in edges:
                if (not isinstance(edge, tuple) or len(edge) != 2
                        or not isinstance(edge[0], str) or not edge[0]
                        or not isinstance(edge[1], tuple)
                        or not edge[1]
                        or any(not isinstance(target, str) or not target for target in edge[1])):
                    raise ValueError(f"{label} edges must be (name, nonempty target tuple)")
                if edge[0] in seen:
                    raise ValueError(f"{label} cannot contain duplicate source names")
                if len(set(edge[1])) != len(edge[1]):
                    raise ValueError(f"{label} cannot contain duplicate target names")
                seen.add(edge[0])

    def stimuli_dict(self) -> dict[str, list[str]]:
        return {source: list(targets) for source, targets in self.stimuli}

    def fibers_dict(self) -> dict[str, list[str]]:
        return {source: list(targets) for source, targets in self.fibers}


def _schedule(first: ProjectionStep, tail: ProjectionStep, rounds: int) -> tuple[ProjectionStep, ...]:
    """Build a finite immutable schedule with one distinguished entry step."""
    return (first,) + (tail,) * (rounds - 1)


def _execute_schedule(steps: tuple[ProjectionStep, ...], brain) -> None:
    """Interpret a validated schedule through the brain projection boundary."""
    for step in steps:
        brain.project(step.stimuli_dict(), step.fibers_dict())
