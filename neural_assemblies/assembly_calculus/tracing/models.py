"""Data models for traced assembly-calculus operations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from neural_assemblies.assembly_calculus.assembly import Assembly


@dataclass(frozen=True)
class TraceStep:
    """One observed assembly state during a traced operation."""

    round_index: int
    operation: str
    area: str
    assembly: Assembly
    drive: str
    sources: tuple[str, ...]
    num_winners: int
    num_ever_fired: int
    num_first_winners: int
    overlap_with_previous: float | None

    def to_record(self) -> dict[str, object]:
        """Return a tabular representation suitable for pandas or CSV."""
        return {
            "round": self.round_index,
            "operation": self.operation,
            "area": self.area,
            "drive": self.drive,
            "sources": " + ".join(self.sources),
            "winner_count": self.num_winners,
            "ever_fired": self.num_ever_fired,
            "new_winners": self.num_first_winners,
            "overlap_prev": self.overlap_with_previous,
        }


@dataclass(frozen=True)
class AssemblyTrace:
    """Round-by-round snapshots from an assembly-calculus operation."""

    operation: str
    target: str
    steps: tuple[TraceStep, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise ValueError("AssemblyTrace requires at least one step")

    @property
    def final(self) -> Assembly:
        """Final assembly in the trace."""
        return self.steps[-1].assembly

    @property
    def assemblies(self) -> tuple[Assembly, ...]:
        """Assembly snapshots in trace order."""
        return tuple(step.assembly for step in self.steps)

    @property
    def overlap_series(self) -> tuple[float | None, ...]:
        """Consecutive-overlap values in trace order."""
        return tuple(step.overlap_with_previous for step in self.steps)

    def to_records(self) -> list[dict[str, object]]:
        """Return trace steps as dictionaries suitable for inspection tables."""
        return [step.to_record() for step in self.steps]

    def stabilized_at(self, threshold: float = 0.95) -> int | None:
        """Return the first round whose previous-overlap reaches *threshold*."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        for step in self.steps:
            if step.overlap_with_previous is not None and step.overlap_with_previous >= threshold:
                return step.round_index
        return None

    def summary(self, threshold: float = 0.95) -> dict[str, object]:
        """Return compact trace diagnostics for notebook tables."""
        final_overlap = self.steps[-1].overlap_with_previous
        return {
            "operation": self.operation,
            "target": self.target,
            "rounds": len(self.steps),
            "final_winners": len(self.final),
            "final_overlap_prev": final_overlap,
            "stabilized_at": self.stabilized_at(threshold),
            "new_winners_final_round": self.steps[-1].num_first_winners,
        }

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[TraceStep]:
        return iter(self.steps)

    def __getitem__(self, index: int) -> TraceStep:
        return self.steps[index]


@dataclass(frozen=True)
class ResponseTrace:
    """A traced target response driven by one source assembly."""

    label: str
    source: str
    trace: AssemblyTrace
    overlap_with_reference: float
    chance_baseline: float

    @property
    def final(self) -> Assembly:
        """Final response assembly."""
        return self.trace.final

    def to_record(self) -> dict[str, object]:
        """Return a compact diagnostic row for this response."""
        return {
            "label": self.label,
            "source": self.source,
            "target": self.trace.target,
            "rounds": len(self.trace),
            "final_winners": len(self.final),
            "overlap_with_reference": self.overlap_with_reference,
            "chance_baseline": self.chance_baseline,
            "stabilized_at": self.trace.stabilized_at(),
        }


@dataclass(frozen=True)
class ResponseDiagnostic:
    """A set of traced source responses compared to one target reference."""

    reference: Assembly
    target: str
    responses: tuple[ResponseTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "responses", tuple(self.responses))
        if not self.responses:
            raise ValueError("ResponseDiagnostic requires at least one response")

    def to_records(self) -> list[dict[str, object]]:
        """Return response diagnostics as dictionaries for inspection tables."""
        return [response.to_record() for response in self.responses]
