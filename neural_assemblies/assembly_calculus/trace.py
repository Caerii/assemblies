"""Trace helpers for inspecting assembly dynamics round by round."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .assembly import Assembly, overlap
from .ops import _snap


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

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[TraceStep]:
        return iter(self.steps)

    def __getitem__(self, index: int) -> TraceStep:
        return self.steps[index]


def snapshot_area(brain, area: str) -> Assembly:
    """Snapshot the current winners in *area* using comparable neuron IDs."""
    return _snap(brain, area)


def project_trace(brain, stimulus: str, target: str, rounds: int = 10) -> AssemblyTrace:
    """Project a stimulus and record the target assembly after each round."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    steps: list[TraceStep] = []
    previous: Assembly | None = None

    for round_index in range(1, rounds + 1):
        if round_index == 1:
            brain.project({stimulus: [target]}, {})
            drive = "stimulus"
        else:
            brain.project({stimulus: [target]}, {target: [target]})
            drive = "stimulus + recurrence"
        previous = _append_step(
            steps,
            brain=brain,
            operation="project",
            target=target,
            round_index=round_index,
            drive=drive,
            sources=(stimulus,),
            previous=previous,
        )

    return AssemblyTrace(operation="project", target=target, steps=tuple(steps))


def reciprocal_project_trace(
    brain,
    source: str,
    target: str,
    rounds: int = 10,
    *,
    fix_source: bool = True,
) -> AssemblyTrace:
    """Project an existing source-area assembly into a target and trace it."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    source_was_fixed = brain.areas[source].fixed_assembly
    if fix_source and not source_was_fixed:
        brain.areas[source].fix_assembly()

    steps: list[TraceStep] = []
    previous: Assembly | None = None
    try:
        for round_index in range(1, rounds + 1):
            if round_index == 1:
                brain.project({}, {source: [target]})
                drive = f"{source}"
            else:
                brain.project({}, {source: [target], target: [target]})
                drive = f"{source} + {target} recurrence"
            previous = _append_step(
                steps,
                brain=brain,
                operation="reciprocal_project",
                target=target,
                round_index=round_index,
                drive=drive,
                sources=(source,),
                previous=previous,
            )
    finally:
        if fix_source and not source_was_fixed:
            brain.areas[source].unfix_assembly()

    return AssemblyTrace(operation="reciprocal_project", target=target, steps=tuple(steps))


def merge_trace(
    brain,
    source_a: str,
    source_b: str,
    target: str,
    *,
    stim_a: str | None = None,
    stim_b: str | None = None,
    rounds: int = 10,
) -> AssemblyTrace:
    """Merge two source assemblies into a target and trace each merge round."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    use_fix = stim_a is None and stim_b is None
    source_a_was_fixed = brain.areas[source_a].fixed_assembly
    source_b_was_fixed = brain.areas[source_b].fixed_assembly
    if use_fix and not source_a_was_fixed:
        brain.areas[source_a].fix_assembly()
    if use_fix and not source_b_was_fixed:
        brain.areas[source_b].fix_assembly()

    stim_dict = {}
    if stim_a:
        stim_dict[stim_a] = [source_a]
    if stim_b:
        stim_dict[stim_b] = [source_b]

    steps: list[TraceStep] = []
    previous: Assembly | None = None
    try:
        for round_index in range(1, rounds + 1):
            if round_index == 1:
                brain.project(
                    stim_dict,
                    {source_a: [source_a, target], source_b: [source_b, target]},
                )
                drive = f"{source_a} + {source_b}"
            else:
                brain.project(
                    stim_dict,
                    {
                        source_a: [source_a, target],
                        source_b: [source_b, target],
                        target: [target, source_a, source_b],
                    },
                )
                drive = f"{source_a} + {source_b} + {target} feedback"
            previous = _append_step(
                steps,
                brain=brain,
                operation="merge",
                target=target,
                round_index=round_index,
                drive=drive,
                sources=(source_a, source_b),
                previous=previous,
            )
    finally:
        if use_fix and not source_a_was_fixed:
            brain.areas[source_a].unfix_assembly()
        if use_fix and not source_b_was_fixed:
            brain.areas[source_b].unfix_assembly()

    return AssemblyTrace(operation="merge", target=target, steps=tuple(steps))


def _append_step(
    steps: list[TraceStep],
    *,
    brain,
    operation: str,
    target: str,
    round_index: int,
    drive: str,
    sources: Sequence[str],
    previous: Assembly | None,
) -> Assembly:
    assembly = _snap(brain, target)
    area = brain.areas[target]
    step = TraceStep(
        round_index=round_index,
        operation=operation,
        area=target,
        assembly=assembly,
        drive=drive,
        sources=tuple(sources),
        num_winners=len(assembly),
        num_ever_fired=area.w,
        num_first_winners=max(0, int(area.num_first_winners)),
        overlap_with_previous=None if previous is None else overlap(previous, assembly),
    )
    steps.append(step)
    return assembly
