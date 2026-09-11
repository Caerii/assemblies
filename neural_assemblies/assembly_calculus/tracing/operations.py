"""Traced assembly-calculus operations."""

from __future__ import annotations

from collections.abc import Sequence

from neural_assemblies.assembly_calculus.assembly import Assembly, chance_overlap, overlap
from neural_assemblies.assembly_calculus.contracts import (
    CompletionPlan, OrderedRecallPlan, ProjectionPlan,
    ReciprocalProjectionPlan,
)
from neural_assemblies.assembly_calculus.ops import _snap

from .models import AssemblyTrace, PatternCompletionDiagnostic, TraceStep


def snapshot_area(brain, area: str) -> Assembly:
    """Snapshot the current winners in *area* using comparable neuron IDs."""
    return _snap(brain, area)


def project_trace(brain, stimulus: str, target: str, rounds: int = 10) -> AssemblyTrace:
    """Project a stimulus and record the target assembly after each round."""
    plan = ProjectionPlan(stimulus, target, rounds, recurrent=True)
    if plan.stimulus not in brain.stimuli:
        raise IndexError(f"Not in brain.stimuli: {plan.stimulus}")
    if plan.target not in brain.areas:
        raise IndexError(f"Not in brain.areas: {plan.target}")

    steps: list[TraceStep] = []
    previous: Assembly | None = None

    for round_index, step in enumerate(plan.steps, start=1):
        brain.project(step.stimuli_dict(), step.fibers_dict())
        drive = "stimulus" if round_index == 1 else "stimulus + recurrence"
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
    plan = ReciprocalProjectionPlan(source, target, rounds, fix_source)
    plan.preflight(brain)
    source = plan.source
    target = plan.target
    rounds = plan.rounds
    fix_source = plan.fix_source

    source_was_fixed = brain.areas[source].fixed_assembly
    if fix_source and not source_was_fixed:
        brain.areas[source].fix_assembly()

    steps: list[TraceStep] = []
    previous: Assembly | None = None
    try:
        for round_index, step in enumerate(plan.steps, start=1):
            brain.project(step.stimuli_dict(), step.fibers_dict())
            drive = (f"{source}" if round_index == 1
                     else f"{source} + {target} recurrence")
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


def associate_trace(
    brain,
    source_a: str,
    source_b: str,
    target: str,
    *,
    stim_a: str | None = None,
    stim_b: str | None = None,
    rounds: int = 10,
) -> AssemblyTrace:
    """Associate two sources through a target and trace all three phases."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    use_fix = stim_a is None and stim_b is None
    source_a_was_fixed = brain.areas[source_a].fixed_assembly
    source_b_was_fixed = brain.areas[source_b].fixed_assembly
    if use_fix and not source_a_was_fixed:
        brain.areas[source_a].fix_assembly()
    if use_fix and not source_b_was_fixed:
        brain.areas[source_b].fix_assembly()

    steps: list[TraceStep] = []
    previous: Assembly | None = None
    round_index = 0

    stim_dict_a = {stim_a: [source_a]} if stim_a else {}
    stim_dict_b = {stim_b: [source_b]} if stim_b else {}
    stim_dict_both = {}
    if stim_a:
        stim_dict_both[stim_a] = [source_a]
    if stim_b:
        stim_dict_both[stim_b] = [source_b]

    try:
        for local_round in range(1, rounds + 1):
            round_index += 1
            dst = {source_a: [source_a, target]}
            drive = f"phase 1: {source_a} -> {target}"
            if local_round > 1:
                dst[target] = [target]
                drive = f"phase 1: {source_a} + {target} recurrence"
            brain.project(stim_dict_a, dst)
            previous = _append_step(
                steps,
                brain=brain,
                operation="associate",
                target=target,
                round_index=round_index,
                drive=drive,
                sources=(source_a,),
                previous=previous,
            )

        for local_round in range(1, rounds + 1):
            round_index += 1
            dst = {source_b: [source_b, target]}
            drive = f"phase 2: {source_b} -> {target}"
            if local_round > 1:
                dst[target] = [target]
                drive = f"phase 2: {source_b} + {target} recurrence"
            brain.project(stim_dict_b, dst)
            previous = _append_step(
                steps,
                brain=brain,
                operation="associate",
                target=target,
                round_index=round_index,
                drive=drive,
                sources=(source_b,),
                previous=previous,
            )

        for _ in range(rounds):
            round_index += 1
            if use_fix:
                dst = {source_a: [target], source_b: [target], target: [target]}
            else:
                dst = {
                    source_a: [source_a, target],
                    source_b: [source_b, target],
                    target: [target],
                }
            brain.project(stim_dict_both, dst)
            previous = _append_step(
                steps,
                brain=brain,
                operation="associate",
                target=target,
                round_index=round_index,
                drive=f"phase 3: {source_a} + {source_b} shared drive",
                sources=(source_a, source_b),
                previous=previous,
            )
    finally:
        if use_fix and not source_a_was_fixed:
            brain.areas[source_a].unfix_assembly()
        if use_fix and not source_b_was_fixed:
            brain.areas[source_b].unfix_assembly()

    return AssemblyTrace(operation="associate", target=target, steps=tuple(steps))


def pattern_complete_trace(
    brain,
    area: str,
    *,
    fraction: float = 0.5,
    rounds: int = 5,
    seed: int | None = None,
    observation_mode: str | None = None,
) -> PatternCompletionDiagnostic:
    """Trace the same validated completion protocol as :func:`pattern_complete`."""
    plan = CompletionPlan(area, fraction, rounds, seed, observation_mode)
    prepared = plan.prepare(brain)

    steps: list[TraceStep] = []
    previous: Assembly | None = None
    with plan.observation_scope(brain):
        prepared.inject_cue(brain)
        previous = _append_step(
            steps,
            brain=brain,
            operation="pattern_complete",
            target=area,
            round_index=0,
            drive=f"partial cue keeps {plan.fraction:.2f}",
            sources=(area,),
            previous=previous,
            num_first_winners=0,
        )
        partial = previous

        for round_index, step in enumerate(plan.steps, start=1):
            brain.project(step.stimuli_dict(), step.fibers_dict())
            previous = _append_step(
                steps,
                brain=brain,
                operation="pattern_complete",
                target=area,
                round_index=round_index,
                drive=f"{area} recurrence",
                sources=(area,),
                previous=previous,
            )

    n = brain.areas[area].n
    return PatternCompletionDiagnostic(
        reference=prepared.reference,
        partial=partial,
        trace=AssemblyTrace(operation="pattern_complete", target=area, steps=tuple(steps)),
        kept_fraction=plan.fraction,
        chance_baseline=chance_overlap(len(prepared.reference), n),
    )


def ordered_recall_trace(
    brain,
    area: str,
    cue: str,
    *,
    max_steps: int = 20,
    known_assemblies: Sequence[Assembly] | None = None,
    convergence_threshold: float = 0.9,
    rounds_per_step: int = 1,
    novelty_threshold: float = 0.3,
) -> AssemblyTrace:
    """Recall a sequence with LRI and record each accepted recalled assembly."""
    plan = OrderedRecallPlan(
        area, cue, max_steps, convergence_threshold,
        rounds_per_step, novelty_threshold,
    )
    plan.preflight(brain)
    area = plan.area
    cue = plan.cue
    max_steps = plan.max_steps
    convergence_threshold = plan.convergence_threshold
    rounds_per_step = plan.rounds_per_step
    novelty_threshold = plan.novelty_threshold

    brain.clear_refractory(area)
    brain.project({cue: [area]}, {})

    steps: list[TraceStep] = []
    previous: Assembly | None = None
    previous = _append_step(
        steps,
        brain=brain,
        operation="ordered_recall",
        target=area,
        round_index=1,
        drive=f"cue {cue}",
        sources=(cue,),
        previous=previous,
    )
    recalled = [previous]

    for step_index in range(2, max_steps + 1):
        for _ in range(rounds_per_step):
            brain.project({}, {area: [area]})

        current = _snap(brain, area)
        is_cycle = any(overlap(current, prev) > convergence_threshold for prev in recalled)
        if is_cycle:
            break

        if known_assemblies is not None and len(known_assemblies) > 0:
            max_known_overlap = max(overlap(current, known) for known in known_assemblies)
            if max_known_overlap < novelty_threshold:
                break

        previous = _append_step(
            steps,
            brain=brain,
            operation="ordered_recall",
            target=area,
            round_index=step_index,
            drive=f"{area} self-projection with LRI",
            sources=(area,),
            previous=previous,
        )
        recalled.append(previous)

    return AssemblyTrace(operation="ordered_recall", target=area, steps=tuple(steps))


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
    num_first_winners: int | None = None,
) -> Assembly:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-trace-counts"""
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
        num_ever_fired=area.get_num_ever_fired(),
        num_first_winners=(max(0, int(area.num_first_winners))
                           if num_first_winners is None else num_first_winners),
        overlap_with_previous=None if previous is None else overlap(previous, assembly),
    )
    steps.append(step)
    return assembly
