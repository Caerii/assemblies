"""Traced assembly-calculus operations."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from neural_assemblies.assembly_calculus.assembly import Assembly, chance_overlap, overlap
from neural_assemblies.assembly_calculus.ops import _snap

from .models import AssemblyTrace, PatternCompletionDiagnostic, TraceStep


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
) -> PatternCompletionDiagnostic:
    """Trace recurrent recovery from a partial activation cue."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    reference = _snap(brain, area)
    if len(reference) == 0:
        raise ValueError(f"area {area!r} has no assembly to complete")

    compact_winners = list(brain.areas[area].winners)
    rng = random.Random(seed)
    subsample_size = max(1, int(len(reference) * fraction))
    subsample = rng.sample(compact_winners, subsample_size)
    brain.areas[area].winners = np.array(subsample, dtype=np.uint32)

    steps: list[TraceStep] = []
    previous: Assembly | None = None
    previous = _append_step(
        steps,
        brain=brain,
        operation="pattern_complete",
        target=area,
        round_index=0,
        drive=f"partial cue keeps {fraction:.2f}",
        sources=(area,),
        previous=previous,
    )
    partial = previous

    for round_index in range(1, rounds + 1):
        brain.project({}, {area: [area]})
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
        reference=reference,
        partial=partial,
        trace=AssemblyTrace(operation="pattern_complete", target=area, steps=tuple(steps)),
        kept_fraction=fraction,
        chance_baseline=chance_overlap(len(reference), n),
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
) -> AssemblyTrace:
    """Recall a sequence with LRI and record each accepted recalled assembly."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if rounds_per_step <= 0:
        raise ValueError("rounds_per_step must be positive")

    area_obj = brain.areas[area]
    if area_obj.refractory_period == 0:
        raise ValueError(
            f"ordered_recall_trace requires refractory_period > 0 for area {area!r}. "
            "Add the area with refractory_period=N to enable LRI."
        )

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
            if max_known_overlap < 0.3:
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
