"""Higher-level diagnostics built from traced operations."""

from __future__ import annotations

import copy
from collections.abc import Sequence

from neural_assemblies.assembly_calculus.assembly import Assembly, chance_overlap, overlap

from .models import ResponseDiagnostic, ResponseTrace
from .operations import reciprocal_project_trace, snapshot_area


def source_response_traces(
    brain,
    sources: Sequence[str],
    target: str,
    *,
    reference: Assembly | None = None,
    rounds: int = 5,
    labels: Sequence[str] | None = None,
    copy_brain: bool = True,
) -> ResponseDiagnostic:
    """Trace target responses when each source assembly drives the target alone."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if not sources:
        raise ValueError("sources must not be empty")
    if labels is not None and len(labels) != len(sources):
        raise ValueError("labels must match sources length")

    reference_assembly = snapshot_area(brain, target) if reference is None else reference
    n = brain.areas[target].n
    baseline = chance_overlap(len(reference_assembly), n) if len(reference_assembly) else 0.0

    response_traces: list[ResponseTrace] = []
    for index, source in enumerate(sources):
        response_label = labels[index] if labels is not None else source
        run_brain = copy.deepcopy(brain) if copy_brain else brain
        trace = reciprocal_project_trace(run_brain, source, target, rounds=rounds)
        response_traces.append(
            ResponseTrace(
                label=response_label,
                source=source,
                trace=trace,
                overlap_with_reference=overlap(reference_assembly, trace.final),
                chance_baseline=baseline,
            )
        )

    return ResponseDiagnostic(
        reference=reference_assembly,
        target=target,
        responses=tuple(response_traces),
    )
