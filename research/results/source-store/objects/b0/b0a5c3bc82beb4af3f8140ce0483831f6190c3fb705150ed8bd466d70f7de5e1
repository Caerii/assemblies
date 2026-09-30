"""Trace and diagnostic helpers for assembly-calculus dynamics.

The operations in :mod:`~neural_assemblies.assembly_calculus.ops` return only
their final assembly, because that is all a caller computing with them needs.
But the scientifically interesting claims of the model are about the
TRAJECTORY, not the endpoint: that an assembly stabilises in O(log n) rounds,
that new winners stop appearing before the winner set stops churning, that a
partial cue climbs back to its attractor.  None of those are visible in a
final snapshot.

This package re-implements the same operations round by round, recording a
:class:`~.models.TraceStep` after each projection.  The duplication is
deliberate and the invariant that matters is that a ``*_trace`` function must
open exactly the same fibers on exactly the same rounds as its counterpart in
``ops`` -- if the schedules drift apart, the trace stops describing the
operation it is named after.  Check this first when a trace disagrees with a
direct call.

Each step records enough to distinguish the two ways an area can be "done":

    num_winners        size of the current winner set
    num_first_winners  how many fired for the FIRST time this round; reaching
                       zero means recruitment has finished
    overlap_with_previous  how much the winner set moved; reaching ~1 means
                       the set itself has stopped changing

Those two can and do decouple -- recruitment can end while the firing set is
still churning -- which is precisely the distinction the E%-WTA formation
criterion in :mod:`~neural_assemblies.assembly_calculus.epwta` is built on.
"""

from .diagnostics import source_response_traces
from .models import (
    AssemblyTrace,
    PatternCompletionDiagnostic,
    ResponseDiagnostic,
    ResponseTrace,
    TraceStep,
)
from .operations import (
    associate_trace,
    merge_trace,
    ordered_recall_trace,
    pattern_complete_trace,
    project_trace,
    reciprocal_project_trace,
    snapshot_area,
)
from .sweeps import (
    ProjectionSweepConfig,
    RecallSweepConfig,
    lri_recall_sweep,
    projection_sweep,
)

__all__ = [
    "AssemblyTrace",
    "PatternCompletionDiagnostic",
    "ProjectionSweepConfig",
    "RecallSweepConfig",
    "ResponseDiagnostic",
    "ResponseTrace",
    "TraceStep",
    "associate_trace",
    "lri_recall_sweep",
    "merge_trace",
    "ordered_recall_trace",
    "pattern_complete_trace",
    "project_trace",
    "projection_sweep",
    "reciprocal_project_trace",
    "snapshot_area",
    "source_response_traces",
]
