"""Trace and diagnostic helpers for assembly-calculus dynamics."""

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
