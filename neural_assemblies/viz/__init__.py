"""Notebook-friendly visualization helpers for assembly calculus objects."""

from .flows import plot_binding_story, plot_projection_flow
from .grids import (
    DEFAULT_COLORS,
    assembly_coordinates,
    plot_assemblies,
    plot_assembly,
)
from .overlap import assembly_overlap_matrix, plot_overlap_matrix, plot_recall_trace
from .tracing import (
    animate_assembly_trace,
    plot_merge_diagnostic,
    plot_response_overlap,
    plot_trace_metrics,
    plot_winner_turnover,
)

__all__ = [
    "DEFAULT_COLORS",
    "assembly_coordinates",
    "assembly_overlap_matrix",
    "animate_assembly_trace",
    "plot_binding_story",
    "plot_assembly",
    "plot_assemblies",
    "plot_merge_diagnostic",
    "plot_overlap_matrix",
    "plot_projection_flow",
    "plot_recall_trace",
    "plot_response_overlap",
    "plot_trace_metrics",
    "plot_winner_turnover",
]
