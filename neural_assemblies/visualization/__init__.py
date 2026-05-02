"""Compatibility namespace for notebook-friendly visualization helpers.

Use ``neural_assemblies.viz`` for new code. This namespace remains so
``import neural_assemblies.visualization`` does not point at missing modules.
"""

from neural_assemblies.viz import (
    DEFAULT_COLORS,
    animate_assembly_trace,
    assembly_coordinates,
    assembly_overlap_matrix,
    plot_binding_story,
    plot_assemblies,
    plot_assembly,
    plot_merge_diagnostic,
    plot_overlap_matrix,
    plot_projection_flow,
    plot_recall_trace,
    plot_response_overlap,
    plot_trace_metrics,
    plot_winner_turnover,
)

__all__ = [
    "DEFAULT_COLORS",
    "animate_assembly_trace",
    "assembly_coordinates",
    "assembly_overlap_matrix",
    "plot_binding_story",
    "plot_assemblies",
    "plot_assembly",
    "plot_merge_diagnostic",
    "plot_overlap_matrix",
    "plot_projection_flow",
    "plot_recall_trace",
    "plot_response_overlap",
    "plot_trace_metrics",
    "plot_winner_turnover",
]
