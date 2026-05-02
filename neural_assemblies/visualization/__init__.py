"""Compatibility namespace for notebook-friendly visualization helpers.

Use ``neural_assemblies.viz`` for new code. This namespace remains so
``import neural_assemblies.visualization`` does not point at missing modules.
"""

from neural_assemblies.viz import (
    DEFAULT_COLORS,
    assembly_coordinates,
    assembly_overlap_matrix,
    plot_assemblies,
    plot_assembly,
    plot_overlap_matrix,
    plot_projection_flow,
    plot_recall_trace,
)

__all__ = [
    "DEFAULT_COLORS",
    "assembly_coordinates",
    "assembly_overlap_matrix",
    "plot_assemblies",
    "plot_assembly",
    "plot_overlap_matrix",
    "plot_projection_flow",
    "plot_recall_trace",
]
