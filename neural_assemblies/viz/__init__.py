"""Notebook-friendly visualization helpers for assembly calculus objects.

These functions are intentionally small and Matplotlib-based. They are meant
for teaching, debugging, and lightweight diagnostics, not for replacing the
research experiment pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neural_assemblies.assembly_calculus import Assembly, overlap


DEFAULT_COLORS = (
    "#4d9de0",
    "#e15554",
    "#59a14f",
    "#f2c14e",
    "#7768ae",
    "#2d728f",
)


def assembly_coordinates(assembly: Assembly, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Map an assembly's winner indices onto a square plotting grid.

    Args:
        assembly: Assembly snapshot to plot.
        n: Number of neurons in the area. The grid side is ``ceil(sqrt(n))``.

    Returns:
        ``(x, y, side)`` where ``x`` and ``y`` are winner coordinates and
        ``side`` is the square grid width.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    side = math.ceil(math.sqrt(n))
    winners = np.asarray(assembly.winners, dtype=np.uint32)
    return winners % side, winners // side, side


def plot_assembly(
    assembly: Assembly,
    n: int,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    color: str = DEFAULT_COLORS[0],
    marker_size: float = 14,
) -> Axes:
    """Plot one assembly as active winner dots on a square neuron grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    x, y, side = assembly_coordinates(assembly, n)
    ax.scatter(x, y, s=marker_size, c=color, alpha=0.82, edgecolors="none")
    ax.set_title(title or f"{assembly.area}: {len(assembly)} winners")
    ax.set_xlim(-1, side)
    ax.set_ylim(side, -1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f7f7f2")
    return ax


def plot_assemblies(
    assemblies: Sequence[Assembly],
    n: int,
    *,
    titles: Sequence[str] | None = None,
    colors: Sequence[str] = DEFAULT_COLORS,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot several assemblies side by side."""
    if not assemblies:
        raise ValueError("assemblies must not be empty")
    if not colors:
        raise ValueError("colors must not be empty")
    if titles is not None and len(titles) != len(assemblies):
        raise ValueError("titles must match assemblies length")

    width = max(4.0, 3.8 * len(assemblies))
    fig, axes = plt.subplots(1, len(assemblies), figsize=figsize or (width, 3.8))
    axes_array = np.atleast_1d(axes)

    for index, assembly in enumerate(assemblies):
        plot_assembly(
            assembly,
            n,
            ax=axes_array[index],
            title=titles[index] if titles else None,
            color=colors[index % len(colors)],
        )

    plt.tight_layout()
    return fig, axes_array


def plot_projection_flow(
    *,
    stimulus_labels: Sequence[str],
    source_labels: Sequence[str],
    target_label: str,
    title: str = "Projection and merge flow",
) -> Figure:
    """Draw a simple two-source projection/merge flow diagram."""
    if len(stimulus_labels) != 2 or len(source_labels) != 2:
        raise ValueError("plot_projection_flow currently expects exactly two sources")

    fig, ax = plt.subplots(figsize=(8, 2.9))
    ax.axis("off")

    boxes = [
        (stimulus_labels[0], 0.12, 0.70, "#f2c14e"),
        (source_labels[0], 0.38, 0.70, "#4d9de0"),
        (stimulus_labels[1], 0.12, 0.30, "#f2c14e"),
        (source_labels[1], 0.38, 0.30, "#e15554"),
        (target_label, 0.72, 0.50, "#59a14f"),
    ]

    for label, x, y, color in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "fc": color, "ec": "black", "lw": 1.2},
        )

    arrows = [
        ((0.21, 0.70), (0.31, 0.70)),
        ((0.21, 0.30), (0.31, 0.30)),
        ((0.48, 0.70), (0.62, 0.53)),
        ((0.48, 0.30), (0.62, 0.47)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2})

    ax.set_title(title)
    ax.text(0.5, 0.06, "project each stimulus, then merge the two source assemblies", ha="center")
    return fig


def assembly_overlap_matrix(assemblies: Sequence[Assembly]) -> np.ndarray:
    """Compute pairwise overlap for a list of assemblies."""
    if not assemblies:
        raise ValueError("assemblies must not be empty")

    matrix = np.zeros((len(assemblies), len(assemblies)), dtype=float)
    for i, left in enumerate(assemblies):
        for j, right in enumerate(assemblies):
            matrix[i, j] = overlap(left, right)
    return matrix


def plot_overlap_matrix(
    assemblies: Sequence[Assembly],
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
) -> tuple[Axes, np.ndarray]:
    """Plot pairwise assembly overlap as a heatmap."""
    matrix = assembly_overlap_matrix(assemblies)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    names = list(labels) if labels else [assembly.area for assembly in assemblies]
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Assembly overlap")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white")

    return ax, matrix


def plot_recall_trace(
    recalled: Sequence[Assembly],
    known: Sequence[Assembly],
    *,
    recalled_labels: Sequence[str] | None = None,
    known_labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> tuple[Axes, np.ndarray]:
    """Plot how recalled assemblies match known reference assemblies."""
    if not recalled:
        raise ValueError("recalled must not be empty")
    if not known:
        raise ValueError("known must not be empty")

    matrix = np.zeros((len(recalled), len(known)), dtype=float)
    for i, recalled_assembly in enumerate(recalled):
        for j, known_assembly in enumerate(known):
            matrix[i, j] = overlap(recalled_assembly, known_assembly)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, len(known) * 1.4), max(3, len(recalled) * 0.9)))

    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(known)), known_labels or [f"known {i}" for i in range(len(known))])
    ax.set_yticks(
        range(len(recalled)),
        recalled_labels or [f"recall {i}" for i in range(len(recalled))],
    )
    ax.set_title("Recall-to-known overlap")
    return ax, matrix


__all__ = [
    "DEFAULT_COLORS",
    "assembly_coordinates",
    "assembly_overlap_matrix",
    "plot_assembly",
    "plot_assemblies",
    "plot_overlap_matrix",
    "plot_projection_flow",
    "plot_recall_trace",
]
