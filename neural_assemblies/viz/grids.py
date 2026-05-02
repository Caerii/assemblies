"""Sparse assembly grid plots."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neural_assemblies.assembly_calculus import Assembly


DEFAULT_COLORS = (
    "#4d9de0",
    "#e15554",
    "#59a14f",
    "#f2c14e",
    "#7768ae",
    "#2d728f",
)


def assembly_coordinates(assembly: Assembly, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Map an assembly's winner indices onto a square plotting grid."""
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
    subtitle: str | None = None,
    annotation: str | None = None,
    color: str = DEFAULT_COLORS[0],
    marker_size: float = 14,
) -> Axes:
    """Plot one assembly as active winner dots on a square neuron grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    x, y, side = assembly_coordinates(assembly, n)
    ax.scatter(x, y, s=marker_size, c=color, alpha=0.82, edgecolors="none")
    ax.set_title(title or f"{assembly.area}: {len(assembly)} winners")
    if subtitle:
        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    if annotation:
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85},
        )
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
    subtitles: Sequence[str] | None = None,
    colors: Sequence[str] = DEFAULT_COLORS,
    marker_size: float = 14,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot several assemblies side by side."""
    if not assemblies:
        raise ValueError("assemblies must not be empty")
    if not colors:
        raise ValueError("colors must not be empty")
    if titles is not None and len(titles) != len(assemblies):
        raise ValueError("titles must match assemblies length")
    if subtitles is not None and len(subtitles) != len(assemblies):
        raise ValueError("subtitles must match assemblies length")

    width = max(4.0, 3.7 * len(assemblies))
    fig, axes = plt.subplots(1, len(assemblies), figsize=figsize or (width, 3.8))
    axes_array = np.atleast_1d(axes)

    for index, assembly in enumerate(assemblies):
        plot_assembly(
            assembly,
            n,
            ax=axes_array[index],
            title=titles[index] if titles else None,
            subtitle=subtitles[index] if subtitles else None,
            color=colors[index % len(colors)],
            marker_size=marker_size,
        )

    plt.tight_layout()
    return fig, axes_array
