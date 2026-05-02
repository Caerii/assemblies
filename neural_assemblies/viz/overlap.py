"""Overlap and recall diagnostic plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from neural_assemblies.assembly_calculus import Assembly, overlap


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
