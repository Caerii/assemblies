"""Parameter-sweep plotting helpers."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_parameter_heatmap(
    matrix,
    *,
    x_labels: Sequence[object],
    y_labels: Sequence[object],
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    value_format: str = ".2f",
    cbar_label: str | None = None,
) -> tuple[Axes, np.ndarray]:
    """Plot a small parameter sweep matrix with numeric cell labels."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if values.shape != (len(y_labels), len(x_labels)):
        raise ValueError("matrix shape must match y_labels by x_labels")

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4.5, len(x_labels) * 1.0), max(3.2, len(y_labels) * 0.7)))

    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)

    ax.set_xticks(range(len(x_labels)), [str(label) for label in x_labels])
    ax.set_yticks(range(len(y_labels)), [str(label) for label in y_labels])
    if title:
        ax.set_title(title)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                format(values[row, col], value_format),
                ha="center",
                va="center",
                color="white",
            )

    return ax, values
