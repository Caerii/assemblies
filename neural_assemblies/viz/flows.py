"""Conceptual flow diagrams for teaching notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .grids import DEFAULT_COLORS


def plot_binding_story(
    *,
    stimulus_labels: Sequence[str],
    source_labels: Sequence[str],
    target_label: str,
    title: str = "Two-source binding story",
    caption: str | None = None,
    stimulus_color: str = "#f2c14e",
    source_colors: Sequence[str] = (DEFAULT_COLORS[0], DEFAULT_COLORS[1]),
    target_color: str = DEFAULT_COLORS[2],
) -> Figure:
    """Draw a two-source projection/merge story for teaching notebooks."""
    if len(stimulus_labels) != 2 or len(source_labels) != 2:
        raise ValueError("plot_binding_story currently expects exactly two sources")
    if len(source_colors) != 2:
        raise ValueError("source_colors must contain exactly two colors")

    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.12, 0.92, "cues", ha="center", va="center", fontsize=9, color="#555555")
    ax.text(
        0.42,
        0.92,
        "source assemblies",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    ax.text(0.78, 0.92, "target assembly", ha="center", va="center", fontsize=9, color="#555555")

    boxes = [
        (stimulus_labels[0], 0.12, 0.68, stimulus_color),
        (source_labels[0], 0.42, 0.68, source_colors[0]),
        (stimulus_labels[1], 0.12, 0.32, stimulus_color),
        (source_labels[1], 0.42, 0.32, source_colors[1]),
        (target_label, 0.78, 0.50, target_color),
    ]

    for label, x, y, color in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": color,
                "ec": "#222222",
                "lw": 1.15,
                "alpha": 0.95,
            },
        )

    arrows = [
        ((0.22, 0.68), (0.32, 0.68), source_colors[0], "project"),
        ((0.22, 0.32), (0.32, 0.32), source_colors[1], "project"),
        ((0.53, 0.68), (0.66, 0.53), target_color, "merge"),
        ((0.53, 0.32), (0.66, 0.47), target_color, "merge"),
    ]
    for start, end, color, label in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 2.2, "color": color},
        )
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.045, label, ha="center", va="center", fontsize=8, color="#444444")

    ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)
    if caption:
        ax.text(0.5, 0.06, caption, ha="center", va="center", fontsize=9, color="#444444")
    return fig


def plot_projection_flow(
    *,
    stimulus_labels: Sequence[str],
    source_labels: Sequence[str],
    target_label: str,
    title: str = "Projection and merge flow",
    caption: str = "project each stimulus, then merge the two source assemblies",
) -> Figure:
    """Draw a two-source projection/merge flow diagram."""
    return plot_binding_story(
        stimulus_labels=stimulus_labels,
        source_labels=source_labels,
        target_label=target_label,
        title=title,
        caption=caption,
    )
