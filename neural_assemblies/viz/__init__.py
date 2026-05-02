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
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from neural_assemblies.assembly_calculus import Assembly, AssemblyTrace, chance_overlap, overlap


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
    """Draw a two-source projection/merge story for teaching notebooks.

    The diagram is conceptual: it shows routing between stimuli, source
    assemblies, and a target assembly. It is not an anatomical circuit map.
    """
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


def plot_merge_diagnostic(
    before: Assembly,
    after: Assembly,
    replay: Assembly,
    *,
    n: int,
    title: str = "What changed after merge?",
    colors: tuple[str, str, str] = ("#9a9a9a", DEFAULT_COLORS[2], "#6f6f6f"),
) -> Figure:
    """Plot target activity before/after merge and same-area replay overlap."""
    if n <= 0:
        raise ValueError("n must be positive")
    if before.area != after.area:
        raise ValueError("before and after assemblies must live in the same area")
    if replay.area != after.area:
        raise ValueError("replay assembly must live in the same area as after")
    if len(colors) != 3:
        raise ValueError("colors must contain exactly three colors")

    replay_overlap = overlap(after, replay)
    chance = chance_overlap(len(after), n) if len(after) else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.2))

    counts = [len(before), len(after)]
    bars = axes[0].bar(
        ["before merge", "after merge"],
        counts,
        color=[colors[0], colors[1]],
    )
    axes[0].bar_label(bars, labels=[str(value) for value in counts], padding=3)
    axes[0].set_ylim(0, max(1, max(counts)) * 1.25)
    axes[0].set_ylabel("active winners")
    axes[0].set_title(f"{after.area} activity")

    scores = [chance, replay_overlap]
    bars = axes[1].bar(
        ["chance baseline", "same-pair replay"],
        scores,
        color=[colors[2], colors[1]],
    )
    axes[1].bar_label(bars, labels=[f"{value:.2f}" for value in scores], padding=3)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel(f"overlap in {after.area}")
    axes[1].set_title("Binding stability check")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, y=1.04, fontsize=14, fontweight="semibold")
    plt.tight_layout()
    return fig


def plot_trace_metrics(
    trace: AssemblyTrace,
    *,
    axes: Sequence[Axes] | None = None,
    title: str | None = None,
    color: str = DEFAULT_COLORS[0],
) -> tuple[Figure, np.ndarray]:
    """Plot simple round-by-round metrics for an AssemblyTrace."""
    if axes is None:
        fig, axes_obj = plt.subplots(1, 2, figsize=(8.8, 3.2))
        axes_array = np.atleast_1d(axes_obj)
    else:
        axes_array = np.asarray(list(axes), dtype=object)
        if len(axes_array) != 2:
            raise ValueError("axes must contain exactly two axes")
        fig = axes_array[0].figure

    rounds = [step.round_index for step in trace]
    overlaps = [
        np.nan if step.overlap_with_previous is None else step.overlap_with_previous
        for step in trace
    ]
    new_winners = [step.num_first_winners for step in trace]

    axes_array[0].plot(rounds, overlaps, marker="o", color=color)
    axes_array[0].set_ylim(0.0, 1.05)
    axes_array[0].set_xlabel("round")
    axes_array[0].set_ylabel("overlap with previous")
    axes_array[0].set_title("Stabilization")

    axes_array[1].bar(rounds, new_winners, color=color, alpha=0.82)
    axes_array[1].set_ylim(0, max(1, max(new_winners)) * 1.2)
    axes_array[1].set_xlabel("round")
    axes_array[1].set_ylabel("new winners")
    axes_array[1].set_title("Winner turnover")

    for ax in axes_array:
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title or f"{trace.operation} trace in {trace.target}", y=1.03)
    plt.tight_layout()
    return fig, axes_array


def animate_assembly_trace(
    trace: AssemblyTrace,
    n: int,
    *,
    title: str | None = None,
    color: str = DEFAULT_COLORS[0],
    marker_size: float = 18,
    interval: int = 700,
    repeat: bool = True,
) -> tuple[Figure, FuncAnimation]:
    """Animate active winners across an AssemblyTrace."""
    if n <= 0:
        raise ValueError("n must be positive")

    first = trace[0].assembly
    _, _, side = assembly_coordinates(first, n)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    scatter = ax.scatter([], [], s=marker_size, c=color, alpha=0.84, edgecolors="none")
    annotation = ax.text(
        0.04,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.88},
    )

    ax.set_xlim(-1, side)
    ax.set_ylim(side, -1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f7f7f2")

    base_title = title or f"{trace.operation} trace in {trace.target}"

    def update(frame: int):
        step = trace[frame]
        x, y, _ = assembly_coordinates(step.assembly, n)
        offsets = np.column_stack([x, y]) if len(x) else np.empty((0, 2))
        scatter.set_offsets(offsets)
        overlap_text = (
            "overlap prev: --"
            if step.overlap_with_previous is None
            else f"overlap prev: {step.overlap_with_previous:.2f}"
        )
        annotation.set_text(
            f"round {step.round_index}\n"
            f"{step.drive}\n"
            f"winners: {step.num_winners}\n"
            f"new: {step.num_first_winners}\n"
            f"{overlap_text}"
        )
        ax.set_title(base_title)
        return scatter, annotation

    animation = FuncAnimation(
        fig,
        update,
        frames=len(trace),
        interval=interval,
        blit=False,
        repeat=repeat,
    )
    update(0)
    return fig, animation


def plot_response_overlap(
    reference: Assembly,
    responses: Sequence[Assembly],
    *,
    n: int,
    labels: Sequence[str],
    ax: Axes | None = None,
    color: str = DEFAULT_COLORS[2],
    baseline_color: str = "#6f6f6f",
    title: str = "Response overlap",
) -> tuple[Axes, np.ndarray]:
    """Compare same-area responses against a reference assembly and chance."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not responses:
        raise ValueError("responses must not be empty")
    if len(labels) != len(responses):
        raise ValueError("labels must match responses length")
    for response in responses:
        if response.area != reference.area:
            raise ValueError("responses must live in the same area as reference")

    scores = np.array([overlap(reference, response) for response in responses], dtype=float)
    chance = chance_overlap(len(reference), n) if len(reference) else 0.0
    names = ["chance"] + list(labels)
    values = np.concatenate(([chance], scores))
    colors = [baseline_color] + [color] * len(scores)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4.8, 1.4 * len(names)), 3.2))

    bars = ax.bar(names, values, color=colors)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"overlap in {reference.area}")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax, values


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
]
