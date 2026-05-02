"""Trace-specific plots and animations."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neural_assemblies.assembly_calculus import Assembly, AssemblyTrace, chance_overlap, overlap

from ..grids import DEFAULT_COLORS, assembly_coordinates


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


def plot_winner_turnover(
    trace: AssemblyTrace,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "Greens",
) -> tuple[Axes, np.ndarray]:
    """Plot which winner IDs appear at each trace step."""
    ordered_winners: list[int] = []
    seen: set[int] = set()
    for step in trace:
        for winner in step.assembly.winners.tolist():
            winner_id = int(winner)
            if winner_id not in seen:
                seen.add(winner_id)
                ordered_winners.append(winner_id)

    matrix = np.zeros((len(trace), len(ordered_winners)), dtype=float)
    winner_to_column = {winner: column for column, winner in enumerate(ordered_winners)}
    for row, step in enumerate(trace):
        for winner in step.assembly.winners.tolist():
            matrix[row, winner_to_column[int(winner)]] = 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5.0, len(ordered_winners) * 0.08), 3.2))

    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(trace)), [step.round_index for step in trace])
    ax.set_xticks([])
    ax.set_xlabel("winner IDs ordered by first appearance")
    ax.set_ylabel("round")
    ax.set_title(title or f"{trace.operation} winner turnover")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax, matrix


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
