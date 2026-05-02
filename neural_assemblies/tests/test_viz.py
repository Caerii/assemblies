from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from neural_assemblies.assembly_calculus import Assembly, AssemblyTrace, TraceStep
from neural_assemblies.viz import (
    animate_assembly_trace,
    assembly_coordinates,
    assembly_overlap_matrix,
    plot_binding_story,
    plot_assemblies,
    plot_assembly,
    plot_merge_diagnostic,
    plot_overlap_matrix,
    plot_parameter_heatmap,
    plot_projection_flow,
    plot_recall_trace,
    plot_response_overlap,
    plot_trace_metrics,
    plot_winner_turnover,
)


def _assembly(area: str, winners: list[int]) -> Assembly:
    return Assembly(area, np.array(winners, dtype=np.uint32))


def _trace() -> AssemblyTrace:
    steps = []
    previous = None
    for index, winners in enumerate(([0, 1, 2], [1, 2, 3], [1, 2, 4]), start=1):
        assembly = _assembly("A", list(winners))
        steps.append(
            TraceStep(
                round_index=index,
                operation="project",
                area="A",
                assembly=assembly,
                drive="stimulus" if index == 1 else "stimulus + recurrence",
                sources=("stim",),
                num_winners=len(assembly),
                num_ever_fired=len(set(winners)),
                num_first_winners=3 if previous is None else 1,
                overlap_with_previous=None if previous is None else previous.overlap(assembly),
            )
        )
        previous = assembly
    return AssemblyTrace(operation="project", target="A", steps=tuple(steps))


def test_assembly_coordinates_map_winners_to_square_grid():
    assembly = _assembly("A", [0, 1, 4, 7])

    x, y, side = assembly_coordinates(assembly, n=9)

    assert side == 3
    assert x.tolist() == [0, 1, 1, 1]
    assert y.tolist() == [0, 0, 1, 2]


def test_assembly_overlap_matrix_is_pairwise_and_symmetric():
    a = _assembly("A", [1, 2, 3])
    b = _assembly("A", [2, 3, 4])

    matrix = assembly_overlap_matrix([a, b])

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert matrix[0, 1] == matrix[1, 0]
    assert round(matrix[0, 1], 3) == 0.667


def test_plot_helpers_create_matplotlib_artists():
    a = _assembly("A", [0, 3, 5])
    b = _assembly("B", [1, 4, 6])

    fig, axes = plot_assemblies(
        [a, b],
        n=16,
        subtitles=["3 winners | 18.75% density", "3 winners | 18.75% density"],
        marker_size=20,
    )
    assert len(axes) == 2
    assert axes[0].texts
    plt.close(fig)

    fig, ax = plt.subplots()
    returned = plot_assembly(a, n=16, ax=ax, subtitle="caption", annotation="sample")
    assert returned is ax
    assert ax.collections
    assert len(ax.texts) == 2
    plt.close(fig)


def test_flow_overlap_and_recall_plots_are_headless_safe():
    a = _assembly("A", [0, 1, 2])
    b = _assembly("A", [1, 2, 3])

    fig = plot_projection_flow(
        stimulus_labels=["red cue", "triangle cue"],
        source_labels=["COLOR area", "SHAPE area"],
        target_label="OBJECT area",
    )
    assert fig.axes
    plt.close(fig)

    fig = plot_binding_story(
        stimulus_labels=["red cue", "triangle cue"],
        source_labels=["COLOR", "SHAPE"],
        target_label="OBJECT",
        caption="toy binding story",
    )
    assert fig.axes
    plt.close(fig)

    fig = plot_merge_diagnostic(
        _assembly("OBJECT", []),
        _assembly("OBJECT", [0, 1, 2]),
        _assembly("OBJECT", [0, 1, 3]),
        n=16,
    )
    assert len(fig.axes) == 2
    plt.close(fig)

    fig, axes = plot_trace_metrics(_trace())
    assert len(axes) == 2
    plt.close(fig)

    ax, turnover = plot_winner_turnover(_trace())
    assert turnover.shape == (3, 5)
    plt.close(ax.figure)

    fig, animation = animate_assembly_trace(_trace(), n=16)
    assert isinstance(animation, FuncAnimation)
    animation._draw_was_started = True
    plt.close(fig)

    ax, values = plot_response_overlap(
        _assembly("OBJECT", [0, 1, 2]),
        [_assembly("OBJECT", [0, 1, 3]), _assembly("OBJECT", [4, 5, 6])],
        n=16,
        labels=["color alone", "shape alone"],
    )
    assert values.shape == (3,)
    plt.close(ax.figure)

    ax, overlap_matrix = plot_overlap_matrix([a, b], labels=["a", "b"])
    assert overlap_matrix.shape == (2, 2)
    plt.close(ax.figure)

    ax, recall_matrix = plot_recall_trace([a], [a, b])
    assert recall_matrix.shape == (1, 2)
    plt.close(ax.figure)

    ax, heatmap = plot_parameter_heatmap(
        [[0.1, 0.2], [0.3, 0.4]],
        x_labels=["low", "high"],
        y_labels=["a", "b"],
    )
    assert heatmap.shape == (2, 2)
    plt.close(ax.figure)


def test_legacy_visualization_namespace_reexports_viz_helpers():
    import neural_assemblies.visualization as visualization

    assert visualization.plot_assembly is plot_assembly


def test_plot_helpers_reject_empty_inputs():
    with pytest.raises(ValueError, match="assemblies"):
        assembly_overlap_matrix([])

    with pytest.raises(ValueError, match="assemblies"):
        plot_assemblies([], n=16)

    with pytest.raises(ValueError, match="subtitles"):
        plot_assemblies([_assembly("A", [1])], n=16, subtitles=["one", "two"])

    with pytest.raises(ValueError, match="same area"):
        plot_merge_diagnostic(
            _assembly("A", []),
            _assembly("B", [1]),
            _assembly("B", [1]),
            n=16,
        )

    with pytest.raises(ValueError, match="three colors"):
        plot_merge_diagnostic(
            _assembly("A", []),
            _assembly("A", [1]),
            _assembly("A", [1]),
            n=16,
            colors=("#111111", "#222222"),
        )

    with pytest.raises(ValueError, match="labels"):
        plot_response_overlap(
            _assembly("A", [1]),
            [_assembly("A", [1])],
            n=16,
            labels=["one", "two"],
        )

    with pytest.raises(ValueError, match="recalled"):
        plot_recall_trace([], [_assembly("A", [1])])

    with pytest.raises(ValueError, match="known"):
        plot_recall_trace([_assembly("A", [1])], [])

    with pytest.raises(ValueError, match="shape"):
        plot_parameter_heatmap([[1, 2, 3]], x_labels=["a"], y_labels=["b"])
