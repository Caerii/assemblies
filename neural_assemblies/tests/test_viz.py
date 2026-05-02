from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from neural_assemblies.assembly_calculus import Assembly
from neural_assemblies.viz import (
    assembly_coordinates,
    assembly_overlap_matrix,
    plot_assemblies,
    plot_assembly,
    plot_overlap_matrix,
    plot_projection_flow,
    plot_recall_trace,
)


def _assembly(area: str, winners: list[int]) -> Assembly:
    return Assembly(area, np.array(winners, dtype=np.uint32))


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

    fig, axes = plot_assemblies([a, b], n=16)
    assert len(axes) == 2
    plt.close(fig)

    fig, ax = plt.subplots()
    returned = plot_assembly(a, n=16, ax=ax)
    assert returned is ax
    assert ax.collections
    plt.close(fig)


def test_flow_overlap_and_recall_plots_are_headless_safe():
    a = _assembly("A", [0, 1, 2])
    b = _assembly("A", [1, 2, 3])

    fig = plot_projection_flow(
        stimulus_labels=["stimulus s1", "stimulus s2"],
        source_labels=["area A1", "area A2"],
        target_label="merged area B",
    )
    assert fig.axes
    plt.close(fig)

    ax, overlap_matrix = plot_overlap_matrix([a, b], labels=["a", "b"])
    assert overlap_matrix.shape == (2, 2)
    plt.close(ax.figure)

    ax, recall_matrix = plot_recall_trace([a], [a, b])
    assert recall_matrix.shape == (1, 2)
    plt.close(ax.figure)


def test_legacy_visualization_namespace_reexports_viz_helpers():
    import neural_assemblies.visualization as visualization

    assert visualization.plot_assembly is plot_assembly


def test_plot_helpers_reject_empty_inputs():
    with pytest.raises(ValueError, match="assemblies"):
        assembly_overlap_matrix([])

    with pytest.raises(ValueError, match="assemblies"):
        plot_assemblies([], n=16)

    with pytest.raises(ValueError, match="recalled"):
        plot_recall_trace([], [_assembly("A", [1])])

    with pytest.raises(ValueError, match="known"):
        plot_recall_trace([_assembly("A", [1])], [])
