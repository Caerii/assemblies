"""Smoke tests for patch graph and vision data modules."""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.programs.patch_graph import (
    build_fashion_salient_patch_graph,
    build_grid_patch_graph,
    build_halves_patch_graph,
    halves_fields_from_pattern,
)
from neural_assemblies.programs.vision_data import (
    VisionDataset,
    load_vision_examples,
    vision_data_info,
)


def test_halves_patch_graph():
    g = build_halves_patch_graph()
    assert len(g.patches) == 2
    assert g.bind_pairs == ((0, 1),)
    pat = np.ones(784, dtype=np.float32)
    top, bot = halves_fields_from_pattern(pat)
    assert top[:392].sum() > 0
    assert bot[392:].sum() > 0
    masked = g.apply_absence(pat, "top_half")
    assert masked[:392].sum() == 0
    assert masked[392:].sum() > 0


def test_grid_patch_absence():
    g = build_grid_patch_graph(grid=3, radii=(2,))
    assert len(g.patches) == 9
    pat = np.random.default_rng(0).random(784).astype(np.float32)
    if g.absence_protocols:
        proto = next(iter(g.absence_protocols))
        out = g.apply_absence(pat, proto)
        assert not np.allclose(pat, out)


def test_fashion_salient_graph():
    g = build_fashion_salient_patch_graph()
    assert len(g.patches) >= 5
    assert "collar_occlude" in g.absence_protocols


def test_vision_data_mnist():
    examples, info = load_vision_examples(VisionDataset.MNIST, n_examples=5)
    assert examples.shape == (10, 5, 784)
    assert info.n_classes == 10


def test_vision_data_fashion():
    info = vision_data_info(VisionDataset.FASHION_MNIST)
    assert len(info.class_names) == 10
    assert len(info.confused_pairs) >= 3
