"""Smoke tests for patch merge and fashion ventral."""

from __future__ import annotations

import pytest

from neural_assemblies.programs.patch_merge import run_grid_patch_merge_mnist
from neural_assemblies.programs.patch_registry import resolve_patch_graph
from neural_assemblies.programs.fashion_ventral import run_fashion_spatial_mnist
from neural_assemblies.programs.colt_mnist_tier_util import clear_ventral_bundle_cache


@pytest.fixture
def smoke_kw():
    return dict(seed=42, n_examples=10)


def test_grid_patch_merge_runs(smoke_kw):
    clear_ventral_bundle_cache()
    r = run_grid_patch_merge_mnist(grid=2, **smoke_kw)
    assert r.mean_accuracy >= 0.08
    assert r.extra.get("n_patches") == 4


def test_fashion_spatial_runs(smoke_kw):
    clear_ventral_bundle_cache()
    r = run_fashion_spatial_mnist(**smoke_kw)
    assert r.mean_accuracy >= 0.08
    assert r.extra.get("patch_absence_accuracy") is not None


def test_fashion_bundle_patch_graph(smoke_kw):
    clear_ventral_bundle_cache()
    from neural_assemblies.programs.colt_mnist_tier_util import load_fashion_spatial_bundle

    b = load_fashion_spatial_bundle(use_cache=False, **smoke_kw)
    g = resolve_patch_graph(b)
    assert g is not None
    assert "collar_occlude" in g.absence_protocols
