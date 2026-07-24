"""E2 — Assembly overlap must match binary dot-product overlap."""

from __future__ import annotations

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.assembly import (
    Assembly,
    overlap,
    overlap_from_binary,
)
from neural_assemblies.assembly_calculus.ops import _snap, learn_assembly_from_pattern
from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.colt_mnist_brain_util import set_kcap_winners


@pytest.fixture
def explicit_mnist_brain():
    brain = Brain(p=0.1, save_winners=True, seed=42, engine="numpy_sparse", w_max=1e9)
    brain.add_area("LOW", 784, 200, 1.0, explicit=True)
    brain.add_area("HIGH", 2000, 200, 1.0, explicit=True)
    return brain


def test_overlap_from_binary_matches_assembly_explicit():
    rng = np.random.default_rng(0)
    k, n = 200, 2000
    a = np.zeros(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)
    shared = rng.choice(n, size=k // 2, replace=False)
    only_a = rng.choice(np.setdiff1d(np.arange(n), shared), size=k // 2, replace=False)
    only_b = rng.choice(np.setdiff1d(np.arange(n), np.union1d(shared, only_a)), size=k // 2, replace=False)
    a[np.union1d(shared, only_a)] = 1.0
    b[np.union1d(shared, only_b)] = 1.0

    asm_a = Assembly("HIGH", np.flatnonzero(a > 0).astype(np.uint32))
    asm_b = Assembly("HIGH", np.flatnonzero(b > 0).astype(np.uint32))
    dot_ov = overlap_from_binary(a, b, k)
    asm_ov = overlap(asm_a, asm_b)
    assert asm_ov == pytest.approx(dot_ov, abs=1e-6)
    assert asm_ov == pytest.approx(len(shared) / k, abs=1e-6)


def test_assembly_from_area_matches_dot_on_explicit_brain(explicit_mnist_brain):
    brain = explicit_mnist_brain
    rng = np.random.default_rng(1)
    k = 200
    pat_a = np.zeros(784, dtype=np.float32)
    pat_b = np.zeros(784, dtype=np.float32)
    idx_a = rng.choice(784, k, replace=False)
    idx_b = rng.choice(784, k, replace=False)
    pat_a[idx_a] = 1.0
    pat_b[idx_b] = 1.0

    set_kcap_winners(brain, "LOW", pat_a)
    snap_a = Assembly.from_area(brain, "LOW")
    set_kcap_winners(brain, "LOW", pat_b)
    snap_b = Assembly.from_area(brain, "LOW")

    dot_ov = overlap_from_binary(pat_a, pat_b, k)
    assert overlap(snap_a, snap_b) == pytest.approx(dot_ov, abs=1e-6)


def test_snap_sparse_engine_overlap_matches_dot():
    brain = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
    brain.add_stimulus("S", 100)
    brain.add_area("A", 5000, 80, 0.1)
    brain.add_stimulus("T", 100)
    from neural_assemblies.assembly_calculus.ops import project

    project(brain, "S", "A", rounds=10)
    ref = _snap(brain, "A")
    vec = np.zeros(5000, dtype=np.float32)
    vec[np.asarray(ref.winners, dtype=int)] = 1.0
    project(brain, "T", "A", rounds=10)
    other = _snap(brain, "A")
    vec2 = np.zeros(5000, dtype=np.float32)
    vec2[np.asarray(other.winners, dtype=int)] = 1.0
    assert overlap(ref, other) == pytest.approx(
        overlap_from_binary(vec, vec2, 80), abs=1e-6,
    )


def test_learn_assembly_from_pattern_converges(explicit_mnist_brain):
    brain = explicit_mnist_brain
    rng = np.random.default_rng(2)
    k = 200
    pat = np.zeros(784, dtype=np.float32)
    pat[rng.choice(784, k, replace=False)] = 1.0
    _, epochs, pers = learn_assembly_from_pattern(
        brain, "LOW", pat, "HIGH", max_epochs=10, project_rounds=5, tau=0.85,
    )
    assert epochs <= 10
    assert pers >= 0.5


def test_sparse_pattern_complete_meets_e1_threshold():
    """E1 acceptance: literature sparse brain recovers >=50% at 50% mask."""
    from neural_assemblies.assembly_calculus.ops import project, pattern_complete

    brain = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
    brain.add_stimulus("S", 100)
    brain.add_area("A", 5000, 80, 0.1)
    project(brain, "S", "A", rounds=10)
    _, rec = pattern_complete(brain, "A", fraction=0.5, rounds=5, seed=42)
    assert rec >= 0.50


def test_init_reciprocal_connectome_and_consolidate_pair(explicit_mnist_brain):
    from neural_assemblies.assembly_calculus.ops import consolidate_pair

    brain = explicit_mnist_brain
    brain.init_reciprocal_connectome("LOW", "HIGH", init="transpose_forward")
    rng = np.random.default_rng(3)
    k = 200
    low_pat = np.zeros(784, dtype=np.float32)
    high_pat = np.zeros(2000, dtype=np.float32)
    low_pat[rng.choice(784, k, replace=False)] = 1.0
    high_pat[rng.choice(2000, k, replace=False)] = 1.0
    asm_l = Assembly("LOW", np.flatnonzero(low_pat > 0).astype(np.uint32))
    asm_h = Assembly("HIGH", np.flatnonzero(high_pat > 0).astype(np.uint32))
    saved = brain.disable_plasticity
    brain.disable_plasticity = False
    brain.set_fiber_plasticity("HIGH", "LOW", True)
    try:
        consolidate_pair(brain, "HIGH", asm_h, "LOW", asm_l, rounds=3, a_to_b=True, b_to_a=False)
    finally:
        brain.disable_plasticity = saved
    assert float(np.sum(brain.connectomes["HIGH"]["LOW"].weights)) > 0


def test_fiber_plasticity_mask_blocks_reinforce(explicit_mnist_brain):
    brain = explicit_mnist_brain
    brain.set_fiber_plasticity("LOW", "HIGH", False)
    set_kcap_winners(brain, "LOW", np.ones(784, dtype=np.float32))
    w_before = brain.connectomes["LOW"]["HIGH"].weights.copy()
    brain.reinforce_connectome("LOW", "HIGH", np.arange(200, dtype=np.uint32))
    w_after = brain.connectomes["LOW"]["HIGH"].weights
    assert np.allclose(w_before, w_after)
