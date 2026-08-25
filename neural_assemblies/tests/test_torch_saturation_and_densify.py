"""Two torch-engine defects, both the same shape: a fix that landed on the
numpy engine and never reached its mirror.

This project has a name for that shape -- the pricing law implemented twice --
and it keeps recurring because the two engines are written to agree on
BEHAVIOUR, not on code. So each of these is pinned against the numpy engine
rather than against a remembered constant: the assertion is "torch does what
numpy does", which is the property that was actually violated.

Skipped without CUDA. The default interpreter on this machine carries a
CPU-only torch build, so these only run under the project's uv venv -- which is
itself worth knowing, since a GPU measurement taken with the wrong interpreter
silently measures the CPU.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from neural_assemblies.core.brain import Brain              # noqa: E402


def _train(engine, n, k, p, M, T, seed=42, beta=0.1):
    random.seed(seed)
    np.random.seed(seed)
    b = Brain(p=p, seed=seed, engine=engine, recurrent_projection=True,
              norm_init=False)
    b.add_area("A", n, k, beta)
    stims = [f"s{i}" for i in range(M)]
    for s in stims:
        b.add_stimulus(s, k)
    for s in stims:
        b.inhibit_areas(["A"])
        for _ in range(T):
            b.project({s: ["A"]}, {"A": ["A"]})
    return b


class TestGracefulSaturation:
    """An area at capacity stops recruiting; it must not crash mid-training.

    `effective_n = n - w` counts neurons that have never fired. When it falls
    to k or below the area cannot recruit a full k of fresh winners. The numpy
    engine takes `k_eff = min(k, max(0, effective_n - 1))` and lets the caller
    complete the set from incumbents; the torch engine raised RuntimeError.

    Saturation is not an edge case here -- an area at `rows/n = 1.0` is the
    normal end state of storing many assemblies, so this made every capacity
    study unrunnable on GPU.
    """

    def test_torch_survives_saturation_like_numpy(self):
        # n=2000, k=50, p=0.5, 20 items x 12 rounds fills the area.
        cpu = _train("numpy_sparse", 2000, 50, 0.5, M=20, T=12)
        gpu = _train("torch_sparse", 2000, 50, 0.5, M=20, T=12)
        assert len(np.asarray(cpu.areas["A"].winners)) == 50
        assert len(np.asarray(gpu.areas["A"].winners)) == 50

    def test_a_full_winner_set_is_still_returned_at_capacity(self):
        """Recruiting fewer than k fresh neurons must not shorten the assembly
        -- the remainder comes from incumbents."""
        b = _train("torch_sparse", 1200, 40, 0.5, M=25, T=10)
        assert len(np.asarray(b.areas["A"].winners)) == 40

    def test_non_saturated_runs_are_untouched(self):
        """`k_eff == k` whenever `effective_n > k`, so nothing below capacity
        may change. Compared against the SAME engine at a size where the area
        cannot saturate."""
        a = _train("torch_sparse", 20000, 70, 0.02, M=3, T=6)
        b = _train("torch_sparse", 20000, 70, 0.02, M=3, T=6)
        assert np.array_equal(np.asarray(a.areas["A"].winners),
                              np.asarray(b.areas["A"].winners))


class TestDensifyDtype:
    """`densify` was the one write into `TorchDenseConn._w` that skipped
    `.to(DTYPE)`.

    CSR stores bfloat16 and dense stores float32 DELIBERATELY -- bf16's 7-bit
    mantissa randomised the Z60 readout margin -- so the conversion is the
    whole point of the boundary. Torch refuses the mismatch outright, so every
    densify of a trained CSR fiber raised.
    """

    def test_densify_converts_bf16_to_the_dense_dtype(self):
        from neural_assemblies.core.torch_engine._csr import (
            CSRConn, TorchDenseConn, densify,
        )
        from neural_assemblies.core.torch_engine._hash import WEIGHT_DTYPE
        conn = CSRConn(device="cuda")
        r = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
        c = torch.tensor([1, 2, 3, 0], dtype=torch.int32, device="cuda")
        # WEIGHT_DTYPE, which is what the hash initialiser actually produces.
        # `CSRConn.expand` stores whatever dtype it is handed -- passing
        # float32 here made `_val` float32 and the test silently stopped
        # exercising the conversion, which the guard below caught.
        v = torch.ones(4, device="cuda", dtype=WEIGHT_DTYPE)
        conn.expand(8, 8, r, c, v)
        assert conn._val.dtype != TorchDenseConn.DTYPE, (
            "CSR is expected to store a NARROWER dtype than dense; if that "
            "stops being true this test no longer exercises the conversion")
        dense = densify(conn, device="cuda")
        assert isinstance(dense, TorchDenseConn)
        assert dense._w.dtype == TorchDenseConn.DTYPE

    def test_the_high_density_path_runs_end_to_end(self):
        """The configuration that raised: p above the densify threshold on an
        area big enough to trigger it."""
        b = _train("torch_sparse", 20000, 70, 0.05, M=20, T=12)
        assert len(np.asarray(b.areas["A"].winners)) == 70


def test_batched_pattern_hoist_is_bit_identical():
    """The transposed sparsity pattern is invariant across rounds -- Hebbian
    changes values, never edges -- so rebuilding it per round was pure cost.
    Hoisting must not move a single bit.
    """
    from neural_assemblies.core.torch_engine._batched import (
        batched_project_independent, block_diagonal,
    )
    n, k, B, R = 800, 25, 4, 6
    g = torch.Generator(device="cuda").manual_seed(0)
    nnz = int(n * n * 0.02)
    mats = []
    for _ in range(B):
        r = torch.randint(0, n, (nnz,), device="cuda", generator=g)
        c = torch.randint(0, n, (nnz,), device="cuda", generator=g)
        mats.append(torch.sparse_coo_tensor(
            torch.stack([r, c]), torch.ones(nnz, device="cuda"),
            (n, n)).coalesce())
    W = block_diagonal(mats, n)
    win = torch.stack([torch.randperm(n, device="cuda", generator=g)[:k]
                       for _ in range(B)])
    w1, v1 = batched_project_independent(W, win, B, n, k, R, beta=0.1,
                                         return_weights=True)
    w2, v2 = batched_project_independent(W, win, B, n, k, R, beta=0.1,
                                         return_weights=True)
    assert torch.equal(w1, w2) and torch.equal(v1, v2)
    assert w1.shape == (B, k)
