"""Do the hashed kernels reproduce `numpy_sparse` on all four substrate arms?

Everything else about this path is checked against a reference written for the
purpose. This is checked against THE ENGINE THAT PRODUCED THE SCIENCE, which is
the only comparison that licenses running a study on it.

TWO CONFOUNDS ARE HELD APART DELIBERATELY.

  * The SELECTOR differs by tie policy (canonical vs argpartition+argsort), and
    a round-1 drive is an integer Bernoulli sum, so ties at the bar are the
    common case. Letting both run freely would report a tie-order divergence as
    a substrate error. So the ENGINE's winner trajectory is captured and
    REPLAYED through the kernels, and the DRIVE is compared every round.
  * `numpy_sparse` materialises lazily and prices unmaterialised neurons as
    sampled candidates (`_pricing`); the hashed path has every neuron from the
    start. `materialize_area` removes the sampler
    ([[sampler-is-the-whole-discrepancy]]), which is what makes them comparable.

STORAGE MUST BE DENSE. `materialize_area(storage="csr")` + `synaptic_scaling`
raises inside the engine -- `_scale_columns_now` assigns `w[:rows, valid]`,
which `CSRWeights.__setitem__` refuses. That is an engine limitation this test
documents rather than works around silently.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from neural_assemblies.core.brain import Brain                  # noqa: E402
from neural_assemblies.core.numpy_engine import _seeding        # noqa: E402
from neural_assemblies.core.torch_engine import _fused_cuda     # noqa: E402
from neural_assemblies.core.torch_engine._batched import (      # noqa: E402
    _chain_table)

AREA = "A"
DEV = "cuda"


@pytest.fixture(scope="module")
def mod():
    m = _fused_cuda.load()
    if m is None:
        pytest.skip(f"fused kernels unavailable: {_fused_cuda.last_error()}")
    return m


def _to_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _engine_trace(n, k, p, beta, T, seed, norm_init, scaling, w_max):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=p, seed=seed, engine="numpy_sparse", w_max=w_max,
                  recurrent_projection=True, norm_init=norm_init,
                  synaptic_scaling=scaling)
    brain.add_area(AREA, n, k, beta)
    eng = brain._engine_for(brain.areas[AREA])
    eng.materialize_area(AREA, storage="dense")
    rng = np.random.default_rng(seed)
    eng.set_winners(AREA, np.sort(
        rng.choice(n, k, replace=False)).astype(np.uint32))
    drives, prevs, news = [], [], []
    for _ in range(T):
        prevs.append(np.asarray(eng.get_winners(AREA), dtype=np.int64))
        res = eng.project_into(AREA, [], [AREA], plasticity_enabled=True,
                               record_activation=True)
        drives.append(np.asarray(res.pre_kwta_inputs, dtype=np.float64))
        news.append(np.asarray(eng.get_winners(AREA), dtype=np.int64))
    return drives, prevs, news, _seeding.fnv1a_pair_seed(seed, AREA, AREA)


def _replay(mod, n, p, beta, T, pair, prevs, news, norm_init, scaling, w_max):
    thr = _fused_cuda.threshold_for(p)
    seeds_t = torch.tensor([_to_i32(pair)], dtype=torch.int32, device=DEV)
    rowmask = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    colmask = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    tab = torch.from_numpy(_chain_table(beta, w_max, T)).to(DEV)
    scale = (torch.ones(1, n, dtype=torch.float32, device=DEV)
             if scaling else None)
    setpoint = max(float(n) * float(p), 1e-12)
    dj = mod.hashed_indegree(seeds_t, n, thr, 1.0) if norm_init else None
    colids = torch.arange(n, dtype=torch.int32, device=DEV).view(1, n)
    hist, out = [], []
    for t in range(T):
        rows = torch.from_numpy(
            prevs[t].astype(np.int32)).to(DEV).view(1, -1).contiguous()
        d = mod.hashed_drive(rows, seeds_t, n, thr)
        if hist:
            mod.dev_correct(rows, rowmask, colids, colmask, tab, seeds_t,
                            thr, d)
        if scale is not None:
            d = d * scale
        if dj is not None:
            d = d / dj
        out.append(d[0].cpu().numpy().astype(np.float64))
        bit = 1 << t
        pidx = torch.from_numpy(prevs[t]).to(DEV).view(1, -1)
        sidx = torch.from_numpy(news[t]).to(DEV).view(1, -1)
        rowmask.scatter_(1, pidx, rowmask.gather(1, pidx) | bit)
        colmask.scatter_(1, sidx, colmask.gather(1, sidx) | bit)
        hist.append(sidx.to(torch.int32))
        if scale is not None:
            mass = mod.column_mass(sidx.to(torch.int32).contiguous(), rowmask,
                                   colmask, tab, seeds_t, thr)
            scale.scatter_(1, sidx, setpoint / mass.clamp_min(1e-12))
    return out


@pytest.mark.parametrize("arm,norm_init,scaling", [
    ("NONE", False, False), ("B", True, False),
    ("C", False, True), ("G", True, True),
])
@pytest.mark.parametrize("n,k,p", [(1024, 30, 0.1), (2048, 50, 0.5)])
def test_substrate_arm_reproduces_numpy_sparse(mod, arm, norm_init, scaling,
                                               n, k, p):
    beta, T, w_max = 0.1, 6, 20.0
    d_cpu, prevs, news, pair = _engine_trace(
        n, k, p, beta, T, 7, norm_init, scaling, w_max)
    d_gpu = _replay(mod, n, p, beta, T, pair, prevs, news, norm_init, scaling,
                    w_max)
    worst = 0.0
    for t in range(T):
        m = min(len(d_cpu[t]), len(d_gpu[t]))
        a, b = d_cpu[t][:m], d_gpu[t][:m]
        worst = max(worst, float(np.abs(a - b).max())
                    / max(float(np.abs(a).max()), 1e-12))
    # float32 accumulation against the engine's float64, over T rounds of
    # training. This is a tolerance, not an equality: the drive is NOT
    # bit-identical once plasticity runs, and a study on this path inherits
    # that. Without plasticity arm NONE IS bit-identical (see
    # test_single_round_no_plasticity_is_bit_identical).
    assert worst < 5e-6, f"arm {arm}: relative drive error {worst:.3g}"


def test_single_round_no_plasticity_is_bit_identical(mod):
    """The connectome itself, with no float accumulation in the way."""
    n, k, p = 2048, 50, 0.5
    d_cpu, prevs, _, pair = _engine_trace(n, k, p, 0.0, 1, 7, False, False,
                                          20.0)
    d_gpu = _replay(mod, n, p, 0.0, 1, pair, prevs, [prevs[0]], False, False,
                    20.0)
    assert np.array_equal(d_cpu[0], d_gpu[0])


def test_norm_init_divisor_has_no_unknown_rows_term(mod):
    """`d_j` here is the TRUE in-degree, not `deg + p*(n_pre - rows_known)`.

    Every row exists in a generated connectome, so the estimate term -- the
    site of the fiber-p defect ([[norm-init-fiber-p]], a 6.15x over-scale) --
    is identically zero. Pinned by comparing against a direct count.
    """
    n, p, seed = 2048, 0.5, 12345
    thr = _fused_cuda.threshold_for(p)
    seeds_t = torch.tensor([_to_i32(seed)], dtype=torch.int32, device=DEV)
    got = mod.hashed_indegree(seeds_t, n, thr, 1.0)[0].cpu().numpy()
    ref = np_seed_indegree(n, p, seed)
    assert np.array_equal(got.astype(np.int64), ref.astype(np.int64))
    assert abs(got.mean() - n * p) < 0.05 * n * p


def np_seed_indegree(n, p, seed):
    """Column in-degree straight from the numpy engine's own hash."""
    W = _seeding.hash_bernoulli_2d(0, n, 0, n, seed, p, finalize=True)
    return W.sum(axis=0)
