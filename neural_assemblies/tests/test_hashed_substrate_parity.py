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
from neural_assemblies.core.torch_engine._hashed import (        # noqa: E402
    _chain_table, _gain_table)

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
            mass, _ = mod.column_mass(sidx.to(torch.int32).contiguous(),
                                      rowmask, colmask, tab, seeds_t, thr)
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


# -- the stimulus fiber ----------------------------------------------------

STIM = "s0"


def _engine_trace_stim(n, k, p, beta, T, seed, norm_init, w_max):
    """Same replay method, with a stimulus firing every round.

    The stimulus is the ANCHOR the registered training protocol relies on:
    `project({s: [AREA]}, {AREA: [AREA]})` fires it on every round, and without
    it an assembly has nothing to converge toward.
    """
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=p, seed=seed, engine="numpy_sparse", w_max=w_max,
                  recurrent_projection=True, norm_init=norm_init,
                  synaptic_scaling=False)
    brain.add_area(AREA, n, k, beta)
    brain.add_stimulus(STIM, k)
    eng = brain._engine_for(brain.areas[AREA])
    eng.materialize_area(AREA, storage="dense")
    rng = np.random.default_rng(seed)
    eng.set_winners(AREA, np.sort(
        rng.choice(n, k, replace=False)).astype(np.uint32))
    stim0 = np.asarray(eng._stim_conns[STIM][AREA].weights,
                       dtype=np.float64).copy()
    drives, prevs, news = [], [], []
    for _ in range(T):
        prevs.append(np.asarray(eng.get_winners(AREA), dtype=np.int64))
        res = eng.project_into(AREA, [STIM], [AREA], plasticity_enabled=True,
                               record_activation=True)
        drives.append(np.asarray(res.pre_kwta_inputs, dtype=np.float64))
        news.append(np.asarray(eng.get_winners(AREA), dtype=np.int64))
    return (drives, prevs, news, stim0,
            _seeding.fnv1a_pair_seed(seed, AREA, AREA))


@pytest.mark.parametrize("norm_init", [False, True])
@pytest.mark.parametrize("n,k,p", [(1024, 30, 0.1), (2048, 50, 0.5)])
def test_stimulus_pricing_reproduces_numpy_sparse(mod, norm_init, n, k, p):
    """The engine's own stim base is INJECTED, on purpose.

    `_expand_stim_vectors_fast` draws stimulus weights from
    `self._rng.binomial(...)` consumed in stimulus insertion order, so a
    stimulus connectome is NOT a pure function of position and a generated one
    cannot reproduce it. (`_seeding.hash_stim_counts` exists and is documented
    as "the content-addressed replacement", but the growth path does not use
    it.) Injecting the base isolates the question this CAN answer: is the
    PRICING right -- the `tgt.n` divisor and the `w_max * stim_size * p` cap?
    """
    beta, T, w_max = 0.1, 6, 20.0
    d_cpu, prevs, news, stim0, apair = _engine_trace_stim(
        n, k, p, beta, T, 7, norm_init, w_max)

    thr = _fused_cuda.threshold_for(p)
    a_s = torch.tensor([_to_i32(apair)], dtype=torch.int32, device=DEV)
    rowmask = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    colmask = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    tab = torch.from_numpy(_chain_table(beta, w_max, T)).to(DEV)
    gpow = torch.from_numpy(_gain_table(beta, T)).to(DEV)
    colids = torch.arange(n, dtype=torch.int32, device=DEV).view(1, n)
    dj = mod.hashed_indegree(a_s, n, thr, 1.0) if norm_init else None
    sbase = torch.from_numpy(stim0.astype(np.float32)).to(DEV).view(1, -1)
    spot = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    sdj = (sbase + p * (n - k)).clamp_min(1.0) if norm_init else None
    shi = w_max * max(1.0, k * p)

    worst = 0.0
    for t in range(T):
        rows = torch.from_numpy(
            prevs[t].astype(np.int32)).to(DEV).view(1, -1).contiguous()
        d = mod.hashed_drive(rows, a_s, n, thr)
        if t:
            mod.dev_correct(rows, rowmask, colids, colmask, tab, a_s, thr, d)
        if dj is not None:
            d = d / dj
        sd = (sbase * gpow[spot.clamp_max(gpow.numel() - 1)]).clamp_max(shi)
        if sdj is not None:
            sd = sd / sdj
        d = d + sd
        got = d[0].cpu().numpy().astype(np.float64)
        m = min(len(d_cpu[t]), len(got))
        worst = max(worst, float(np.abs(d_cpu[t][:m] - got[:m]).max())
                    / max(float(np.abs(d_cpu[t][:m]).max()), 1e-12))
        bit = 1 << t
        pidx = torch.from_numpy(prevs[t]).to(DEV).view(1, -1)
        sidx = torch.from_numpy(news[t]).to(DEV).view(1, -1)
        rowmask.scatter_(1, pidx, rowmask.gather(1, pidx) | bit)
        colmask.scatter_(1, sidx, colmask.gather(1, sidx) | bit)
        spot.scatter_add_(1, sidx, torch.ones_like(sidx))
    assert worst < 5e-6, f"norm_init={norm_init}: relative error {worst:.3g}"


def test_norm_init_stim_divisor_is_potentiation_invariant():
    """`_norm_scale`'s own contract, asserted directly.

    Its docstring says "Present synapses are COUNTED, not summed, so the
    divisor is potentiation-invariant". It was not: the snapshot was taken with
    `xp.asarray(w[...], dtype=float32)`, which on an already-float32 array
    returns a VIEW, so every potentiation moved the divisor with the weights.
    The stimulus contribution `w / (w + unknown*p)` then drifts toward 1 --
    exactly the failure the same docstring warns about.
    """
    n, k, p, seed = 1024, 30, 0.1, 7
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=p, seed=seed, engine="numpy_sparse",
                  recurrent_projection=True, norm_init=True,
                  synaptic_scaling=False)
    brain.add_area(AREA, n, k, 0.1)
    brain.add_stimulus(STIM, k)
    eng = brain._engine_for(brain.areas[AREA])
    eng.materialize_area(AREA, storage="dense")
    eng.set_winners(AREA, np.sort(np.random.default_rng(seed).choice(
        n, k, replace=False)).astype(np.uint32))
    conn = eng._stim_conns[STIM][AREA]
    eng.project_into(AREA, [STIM], [AREA], plasticity_enabled=True,
                     record_activation=True)
    assert not np.shares_memory(conn._norm_deg_base, conn.weights)
    before = np.asarray(conn._norm_deg_base, dtype=np.float64).copy()
    conn.weights[:5] *= 3.0
    after = np.asarray(conn._norm_deg_base, dtype=np.float64)
    assert np.array_equal(before, after), (
        "norm_init's stimulus divisor moved with the weights")


# -- multi-episode, against the ENGINE -------------------------------------

@pytest.mark.parametrize("arm,norm_init,scaling", [
    ("NONE", False, False), ("B", True, False),
    ("C", False, True), ("G", True, True),
])
def test_capacity_protocol_reproduces_numpy_sparse_across_episodes(
        mod, arm, norm_init, scaling):
    """The check that would have caught the contaminated capacity run.

    The single-trajectory replays above never read learned state ACROSS
    episodes, and the CSR store's own tests compare against a reference
    written for the purpose. This replays the ENGINE running the capacity
    protocol itself -- inhibit, then T rounds of stimulus+recurrence, per
    assembly -- and compares the hashed path's drive at every round of every
    episode. A store that dropped, double-counted or mis-keyed a cell across
    the episode boundary diverges here against the engine, not against my
    own arithmetic.

    Stimulus bases are INJECTED from the engine (its stimulus connectomes are
    drawn in RNG order, not content-addressed), which is the established
    method from the stimulus-pricing test above.
    """
    from neural_assemblies.core.torch_engine._hashed import (
        AreaFiber, StimulusFiber)

    n, k, p, beta, T, M_eps, w_max, seed = 1024, 30, 0.1, 0.1, 4, 3, 20.0, 7
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=p, seed=seed, engine="numpy_sparse", w_max=w_max,
                  recurrent_projection=True, norm_init=norm_init,
                  synaptic_scaling=scaling)
    brain.add_area(AREA, n, k, beta)
    stims = []
    for a in range(M_eps):
        brain.add_stimulus(f"s{a}", k)
        stims.append(f"s{a}")
    eng = brain._engine_for(brain.areas[AREA])
    eng.materialize_area(AREA, storage="dense")

    # engine trace: per episode, per round -- drive, prev winners, new winners
    trace = []
    stim0 = {}
    for a, sname in enumerate(stims):
        stim0[a] = np.asarray(eng._stim_conns[sname][AREA].weights,
                              dtype=np.float64).copy()
        brain.inhibit_areas([AREA])
        ep = []
        for _ in range(T):
            prev = np.asarray(eng.get_winners(AREA), dtype=np.int64)
            res = eng.project_into(AREA, [sname], [AREA],
                                   plasticity_enabled=True,
                                   record_activation=True)
            ep.append((prev,
                       np.asarray(res.pre_kwta_inputs, dtype=np.float64),
                       np.asarray(eng.get_winners(AREA), dtype=np.int64)))
        trace.append(ep)

    # hashed replay: same fiber pattern (the engine's own hash), same winners
    pair = _seeding.fnv1a_pair_seed(seed, AREA, AREA)
    fiber = AreaFiber([_to_i32(pair)], n, n, p, beta=beta, w_max=w_max,
                      norm_init=norm_init, synaptic_scaling=scaling,
                      max_rounds=M_eps * T)
    worst = 0.0
    for a, ep in enumerate(trace):
        sf = StimulusFiber([0], k, n, p, beta=beta, w_max=w_max,
                           norm_init=norm_init, max_rounds=T)
        sf.base = torch.from_numpy(
            stim0[a].astype(np.float32)).cuda().view(1, -1)
        if norm_init:
            sf.dj = (sf.base + p * (n - k)).clamp_min(1.0)
        fiber.begin_episode()
        for prev, d_cpu, new in ep:
            drive = torch.zeros(1, n, dtype=torch.float32, device="cuda")
            fiber.contribute(drive, torch.from_numpy(prev).cuda().view(1, -1))
            sf.contribute(drive)
            got = drive[0].cpu().numpy().astype(np.float64)
            m = min(len(d_cpu), len(got))
            worst = max(worst, float(np.abs(d_cpu[:m] - got[:m]).max())
                        / max(float(np.abs(d_cpu[:m]).max()), 1e-12))
            pt = torch.from_numpy(prev).cuda().view(1, -1)
            nt = torch.from_numpy(new).cuda().view(1, -1)
            fiber.observe(pt, nt)
            sf.observe(pt, nt)
        fiber.end_episode()
    assert fiber.nnz > 0, "the store never populated -- the test is vacuous"
    assert worst < 5e-6, (
        f"arm {arm}: hashed path diverges from numpy_sparse across episodes: "
        f"relative drive error {worst:.3g}")


def test_refracted_capacity_protocol_reproduces_numpy_sparse(mod):
    """The engine's `refracted` mode, replayed on the hashed area.

    Same multi-episode protocol as above with `set_refracted(AREA, True, beta)`
    on the engine. The engine's `pre_kwta_inputs` snapshot is taken AFTER the
    bias is subtracted, so the comparison is on the NET drive the k-WTA ranks;
    the bias itself is charged from the raw drive at the engine's winners,
    which is what `HashedArea.charge` does. Registered as the parity gate of
    PREREG_refraction_capacity.md: no number there is read before this passes.
    """
    from neural_assemblies.core.torch_engine._hashed import (
        AreaFiber, HashedArea, StimulusFiber)

    n, k, p, beta, T, M_eps, w_max, seed = 1024, 30, 0.1, 0.1, 4, 3, 20.0, 7
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=p, seed=seed, engine="numpy_sparse", w_max=w_max,
                  recurrent_projection=True, norm_init=True,
                  synaptic_scaling=False)
    brain.add_area(AREA, n, k, beta)
    stims = []
    for a in range(M_eps):
        brain.add_stimulus(f"s{a}", k)
        stims.append(f"s{a}")
    eng = brain._engine_for(brain.areas[AREA])
    eng.materialize_area(AREA, storage="dense")
    eng.set_refracted(AREA, True, beta)

    trace, stim0 = [], {}
    for a, sname in enumerate(stims):
        stim0[a] = np.asarray(eng._stim_conns[sname][AREA].weights,
                              dtype=np.float64).copy()
        brain.inhibit_areas([AREA])
        ep = []
        for _ in range(T):
            prev = np.asarray(eng.get_winners(AREA), dtype=np.int64)
            res = eng.project_into(AREA, [sname], [AREA],
                                   plasticity_enabled=True,
                                   record_activation=True)
            ep.append((prev,
                       np.asarray(res.pre_kwta_inputs, dtype=np.float64),
                       np.asarray(eng.get_winners(AREA), dtype=np.int64)))
        trace.append(ep)
    bias_engine = np.asarray(eng._areas[AREA]._cumulative_bias,
                             dtype=np.float64)
    assert float(np.abs(bias_engine).max()) > 0, "engine never charged"

    pair = _seeding.fnv1a_pair_seed(seed, AREA, AREA)
    area = HashedArea(n, k, [_to_i32(pair)], refracted_strength=beta)
    fiber = AreaFiber([_to_i32(pair)], n, n, p, beta=beta, w_max=w_max,
                      norm_init=True, synaptic_scaling=False,
                      max_rounds=M_eps * T)
    worst = 0.0
    for a, ep in enumerate(trace):
        sf = StimulusFiber([0], k, n, p, beta=beta, w_max=w_max,
                           norm_init=True, max_rounds=T)
        sf.base = torch.from_numpy(
            stim0[a].astype(np.float32)).cuda().view(1, -1)
        sf.dj = (sf.base + p * (n - k)).clamp_min(1.0)
        fiber.begin_episode()
        for prev, d_cpu, new in ep:
            raw = torch.zeros(1, n, dtype=torch.float32, device="cuda")
            pt = torch.from_numpy(prev).cuda().view(1, -1)
            fiber.contribute(raw, pt)
            sf.contribute(raw)
            net = area.apply_bias(raw)
            got = net[0].cpu().numpy().astype(np.float64)
            m = min(len(d_cpu), len(got))
            worst = max(worst, float(np.abs(d_cpu[:m] - got[:m]).max())
                        / max(float(np.abs(d_cpu[:m]).max()), 1e-12))
            nt = torch.from_numpy(new).cuda().view(1, -1)
            fiber.observe(pt, nt)
            sf.observe(pt, nt)
            area.charge(raw, nt)
        fiber.end_episode()
    got_bias = area.bias[0].cpu().numpy().astype(np.float64)
    m = min(len(bias_engine), len(got_bias))
    bias_err = float(np.abs(bias_engine[:m] - got_bias[:m]).max()) / max(
        float(np.abs(bias_engine[:m]).max()), 1e-12)
    assert worst < 5e-6, (
        f"refracted hashed path diverges from numpy_sparse on the NET drive: "
        f"relative error {worst:.3g}")
    assert bias_err < 5e-6, (
        f"accumulated bias diverges from the engine's: {bias_err:.3g}")
