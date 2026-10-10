"""The memory studies' fast paths equal the loops they replace, bit for bit
(research/experiments/memory_fast.py): cue permutations from a vectorised Philox, the token-index
read-out, the batched replay check, graphed sleep and calibration, and the graphed sequence writer
(plain and comparator). A fast path is only a speedup if every reading, every count and every
state it leaves is the slow path's."""
from __future__ import annotations

import pytest

from neural_assemblies.tests import _devices

torch = _devices.import_torch()
pytestmark = pytest.mark.requires_cuda

DEV = "cuda"
N, K, P, TAU = 2000, 60, 0.5, 17
SEEDS = [900, 901, 902, 903]


@pytest.fixture(scope="module")
def mod():
    return _devices.fused_kernels()


def _spec():
    from research.experiments import memory_reuse_grammar as rg
    from research.experiments import memory_threshold_law as tl
    return {"n": N, "k": K, "p": P, "tau": TAU, "rho": rg.RHO, "beta": round(tl.theta(N, K, P), 5)}


def test_philox_reproduces_torch_rand(mod):
    from research.experiments import memory_fast as mf
    seeds = [7, 2**33 + 5, 980 * 1_000_003 + 3472 * 100_000 + 216, 2**62 + 11]
    for k in (2, 60, 75, 200):
        mine = mf.philox_uniform(seeds, k, DEV)
        ref = torch.stack([torch.rand(k, device=DEV, generator=torch.Generator(device=DEV).manual_seed(s))
                           for s in seeds])
        assert torch.equal(mine, ref), k
    assert mf.philox_matches_torch(DEV)


def test_cue_index_is_the_cue(mod):
    from research.experiments import memory_fast as mf
    from research.experiments import memory_load_drift as md
    state = torch.randperm(N, device=DEV)[:K]
    gseeds = [sd * 1_000_003 + L for sd in (900, 901) for L in range(0, 4_000_000, 97_301)]
    keep = mf.cue_index(gseeds, K, DEV)
    for i, s in enumerate(gseeds):
        ref = md.cue(state, s // 1_000_003, s % 1_000_003, K, DEV)
        assert torch.equal(state[keep[i]], ref)


def test_token_index_overlaps_equal_the_scan(mod):
    from research.experiments import memory_fast as mf
    gen = torch.Generator(device=DEV).manual_seed(3)
    B, L = 3, 500
    toks = torch.stack([torch.stack([torch.randperm(N, device=DEV, generator=gen)[:K] for _ in range(B)])
                        for _ in range(L)])                              # [L, B, K]
    idx = mf.TokenIndex.build(toks[:300], N, DEV)
    idx.extend(toks[300:])                                               # appended in two calls
    states = torch.stack([toks[t, b] if t % 2 else torch.randperm(N, device=DEV, generator=gen)[:K]
                          for t in range(40) for b in range(B)])
    brains = torch.arange(B, device=DEV).repeat(40)
    hot = torch.zeros(states.shape[0], N, device=DEV)
    hot.scatter_(1, states, 1.0)
    ref = torch.gather(hot.unsqueeze(1).expand(-1, L, N), 2,
                       toks.permute(1, 0, 2)[brains]).sum(2)              # [V, L]
    assert torch.equal(idx.overlaps(states, brains), ref)


@pytest.mark.parametrize("uses", [10, 40])
def test_batched_reliability_is_the_loop(mod, uses):
    from research.experiments import memory_fast as mf
    from research.experiments import memory_sleep as sl
    st = sl.build_store(_spec(), uses, SEEDS, DEV)
    ref = sl.reliability(st, DEV)
    bias, win = st["mem"].area.bias.clone(), st["mem"].area.winners.clone()
    st["mem"].area.bias = torch.ones_like(bias)
    assert mf.reliability(st, DEV) == ref
    assert torch.equal(st["mem"].area.bias, bias) and torch.equal(st["mem"].area.winners, win)


@pytest.mark.parametrize("graph", [False, True])
def test_sleep_is_the_dream_loop(mod, graph):
    from research.experiments import memory_fast as mf
    from research.experiments import memory_sleep as sl
    st = sl.build_store(_spec(), 40, SEEDS, DEV)
    C0 = st["mem"].fiber.C.clone()
    for gate in (1.2, torch.tensor([1.1, 1.2, 1.3, 9.9], device=DEV)):
        st["mem"].fiber.C.copy_(C0)
        g = torch.Generator(device=DEV).manual_seed(sl.NOISE_SEED)
        tot = [0, 0, 0]
        for _ in range(25):
            tot = [a + b for a, b in zip(tot, sl.dream(st["mem"], g, gate, DEV))]
        C_ref, w_ref = st["mem"].fiber.C.clone(), st["mem"].area.winners.clone()
        st["mem"].fiber.C.copy_(C0)
        g = torch.Generator(device=DEV).manual_seed(sl.NOISE_SEED)
        got = mf.sleep(st["mem"], g, gate, 25, DEV, graph=graph)
        assert list(got) == tot and tot[0] > 0
        assert torch.equal(st["mem"].fiber.C, C_ref) and torch.equal(st["mem"].area.winners, w_ref)


@pytest.mark.parametrize("graph", [False, True])
def test_calibration_is_the_dream_loop(mod, graph):
    from research.experiments import memory_fast as mf
    from research.experiments import memory_sleep as sl
    st = sl.build_store(_spec(), 10, SEEDS, DEV)
    g = torch.Generator(device=DEV).manual_seed(sl.CAL_SEED)
    cal = []
    for _ in range(30):
        sl.dream(st["mem"], g, None, DEV, cal)
    g = torch.Generator(device=DEV).manual_seed(sl.CAL_SEED)
    assert torch.equal(mf.calibrate(st["mem"], g, 30, DEV, graph=graph).cpu(), torch.cat(cal))


@pytest.mark.parametrize("compare", [False, True])
def test_graphed_writer_is_the_store_loop(mod, compare):
    from research.experiments import memory_comparator as mc
    from research.experiments import memory_fast as mf
    from research.experiments import memory_load_law as ml
    from research.experiments import memory_reuse_grammar as rg
    from research.experiments import memory_write_separation as ws
    from research.experiments.seq_capacity_scaling import to_i32
    from neural_assemblies.core.numpy_engine import _seeding
    LEN, U = rg.LENGTH, 40
    M = max(1, round(rg.RHO * ml.unit(N, K, P) / LEN))
    V = max(8, round(M * LEN / U))
    words = [rg.walks(sd, M, V, None, 4242) for sd in SEEDS]
    ES = [[[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(SEEDS)]
           for e in range(LEN)] for q in range(M)]
    a = ws.build(N, K, P, TAU, SEEDS, DEV)
    stats = {"judged": 0, "flag": 0, "oracle": 0, "hit": 0}
    sa, stored_a = [], []
    for q in range(M):
        sa.append(mc.store(a, ES[q], stored_a, True, DEV, stats) if compare
                  else ws.store(a, ES[q], stored_a, False, DEV)[0])
    b = ws.build(N, K, P, TAU, SEEDS, DEV)
    w = mf.SequenceWriter(b, DEV, compare=compare)
    sb, stored_b = [], []
    for q in range(M):
        if q == M // 2:                                  # other code replaces the area's tensors
            b.area.bias, b.area.winners = b.area.bias.clone(), b.area.winners.clone()
        sb.append(w.write(ES[q], stored_b))
    w.check()
    assert torch.equal(torch.stack(sa), torch.stack(sb))
    assert all(torch.equal(x, y) for x, y in zip(stored_a, stored_b))
    for x, y in ((a.fiber.C, b.fiber.C), (a.area.bias, b.area.bias), (a.area.ever, b.area.ever),
                 (a.area.winners, b.area.winners)):
        assert torch.equal(x, y)
    assert (a.area.rounds_seen, a.items) == (b.area.rounds_seen, b.items)
    if compare:
        assert stats["flag"] == int(w.flags) > 0 and stats["judged"] == w.judged


def test_vram_guard_refuses_a_full_card(mod):
    from research.experiments import memory_fast as mf
    with pytest.raises(RuntimeError, match="free"):
        mf.vram_guard(10_000.0)
    assert mf.vram_guard(0.01) > 0
