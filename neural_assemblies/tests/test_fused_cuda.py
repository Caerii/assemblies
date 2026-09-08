"""The fused kernels must agree with the ENGINE, not with a transcription.

``_fused_cuda`` regenerates the connectome from the same hash the torch engine
stores, and selects winners without sorting. Both are only useful if they are
the same computation the engine already does, so every assertion here is
pinned against ``_hash``/``_seeding`` rather than against a remembered
constant: a kernel that agrees with a plausible re-derivation of the hash and
disagrees with the engine is a fast wrong answer.

The one place the kernels DO differ is tie order, and that is asserted
explicitly rather than left to be discovered -- see
``test_selection_tie_break_is_canonical_unlike_torch_topk``.

Skipped without CUDA, and skipped without a working nvcc/host-compiler/ninja
toolchain, since the kernels are an optional capability.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

from neural_assemblies.core.torch_engine import _fused_cuda        # noqa: E402
from neural_assemblies.core.torch_engine import _hash as t_hash    # noqa: E402

P = 0.05
SEED = 0x5EED1234


@pytest.fixture(scope="module")
def mod():
    m = _fused_cuda.load()
    if m is None:
        pytest.skip(f"fused kernels unavailable: {_fused_cuda.last_error()}")
    return m


# -- the hash --------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(512, 20), (2048, 60)])
def test_hashed_drive_equals_the_engines_own_connectome(mod, n, k):
    """drive[b,j] == sum over the row-set of the engine's stored W[i,j]."""
    W = t_hash.hash_bernoulli_2d(0, n, 0, n, SEED, P,
                                 device='cuda').float()
    g = torch.Generator(device='cuda').manual_seed(3)
    rows = torch.randint(0, n, (4, k), device='cuda', generator=g,
                         dtype=torch.int32)
    seeds = torch.full((4,), _to_i32(SEED), dtype=torch.int32, device='cuda')
    got = mod.hashed_drive(rows, seeds, n, _fused_cuda.threshold_for(P))
    ref = torch.stack([W[rows[b].long()].sum(0) for b in range(4)])
    assert torch.equal(got, ref), (
        f"max|diff| = {(got - ref).abs().max().item()}")


def test_threshold_convention_matches_the_engine():
    """`int(p * 2**24)`, not a float division -- they differ on the boundary."""
    for p in (0.05, 0.1, 0.017, 1.0 / 3.0):
        assert _fused_cuda.threshold_for(p) == int(p * 16777216.0)


def test_per_brain_seeds_give_independent_connectomes(mod):
    """The batch dimension carries a SEED, which is what makes brains distinct."""
    n, k = 1024, 30
    rows = torch.arange(k, device='cuda', dtype=torch.int32).repeat(2, 1)
    seeds = torch.tensor([_to_i32(SEED), _to_i32(SEED + 1)],
                         dtype=torch.int32, device='cuda')
    d = mod.hashed_drive(rows, seeds, n, _fused_cuda.threshold_for(P))
    assert not torch.equal(d[0], d[1])
    W1 = t_hash.hash_bernoulli_2d(0, k, 0, n, SEED + 1, P,
                                  device='cuda').float()
    assert torch.equal(d[1], W1.sum(0))


# -- the selector ----------------------------------------------------------

def _drive_like(B, n, k, seed=1):
    g = np.random.default_rng(seed)
    base = g.binomial(k, P, size=(B, n)).astype(np.float32)
    m = g.random((B, n)) < 0.06
    bump = (1.1 ** g.integers(1, 12, size=(B, n))).astype(np.float32)
    return (base + np.where(m, bump, 0.0)).astype(np.float32)


@pytest.mark.parametrize("n,k", [(4000, 60), (20000, 70)])
def test_selection_equals_stable_argsort(mod, n, k):
    x = _drive_like(6, n, k)
    out, ovf = mod.topk_select(torch.from_numpy(x).cuda(), k)
    assert int(ovf.max()) == 0, "candidate buffer overflowed"
    got = out.cpu().numpy()
    for b in range(x.shape[0]):
        ref = np.argsort(-x[b], kind='stable')[:k]
        assert np.array_equal(np.sort(ref), np.sort(got[b]))


def test_selection_tie_break_is_canonical_unlike_torch_topk(mod):
    """Ties at the bar are the COMMON case, and the two selectors differ there.

    This is asserted, not hoped: the drive is an integer Bernoulli sum, so the
    k-th largest is routinely tied. `_kwta_prune` records that making the
    tie-break canonical is science-affecting, which is why this kernel is
    opt-in. If this test ever stops finding ties, the fixture stopped being
    representative of a real drive.
    """
    n, k = 4000, 60
    x = _drive_like(4, n, k, seed=7)
    ties = 0
    for b in range(x.shape[0]):
        bar = np.sort(x[b])[-k]
        ties += int((x[b] == bar).sum()) - 1
    assert ties > 0, "fixture has no ties at the bar; it is not drive-like"

    out, _ = mod.topk_select(torch.from_numpy(x).cuda(), k)
    got = out.cpu().numpy()
    for b in range(x.shape[0]):
        ref = np.argsort(-x[b], kind='stable')[:k]
        assert np.array_equal(np.sort(ref), np.sort(got[b]))


def test_overflow_is_reported_not_truncated(mod):
    """A wrong winner set must never be returned silently."""
    n, k = 4096, 8
    x = np.zeros((2, n), dtype=np.float32)       # every column tied
    out, ovf = mod.topk_select(torch.from_numpy(x).cuda(), k)
    assert int(ovf.max()) > 0
    assert int(ovf.max()) >= n


# -- helpers ---------------------------------------------------------------

def _to_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


# -- the batched Phase 3 path ----------------------------------------------

def _reference_rounds(Ws, winners, k, rounds):
    """Stored-connectome reference with the SAME tie policy as the kernel.

    Pinning against `batched_project_independent` directly would conflate two
    differences -- generated-vs-stored connectome, and canonical-vs-unspecified
    tie order. This isolates the first, which is the claim under test.
    """
    idx = [w.copy() for w in winners]
    for _ in range(rounds):
        nxt = []
        for b, W in enumerate(Ws):
            drive = W[idx[b]].sum(axis=0)
            nxt.append(np.sort(np.argsort(-drive, kind='stable')[:k]))
        idx = nxt
    return idx


@pytest.mark.parametrize("n,k,rounds", [(2048, 40, 1), (2048, 40, 3)])
def test_hashed_batched_matches_a_stored_connectome_reference(mod, n, k,
                                                              rounds):
    from neural_assemblies.core.torch_engine._batched import (
        batched_project_hashed)

    B = 4
    seeds = [SEED + 17 * b for b in range(B)]
    g = np.random.default_rng(11)
    w0 = np.stack([np.sort(g.choice(n, k, replace=False)) for _ in range(B)])

    Ws = [t_hash.hash_bernoulli_2d(0, n, 0, n, s, P,
                                   device='cuda').float().cpu().numpy()
          for s in seeds]
    ref = _reference_rounds(Ws, list(w0), k, rounds)

    got = batched_project_hashed(
        n, k, P,
        [_to_i32(s) for s in seeds],
        torch.from_numpy(w0).cuda(),
        rounds).cpu().numpy()

    for b in range(B):
        assert np.array_equal(np.sort(got[b]), ref[b]), (
            f"brain {b} diverges from the stored-connectome reference")


def test_brains_stay_independent_across_rounds(mod):
    """Two brains with the same start and different seeds must not converge."""
    from neural_assemblies.core.torch_engine._batched import (
        batched_project_hashed)
    n, k = 2048, 40
    w0 = torch.arange(k, device='cuda', dtype=torch.int64).repeat(2, 1)
    got = batched_project_hashed(
        n, k, P, [_to_i32(SEED), _to_i32(SEED + 1)], w0, 3).cpu().numpy()
    assert set(got[0].tolist()) != set(got[1].tolist())


# -- plasticity ------------------------------------------------------------

def _reference_rounds_beta(Ws, winners, k, rounds, beta, w_max):
    """Stored-connectome reference using the ENGINE's rule: prev x new.

    Potentiation multiplies existing cells; an ABSENT cell is 0 and stays 0
    under `*= (1+beta)`, which is exactly what the kernel's present() test
    encodes. Clipping is per-round, matching `conn.weights.clamp_`.
    """
    Ws = [W.copy() for W in Ws]
    idx = [w.copy() for w in winners]
    for _ in range(rounds):
        nxt = []
        for b, W in enumerate(Ws):
            drive = W[idx[b]].sum(axis=0)
            nxt.append(np.sort(np.argsort(-drive, kind='stable')[:k]))
        for b, W in enumerate(Ws):
            W[np.ix_(idx[b], nxt[b])] *= np.float32(1.0 + beta)
            if w_max is not None:
                np.minimum(W, np.float32(w_max), out=W)
        idx = nxt
    return idx


@pytest.mark.parametrize("rounds,beta,w_max", [
    (2, 0.10, None), (4, 0.10, None), (4, 0.50, 2.0), (6, 0.25, None),
])
def test_plasticity_matches_a_stored_connectome_reference(mod, rounds, beta,
                                                          w_max):
    from neural_assemblies.core.torch_engine._batched import (
        batched_project_hashed)

    n, k, B = 2048, 40, 4
    seeds = [SEED + 17 * b for b in range(B)]
    g = np.random.default_rng(11)
    w0 = np.stack([np.sort(g.choice(n, k, replace=False)) for _ in range(B)])

    Ws = [t_hash.hash_bernoulli_2d(0, n, 0, n, s, P,
                                   device='cuda').float().cpu().numpy()
          for s in seeds]
    ref = _reference_rounds_beta(Ws, list(w0), k, rounds, beta, w_max)

    got = batched_project_hashed(
        n, k, P, [_to_i32(s) for s in seeds], torch.from_numpy(w0).cuda(),
        rounds, beta=beta, w_max=w_max).cpu().numpy()

    for b in range(B):
        assert np.array_equal(np.sort(got[b]), ref[b]), (
            f"brain {b}: {len(set(ref[b]) - set(got[b]))} of {k} winners "
            f"differ at rounds={rounds} beta={beta} w_max={w_max}")


def test_beta_actually_changes_the_trajectory(mod):
    """A plasticity path that silently did nothing would pass every parity
    test above against a reference that also did nothing. Pin that it moves."""
    from neural_assemblies.core.torch_engine._batched import (
        batched_project_hashed)
    n, k, B, rounds = 2048, 40, 2, 5
    seeds = [_to_i32(SEED + 17 * b) for b in range(B)]
    g = np.random.default_rng(3)
    w0 = torch.from_numpy(np.stack(
        [np.sort(g.choice(n, k, replace=False)) for _ in range(B)])).cuda()
    a = batched_project_hashed(n, k, P, seeds, w0, rounds, beta=0.0)
    b_ = batched_project_hashed(n, k, P, seeds, w0, rounds, beta=0.5)
    assert not torch.equal(a.sort(dim=1).values, b_.sort(dim=1).values)


def test_learning_rounds_are_bounded_by_the_mask_width(mod):
    from neural_assemblies.core.torch_engine._batched import (
        batched_project_hashed)
    from neural_assemblies.core.torch_engine._hashed import AreaFiber
    w0 = torch.arange(8, device='cuda', dtype=torch.int64).view(1, 8)
    with pytest.raises(ValueError, match="64-bit word"):
        batched_project_hashed(512, 8, P, [1], w0,
                               AreaFiber.MAX_EPISODE_ROUNDS + 1, beta=0.1)


# -- the CSR deviation store, across EPISODES ------------------------------

def _reference_episodes(Ws, cues, k, rounds, beta, w_max, norm_init=False):
    """Multi-episode stored-connectome reference, ENGINE rule (prev x new).

    Each episode starts from its own cue -- the area is inhibited between
    assemblies -- and the connectome CARRIES OVER, which is the whole point:
    it is what makes the assemblies compete, and what puts past episodes in
    the store rather than in the live mask.
    """
    # norm_init's divisor is the BASE in-degree and is potentiation-invariant,
    # so it is taken once, before any training.
    djs = [np.maximum((W != 0).sum(axis=0), 1.0).astype(np.float64)
           if norm_init else None for W in Ws]
    Ws = [W.copy() for W in Ws]
    finals = []
    for cue in cues:
        idx = [c.copy() for c in cue]
        for _ in range(rounds):
            nxt = []
            for b, W in enumerate(Ws):
                drive = W[idx[b]].sum(axis=0)
                if djs[b] is not None:
                    drive = drive / djs[b]
                nxt.append(np.sort(np.argsort(-drive, kind='stable')[:k]))
            for b, W in enumerate(Ws):
                W[np.ix_(idx[b], nxt[b])] *= np.float32(1.0 + beta)
                if w_max is not None:
                    np.minimum(W, np.float32(w_max), out=W)
            idx = nxt
        finals.append([i.copy() for i in idx])
    return finals


@pytest.mark.parametrize("norm_init", [False, True])
@pytest.mark.parametrize("episodes,rounds,beta,w_max", [
    (2, 3, 0.10, None), (4, 3, 0.10, 20.0), (6, 2, 0.25, None),
])
def test_csr_store_matches_reference_across_episodes(mod, episodes, rounds,
                                                     beta, w_max, norm_init):
    """Exercises the CSR READ path, which a single-episode test never does.

    Within an episode the correction comes from a one-word mask; everything
    earlier comes from the store. A store that dropped, double-counted or
    mis-keyed a cell shows up here and nowhere else.
    """
    from neural_assemblies.core.torch_engine._batched import (
        batched_project_hashed)

    n, k, B = 2048, 40, 4
    seeds = [SEED + 17 * b for b in range(B)]
    g = np.random.default_rng(5)
    cues = [np.stack([np.sort(g.choice(n, k, replace=False))
                      for _ in range(B)]) for _ in range(episodes)]

    Ws = [t_hash.hash_bernoulli_2d(0, n, 0, n, s, P,
                                   device='cuda').float().cpu().numpy()
          for s in seeds]
    ref = _reference_episodes(Ws, [list(c) for c in cues], k, rounds, beta,
                              w_max, norm_init=norm_init)

    state = None
    got = []
    for cue in cues:
        out, state = batched_project_hashed(
            n, k, P, [_to_i32(s) for s in seeds],
            torch.from_numpy(cue).cuda(), rounds, beta=beta, w_max=w_max,
            norm_init=norm_init, state=state,
            max_rounds=episodes * rounds, return_state=True)
        got.append(out.cpu().numpy())

    for e in range(episodes):
        for b in range(B):
            assert np.array_equal(np.sort(got[e][b]), ref[e][b]), (
                f"episode {e}, brain {b}: diverges from the reference "
                f"(episodes={episodes} rounds={rounds} beta={beta} "
                f"norm_init={norm_init})")


def test_store_grows_and_is_read(mod):
    """A store that stayed empty would pass every parity test vacuously."""
    from neural_assemblies.core.torch_engine._batched import (
        batched_project_hashed)
    n, k, B, rounds = 2048, 40, 2, 3
    seeds = [_to_i32(SEED + 17 * b) for b in range(B)]
    g = np.random.default_rng(9)
    state, sizes = None, []
    for _ in range(4):
        cue = torch.from_numpy(np.stack(
            [np.sort(g.choice(n, k, replace=False)) for _ in range(B)])).cuda()
        _, state = batched_project_hashed(
            n, k, P, seeds, cue, rounds, beta=0.1, state=state,
            max_rounds=12, return_state=True)
        sizes.append(state["fiber"].nnz)
    assert sizes[0] > 0, "store never populated"
    assert sizes == sorted(sizes), f"store shrank across episodes: {sizes}"
    assert sizes[-1] > sizes[0], f"store stopped growing: {sizes}"


def test_one_round_episodes_equal_one_multi_round_episode(mod):
    """Three 1-round episodes of the same co-firing must read as count 3.

    The store's LSM merge sums counts across runs; the chain table must then
    price count 3. A fiber whose table was sized by the EPISODE (max_rounds=1)
    silently clamped to tab[1] and its drive stopped growing -- the aligner
    parity gate caught it. Now the overrun raises, and the sized fiber agrees
    with a single 3-round episode exactly.
    """
    from neural_assemblies.core.torch_engine._hashed import AreaFiber
    n, p, beta = 256, 0.2, 0.1
    seeds = [12345]
    prev = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    new = torch.tensor([[10, 11, 12, 13, 14]], device="cuda")

    def read(f):
        d = torch.zeros(1, n, device="cuda")
        f.contribute(d, prev)
        return d[0, new[0]].cpu()

    one = AreaFiber(seeds, n, n, p, beta=beta, w_max=20.0, max_rounds=3)
    one.begin_episode()
    for _ in range(3):
        one.observe(prev, new)
    one.end_episode()

    three = AreaFiber(seeds, n, n, p, beta=beta, w_max=20.0, max_rounds=3)
    for _ in range(3):
        three.begin_episode()
        three.observe(prev, new)
        three.end_episode()
    assert three.store.max_count == 3
    torch.testing.assert_close(read(three), read(one), rtol=0, atol=0)

    short = AreaFiber(seeds, n, n, p, beta=beta, w_max=20.0, max_rounds=1)
    short.begin_episode()
    short.observe(prev, new)
    short.end_episode()
    short.begin_episode()
    short.observe(prev, new)
    with pytest.raises(ValueError, match="chain table"):
        short.end_episode()


def test_relative_pricing_equals_absolute_where_both_are_exact(mod):
    """Max-relative pricing is the same number as the absolute chain.

    With column scaling and NO clip, a weight is base*(1+beta)^c*s_j, so a
    column is a share distribution and only count differences matter. The
    absolute form overflows float32 near c ~ 900; the relative form
    (1+beta)^(c - cmax_j) is bounded. Where both are representable they must
    agree: a fiber at w_max=None (relative) against one with a clip that can
    never bind (absolute, opted in), same writes, same drive to 1e-5.
    """
    from neural_assemblies.core.torch_engine._hashed import AreaFiber
    n, p, beta = 256, 0.2, 0.1
    seeds = [777]
    rows_a = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    rows_b = torch.tensor([[6, 7, 8, 9, 10]], device="cuda")
    cols_a = torch.tensor([[10, 11, 12, 13, 14]], device="cuda")
    cols_b = torch.tensor([[12, 13, 14, 15, 16]], device="cuda")
    rel = AreaFiber(seeds, n, n, p, beta=beta, w_max=None, norm_init=True,
                    synaptic_scaling=True, max_rounds=64)
    absf = AreaFiber(seeds, n, n, p, beta=beta, w_max=1e9, norm_init=True,
                     synaptic_scaling=True, max_rounds=64,
                     scaling_allows_clip=True)
    assert rel.relative and not absf.relative
    for f in (rel, absf):
        for _ in range(3):
            f.begin_episode()
            for _ in range(4):
                f.observe(rows_a, cols_a)
            f.observe(rows_b, cols_b)
            f.end_episode()

    def read(f, rows):
        d = torch.zeros(1, n, device="cuda")
        f.contribute(d, rows)
        return d[0].cpu()

    for rows in (rows_a, rows_b):
        a, b = read(rel, rows), read(absf, rows)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    assert int(rel.cmax.max()) == 12


def test_relative_pricing_survives_deep_counts(mod):
    """A cell co-firing 1200 times must still price finitely and keep its
    column a share distribution -- the case the absolute chain cannot hold."""
    from neural_assemblies.core.torch_engine._hashed import AreaFiber
    n, p, beta = 128, 0.3, 0.1
    f = AreaFiber([99], n, n, p, beta=beta, w_max=None, norm_init=False,
                  synaptic_scaling=True, max_rounds=2048)
    # 24 source rows at p=0.3: every written column has several synapses
    # from them, so its whole setpoint must come back from these rows.
    rows = torch.arange(24, device="cuda").view(1, -1)
    cols = torch.tensor([[7, 8, 9]], device="cuda")
    for _ in range(20):
        f.begin_episode()
        for _ in range(60):
            f.observe(rows, cols)
        f.end_episode()
    assert f.store.max_count == 1200
    d = torch.zeros(1, n, device="cuda")
    f.contribute(d, rows)
    d = d[0]
    assert torch.isfinite(d).all()
    # the written columns carry (nearly) their whole setpoint from these rows
    assert float(d[cols[0]].min()) > 0.9 * f.setpoint



def test_presence_mask_is_the_hash(mod):
    """GATE-4 (DESIGN_dense_floor.md): the bitmask's popcount drive equals
    `hashed_drive` -- the same connectome, stored instead of re-derived."""
    seeds = torch.tensor([11, 12, 13], dtype=torch.int32, device="cuda")
    n_pre, n_post, p = 300, 1000, 0.05
    thr = _fused_cuda.threshold_for(p)
    pres = mod.hashed_presence(seeds, n_pre, n_post, thr)          # [B, n_pre, W]
    assert pres.shape == (3, n_pre, (n_post + 31) // 32)
    bits = ((pres.unsqueeze(-1) >> torch.arange(32, device="cuda")) & 1)
    bits = bits.reshape(3, n_pre, -1)[:, :, :n_post].to(torch.float32)
    g = torch.Generator(device="cpu").manual_seed(0)
    rows = torch.randint(0, n_pre, (3, 40), generator=g).to("cuda")
    want = mod.hashed_drive(rows.to(torch.int32), seeds, n_post, thr)
    got = torch.stack([bits[b, rows[b]].sum(0) for b in range(3)])
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert 0.03 < float(bits.mean()) < 0.07




def test_present_lists_are_the_mask(mod):
    """GATE-4b (DESIGN_present_only.md): every row's entries are exactly its
    set bits, ascending, count 0; the rest is padding."""
    from neural_assemblies.core.torch_engine._hashed import PresentFiber
    f = PresentFiber([11, 12], 300, 1000, 0.05)
    pres = mod.hashed_presence(f.seeds, 300, 1000, f.threshold)
    bits = ((pres.unsqueeze(-1) >> torch.arange(32, device="cuda")) & 1)
    bits = bits.reshape(2, 300, -1)[:, :, :1000].bool()
    cols = f.columns()
    for b in range(2):
        for i in (0, 7, 150, 299):
            want = torch.nonzero(bits[b, i]).flatten()
            got = cols[b, i]
            got = got[got >= 0]
            assert torch.equal(got, want)
    assert int(f.counts().sum()) == 0 and f.DMAX == int(bits.sum(2).max())


def test_present_fiber_equals_store_fiber(mod):
    """GATE-3 (DESIGN_present_only.md): `PresentFiber` and `AreaFiber`
    (unclipped, scaled, relative) are the same numbers on the same writes."""
    from neural_assemblies.core.torch_engine._hashed import (
        AreaFiber, PresentFiber)
    n, p, beta = 256, 0.2, 0.1
    seeds = [4242, 4243]
    rows_a = torch.tensor([[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]], device="cuda")
    rows_b = torch.tensor([[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]], device="cuda")
    cols_a = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
                          device="cuda")
    cols_b = torch.tensor([[12, 13, 14, 15, 16], [22, 23, 24, 25, 26]],
                          device="cuda")
    store = AreaFiber(seeds, n, n, p, beta=beta, w_max=None, norm_init=True,
                      synaptic_scaling=True, max_rounds=64)
    pf = PresentFiber(seeds, n, n, p, beta=beta, norm_init=True,
                      synaptic_scaling=True, max_rounds=64)
    for f in (store, pf):
        for _ in range(3):
            f.begin_episode()
            for _ in range(4):
                f.observe(rows_a, cols_a)
            f.observe(rows_b, cols_b)
            f.end_episode()

    def read(f, rows):
        d = torch.zeros(2, n, device="cuda")
        f.contribute(d, rows)
        return d.cpu()

    for rows in (rows_a, rows_b):
        torch.testing.assert_close(read(pf, rows), read(store, rows),
                                   rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pf.cmax.cpu(), store.cmax.cpu())
    assert 0 < pf.nnz < store.nnz



def test_present_fiber_flags_count_overflow(mod):
    from neural_assemblies.core.torch_engine._hashed import PresentFiber
    f = PresentFiber([5], 64, 64, 0.5, beta=0.1, max_rounds=8)
    rows = torch.arange(8, device="cuda").view(1, 8)
    edge = (f.ent & 0xFFFF) | (f.MAX_COUNT << 16)
    f.ent = torch.where(f.ent == -1, f.ent, edge)
    f.cmax.fill_(f.MAX_COUNT)
    f.observe(rows, rows)
    with pytest.raises(OverflowError):
        f.check()


def test_present_probe_runs(mod):
    from neural_assemblies.core.torch_engine._hashed import PresentFiber
    f = PresentFiber([1, 2, 3, 4], 200, 1000, 0.05)
    S = torch.randint(0, 200, (4, 50), dtype=torch.int32, device="cuda")
    out = mod.present_probe(f.ent, S, 3, 4)
    assert out.shape == (4, 32) and torch.isfinite(out).all()


def test_present_fiber_absolute_equals_store_fiber(mod):
    """The organ's regime -- clip, no scaling, norm_init -- priced by count
    from the chain table: `PresentFiber(absolute)` equals `AreaFiber`."""
    from neural_assemblies.core.torch_engine._hashed import (
        AreaFiber, PresentFiber)
    n, p, beta, w_max = 512, 0.1, 0.1, 20.0
    seeds = [4242, 4243]
    g = torch.Generator(device="cpu").manual_seed(5)
    store = AreaFiber(seeds, n, n, p, beta=beta, w_max=w_max, norm_init=True,
                      synaptic_scaling=False, max_rounds=64)
    pf = PresentFiber(seeds, n, n, p, beta=beta, norm_init=True,
                      synaptic_scaling=False, w_max=w_max, max_rounds=64)
    assert pf.absolute

    def sets():
        return torch.stack([torch.randperm(n, generator=g)[:20] for _ in range(2)]).to("cuda")

    rows_a, cols_a = sets(), sets()
    for f in (store, pf):
        for _ in range(3):
            f.begin_episode()
            for _ in range(12):                       # past the clip at c ~ 31? no: 36 rounds
                f.observe(rows_a, cols_a)
            f.end_episode()

    def read(f, rows):
        d = torch.zeros(2, n, device="cuda")
        f.contribute(d, rows)
        return d.cpu()

    for rows in (rows_a, sets()):
        torch.testing.assert_close(read(pf, rows), read(store, rows),
                                   rtol=1e-5, atol=1e-6)
    assert int(pf.counts().max()) == 36


def test_organ_fiber_equals_store_fiber(mod):
    """DESIGN_sequence_port.md: the dense organ fiber (p = 0.2, K = 40 rows,
    clip, norm_init, no scaling) equals `AreaFiber`, and skips -1 rows and
    winners as the dead-brain convention requires."""
    from neural_assemblies.core.torch_engine._hashed import (
        AreaFiber, DenseOrganFiber)
    n, p, beta, w_max = 512, 0.2, 0.1, 20.0
    seeds = [4242, 4243]
    g = torch.Generator(device="cpu").manual_seed(9)
    store = AreaFiber(seeds, n, n, p, beta=beta, w_max=w_max, norm_init=True,
                      synaptic_scaling=False, max_rounds=64)
    org = DenseOrganFiber(seeds, n, n, p, beta=beta, w_max=w_max, norm_init=True,
                          max_rounds=64)

    def sets():
        return torch.stack([torch.randperm(n, generator=g)[:40] for _ in range(2)]).to("cuda")

    for _ in range(6):
        rows, cols = sets(), sets()
        for f in (store, org):
            f.begin_episode()
            for _ in range(5):
                f.observe(rows, cols)
            f.end_episode()

    def read(f, rows):
        d = torch.zeros(2, n, device="cuda")
        f.contribute(d, rows)
        return d.cpu()

    rows = sets()
    torch.testing.assert_close(read(org, rows), read(store, rows), rtol=1e-5, atol=1e-6)
    # -1 rows are skipped: a half-dead row set reads like the live half alone
    half = rows.clone()
    half[:, 20:] = -1
    live = rows[:, :20]
    torch.testing.assert_close(read(org, half), read(store, live), rtol=1e-5, atol=1e-6)
    # a dead brain's write is a no-op
    before = org.C.clone()
    dead_rows = torch.full((2, 40), -1, dtype=torch.int64, device="cuda")
    org.observe(dead_rows, sets())
    org.observe(sets(), dead_rows)
    assert torch.equal(org.C, before)
    assert org.nnz > 0


def test_topk_select_ranks_negative_drives(mod):
    """A refracted area's net drive is raw - bias and goes NEGATIVE. The
    selector's key must order floats, not their bit patterns: the most
    negative values must never be chosen (they were, and the organ's arc
    collapsed onto its most-biased neurons)."""
    g = torch.Generator(device="cpu").manual_seed(2)
    x = (torch.randn(3, 4000, generator=g) * 3).to("cuda")
    x[0, :50] = -60.0                                  # heavily refracted
    x[1, ::7] = -0.0
    sel, ovf = mod.topk_select(x, 100)
    assert int(ovf.max()) == 0
    want = torch.topk(x, 100, dim=1).indices
    for b in range(3):
        assert set(sel[b].tolist()) == set(want[b].tolist())
    assert not (sel[0] < 50).any()


def test_stop_when_stable_equals_ungated_prefix(mod):
    """PREREG_refraction_memory.md Amendment 5: gating the rounds per brain
    on convergence writes exactly what the ungated run writes up to each
    brain's convergence round, and nothing after -- the count matrix and the
    refraction bias after ONE item, against an ungated run of the same item
    cut at that brain's own round count (each brain re-run alone)."""
    from neural_assemblies.core.torch_engine._batched import batched_project_hashed
    from neural_assemblies.core.numpy_engine import _seeding
    n, k, p, beta, w_max = 1000, 40, 0.5, 0.1, 20.0

    def i32(v):
        v &= 0xFFFFFFFF
        return v - 0x100000000 if v >= 0x80000000 else v

    def run(brains, rounds, gate):
        sd = [i32(_seeding.fnv1a_pair_seed(42 + b, "A", "A")) for b in brains]
        ss = [i32(_seeding.fnv1a_pair_seed(42 + b, "s0", "A")) for b in brains]
        cue = torch.zeros(len(brains), 0, dtype=torch.int64, device="cuda")
        win, st = batched_project_hashed(
            n, k, p, sd, cue, rounds, beta=beta, w_max=w_max, norm_init=True,
            synaptic_scaling=False, stim_seeds=ss, stim_size=k,
            max_rounds=64, return_state=True, refracted_strength=0.5 * beta,
            stop_when_stable=gate)
        return win, st

    brains = list(range(6))
    win_g, st_g = run(brains, 8, True)
    used = st_g["area"].rounds_used.tolist()
    assert min(used) >= 2 and max(used) <= 8
    assert any(u < 8 for u in used), "no brain converged before T_max; vacuous"
    for b, u in zip(brains, used):
        win_u, st_u = run([b], u, False)
        assert torch.equal(win_u[0], win_g[b])
        assert torch.equal(st_u["fiber"].C[0], st_g["fiber"].C[b])
        assert torch.equal(st_u["area"].bias[0], st_g["area"].bias[b])
        if u < 8:
            # one more ungated round would have written something more
            _, st_x = run([b], u + 1, False)
            assert not torch.equal(st_x["fiber"].C[0], st_g["fiber"].C[b])
