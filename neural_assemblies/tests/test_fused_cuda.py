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
