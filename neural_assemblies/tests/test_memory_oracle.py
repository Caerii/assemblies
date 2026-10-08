"""The sequence memory against an INDEPENDENT oracle.

Every sequence study from Amendment 26 on rests on paths added to
AssemblyMemory during that program -- ``store_sequence``, the recovering
bias (``bias_decay``), ``bias_reset``, forward and reverse counts -- and on
the one-round masked ``recall`` that replays them. The other tests of these
paths are RELATIONS (an option that never fires leaves the write bit for
bit; decay 1 is the cumulative bias); none says the write computes what the
registrations say it computes. A torch hash once shipped without its
finalizer and every density test stayed green.

So this file writes the memory again, from its stated semantics, in float64
numpy with dense matrices and none of the engine's kernels, chain tables or
count storage:

    raw_j    = sum_{i in winners} P_ij min((1 + beta)^C_ij, w_max) / deg_j
             + min(base_j (1 + beta)^pot_j, w_max max(1, size p)) / d_j
    net      = raw - bias;  winners = top k of net (lowest index on a tie)
    write    C_ij += 1 for i in the previous winners, j in the new (P_ij);
             pot_j += 1 at the new winners
    bias     *= decay at the start of every writing round;
             bias_j += s raw_j at the new winners
    recall   one frozen round of the recurrent term alone (bias masked)

The connectome alone is shared: the hash, by `_hash.hash_bernoulli_2d` on the
CPU, which test_fused_cuda pins to the kernels. The oracle runs in LOCKSTEP
with the engine: at every round it predicts the engine's winners from the
engine's previous ones, and its state follows the engine's winners, so one
near-tie cannot fork the comparison. A prediction that differs from the
engine's choice only among neurons whose net drive lies within float32
rounding of the k-th value is a NEAR TIE (counted); any other difference is a
failure. At the end the counts must be equal and the bias close.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from neural_assemblies.tests import _devices

torch = _devices.import_torch()
pytestmark = pytest.mark.requires_cuda

from neural_assemblies.core.numpy_engine import _seeding               # noqa: E402

W_MAX = 20.0
TIE = 1e-5          # relative: float32 rounding of a sum of ~k terms


@pytest.fixture(scope="module")
def mod():
    return _devices.fused_kernels()


def _i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _theta(n, k, p):
    return math.sqrt((1 - p) * math.log(n) / (p * k))


def _hash(rows, n, seed, p):
    from neural_assemblies.core.torch_engine import _hash as t_hash
    return t_hash.hash_bernoulli_2d(0, rows, 0, n, seed, p, device="cpu").float().numpy() > 0


class Oracle:
    """One brain's memory, from the semantics above."""

    def __init__(self, seed, n, k, p, beta, strength, decay):
        self.n, self.k, self.p, self.beta = n, k, p, beta
        self.P = _hash(n, n, seed, p)                       # [pre, post]
        self.deg = np.maximum(self.P.sum(axis=0), 1).astype(np.float64)
        self.C = np.zeros((n, n), dtype=np.int64)
        self.bias = np.zeros(n)
        self.s = strength * beta
        self.decay = decay
        self.near_ties = 0
        self.steps = 0

    def weight(self, c):
        return np.minimum((1.0 + self.beta) ** c, W_MAX)

    def recurrent(self, rows):
        if len(rows) == 0:
            return np.zeros(self.n)
        rows = np.asarray(rows)
        return (self.P[rows] * self.weight(self.C[rows])).sum(axis=0) / self.deg

    def check(self, net, engine):
        """Assert the engine's winners are the top k of ``net`` up to ties."""
        self.steps += 1
        order = np.lexsort((np.arange(self.n), -net))        # descending, lowest index first
        mine = set(order[:self.k].tolist())
        theirs = set(int(j) for j in engine)
        if mine == theirs:
            return
        kth = net[order[self.k - 1]]
        off = mine ^ theirs
        gap = max(abs(net[j] - kth) for j in off)
        assert gap <= TIE * max(1.0, abs(kth)), (
            f"step {self.steps}: winners differ by {len(off) // 2} neurons, "
            f"drive gap {gap:.3g} at the bar {kth:.6g}")
        self.near_ties += 1

    def element(self, stim_seed, size, rounds, prev, engine_rounds):
        """One element's rounds; ``engine_rounds`` [rounds, k] the engine's
        winners. Returns the last round's (engine) winners."""
        S = _hash(size, self.n, stim_seed, self.p)
        base = S.sum(axis=0).astype(np.float64)
        dj = np.maximum(base + self.p * (self.n - size), 1.0)
        hi = W_MAX * max(1.0, size * self.p)
        pot = np.zeros(self.n)
        for new in engine_rounds:
            if self.decay is not None:
                self.bias *= self.decay
            raw = (self.recurrent(prev)
                   + np.minimum(base * (1.0 + self.beta) ** pot, hi) / dj)
            self.check(raw - self.bias, new)
            new = np.asarray(new)
            if len(prev):
                self.C[np.ix_(prev, new)] += self.P[np.ix_(prev, new)]
            pot[new] += 1
            self.bias[new] += self.s * raw[new]
            prev = new
        return prev

    def counts(self, a, b, times):
        for _ in range(times):
            self.C[np.ix_(a, b)] += self.P[np.ix_(a, b)]

    def recall(self, cue, engine):
        self.check(self.recurrent(cue), engine)


def _run(n, k, p, *, L, strength=0.5, decay=None, rounds=1, bias_reset=None,
         forward=0, reverse=0, brains=2, size=None):
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    beta = round(_theta(n, k, p), 5)
    seeds = [_i32(_seeding.fnv1a_pair_seed(700 + b, "A", "A")) for b in range(brains)]
    elements = [[_i32(_seeding.fnv1a_pair_seed(700 + b, f"o{e}", "A")) for b in range(brains)]
                for e in range(L)]
    mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=W_MAX, norm_init=True, rounds=1,
                         strength=strength, max_items=4, bias_decay=decay)
    states, every = mem.store_sequence(elements, rounds_per_element=rounds, stim_size=size,
                                       bias_reset=bias_reset, forward_counts=forward,
                                       reverse_counts=reverse)
    mem.check()
    every = every.cpu().numpy()                            # [L * rounds, B, k]
    st = states.cpu().numpy()
    oracles = []
    for b, seed in enumerate(seeds):
        o = Oracle(seed, n, k, p, beta, strength, decay)
        prev = np.zeros(0, dtype=np.int64)
        for e in range(L):
            if bias_reset and e and e % bias_reset == 0:
                o.bias[:] = 0.0
            prev = o.element(elements[e][b], k if size is None else size, rounds, prev,
                             every[e * rounds:(e + 1) * rounds, b])
        for e in range(1, L):
            o.counts(st[e - 1, b], st[e, b], forward)
        for e in range(1, L):
            o.counts(st[e, b], st[e - 1, b], reverse)
        assert np.array_equal(o.C, mem.fiber.C[b].cpu().numpy().astype(np.int64)), \
            "the engine's counts are not the oracle's"
        if mem.bias is None:                               # the Hebbian control has none
            assert not o.bias.any()
        else:
            np.testing.assert_allclose(mem.bias[b].cpu().numpy(), o.bias, rtol=1e-4, atol=1e-6)
        oracles.append(o)
    # one-round masked recall from every element's half cue
    cues = states[:, :, :k // 2]
    for e in range(L):
        got = mem.recall(cues[e], rounds=1).cpu().numpy()
        for b, o in enumerate(oracles):
            o.recall(st[e, b, :k // 2], got[b])
    print(f"\n({n}, {k}, {p}) L={L}: near ties per brain "
          f"{[(o.near_ties, o.steps) for o in oracles]}")
    return oracles


def test_the_oracle_shares_the_engines_connectome_and_indegree(mod):
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    from research.experiments import memory_pattern_efficiency as pe
    n, k, p = 600, 30, 0.5
    seeds = [_i32(_seeding.fnv1a_pair_seed(700 + b, "A", "A")) for b in range(2)]
    mem = AssemblyMemory(seeds, n, k, p, beta=0.3, w_max=W_MAX, norm_init=True, rounds=1,
                         strength=0.5, max_items=4)
    for b, seed in enumerate(seeds):
        o = Oracle(seed, n, k, p, 0.3, 0.5, None)
        assert np.array_equal(pe.presence_of(mem.fiber.pres, b, n).cpu().numpy(), o.P)
        assert np.array_equal(mem.fiber.dj[b].cpu().numpy().astype(np.float64), o.deg)


def test_recall_returns_its_winners_strongest_first(mod):
    """The k-WTA's winners come back ordered by drive, strongest first, so
    ``winners[:, :m]`` is the m MOST driven -- not an arbitrary subset.
    Amendment 34's noise replaced those slots (Amendment 36 erratum)."""
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    n, k, p = 600, 30, 0.5
    seeds = [_i32(_seeding.fnv1a_pair_seed(700 + b, "A", "A")) for b in range(2)]
    mem = AssemblyMemory(seeds, n, k, p, beta=round(_theta(n, k, p), 5), w_max=W_MAX,
                         norm_init=True, rounds=1, strength=0.5, max_items=4)
    elements = [[_i32(_seeding.fnv1a_pair_seed(700 + b, f"o{e}", "A")) for b in range(2)]
                for e in range(12)]
    states, _ = mem.store_sequence(elements)
    cue = states[3][:, :k // 2]
    got = mem.recall(cue, rounds=1)
    raw = torch.zeros(2, n, device="cuda")
    mem.fiber.contribute(raw, cue)
    drive = torch.gather(raw, 1, got)
    assert bool((drive[:, 1:] <= drive[:, :-1]).all())


@pytest.mark.parametrize("n_pre,n_post", [(400, 800), (800, 400), (600, 600)])
def test_a_cross_area_fiber_equals_the_oracle(mod, n_pre, n_post):
    """A fiber between areas of different sizes (Amendment 33's C -> S):
    its norm_init divisor is the in-degree over the SOURCE's rows, and its
    write and drive are the oracle's. Until 2026-10-07 the divisor counted
    n_post rows (Amendment 35); a square fiber was exact."""
    from neural_assemblies.core.torch_engine._hashed import DenseOrganFiber
    from research.experiments import memory_pattern_efficiency as pe
    p, beta, k = 0.5, 0.3, 30
    seeds = [_i32(_seeding.fnv1a_pair_seed(700 + b, "C->S", "A")) for b in range(2)]
    f = DenseOrganFiber(seeds, n_pre, n_post, p, beta=beta, w_max=W_MAX, norm_init=True,
                        device="cuda")
    g = np.random.default_rng(5)
    pairs = [(g.choice(n_pre, k, replace=False), g.choice(n_post, k, replace=False))
             for _ in range(12)]
    for _ in range(3):                                     # repeats reach the clip
        for a, b in pairs:
            rows = torch.as_tensor(np.stack([a, a]), device="cuda")
            f.observe(rows, torch.as_tensor(np.stack([b, b]), device="cuda"))
    f.check()
    for s, seed in enumerate(seeds):
        P = _hash(n_pre, n_post, seed, p)
        deg = np.maximum(P.sum(axis=0), 1).astype(np.float64)
        assert np.array_equal(pe.presence_of(f.pres, s, n_post).cpu().numpy(), P)
        assert np.array_equal(f.dj[s].cpu().numpy().astype(np.float64), deg)
        C = np.zeros((n_pre, n_post), dtype=np.int64)
        for _ in range(3):
            for a, b in pairs:
                C[np.ix_(a, b)] += P[np.ix_(a, b)]
        assert np.array_equal(C, f.C[s].cpu().numpy().astype(np.int64))
        for a, _ in pairs:
            raw = torch.zeros(2, n_post, device="cuda")
            f.contribute(raw, torch.as_tensor(np.stack([a, a]), device="cuda"))
            want = (P[a] * np.minimum((1.0 + beta) ** C[a], W_MAX)).sum(axis=0) / deg
            np.testing.assert_allclose(raw[s].cpu().numpy(), want, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("case", [
    dict(),                                       # the cumulative bias (A26-A28)
    dict(decay=math.exp(-1 / 33)),                # the recovering bias (A29-A34)
    dict(decay=math.exp(-1 / 8), rounds=2),       # decay per ROUND, two rounds per element
    dict(bias_reset=16),                          # a reset bias (A28)
    dict(strength=0.0),                           # the Hebbian control (A27)
    dict(forward=1, reverse=2),                   # a two-way chain (A31, A32)
    dict(size=45, decay=math.exp(-1 / 33)),       # a stimulus larger than k
], ids=["cumulative", "recovering", "two-rounds", "reset", "hebbian", "two-way", "stim-45"])
def test_store_sequence_and_recall_equal_the_oracle(mod, case):
    oracles = _run(600, 30, 0.5, L=48, **case)
    for o in oracles:
        assert o.steps > 0
        # near ties are rare, not the rule
        assert o.near_ties <= 0.05 * o.steps


@pytest.mark.slow
@pytest.mark.parametrize("decay", [None, math.exp(-1 / 33)], ids=["cumulative", "recovering"])
def test_a_study_cell_past_the_tiling_wrap_equals_the_oracle(mod, decay):
    """(4000, 60, 0.5), the cell of Amendments 28-34, for 200 elements --
    three wraps of n/k -- where the clip binds and neurons are reused."""
    oracles = _run(4000, 60, 0.5, L=200, decay=decay)
    for o in oracles:
        assert o.near_ties <= 0.05 * o.steps
        assert int(o.C.max()) >= 2
