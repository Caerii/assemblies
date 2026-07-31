"""The CSR drive mirror must be a pure speedup: same brain, less time.

WHAT THIS GUARDS.  ``_csr_row_sum`` mirrors a dense area->area block as CSR and
uses it for the ``w[winners].sum(axis=0)`` read in ``project_into``. A
materialised block is ~``p`` occupied (measured 4.99% at ``p=0.05``), so the
dense gather spends most of its memory bandwidth on zeros; CSR is 3-11x faster
on that line and 4x end-to-end at ``n=4000``.

The whole thing is only safe because of an invariant that is easy to break
later: **the mirror is populated and read only while plasticity is off, and
every write path drops it.** A missed invalidation would compute drive from
stale weights -- silently, with plausible numbers, which is the failure mode
this engine surfaces worst. So the tests below assert on EXACT WINNER SETS
rather than on any downstream summary, and one of them deliberately interleaves
training with frozen reads to exercise the invalidation.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine import _sparse as SP

pytestmark = pytest.mark.skipif(SP.sp is None, reason="scipy not installed")

N, K, P, BETA = 1200, 120, 0.05, 1.0
SETTLE = 6


def _no_csr(*_a, **_k):
    """Force the dense path, whatever the block looks like."""
    return None


def _build(seed):
    b = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    b.add_stimulus("s", K)
    b.add_area("C", N, K, BETA)
    for _ in range(6):
        b.project({"s": ["C"]}, {})
    b._engine.materialize_area("C")
    return b


def _settle_from(b, seed, rounds=SETTLE):
    """Seed a uniform k-subset of all n, settle frozen, return the winners."""
    rng = np.random.default_rng(seed)
    init = rng.choice(N, size=K, replace=False).astype(np.uint32)
    b.areas["C"]._winners = init
    b._engine.set_winners("C", init)
    with b.frozen():
        for _ in range(rounds):
            b.project({}, {"C": ["C"]})
    return sorted(int(x) for x in b.areas["C"].winners)


def _run(monkeypatch, use_csr, seed=3, flips=4):
    if not use_csr:
        monkeypatch.setattr(SP.NumpySparseEngine, "_csr_row_sum", _no_csr)
    b = _build(seed)
    return [_settle_from(b, 700 + i) for i in range(flips)]


def test_csr_and_dense_settle_to_the_same_winners(monkeypatch):
    """THE load-bearing assertion: identical winner sets, not just verdicts.

    A coin flip collapses 120 winners to one bit, so comparing the 0/1 would
    pass even if the mirror were substantially wrong. Compare the assemblies.
    """
    with monkeypatch.context() as m:
        dense = _run(m, use_csr=False)
    csr = _run(monkeypatch, use_csr=True)
    assert dense == csr, (
        "the CSR mirror settled to different winners than the dense path -- "
        "it is not a pure speedup")


def test_mirror_is_actually_used(monkeypatch):
    """A speedup test that silently stops exercising the fast path is not one.

    If the thresholds or the plasticity guard later exclude this case, the
    equality test above would still pass -- against the dense path, twice.
    """
    b = _build(seed=3)
    assert not b._engine._csr_drive, "mirror populated before any frozen read"
    _settle_from(b, 700)
    assert b._engine._csr_drive, (
        f"no CSR mirror was built for an n={N} materialised block; "
        f"_csr_row_sum declined and the fast path is dead")


def test_training_invalidates_the_mirror(monkeypatch):
    """Interleave frozen reads with LEARNING; the mirror must not persist.

    This is the regression that matters. A mirror surviving a plasticity round
    would keep answering from pre-training weights, and every downstream number
    would still look entirely reasonable.
    """
    def sequence(b):
        out = [_settle_from(b, 700)]
        for _ in range(3):                      # writes weights
            b.project({"s": ["C"]}, {"C": ["C"]})
        out.append(_settle_from(b, 701))
        for _ in range(3):
            b.project({"s": ["C"]}, {"C": ["C"]})
        out.append(_settle_from(b, 702))
        return out

    with monkeypatch.context() as m:
        m.setattr(SP.NumpySparseEngine, "_csr_row_sum", _no_csr)
        dense = sequence(_build(seed=5))
    csr = sequence(_build(seed=5))
    assert dense == csr, (
        "results diverged once training was interleaved with frozen reads -- "
        "a CSR mirror outlived the weights it mirrors")


def test_plasticity_round_clears_the_cache():
    """The guarantee stated directly, so its failure is legible."""
    b = _build(seed=7)
    _settle_from(b, 700)
    assert b._engine._csr_drive
    b.project({"s": ["C"]}, {"C": ["C"]})       # plasticity ON
    assert not b._engine._csr_drive, (
        "a learning round left CSR mirrors in place")


def test_explicit_invalidation_is_targetable():
    b = _build(seed=7)
    _settle_from(b, 700)
    assert b._engine._csr_drive
    b._engine.invalidate_csr_drive(src="nonexistent")
    assert b._engine._csr_drive, "cleared an unrelated key"
    b._engine.invalidate_csr_drive(src="C", tgt="C")
    assert not b._engine._csr_drive


def test_dense_blocks_below_threshold_are_left_alone():
    """Small blocks must not pay for a mirror they cannot amortise."""
    b = Brain(p=P, save_winners=True, seed=1, engine="numpy_sparse")
    b.add_stimulus("s", 20)
    b.add_area("A", 200, 20, BETA)
    for _ in range(4):
        b.project({"s": ["A"]}, {})
    b._engine.materialize_area("A")
    with b.frozen():
        b.project({}, {"A": ["A"]})
    assert not b._engine._csr_drive, (
        f"built a CSR mirror for a 200x200 block, below the "
        f"{SP._CSR_MIN_CELLS:,}-cell floor")
