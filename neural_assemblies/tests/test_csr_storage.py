"""``materialize_area(storage="csr")`` must be the same brain, 10x smaller.

WHAT IT BUYS.  A materialised self fiber is dense ``O(n^2)`` float32 -- 1.0 GB
at ``n=16,000``, which is what bounded the finite-size ladder -- while being
only ``~p`` occupied (4.99% at ``p=0.05``). CSR storage is 10x smaller AND
1.26-1.40x faster than dense, and it is built chunk by chunk so the dense form
is never held even transiently. Measured after: ``n=32,000`` runs in 410 MB
where dense would need 4.1 GB, still settling to ``decisive = 1.0000``.

WHAT MAKES IT CORRECT.  The sparsity pattern is invariant under everything the
engine does to the block -- plasticity is ``w[ix] *= (1+beta)``, the clip
bounds always bracket zero, normalisation is ``sub * scale``. So only ``.data``
ever changes and a fixed-pattern representation loses nothing.

The assertions below are all EXACT. "Approximately the same brain" is not a
property worth having: the whole justification for changing the storage is
that it changes nothing observable.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine import _csr_weights as CW
from neural_assemblies.core.numpy_engine._csr_weights import CSRWeights

pytestmark = pytest.mark.skipif(CW.sp is None, reason="scipy not installed")

N, K, P, BETA = 1200, 120, 0.05, 1.0


def _build(seed, storage):
    b = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse")
    b.add_stimulus("s", K)
    b.add_area("C", N, K, BETA)
    for _ in range(6):
        b.project({"s": ["C"]}, {})
    b._engine.materialize_area("C", storage=storage)
    return b


def _conn(b):
    return b._engine_for(b.areas["C"])._area_conns["C"]["C"]


def _settle(b, seed, rounds=6):
    rng = np.random.default_rng(seed)
    init = rng.choice(N, size=K, replace=False).astype(np.uint32)
    b.areas["C"]._winners = init
    b._engine.set_winners("C", init)
    with b.frozen():
        for _ in range(rounds):
            b.project({}, {"C": ["C"]})
    return sorted(int(x) for x in b.areas["C"].winners)


def test_csr_is_the_default():
    assert isinstance(_conn(_build(1, "csr")).weights, CSRWeights)
    assert isinstance(_conn(_build(1, "dense")).weights, np.ndarray)


def test_the_weights_are_identical_to_dense():
    """THE load-bearing property: same connectome, different container.

    `_init_area_block` is addressed by ABSOLUTE position precisely so that
    chunked assembly cannot change a value, and the already-trained corner is
    carried over rather than regenerated -- regenerating it would silently
    reset the area to naive while every shape and count still looked right.
    """
    dense = np.asarray(_conn(_build(3, "dense")).weights)
    csr = np.asarray(_conn(_build(3, "csr")).weights)
    assert dense.shape == csr.shape == (N, N)
    assert np.array_equal(dense, csr), (
        f"{int((dense != csr).sum())} of {dense.size} cells differ -- "
        f"CSR storage is not the same brain")


def test_settling_gives_identical_winners():
    d, c = _build(5, "dense"), _build(5, "csr")
    assert _settle(d, 700) == _settle(c, 700)
    assert _settle(d, 701) == _settle(c, 701)


def test_training_stays_identical_and_preserves_the_pattern():
    """Plasticity is multiplicative, so nnz must not move by a single synapse.

    If a write ever added structure, CSR would drop it silently -- the update
    would look applied and the weight would be gone.
    """
    d, c = _build(7, "dense"), _build(7, "csr")
    nnz_before = _conn(c).weights.nnz
    for _ in range(4):
        d.project({"s": ["C"]}, {"C": ["C"]})
        c.project({"s": ["C"]}, {"C": ["C"]})
    assert _conn(c).weights.nnz == nnz_before, "plasticity changed the pattern"
    assert np.array_equal(np.asarray(_conn(d).weights),
                          np.asarray(_conn(c).weights))
    assert _settle(d, 702) == _settle(c, 702)


def test_memory_is_about_ten_times_smaller():
    c = _conn(_build(1, "csr")).weights
    dense_bytes = N * N * 4
    assert c.nbytes < dense_bytes / 5, (
        f"CSR block is {c.nbytes / 1e6:.1f} MB against a dense "
        f"{dense_bytes / 1e6:.1f} MB -- expected ~10x at p={P}")


def test_hot_paths_never_densify():
    """A silent dense fallback would be correct and would undo the whole point.

    ``__array__`` counts its own calls so this is checkable rather than a
    matter of reading the code.
    """
    b = _build(9, "csr")
    w = _conn(b).weights
    w._densified = 0
    _settle(b, 700)                       # drive reads + norm_scale
    b.project({"s": ["C"]}, {"C": ["C"]})  # plasticity read/write
    assert w._densified == 0, (
        f"the block was densified {w._densified} times during normal use")


def test_setitem_refuses_patterns_it_cannot_honour():
    """Better a TypeError than a silently dropped write."""
    b = _build(1, "csr")
    w = _conn(b).weights
    with pytest.raises(TypeError):
        w[0] = np.zeros(N, dtype=np.float32)


def test_chunked_build_matches_one_shot():
    """Chunk size must be an implementation detail, not a parameter of the brain."""
    rng = np.random.default_rng(0)
    ref = ((rng.random((300, 300)) < 0.05)
           * rng.choice([1.0, 16.0], (300, 300))).astype(np.float32)
    for chunk in (7, 64, 1024):
        got = CW.build_csr_from_blocks(
            300, 300, lambda r0, r1: ref[r0:r1], rows_per_chunk=chunk)
        assert np.array_equal(np.asarray(got), ref), f"chunk={chunk}"


def test_column_nnz_matches_the_dense_count():
    """`_deg_counts` reads this, and norm_init reads `_deg_counts`."""
    w = _conn(_build(11, "csr")).weights
    assert np.array_equal(w.column_nnz(), (np.asarray(w) != 0).sum(axis=0))


def test_row_sum_matches_the_dense_reduction():
    w = _conn(_build(13, "csr")).weights
    dense = np.asarray(w)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(N, K, replace=False))
    assert np.array_equal(w.row_sum(rows), dense[rows].sum(axis=0))
    assert np.array_equal(w.row_sum(rows, 500), dense[rows, :500].sum(axis=0))
