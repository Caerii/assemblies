"""The Rust accelerator must be the SAME BRAIN, byte for byte.

WHY THESE ASSERTIONS ARE EXACT.  ``crates/na-kernels`` exists only to compute
what ``_seeding.py`` already computes, faster. "Statistically equivalent" would
be worthless here: a synapse's initial weight is a pure function of its
position, that property is what makes materialisation order-free, and
order-freedom is what makes probe isolation sound. An accelerator that got the
distribution right and the cells wrong would silently produce a different brain
under a flag most runs never look at.

Exactness is achievable rather than aspirational because the computation is all
``u32`` arithmetic with wrapping multiplies -- which Rust reproduces by
definition -- plus one division of an integer below 2^24 by 2^24, which is
exact in float32. The two places that could still drift are the COMPARISONS,
where numpy's NEP 50 narrowing rules decide the type, so those are tested
head-on: ``hash_uniform_2d`` compares ``m/2^24 < p`` while ``hash_stim_counts``
compares ``m < p*2^24``, and that asymmetry is deliberate.

If the extension is not built these tests skip. That is the right behaviour --
it is optional at every level -- but it also means a green run here does not
prove the accelerator was exercised, hence ``test_extension_is_actually_loaded``
reports which arm ran.
"""

import numpy as np
import pytest

from neural_assemblies.core.brain import Brain
from neural_assemblies.core.numpy_engine import _seeding as S

rust = S.rust_kernels()
# Two very different reasons to skip, and saying the wrong one sends the reader
# to build an extension that is already built.
_why = ("na-kernels disabled by NEURAL_ASSEMBLIES_NO_RUST"
        if S._rust is not None
        else "na-kernels extension not built (see docs/RUST_KERNELS.md)")
pytestmark = pytest.mark.skipif(rust is None, reason=_why)


def _numpy_arm(monkeypatch, fn, *a, **kw):
    """Run *fn* with the accelerator forced off, in-process."""
    monkeypatch.setenv("NEURAL_ASSEMBLIES_NO_RUST", "1")
    try:
        return np.asarray(fn(*a, **kw))
    finally:
        monkeypatch.setenv("NEURAL_ASSEMBLIES_NO_RUST", "0")


BLOCKS = [
    (0, 64, 0, 64),          # square, aligned
    (7, 130, 3, 91),         # offset origin -- catches absolute-vs-relative
    (0, 1, 0, 1),            # single cell
    (0, 257, 0, 3),          # very tall and thin
    (1000, 1099, 2000, 2222),  # far from the origin, in both axes
]


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.3, 1.0])
def test_area_weights_are_byte_identical(monkeypatch, block, p):
    r0, r1, c0, c1 = block
    ref = _numpy_arm(monkeypatch, S.hash_area_weights, r0, r1, c0, c1, 0xDEADBEEF, p)
    got = np.asarray(S.hash_area_weights(r0, r1, c0, c1, 0xDEADBEEF, p))
    assert ref.shape == got.shape
    assert np.array_equal(ref, got), (
        f"{int((ref != got).sum())} of {ref.size} cells differ at p={p}")


@pytest.mark.parametrize("inhib_prob,inhib_w", [(0.2, -1.0), (0.5, -2.5), (1.0, -1.0)])
def test_the_salted_inhibitory_draw_matches(monkeypatch, inhib_prob, inhib_w):
    """The second draw is salted; an unsalted port would make EVERY present
    synapse inhibitory whenever inhibitory_prob >= p, and still look plausible."""
    args = (0, 300, 0, 300, 777, 0.1, inhib_prob, inhib_w)
    ref = _numpy_arm(monkeypatch, S.hash_area_weights, *args)
    got = np.asarray(S.hash_area_weights(*args))
    assert np.array_equal(ref, got)
    assert (got < 0).any(), "no inhibitory synapse was produced -- vacuous test"


def test_unfinalized_hash_matches(monkeypatch):
    """`finalize=False` is the raw CUDA-kernel hash, a separate code path."""
    args = (0, 200, 0, 200, 99, 0.05, 0.0, -1.0, False)
    assert np.array_equal(_numpy_arm(monkeypatch, S.hash_area_weights, *args),
                          np.asarray(S.hash_area_weights(*args)))


@pytest.mark.parametrize("size,n,p", [(100, 500, 0.05), (2048, 300, 0.1), (50, 64, 0.5)])
def test_stim_counts_match_including_the_threshold_convention(monkeypatch, size, n, p):
    """`hash_stim_counts` compares `m < p*2^24`, not `m/2^24 < p`.

    Mathematically the same predicate, different rounding at the boundary. The
    Rust kernel reproduces the convention rather than unifying it; unifying
    would move a handful of synapses per block and nothing would fail loudly.
    """
    ref = _numpy_arm(monkeypatch, S.hash_stim_counts, size, 0, n, 31337, p)
    got = np.asarray(S.hash_stim_counts(size, 0, n, 31337, p))
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("n,p", [(512, 0.05), (400, 0.1)])
def test_csr_direct_build_equals_the_dense_block(n, p):
    """`area_weights_csr_rows` never forms the dense block; it must still be it."""
    indptr, indices, data = rust.area_weights_csr_rows(0, n, n, 4, p)
    dense = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dense[i, indices[indptr[i]:indptr[i + 1]]] = data[indptr[i]:indptr[i + 1]]
    assert np.array_equal(np.asarray(S.hash_area_weights(0, n, 0, n, 4, p)), dense)


def test_row_offset_is_absolute_not_relative():
    """Rows [a, b) must equal that slice of rows [0, b) -- the whole point of
    content addressing. A kernel that indexed from the block start instead
    would pass every fixed-origin test above and still break order-freedom."""
    full = np.asarray(S.hash_area_weights(0, 400, 0, 128, 5150, 0.05))
    part = np.asarray(S.hash_area_weights(150, 400, 0, 128, 5150, 0.05))
    assert np.array_equal(full[150:], part)


def _brain(n=1500, k=150, p=0.05):
    b = Brain(p=p, save_winners=True, seed=11, engine="numpy_sparse")
    b.add_stimulus("s", k)
    b.add_area("C", n, k, 1.0)
    for _ in range(5):
        b.project({"s": ["C"]}, {})
    b._engine.materialize_area("C")
    rng = np.random.default_rng(3)
    init = rng.choice(n, size=k, replace=False).astype(np.uint32)
    b.areas["C"]._winners = init
    b._engine.set_winners("C", init)
    with b.frozen():
        for _ in range(8):
            b.project({}, {"C": ["C"]})
    return b


def test_a_whole_materialized_brain_is_unchanged(monkeypatch):
    """End to end, through `materialize_area`'s CSR-direct path.

    The unit tests above cover the kernel; this covers the WIRING, which is
    where the real risk is -- the accelerated path splits the block at the
    trained corner and stacks two CSR matrices, and getting that split wrong
    would corrupt exactly the rows that carry everything the area has learned.
    """
    monkeypatch.setenv("NEURAL_ASSEMBLIES_NO_RUST", "1")
    ref = _brain()
    ref_w = np.asarray(ref._engine._area_conns["C"]["C"].weights)
    ref_win = sorted(int(x) for x in ref.areas["C"].winners)

    monkeypatch.setenv("NEURAL_ASSEMBLIES_NO_RUST", "0")
    got = _brain()
    got_w = np.asarray(got._engine._area_conns["C"]["C"].weights)

    assert ref_w.shape == got_w.shape
    assert np.array_equal(ref_w, got_w), (
        f"{int((ref_w != got_w).sum())} connectome cells differ")
    assert ref_win == sorted(int(x) for x in got.areas["C"].winners)


def test_extension_is_actually_loaded():
    """Guards against a green suite that never ran the accelerator."""
    assert S.rust_kernels() is not None
    assert hasattr(rust, "area_weights_block")
