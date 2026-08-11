"""Where a synapse's initial weight comes from.

Two disciplines were in use, and only one of them is order-free.

CONTENT-ADDRESSED -- a synapse's value is a pure function of its endpoints.
Materialising block A before block B gives the same answer as B before A,
because neither consults a shared cursor.

STREAM-ADDRESSED -- draws come from a single ``Generator`` owned by the engine,
so a synapse's value depends on HOW MANY DRAWS HAPPENED FIRST. Two runs that
build the same connectome in a different order get different weights.

The second is what made probing unsound. Parsing three items as [X,Y,Z] and as
[Z,Y,X] produced connectomes differing in 13165 / 9643 / 7167 cells across
three core->role blocks, because each parse materialises rows and thereby moves
the cursor for everything after it. That is also the mechanism behind the
long-standing observation that candidate metrics contaminate each other: a
probe is not read-only, and its cost is paid by whatever is measured next.

So the split this module draws is between INITIALISATION and DYNAMICS:

* initialisation (which synapses exist, and their pre-training weights) is
  content-addressed here, and consumes no stream at all;
* dynamics (input noise, tie-breaking, subsampling) legitimately depends on
  history and keeps using the engine's ``Generator`` -- but that generator's
  state is a small serialisable object, so it can be snapshotted and restored
  around a probe. Order-freedom plus a restorable cursor is what makes cheap
  probe isolation sound; neither alone is enough.

``CudaImplicitEngine`` already worked this way ("No RNG state consumed for
connectivity initialization"). This module lifts that property out of the CUDA
path so the default numpy engine can share it rather than reimplement it.
"""

from __future__ import annotations

import os
import zlib

import numpy as np

# -- optional Rust accelerator ---------------------------------------------
#
# `crates/na-kernels` computes the blocks below in one pass, with no
# temporaries, across all cores. Measured 8-32x depending on block shape, and
# BYTE-IDENTICAL -- which is a property of the computation rather than of the
# tuning: it is exact u32 arithmetic with wrapping multiplies, and the one
# float step (`m / 2^24` for integer m < 2^24) is exactly representable in f32.
# `tests/test_rust_kernels.py` asserts the equality rather than assuming it.
#
# THE NUMPY CODE BELOW REMAINS THE SPECIFICATION. The extension is optional at
# every level: absent, unbuildable, or disabled by
# NEURAL_ASSEMBLIES_NO_RUST=1, everything still runs and produces the same
# numbers -- just slower.
try:
    import na_kernels as _rust
except ImportError:                     # pragma: no cover - accelerator absent
    _rust = None


def rust_kernels():
    """The accelerator module, or None. Checked per call, not per import.

    The env var is read here rather than cached so a test can toggle it and
    A/B the two implementations inside one process.
    """
    if _rust is None or os.environ.get("NEURAL_ASSEMBLIES_NO_RUST", "0") != "0":
        return None
    return _rust

# Salt distinguishing the presence draw from the inhibitory draw for the same
# (row, col). Without it both would hash identically and every present synapse
# would be inhibitory whenever inhibitory_prob >= p.
_INHIBITORY_SALT = np.uint32(0x9E3779B9)

_MUL_ROW = np.uint32(2654435761)
_MUL_COL = np.uint32(2246822519)
_MANTISSA = np.uint32(0xFFFFFF)      # 24 bits
_MANTISSA_SCALE = 16777216.0         # 2**24


def stable_seed(*parts) -> int:
    """A 32-bit seed from `parts` that is identical in every process.

    MUST be used instead of ``hash(...)`` for anything that seeds an RNG.
    Python randomizes ``hash()`` of str/bytes per process (PEP 456), so
    ``hash((src, tgt, nr, nc))`` is stable WITHIN a run and different across
    runs. Three lazy-connectome sites seeded ``default_rng`` that way and were
    commented "deterministic per-pair seed" -- they were not, and the result
    was that `Brain(seed=42)` trained different weights from one process to the
    next. Measured on the Geschwind lesion study: ~1/3 of PYTHONHASHSEED values
    changed the reported accuracy (1.00 vs 0.50), and one crashed.
    """
    return zlib.crc32(repr(parts).encode("utf-8")) & 0xFFFFFFFF


def fnv1a_pair_seed(global_seed: int, source: str, target: str) -> int:
    """FNV-1a 32-bit seed for one (source, target) fiber.

    Byte-identical to the CUDA engine's ``_fnv1a_pair_seed`` so both engines
    place the same fiber at the same seed. The NUL separator is what stops
    ("ab", "c") and ("a", "bc") from colliding.
    """
    h = 0x811C9DC5
    for byte in int(global_seed).to_bytes(4, "little"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    for byte in source.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    h = ((h ^ 0x00) * 0x01000193) & 0xFFFFFFFF
    for byte in target.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h


def _raw_hash(rows, cols, pair_seed):
    """The CUDA kernels' hash: ``(r*A) ^ (c*B) ^ seed``, no finalizer.

    Kept callable on its own because its statistical quality is the thing that
    has to be measured rather than assumed -- see ``mix32`` below.
    """
    with np.errstate(over="ignore"):
        h = (rows * _MUL_ROW) ^ (cols * _MUL_COL)
    return h ^ np.uint32(pair_seed)


def mix32(h, copy: bool = True):
    """Murmur3 ``fmix32`` avalanche finalizer.

    ``(r*A) ^ (c*B)`` is cheap but its low bits are close to a function of the
    low bits of r and c alone, and the Bernoulli test reads exactly those low
    24 bits. This scrambles high bits back down into them so that adjacent
    cells are independent. Whether the raw hash actually needs it is an
    empirical question, answered in ``tests/test_seeding.py``.

    WRITTEN IN PLACE, because this runs over an ``n x n`` block and the naive
    spelling allocates five fresh arrays of it -- 64 MB each at ``n=4000``, 1 GB
    at ``n=16,000``. Reusing one temporary and mutating in place is **1.52x
    faster** and bit-identical (asserted in ``tests/test_seeding.py``); it is
    56% of ``materialize_area``'s cost, which is what bounds the ladder.

    ``copy=False`` additionally consumes *h* itself, for callers that built it
    solely to be mixed.
    """
    if copy:
        h = h.copy()
    t = np.empty_like(h)
    with np.errstate(over="ignore"):
        np.right_shift(h, np.uint32(16), out=t)
        h ^= t
        h *= np.uint32(0x85EBCA6B)
        np.right_shift(h, np.uint32(13), out=t)
        h ^= t
        h *= np.uint32(0xC2B2AE35)
        np.right_shift(h, np.uint32(16), out=t)
        h ^= t
    return h


def hash_uniform_2d(row_start, row_end, col_start, col_end, pair_seed,
                    finalize: bool = True):
    """Uniform [0, 1) draws for a rectangular block, addressed by (row, col).

    The value at absolute cell (i, j) does not depend on which block it was
    requested in, so growing a connectome by rows then columns gives the same
    matrix as growing it by columns then rows.
    """
    nr, nc = int(row_end - row_start), int(col_end - col_start)
    if nr <= 0 or nc <= 0:
        return np.empty((max(nr, 0), max(nc, 0)), dtype=np.float32)
    rows = np.arange(row_start, row_end, dtype=np.uint32).reshape(nr, 1)
    cols = np.arange(col_start, col_end, dtype=np.uint32).reshape(1, nc)
    h = _raw_hash(rows, cols, pair_seed)
    if finalize:
        # _raw_hash just built h for us, so mixing may consume it.
        h = mix32(h, copy=False)
    return ((h & _MANTISSA).astype(np.float32) / _MANTISSA_SCALE)


def hash_bernoulli_2d(row_start, row_end, col_start, col_end, pair_seed, p,
                      finalize: bool = True):
    """Bernoulli(p) block, addressed by (row, col). 1.0 where connected."""
    u = hash_uniform_2d(row_start, row_end, col_start, col_end, pair_seed,
                        finalize=finalize)
    return (u < p).astype(np.float32)


def hash_area_weights(row_start, row_end, col_start, col_end, pair_seed,
                      p: float, inhibitory_prob: float = 0.0,
                      inhibitory_weight: float = -1.0,
                      finalize: bool = True):
    """Content-addressed equivalent of ``_sample_area_weights``.

    Each present synapse (probability ``p``) is excitatory (weight 1) with
    probability ``1 - inhibitory_prob`` or inhibitory (``inhibitory_weight``)
    otherwise -- the same two-draw structure as the streamed version, with the
    second draw salted so it is independent of the first.
    """
    rust = rust_kernels()
    if rust is not None:
        return rust.area_weights_block(
            int(row_start), int(row_end), int(col_start), int(col_end),
            int(pair_seed) & 0xFFFFFFFF, float(p),
            float(inhibitory_prob), float(inhibitory_weight), bool(finalize),
        )
    present = hash_bernoulli_2d(row_start, row_end, col_start, col_end,
                                pair_seed, p, finalize=finalize)
    if inhibitory_prob <= 0.0:
        return present
    u_inh = hash_uniform_2d(row_start, row_end, col_start, col_end,
                            np.uint32(pair_seed) ^ _INHIBITORY_SALT,
                            finalize=finalize)
    w = present.copy()
    w[(present > 0) & (u_inh < inhibitory_prob)] = inhibitory_weight
    return w


def hash_area_weights_rows(row_ids, col_start, col_end, pair_seed,
                           p: float, inhibitory_prob: float = 0.0,
                           inhibitory_weight: float = -1.0,
                           finalize: bool = True):
    """`hash_area_weights` for a SCATTERED set of rows, one vectorised call.

    A cell's value is a pure function of its absolute (row, col), so the
    contiguous kernel's ``arange(row_start, row_end)`` is just a special case
    of an arbitrary row-id vector -- the broadcasting and every elementwise op
    are identical, and `test_seeding` asserts the equality rather than
    trusting this sentence.

    WHY IT EXISTS. `VirtualWeights.row_sum` regenerates the k WINNER rows of
    a fiber per projection. Calling the contiguous kernel once per row cost k
    Python-level round trips and made the virtual path ~5x slower than dense;
    this replaces them with one call over a (k, n_cols) block. Kept in this
    module so the two spellings share `_raw_hash`/`mix32` and cannot drift.
    Rust path: `area_rows_block` (already in na-kernels, rayon over rows)
    measured 15.6x over the numpy spelling at (70, 20000) on this box --
    2.37 ms vs 37.1 ms -- and byte-identical, asserted in `test_seeding`.
    """
    rust = rust_kernels()
    if rust is not None:
        return rust.area_rows_block(
            np.ascontiguousarray(row_ids, dtype=np.int64),
            int(col_start), int(col_end), int(pair_seed) & 0xFFFFFFFF,
            float(p), float(inhibitory_prob), float(inhibitory_weight),
            bool(finalize))
    row_ids = np.asarray(row_ids, dtype=np.uint32).reshape(-1, 1)
    nc = int(col_end - col_start)
    if len(row_ids) == 0 or nc <= 0:
        return np.empty((len(row_ids), max(nc, 0)), dtype=np.float32)
    cols = np.arange(col_start, col_end, dtype=np.uint32).reshape(1, nc)
    h = _raw_hash(row_ids, cols, pair_seed)
    if finalize:
        h = mix32(h, copy=False)
    u = (h & _MANTISSA).astype(np.float32) / _MANTISSA_SCALE
    present = (u < p).astype(np.float32)
    if inhibitory_prob <= 0.0:
        return present
    h2 = _raw_hash(row_ids, cols, np.uint32(pair_seed) ^ _INHIBITORY_SALT)
    if finalize:
        h2 = mix32(h2, copy=False)
    u_inh = (h2 & _MANTISSA).astype(np.float32) / _MANTISSA_SCALE
    w = present.copy()
    w[(present > 0) & (u_inh < inhibitory_prob)] = inhibitory_weight
    return w


def hash_stim_counts(stim_size: int, neuron_start: int, neuron_end: int,
                     pair_seed, p: float, finalize: bool = True,
                     chunk: int = 1024):
    """Per-neuron count of connected stimulus fibers.

    The content-addressed replacement for ``rng.binomial(stim_size, p, size)``:
    it materialises the actual stim x neuron connectivity and counts it, so the
    count for neuron j is fixed by j rather than by draw order. Chunked over
    the stimulus axis to bound peak memory for large stimuli.
    """
    n = int(neuron_end - neuron_start)
    if n <= 0 or stim_size <= 0:
        return np.zeros(max(n, 0), dtype=np.float32)
    rust = rust_kernels()
    if rust is not None:
        # Rust fuses the count into the hash loop, so nothing
        # O(stim_size * n) is allocated and the chunking below is moot.
        return rust.stim_counts(
            int(stim_size), int(neuron_start), int(neuron_end),
            int(pair_seed) & 0xFFFFFFFF, float(p), bool(finalize),
        )
    out = np.zeros(n, dtype=np.float32)
    neurons = np.arange(neuron_start, neuron_end, dtype=np.uint32).reshape(1, n)
    for start in range(0, int(stim_size), chunk):
        stop = min(start + chunk, int(stim_size))
        stim = np.arange(start, stop, dtype=np.uint32).reshape(stop - start, 1)
        h = _raw_hash(stim, neurons, pair_seed)
        if finalize:
            h = mix32(h)
        out += (h & _MANTISSA).astype(np.float32).__lt__(
            p * _MANTISSA_SCALE).sum(axis=0)
    return out


def hash_area_rows(rows, n_cols: int, pair_seed, p: float,
                   inhibitory_prob: float = 0.0, inhibitory_weight: float = -1.0,
                   finalize: bool = True, want_sum: bool = False):
    """Initial weights for an ARBITRARY set of rows -- block, or column sums.

    An assembly's winners are a scattered index set, not a contiguous range, so
    `hash_area_weights` has to be called once per row. At n=1e4, k=200 that is
    200 calls and 200 allocations per projection round for ~1.7 ms of actual
    hashing. The Rust kernels take the row set directly and, with
    `want_sum=True`, fuse the column sum so nothing of size k*n is allocated at
    all -- which is what the un-potentiated drive actually needs.

    Falls back to per-row `hash_area_weights` when the accelerator is absent,
    so behaviour is identical either way.
    """
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    rust = rust_kernels()
    if rust is not None:
        fn = rust.area_rows_sum if want_sum else rust.area_rows_block
        return fn(rows, 0, int(n_cols), int(pair_seed) & 0xFFFFFFFF, float(p),
                  float(inhibitory_prob), float(inhibitory_weight),
                  bool(finalize))
    if want_sum:
        acc = np.zeros(int(n_cols), dtype=np.float64)
        for r in rows:
            acc += hash_area_weights(int(r), int(r) + 1, 0, int(n_cols),
                                     pair_seed, p, inhibitory_prob,
                                     inhibitory_weight, finalize).reshape(-1)
        return acc
    out = np.empty((rows.size, int(n_cols)), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i] = hash_area_weights(int(r), int(r) + 1, 0, int(n_cols),
                                   pair_seed, p, inhibitory_prob,
                                   inhibitory_weight, finalize).reshape(-1)
    return out


def hash_area_indegree(n_rows: int, n_cols: int, pair_seed, p: float,
                       finalize: bool = True, chunk: int = 512):
    """Per-column count of present synapses over rows [0, n_rows).

    The exact `norm_init` divisor `d_j`. Only computable at all because every
    row exists; the lazy engines have to estimate the not-yet-materialised part
    (`inverse_indegree`'s ambient term). Counted with the SAME predicate the
    weights are drawn with, never `hash_stim_counts`' -- see the kernel's note.
    """
    rust = rust_kernels()
    if rust is not None:
        return rust.area_indegree(int(n_rows), 0, int(n_cols),
                                  int(pair_seed) & 0xFFFFFFFF, float(p),
                                  bool(finalize))
    deg = np.zeros(int(n_cols), dtype=np.float32)
    for r0 in range(0, int(n_rows), chunk):
        r1 = min(r0 + chunk, int(n_rows))
        blk = hash_area_weights(r0, r1, 0, int(n_cols), pair_seed, p,
                                0.0, -1.0, finalize)
        deg += (np.asarray(blk) != 0).sum(axis=0)
    return deg


def hash_area_cells(rows, cols, pair_seed, p: float,
                    inhibitory_prob: float = 0.0, inhibitory_weight: float = -1.0,
                    finalize: bool = True):
    """Initial weights on an arbitrary rows x cols GATHER.

    Potentiation reaches a scattered set of columns, so the correction term in
    the exact drive lives on a gather, not on a contiguous slice -- a
    bounding range would span nearly all of `n` and save nothing.
    """
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    rust = rust_kernels()
    if rust is not None:
        return rust.area_cells_block(
            rows, cols, int(pair_seed) & 0xFFFFFFFF, float(p),
            float(inhibitory_prob), float(inhibitory_weight), bool(finalize))
    out = np.empty((rows.size, cols.size), dtype=np.float32)
    for i, r in enumerate(rows):
        full = hash_area_weights(int(r), int(r) + 1, 0, int(cols.max()) + 1,
                                 pair_seed, p, inhibitory_prob,
                                 inhibitory_weight, finalize).reshape(-1)
        out[i] = full[cols]
    return out


def hash_area_csr(rows, n_cols: int, pair_seed, p: float,
                  inhibitory_prob: float = 0.0, inhibitory_weight: float = -1.0,
                  finalize: bool = True):
    """Edge list (CSR) of a fiber restricted to `rows`. `(indptr, indices, data)`.

    G(n,p) is drawn once at t=0 and never changes, so a row's neighbours are a
    fixed fact. Enumerating them once and reusing turns the drive from
    `k * n` hash evaluations into `k * n * p` scatter-adds -- at n=1e4, p=0.05
    that is 100k updates instead of 2M hashes.
    """
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    rust = rust_kernels()
    if rust is not None:
        return rust.area_rows_csr(
            rows, int(n_cols), int(pair_seed) & 0xFFFFFFFF, float(p),
            float(inhibitory_prob), float(inhibitory_weight), bool(finalize))
    indptr = np.zeros(rows.size + 1, dtype=np.int64)
    idx_parts, dat_parts = [], []
    for i, r in enumerate(rows):
        w = hash_area_weights(int(r), int(r) + 1, 0, int(n_cols), pair_seed, p,
                              inhibitory_prob, inhibitory_weight,
                              finalize).reshape(-1)
        nz = np.flatnonzero(w)
        idx_parts.append(nz.astype(np.int32))
        dat_parts.append(np.asarray(w)[nz].astype(np.float32))
        indptr[i + 1] = indptr[i] + nz.size
    empty_i = np.zeros(0, dtype=np.int32)
    return (indptr,
            np.concatenate(idx_parts) if idx_parts else empty_i,
            np.concatenate(dat_parts) if dat_parts else empty_i.astype(np.float32))


def csr_drive(indptr, indices, rows, n_cols: int):
    """Accumulate cached CSR rows into an `n_cols` drive vector.

    Only valid where every present weight is 1 (``inhibitory_prob == 0``), in
    which case the drive IS a count. Fused in Rust because doing it from numpy
    does not pay: the accumulation is cheap but GATHERING the rows out of the
    CSR costs more than the dense rescan (measured 1.81 ms against 0.74 ms).
    """
    rust = rust_kernels()
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if rust is not None:
        return rust.csr_row_counts(indptr, indices, rows, int(n_cols))
    out = np.zeros(int(n_cols), dtype=np.float32)
    for r in rows:
        a, b = int(indptr[int(r)]), int(indptr[int(r) + 1])
        np.add.at(out, indices[a:b], 1.0)
    return out
