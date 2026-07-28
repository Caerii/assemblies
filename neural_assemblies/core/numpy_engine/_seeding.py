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

import zlib

import numpy as np

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


def mix32(h):
    """Murmur3 ``fmix32`` avalanche finalizer.

    ``(r*A) ^ (c*B)`` is cheap but its low bits are close to a function of the
    low bits of r and c alone, and the Bernoulli test reads exactly those low
    24 bits. This scrambles high bits back down into them so that adjacent
    cells are independent. Whether the raw hash actually needs it is an
    empirical question, answered in ``tests/test_seeding.py``.
    """
    with np.errstate(over="ignore"):
        h = h ^ (h >> np.uint32(16))
        h = h * np.uint32(0x85EBCA6B)
        h = h ^ (h >> np.uint32(13))
        h = h * np.uint32(0xC2B2AE35)
        h = h ^ (h >> np.uint32(16))
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
        h = mix32(h)
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
