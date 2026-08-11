"""A connectome that stores DEVIATIONS from a recomputable base, not weights.

Implements `research/notes/DESIGN_virtual_connectome.md`, amended by reading
`_expand_connectomes` rather than trusting the design's own premise: the base
is NOT pure hash. Three facts survived contact with the code, one did not.

WHAT A MATERIALISED FIBER ACTUALLY IS. Under content-addressed init
([[content-addressed-synapse-init]]), a cell's value is

    w[i, j] = CHAIN( OVERRIDE(i, j) ? 1.0 : BASE(i, j),  n[i, j] )

  BASE      `hash_area_weights(i, j, pair_seed, p_fiber, ...)` -- a pure
            function of position, zero storage.
  OVERRIDE  the sampled first-winner edges `_expand_connectomes` writes
            explicitly (`conn.weights[chosen, col] = 1.0`, drawn from a
            growth-point-keyed RNG). Within a recruitment round plasticity
            runs BEFORE expansion, so the override lands on an
            already-potentiated column and dense assignment CLOBBERS that
            history -- `override` therefore resets the cell's event count
            (the first version refused this order and the fingerprint run
            refuted it immediately). Sparse: <= alloc (~k) cells per
            recruited neuron.
  CHAIN     the Hebbian event sequence. The dense engine does
            ``w *= (1 + beta)`` in float32 THEN clips, per event, so byte
            identity requires replaying n float32 multiply-clip rounds --
            NOT a float64 closed-form power, which differs in ulps and the
            fingerprint digests at round(9), finer than float32 resolution.
  n         an integer co-firing count per cell. Multiplicative updates
            never create a nonzero, so the SPARSITY pattern is
            base-plus-overrides, never the chain.

VECTORISED, SECOND PASS. The first implementation was semantically exact and
~5x SLOWER than dense: `row_sum` regenerated base rows one hash call at a
time and walked per-cell dicts, `bump` incremented dict-of-dicts one cell at
a time. This version stores per-row SORTED ARRAYS (cols, counts), generates
all k rows in ONE `hash_area_weights_rows` call, applies deviations with one
fancy-index scatter, and keeps a small LRU of raw base blocks -- the drive
and the plasticity of one projection use the SAME winner rows, so the block
is hashed once and read twice. The float32 summation order is unchanged
(same rows, same `sum(axis=0)`), so byte identity is preserved; the tests
that pin it did not change.

WHAT THE REPRESENTATION REFUSES, structurally (`supports`):

  * homeostatic synaptic scaling -- per-column rescales interleave with
    clips and do not factor into any per-cell summary;
  * the legacy (non-content-addressed) initializer -- its values come from a
    shared RNG stream a positional function cannot reproduce;
  * a CHANGED fiber beta after potentiation has begun -- one exponent count
    cannot carry two growth factors ([[same-name-two]]). `bump` raises; the
    engine's escape hatch is densify-and-swap, never silent mixing.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional

import numpy as np

from ._seeding import hash_area_weights, hash_area_weights_rows

_CACHE_BLOCKS = 4


class VirtualWeights:
    """One area->area fiber: hash base + sparse overrides + sparse exponents.

    All coordinates are ABSOLUTE row/column indices into the fiber's logical
    block -- the same indices the dense buffer would use -- so the engine's
    existing compact bookkeeping maps onto this unchanged.
    """

    __slots__ = ("n_rows", "n_cols", "pair_seed", "p", "beta",
                 "w_lo", "w_hi", "inhibitory_prob", "inhibitory_weight",
                 "_exp", "_ovr", "_ovr_sorted", "_potentiated", "_cache")

    def __init__(self, n_rows: int, n_cols: int, pair_seed: int, p: float,
                 beta: float, w_lo: Optional[float], w_hi: Optional[float],
                 inhibitory_prob: float = 0.0,
                 inhibitory_weight: float = -1.0):
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.pair_seed = int(pair_seed)
        self.p = float(p)
        self.beta = float(beta)
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.inhibitory_prob = float(inhibitory_prob)
        self.inhibitory_weight = float(inhibitory_weight)
        #: row -> [sorted col ids (int64), event counts (int64)]
        self._exp: Dict[int, list] = {}
        #: row -> col ids forced to 1.0 at recruitment. WRITE side is a
        #: python set (override storms during recruitment cost 5.2s of a
        #: 16.9s training profile as sorted np.insert allocations); the READ
        #: side sorts lazily via `_ovr_sorted`.
        self._ovr: Dict[int, set] = {}
        self._ovr_sorted: Dict[int, np.ndarray] = {}
        self._potentiated = False
        #: LRU of RAW base blocks (no overrides): rows-bytes -> (rows, block)
        self._cache: OrderedDict = OrderedDict()

    # -- capability gate ----------------------------------------------------

    @staticmethod
    def supports(*, synaptic_scaling: bool, content_init: bool) -> bool:
        return content_init and not synaptic_scaling

    # -- shape --------------------------------------------------------------

    @property
    def shape(self):
        return (self.n_rows, self.n_cols)

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        """Logical cell count. TWO consumers branch on it, and its absence
        was a silent norm kill-switch: `_norm_scale` early-returns on
        ``getattr(w, 'size', 0) == 0``, so without this property every
        virtual fiber skipped normalization. `clone` also branches on
        ``size > 0`` to decide copy-vs-share."""
        return self.n_rows * self.n_cols

    def copy(self) -> "VirtualWeights":
        """Deep copy for `clone` -- sharing the stores would alias two
        brains' plasticity onto one fiber."""
        out = VirtualWeights(self.n_rows, self.n_cols, self.pair_seed,
                             self.p, self.beta, self.w_lo, self.w_hi,
                             self.inhibitory_prob, self.inhibitory_weight)
        out._exp = {r: [v[0].copy(), v[1].copy()]
                    for r, v in self._exp.items()}
        out._ovr = {r: set(v) for r, v in self._ovr.items()}
        out._potentiated = self._potentiated
        return out

    def resize(self, n_rows: int, n_cols: int) -> None:
        """Growth. The dense path's 2x-peak reallocation becomes two ints."""
        if n_rows < self.n_rows or n_cols < self.n_cols:
            raise ValueError("virtual fibers never shrink")
        if n_cols > self.n_cols:
            self._cache.clear()          # cached blocks are col_end-shaped
        self.n_rows, self.n_cols = int(n_rows), int(n_cols)

    # -- base ---------------------------------------------------------------

    def _raw_rows(self, rows: np.ndarray) -> np.ndarray:
        """RAW hash block for *rows* (no overrides), LRU-cached.

        The drive (`row_sum`) and the plasticity (`bump`) of one training
        projection use the SAME source winners, so the second call is a hit;
        the returned array is shared and must be treated as read-only.
        """
        key = rows.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        block = hash_area_weights_rows(rows, 0, self.n_cols, self.pair_seed,
                                       self.p, self.inhibitory_prob,
                                       self.inhibitory_weight)
        self._cache[key] = block
        if len(self._cache) > _CACHE_BLOCKS:
            self._cache.popitem(last=False)
        return block

    def _deviation_triples(self, rows: np.ndarray, cols: int):
        """Concatenated (local row idx, col, count) over the selected rows."""
        loc, col_parts, cnt_parts = [], [], []
        for i, r in enumerate(rows):
            entry = self._exp.get(int(r))
            if entry is None:
                continue
            c, n = entry
            if len(c) and c[-1] >= cols:
                keep = c < cols
                c, n = c[keep], n[keep]
            if len(c):
                loc.append(np.full(len(c), i, dtype=np.int64))
                col_parts.append(c)
                cnt_parts.append(n)
        if not loc:
            return None
        return (np.concatenate(loc), np.concatenate(col_parts),
                np.concatenate(cnt_parts))

    def _ovr_arr(self, r: int):
        """Sorted override cols for row *r*, built lazily after writes."""
        arr = self._ovr_sorted.get(r)
        if arr is None:
            src = self._ovr.get(r)
            if not src:
                return None
            arr = np.fromiter(src, dtype=np.int64, count=len(src))
            arr.sort()
            self._ovr_sorted[r] = arr
        return arr

    def _apply_overrides(self, block: np.ndarray, rows: np.ndarray,
                         cols: int) -> None:
        for i, r in enumerate(rows):
            ovr = self._ovr_arr(int(r))
            if ovr is not None and len(ovr):
                idx = ovr[ovr < cols] if ovr[-1] >= cols else ovr
                if len(idx):
                    block[i, idx] = 1.0

    def _chain(self, values: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Replay the dense engine's per-event float32 multiply-then-clip."""
        v = values.astype(np.float32, copy=True)
        g = np.float32(1.0 + self.beta)
        rounds = int(counts.max()) if len(counts) else 0
        for r in range(1, rounds + 1):
            mask = counts >= r
            v[mask] = v[mask] * g
            if self.w_hi is not None:
                np.clip(v, self.w_lo, self.w_hi, out=v)
        return v

    def _materialize_rows(self, rows: np.ndarray, cols: int) -> np.ndarray:
        """The (len(rows), cols) float32 block exactly as dense would hold it."""
        block = self._raw_rows(rows)[:, :cols].copy()
        self._apply_overrides(block, rows, cols)
        dev = self._deviation_triples(rows, cols)
        if dev is not None:
            loc, c, n = dev
            block[loc, c] = self._chain(block[loc, c], n)
        return block

    # -- reads --------------------------------------------------------------

    def row_sum(self, rows, col_end: Optional[int] = None) -> np.ndarray:
        """``w[rows, :col_end].sum(axis=0)`` -- the drive kernel.

        One batched hash for all k rows, one scatter for the deviations, and
        the SAME float32 ``sum(axis=0)`` the dense path performs -- the
        summation order is part of the contract (ulps flip k-WTA tie order).
        """
        rows = np.asarray(rows, dtype=np.int64)
        cols = self.n_cols if col_end is None else min(int(col_end),
                                                       self.n_cols)
        return self._materialize_rows(rows, cols).sum(axis=0)

    def cell(self, row: int, col: int) -> float:
        block = self._materialize_rows(np.asarray([row], dtype=np.int64),
                                       int(col) + 1)
        return float(block[0, col])

    def column_nnz(self, rows_known: Optional[int] = None) -> np.ndarray:
        """Per-column nonzero counts -- what `_norm_scale` divides by.

        A property of base-plus-overrides only: the chain never creates a
        nonzero. Chunked contiguous regeneration; overrides that landed on
        base-zero cells add one each.
        """
        rows_known = self.n_rows if rows_known is None else int(rows_known)
        counts = np.zeros(self.n_cols, dtype=np.int64)
        chunk = 4096
        for start in range(0, rows_known, chunk):
            end = min(start + chunk, rows_known)
            block = hash_area_weights(start, end, 0, self.n_cols,
                                      self.pair_seed, self.p,
                                      self.inhibitory_prob,
                                      self.inhibitory_weight)
            counts += (block != 0).sum(axis=0)
            for r in range(start, end):
                ovr = self._ovr_arr(r)
                if ovr is not None and len(ovr):
                    zero_base = ovr[block[r - start, ovr] == 0]
                    counts[zero_base] += 1
        return counts

    # -- writes -------------------------------------------------------------

    def override(self, rows, col: int) -> None:
        """The recruitment write: assign 1.0, CLOBBERING any history.

        Within a recruitment round `_apply_plasticity` executes BEFORE
        `_expand_connectomes`, so the new winner's column is potentiated and
        THEN assigned 1.0. Dense assignment erases that history, so the
        exponent count resets here -- future events chain from 1.0.
        """
        col = int(col)
        for r in np.asarray(rows, dtype=np.int64):
            r = int(r)
            entry = self._exp.get(r)
            if entry is not None:
                c, n = entry
                pos = int(np.searchsorted(c, col))
                if pos < len(c) and c[pos] == col:
                    keep = np.ones(len(c), dtype=bool)
                    keep[pos] = False
                    self._exp[r] = [c[keep], n[keep]]
            self._ovr.setdefault(r, set()).add(col)
            self._ovr_sorted.pop(r, None)

    def bump(self, rows, cols, beta: float) -> None:
        """One Hebbian event on the co-firing block: count += 1 per nonzero.

        Base-zero, non-overridden cells are skipped -- multiplying zero is
        what the dense engine does, and an exponent there would claim a
        synapse that does not exist. Vectorised: one (possibly cached) hash
        block for the row set, one sorted-merge per row.
        """
        if float(beta) != self.beta:
            raise ValueError(
                f"fiber constructed at beta={self.beta}, bump at {beta}: one "
                f"exponent store cannot carry two growth factors -- densify")
        self._potentiated = True
        rows = np.asarray(rows, dtype=np.int64)
        cols_sorted = np.sort(np.asarray(cols, dtype=np.int64))
        raw = self._raw_rows(rows)
        for i, r in enumerate(rows):
            r = int(r)
            live = cols_sorted[raw[i, cols_sorted] != 0]
            ovr = self._ovr_arr(r)
            if ovr is not None and len(ovr):
                extra = cols_sorted[np.isin(cols_sorted, ovr,
                                            assume_unique=True)]
                if len(extra):
                    live = np.union1d(live, extra)
            if not len(live):
                continue
            entry = self._exp.get(r)
            if entry is None:
                self._exp[r] = [live.copy(),
                                np.ones(len(live), dtype=np.int64)]
                continue
            c, n = entry
            merged = np.union1d(c, live)
            counts = np.zeros(len(merged), dtype=np.int64)
            counts[np.searchsorted(merged, c)] = n
            counts[np.searchsorted(merged, live)] += 1
            self._exp[r] = [merged, counts]

    # -- bookkeeping ---------------------------------------------------------

    @property
    def deviations(self) -> int:
        return sum(len(v[0]) for v in self._exp.values())

    @property
    def overrides(self) -> int:
        return sum(len(v) for v in self._ovr.values())

    @property
    def nbytes(self) -> int:
        return (self.deviations * 16 + self.overrides * 8) + len(self._exp) * 64

    def todense(self) -> np.ndarray:
        """The full matrix, for VERIFICATION only."""
        out = np.empty((self.n_rows, self.n_cols), dtype=np.float32)
        chunk = 2048
        for start in range(0, self.n_rows, chunk):
            end = min(start + chunk, self.n_rows)
            rows = np.arange(start, end, dtype=np.int64)
            out[start:end] = self._materialize_rows(rows, self.n_cols)
        return out
