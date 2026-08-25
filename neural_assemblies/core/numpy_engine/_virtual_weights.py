"""A connectome that stores DEVIATIONS from a recomputable base, not weights.

Implements `research/notes/DESIGN_virtual_connectome.md` and, as of the third
pass, `research/notes/PREREG_drive_semantics_v2.md`. A cell's value is

    w[i, j] = CHAIN( OVERRIDE(i, j) ? 1.0 : BASE(i, j),  n[i, j] )

  BASE      `hash_area_weights(i, j, pair_seed, p_fiber, ...)` -- a pure
            function of position, zero storage, IMMUTABLE.
  OVERRIDE  the sampled first-winner edges recruitment writes explicitly.
            Plasticity precedes expansion within a round, so dense assignment
            CLOBBERS potentiated history; `override` resets the event count.
  CHAIN     n float32 multiply-then-clip rounds, replaying the dense engine's
            per-event arithmetic exactly. Per-cell semantics are UNCHANGED
            across all three passes and stay pinned by `test_virtual_weights`.

DRIVE SEMANTICS v2 (the third pass, and the algorithmic point). Because BASE
is immutable, a row-set's base drive is a pure function of the row-set:

    row_sum(rows) = float32( base_sum_f64(rows)[:cols] + delta_f64(rows) )

`base_sum` is computed once per row-set (float64 pairwise over the raw block,
full width) and MEMOIZED under a byte budget; `delta` sums, in fixed row
order, `final - raw` over the deviation and override cells of the selected
rows -- all of whose raw values are STORED AT WRITE TIME, so the memoized
path performs no hashing at all. Training presents the same row-sets ~15x
and eval revisits the same states hundreds of times: amortized drive cost
falls from O(k * n_cols) to O(deviations) + top-k. Dense storage cannot do
this -- plasticity mutates its matrix, so there is nothing immutable to
memoize. The cost, stated in the prereg: this summation order differs from
dense float32 pairwise reduction by ulps, so v2 winners may differ from
dense at k-WTA TIES. The registered science-invariance bar (V-S5) is what
licenses that; the cold and memoized paths are the SAME computation, so
self-consistency is exact by construction.

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

from ._seeding import (
    hash_area_weights, hash_area_weights_at, hash_area_weights_rows,
)

_CACHE_BLOCKS = 4
_SUM_CACHE_BYTES = 64 << 20          # 64 MB of memoized f64 base sums


class VirtualWeights:
    """One area->area fiber: hash base + sparse overrides + sparse exponents.

    All coordinates are ABSOLUTE row/column indices into the fiber's logical
    block -- the same indices the dense buffer would use -- so the engine's
    existing compact bookkeeping maps onto this unchanged.
    """

    __slots__ = ("n_rows", "n_cols", "pair_seed", "p", "beta",
                 "w_lo", "w_hi", "inhibitory_prob", "inhibitory_weight",
                 "_exp", "_ovr", "_ovr_sorted", "_potentiated",
                 "_cache", "_sum_cache", "_sum_bytes")

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
        #: row -> [sorted cols (i64), counts (i64), chain base eff (f32:
        #: 1.0 if the cell was overridden when first bumped, else raw),
        #: raw base (f32)]. Raw is stored AT WRITE TIME so the memoized
        #: drive path never hashes.
        self._exp: Dict[int, list] = {}
        #: row -> {col: raw base value}. Write side is a dict (the sorted
        #: np.insert version cost 5.2s of a 16.9s profile); read side sorts
        #: lazily via `_ovr_sorted`.
        self._ovr: Dict[int, dict] = {}
        self._ovr_sorted: Dict[int, np.ndarray] = {}
        self._potentiated = False
        #: LRU of RAW base blocks (verification/materialize path only).
        self._cache: OrderedDict = OrderedDict()
        #: byte-budgeted memo of f64 base sums, keyed on the row-set bytes.
        self._sum_cache: OrderedDict = OrderedDict()
        self._sum_bytes = 0

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
        """Logical cell count. `_norm_scale` early-returns on
        ``getattr(w, 'size', 0) == 0`` (its absence was a silent norm
        kill-switch) and `clone` branches on it for copy-vs-share."""
        return self.n_rows * self.n_cols

    def copy(self) -> "VirtualWeights":
        """Deep copy for `clone` -- sharing the stores would alias two
        brains' plasticity onto one fiber."""
        out = VirtualWeights(self.n_rows, self.n_cols, self.pair_seed,
                             self.p, self.beta, self.w_lo, self.w_hi,
                             self.inhibitory_prob, self.inhibitory_weight)
        out._exp = {r: [a.copy() for a in v] for r, v in self._exp.items()}
        out._ovr = {r: dict(v) for r, v in self._ovr.items()}
        out._potentiated = self._potentiated
        return out

    def resize(self, n_rows: int, n_cols: int) -> None:
        """Growth. The dense path's 2x-peak reallocation becomes two ints."""
        if n_rows < self.n_rows or n_cols < self.n_cols:
            raise ValueError("virtual fibers never shrink")
        if n_cols > self.n_cols:
            # Cached blocks and sums are full-width objects.
            self._cache.clear()
            self._sum_cache.clear()
            self._sum_bytes = 0
        self.n_rows, self.n_cols = int(n_rows), int(n_cols)

    # -- base ---------------------------------------------------------------

    def _raw_rows(self, rows: np.ndarray) -> np.ndarray:
        """RAW hash block for *rows* (no overrides), LRU-cached, read-only."""
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

    def _raw_at(self, rows, cols) -> np.ndarray:
        return hash_area_weights_at(rows, cols, self.pair_seed, self.p,
                                    self.inhibitory_prob,
                                    self.inhibitory_weight)

    def _base_sum(self, rows: np.ndarray) -> np.ndarray:
        """Memoized f64 base drive of a row-set, full fiber width.

        Immutability is what makes this sound: the raw base never changes,
        so an entry is valid for the fiber's lifetime (column growth clears
        the cache in `resize`, since entries are full-width objects).
        """
        key = rows.tobytes()
        hit = self._sum_cache.get(key)
        if hit is not None:
            self._sum_cache.move_to_end(key)
            return hit
        s = self._raw_rows(rows).sum(axis=0, dtype=np.float64)
        self._sum_cache[key] = s
        self._sum_bytes += s.nbytes
        while self._sum_bytes > _SUM_CACHE_BYTES and len(self._sum_cache) > 1:
            _, old = self._sum_cache.popitem(last=False)
            self._sum_bytes -= old.nbytes
        return s

    @staticmethod
    def _sorted_isin(needles, haystack):
        """`np.isin(needles, haystack, assume_unique=True)` for SORTED inputs.

        Same boolean, by binary search instead of isin's internal
        sort-and-concatenate. `_ovr_arr` and the exponent column arrays are
        both maintained sorted, so the precondition holds at every call site.
        `np.isin` was 86,207 calls and ~3.0s of a 30.7s virtual build --
        dispatch cost on tiny arrays, not arithmetic.
        """
        out = np.zeros(len(needles), dtype=bool)
        if haystack is None or len(haystack) == 0 or len(needles) == 0:
            return out
        pos = np.searchsorted(haystack, needles)
        inb = pos < len(haystack)
        if inb.any():
            out[inb] = haystack[pos[inb]] == needles[inb]
        return out

    def _ovr_arr(self, r: int):
        """Sorted override cols for row *r*, built lazily after writes."""
        arr = self._ovr_sorted.get(r)
        if arr is None:
            src = self._ovr.get(r)
            if not src:
                return None
            arr = np.fromiter(src.keys(), dtype=np.int64, count=len(src))
            arr.sort()
            self._ovr_sorted[r] = arr
        return arr

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

    # -- reads --------------------------------------------------------------

    def row_sum(self, rows, col_end: Optional[int] = None) -> np.ndarray:
        """``w[rows, :col_end].sum(axis=0)`` under DRIVE SEMANTICS v2.

        float32( memoized f64 base_sum + f64 delta ), delta accumulated by
        one unbuffered `np.add.at` over the selected rows' deviation cells
        in FIXED order: per row (call order), exponent cells ascending, then
        override-only cells ascending. Memoized and cold paths execute the
        same computation, so self-consistency is exact by construction.
        """
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        # SORTED, and that is part of the v2 definition (second amendment,
        # justified in the golden regeneration): a drive is a SUM over rows,
        # mathematically order-insensitive, but the memo key and the f64
        # accumulation were order-sensitive -- and winner ORDER varies across
        # visits to the same state, so eval missed the cache on every step
        # (measured 16.9s vs 5.6s). Sorting normalizes both at once.
        cols = self.n_cols if col_end is None else min(int(col_end),
                                                       self.n_cols)
        out = self._base_sum(rows)[:cols].copy()

        col_parts, delta_parts = [], []
        for r in rows:
            r = int(r)
            entry = self._exp.get(r)
            exp_cols = None
            if entry is not None:
                c, n, eff, raw = entry
                if len(c) and c[-1] >= cols:
                    keep = c < cols
                    c, n, eff, raw = c[keep], n[keep], eff[keep], raw[keep]
                if len(c):
                    final = self._chain(eff, n)
                    col_parts.append(c)
                    delta_parts.append(final.astype(np.float64)
                                       - raw.astype(np.float64))
                    exp_cols = c
            ovr = self._ovr.get(r)
            if ovr:
                oarr = self._ovr_arr(r)
                if oarr[-1] >= cols:
                    oarr = oarr[oarr < cols]
                if exp_cols is not None and len(oarr):
                    oarr = oarr[~self._sorted_isin(oarr, exp_cols)]
                if len(oarr):
                    raws = np.fromiter((ovr[int(c)] for c in oarr),
                                       dtype=np.float64, count=len(oarr))
                    col_parts.append(oarr)
                    delta_parts.append(1.0 - raws)
        if col_parts:
            np.add.at(out, np.concatenate(col_parts),
                      np.concatenate(delta_parts))
        return out.astype(np.float32)

    def _materialize_rows(self, rows: np.ndarray, cols: int) -> np.ndarray:
        """Per-cell exact block (verification / cell reads), unchanged
        semantics: override then chain, matching dense cell-for-cell."""
        block = self._raw_rows(rows)[:, :cols].copy()
        for i, r in enumerate(rows):
            ovr = self._ovr_arr(int(r))
            if ovr is not None and len(ovr):
                idx = ovr[ovr < cols] if ovr[-1] >= cols else ovr
                if len(idx):
                    block[i, idx] = 1.0
            entry = self._exp.get(int(r))
            if entry is not None:
                c, n = entry[0], entry[1]
                if len(c) and c[-1] >= cols:
                    keep = c < cols
                    c, n = c[keep], n[keep]
                if len(c):
                    block[i, c] = self._chain(block[i, c], n)
        return block

    def cell(self, row: int, col: int) -> float:
        block = self._materialize_rows(np.asarray([row], dtype=np.int64),
                                       int(col) + 1)
        return float(block[0, col])

    def column_nnz(self, rows_known: Optional[int] = None) -> np.ndarray:
        """Per-column nonzero counts -- what `_norm_scale` divides by."""
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

    def override_batch(self, rows_list, cols_list) -> None:
        """`override` for a WHOLE recruitment round, in one pass.

        Final state is identical to calling `override(rows_list[i],
        cols_list[i])` for each i in order -- this is a batching of the same
        writes, not a change to what they do.

        WHY. `override` takes ONE column, so recruitment called it once per
        recruited neuron: 19,999 calls in a 6-presentation Z60 build, each
        running a Python loop over its rows and a SCALAR `np.searchsorted`
        per (row, col) pair. That is 813,858 numpy calls whose cost is
        dispatch overhead, not arithmetic -- `override` was 9.5s of a 40s
        virtual build while the rust hash kernel underneath it was 0.9s.

        Batching collapses two axes at once:
          * ONE `_raw_at` for every (row, col) pair in the round, instead of
            one per column;
          * grouping by row, so each row does ONE `searchsorted` over all the
            columns it is overriding rather than one per column. Distinct
            rows are bounded by the source area, so this is ~k times fewer
            numpy calls.

        ORDER IS PRESERVED where it can matter. The group-by uses a STABLE
        sort, so writes within a row keep their original sequence and a
        repeated (row, col) still ends on its last value. Across rows the
        writes touch disjoint dict entries and commute. Removing a row's
        exponent cells before writing all of its overrides (rather than
        interleaving per column) reaches the same final state, and nothing
        reads the intermediate.
        """
        parts_r, parts_c = [], []
        for rows, col in zip(rows_list, cols_list):
            rows = np.asarray(rows, dtype=np.int64)
            if len(rows) == 0:
                continue
            parts_r.append(rows)
            parts_c.append(np.full(len(rows), int(col), dtype=np.int64))
        if not parts_r:
            return
        all_rows = np.concatenate(parts_r)
        all_cols = np.concatenate(parts_c)
        raws = self._raw_at(all_rows, all_cols)

        order = np.argsort(all_rows, kind="stable")
        sr, sc, sv = all_rows[order], all_cols[order], raws[order]
        cuts = np.flatnonzero(np.diff(sr)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(sr)]))
        for lo, hi in zip(starts.tolist(), ends.tolist()):
            r = int(sr[lo])
            cols_r = sc[lo:hi]
            entry = self._exp.get(r)
            if entry is not None and len(entry[0]):
                c = entry[0]
                pos = np.searchsorted(c, cols_r)
                inb = pos < len(c)
                hit = np.zeros(len(cols_r), dtype=bool)
                if inb.any():
                    hit[inb] = c[pos[inb]] == cols_r[inb]
                if hit.any():
                    keep = np.ones(len(c), dtype=bool)
                    keep[pos[hit]] = False
                    self._exp[r] = [a[keep] for a in entry]
            d = self._ovr.setdefault(r, {})
            for cc, vv in zip(cols_r.tolist(), sv[lo:hi].tolist()):
                d[int(cc)] = float(vv)
            self._ovr_sorted.pop(r, None)

    def override(self, rows, col: int) -> None:
        """The recruitment write: assign 1.0, CLOBBERING any history.

        Plasticity precedes expansion within a recruitment round, so dense
        assignment lands on potentiated cells and erases them; the exponent
        entry resets here. Raw base values are point-hashed and stored so
        the drive path never has to."""
        col = int(col)
        rows = np.asarray(rows, dtype=np.int64)
        raws = self._raw_at(rows, np.full(len(rows), col, dtype=np.int64))
        for i, r in enumerate(rows):
            r = int(r)
            entry = self._exp.get(r)
            if entry is not None:
                c = entry[0]
                pos = int(np.searchsorted(c, col))
                if pos < len(c) and c[pos] == col:
                    keep = np.ones(len(c), dtype=bool)
                    keep[pos] = False
                    self._exp[r] = [a[keep] for a in entry]
            self._ovr.setdefault(r, {})[col] = float(raws[i])
            self._ovr_sorted.pop(r, None)

    def bump(self, rows, cols, beta: float) -> None:
        """One Hebbian event on the co-firing block: count += 1 per nonzero.

        Base-zero, non-overridden cells are skipped (multiplying zero is the
        dense behaviour). Raw values come from ONE point-hash over the k x k
        subgrid -- the full-width block this used to hash was the
        recruitment-phase residue."""
        if float(beta) != self.beta:
            raise ValueError(
                f"fiber constructed at beta={self.beta}, bump at {beta}: one "
                f"exponent store cannot carry two growth factors -- densify")
        self._potentiated = True
        rows = np.asarray(rows, dtype=np.int64)
        cols_sorted = np.sort(np.asarray(cols, dtype=np.int64))
        grid_raw = self._raw_at(
            np.repeat(rows, len(cols_sorted)),
            np.tile(cols_sorted, len(rows)),
        ).reshape(len(rows), len(cols_sorted))
        for i, r in enumerate(rows):
            r = int(r)
            raw_i = grid_raw[i]
            ovr = self._ovr.get(r)
            if ovr:
                oarr = self._ovr_arr(r)
                is_ovr = self._sorted_isin(cols_sorted, oarr)
            else:
                is_ovr = np.zeros(len(cols_sorted), dtype=bool)
            alive = (raw_i != 0) | is_ovr
            live = cols_sorted[alive]
            if not len(live):
                continue
            live_raw = raw_i[alive]
            live_eff = np.where(is_ovr[alive], np.float32(1.0),
                                live_raw).astype(np.float32)
            entry = self._exp.get(r)
            if entry is None:
                self._exp[r] = [live.copy(),
                                np.ones(len(live), dtype=np.int64),
                                live_eff, live_raw.astype(np.float32)]
                continue
            c, n, eff, raw = entry
            merged = np.union1d(c, live)
            counts = np.zeros(len(merged), dtype=np.int64)
            m_eff = np.zeros(len(merged), dtype=np.float32)
            m_raw = np.zeros(len(merged), dtype=np.float32)
            old_pos = np.searchsorted(merged, c)
            counts[old_pos] = n
            m_eff[old_pos] = eff
            m_raw[old_pos] = raw
            new_pos = np.searchsorted(merged, live)
            counts[new_pos] += 1
            fresh = counts[new_pos] == 1
            if fresh.any():
                m_eff[new_pos[fresh]] = live_eff[fresh]
                m_raw[new_pos[fresh]] = live_raw[fresh]
            self._exp[r] = [merged, counts, m_eff, m_raw]

    # -- bookkeeping ---------------------------------------------------------

    @property
    def deviations(self) -> int:
        return sum(len(v[0]) for v in self._exp.values())

    @property
    def overrides(self) -> int:
        return sum(len(v) for v in self._ovr.values())

    @property
    def nbytes(self) -> int:
        return (self.deviations * 24 + self.overrides * 16
                + len(self._exp) * 64 + self._sum_bytes)

    def todense(self) -> np.ndarray:
        """The full matrix, for VERIFICATION only."""
        out = np.empty((self.n_rows, self.n_cols), dtype=np.float32)
        chunk = 2048
        for start in range(0, self.n_rows, chunk):
            end = min(start + chunk, self.n_rows)
            rows = np.arange(start, end, dtype=np.int64)
            out[start:end] = self._materialize_rows(rows, self.n_cols)
        return out
