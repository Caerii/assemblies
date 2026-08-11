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
            growth-point-keyed RNG). An override can only land on a
            FIRST-TIME winner's column, which has never been potentiated, so
            override-then-potentiate is the only event order. Sparse:
            <= alloc (~k) cells per recruited neuron.
  CHAIN     the Hebbian event sequence. The dense engine does
            ``w *= (1 + beta)`` in float32 THEN clips, per event, so byte
            identity requires replaying n float32 multiply-clip rounds --
            NOT a float64 closed-form power, which differs in ulps and the
            fingerprint digests at round(9), finer than float32 resolution.
  n         an integer co-firing count per cell. Multiplicative updates
            never create a nonzero, so the SPARSITY pattern is
            base-plus-overrides, never the chain.

WHAT THE REPRESENTATION REFUSES, structurally (`supports`):

  * homeostatic synaptic scaling -- per-column rescales interleave with
    clips and do not factor into any per-cell summary;
  * the legacy (non-content-addressed) initializer -- its values come from a
    shared RNG stream a positional function cannot reproduce;
  * a CHANGED fiber beta after potentiation has begun -- one exponent count
    cannot carry two growth factors ([[same-name-two]]). `bump` raises; the
    engine's escape hatch is densify-and-swap, never silent mixing.

Memory: the S5 organ's arc<->state pair is 2.69 GB dense; here it is the
override and exponent stores, ~10-20 MB. Growth costs nothing: `resize` just
raises the logical bounds, where the dense path's reallocation (new buffer
live beside the old) is the 2x peak that OOM'd two studies today.

Unwired until the engine integration lands behind
`test_connectome_representation_fingerprint` -- whose golden, regenerated on
the fixed `_norm_scale` engine, is the byte-identity target.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np

from ._seeding import hash_area_weights


class VirtualWeights:
    """One area->area fiber: hash base + sparse overrides + sparse exponents.

    All coordinates are ABSOLUTE row/column indices into the fiber's logical
    block -- the same indices the dense buffer would use -- so the engine's
    existing compact bookkeeping maps onto this unchanged.

    Per-ROW dicts, because every hot operation (drive, bump) touches ~k rows:
    `row_sum` then walks only the selected rows' deviations (~10^2 entries)
    instead of the fiber's full store (~10^6).
    """

    __slots__ = ("n_rows", "n_cols", "pair_seed", "p", "beta",
                 "w_lo", "w_hi", "inhibitory_prob", "inhibitory_weight",
                 "_exp", "_ovr", "_potentiated")

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
        #: row -> {col: event count}
        self._exp: Dict[int, Dict[int, int]] = {}
        #: row -> {col} forced to 1.0 at recruitment
        self._ovr: Dict[int, Set[int]] = {}
        self._potentiated = False

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

    def resize(self, n_rows: int, n_cols: int) -> None:
        """Growth. The dense path's 2x-peak reallocation becomes two ints."""
        if n_rows < self.n_rows or n_cols < self.n_cols:
            raise ValueError("virtual fibers never shrink")
        self.n_rows, self.n_cols = int(n_rows), int(n_cols)

    # -- base ---------------------------------------------------------------

    def _base_row(self, row: int, col_end: Optional[int] = None) -> np.ndarray:
        cols = self.n_cols if col_end is None else col_end
        out = hash_area_weights(row, row + 1, 0, cols, self.pair_seed,
                                self.p, self.inhibitory_prob,
                                self.inhibitory_weight)[0]
        ovr = self._ovr.get(row)
        if ovr:
            idx = [c for c in ovr if c < cols]
            if idx:
                out[idx] = 1.0
        return out

    def _chain(self, values: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Replay the dense engine's per-event float32 multiply-then-clip.

        Vectorised by round: cells with count >= r take round r. Bounded by
        the maximum event count, ~presentations in practice.
        """
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
        """``w[rows, :col_end].sum(axis=0)`` -- the drive kernel.

        Regenerates the k selected base rows (k x n_cols cells, against the
        dense path's full-block traversal) and adds each row's deviation
        surplus from its own small dict.
        """
        rows = np.asarray(rows, dtype=np.int64)
        cols = self.n_cols if col_end is None else min(int(col_end),
                                                       self.n_cols)
        out = np.zeros(cols, dtype=np.float64)
        for r in rows:
            base = self._base_row(int(r), cols)
            out += base
            dev = self._exp.get(int(r))
            if dev:
                idx = np.fromiter((c for c in dev if c < cols),
                                  dtype=np.int64)
                if len(idx):
                    counts = np.fromiter((dev[int(c)] for c in idx),
                                         dtype=np.int64)
                    chained = self._chain(base[idx], counts)
                    out[idx] += chained.astype(np.float64) - base[idx]
        return out.astype(np.float32)

    def cell(self, row: int, col: int) -> float:
        base = float(self._base_row(int(row), int(col) + 1)[int(col)])
        n = self._exp.get(int(row), {}).get(int(col), 0)
        if n == 0:
            return base
        return float(self._chain(np.asarray([base], dtype=np.float32),
                                 np.asarray([n]))[0])

    def column_nnz(self, rows_known: Optional[int] = None) -> np.ndarray:
        """Per-column nonzero counts -- what `_norm_scale` divides by.

        A property of base-plus-overrides only: the chain never creates a
        nonzero. Regenerated in chunks; overrides that landed on base-zero
        cells add one each.
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
                ovr = self._ovr.get(r)
                if ovr:
                    for c in ovr:
                        if block[r - start, c] == 0:
                            counts[c] += 1
        return counts

    # -- writes -------------------------------------------------------------

    def override(self, rows, col: int) -> None:
        """The recruitment write: force cells to 1.0 at a fresh column.

        Refused once the column has been potentiated -- in the dense engine
        that order cannot occur (a first-time winner's column has no events),
        so hitting this guard means the caller's invariant broke, not ours.
        """
        col = int(col)
        for r in np.asarray(rows, dtype=np.int64):
            if self._exp.get(int(r), {}).get(col):
                raise ValueError(
                    f"override at ({int(r)}, {col}) after potentiation: "
                    f"the dense engine cannot produce this order")
            self._ovr.setdefault(int(r), set()).add(col)

    def bump(self, rows, cols, beta: float) -> None:
        """One Hebbian event on the co-firing block: count += 1 per nonzero.

        Base-zero, non-overridden cells are skipped -- multiplying zero is
        what the dense engine does, and an exponent there would claim a
        synapse that does not exist.
        """
        if float(beta) != self.beta:
            raise ValueError(
                f"fiber constructed at beta={self.beta}, bump at {beta}: one "
                f"exponent store cannot carry two growth factors -- densify")
        self._potentiated = True
        cols = np.asarray(cols, dtype=np.int64)
        for r in np.asarray(rows, dtype=np.int64):
            base = self._base_row(int(r))
            live = cols[base[cols] != 0]
            if len(live):
                row_exp = self._exp.setdefault(int(r), {})
                for c in live:
                    row_exp[int(c)] = row_exp.get(int(c), 0) + 1

    # -- bookkeeping ---------------------------------------------------------

    @property
    def deviations(self) -> int:
        return sum(len(d) for d in self._exp.values())

    @property
    def overrides(self) -> int:
        return sum(len(s) for s in self._ovr.values())

    @property
    def nbytes(self) -> int:
        return (self.deviations + self.overrides) * 24     # approximate

    def todense(self) -> np.ndarray:
        """The full matrix, for VERIFICATION only."""
        out = np.empty((self.n_rows, self.n_cols), dtype=np.float32)
        for r in range(self.n_rows):
            base = self._base_row(r)
            dev = self._exp.get(r)
            if dev:
                idx = np.fromiter(dev.keys(), dtype=np.int64)
                counts = np.fromiter(dev.values(), dtype=np.int64)
                base = base.copy()
                base[idx] = self._chain(base[idx], counts)
            out[r] = base
        return out
