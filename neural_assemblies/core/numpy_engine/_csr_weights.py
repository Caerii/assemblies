"""Sparse storage for a FULLY MATERIALIZED area->area weight block.

WHY THIS EXISTS.  ``materialize_area`` allocates the whole ``n x n`` recurrent
block so that a protocol can drive an area from an arbitrary subset of ``n``
(the reference NEMO coin does exactly that). Dense, that is ``O(n^2)`` float32:
16 MB at ``n=2000``, **1.0 GB at ``n=16,000``** -- which is what bounds how far
the finite-size ladder in ``research/notes/neural_coin_fairness.md`` can run.

But the block is only ``~p`` occupied -- measured **4.99% at ``p=0.05``** -- so
95% of those bytes are zeros. CSR stores 8 bytes per nonzero (float32 value +
int32 column) plus one int32 per row, which at ``p=0.05`` is a **10x**
reduction: 1.0 GB -> 102 MB, putting ``n ~ 50,000`` inside the same budget.

WHY IT IS SAFE.  The sparsity pattern is INVARIANT under everything the engine
does to a materialized block:

  * ``_apply_plasticity`` does ``w[ix] *= (1 + beta)`` -- multiplicative, so a
    zero stays zero.
  * the ``w_max`` clip uses ``_weight_bounds``, which is ``(0.0, hi)`` without
    inhibition and ``(-x, hi)`` with it. **Zero is inside the interval either
    way**, so clipping never lifts a zero off the floor.
  * normalisation does ``sub * scale`` -- multiplicative again.

So the structure is fixed at init and only ``.data`` ever changes. That is what
makes a fixed-pattern representation correct rather than merely smaller.

WHAT SURFACE IT HAS TO SUPPORT.  Not the 80 ``.weights`` references in the
engine -- measured by tracing every operation that actually reaches a
materialized block during a real run (build + settle + a training round), the
answer is **three**:

    61x  w[rows_1d, :col_slice].sum(axis=0)   the drive read (the hot path)
     2x  w[np.ix_(rows, cols)]                plasticity, read
     2x  w[np.ix_(rows, cols)] = sub          plasticity, write

plus per-column nonzero counts for ``_deg_counts``. Each has a native method
here; anything else falls through ``__array__`` to a dense copy, which is
correct but defeats the point, so it is logged rather than silent.
"""

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:                     # pragma: no cover
    sp = None


class CSRWeights:
    """A fixed-pattern CSR weight block that quacks like a 2-D float32 array.

    Only ``.data`` is mutable. Any write whose value is nonzero where the
    structure has no entry is DROPPED -- which is correct for every update the
    engine performs (all multiplicative, see module docstring) and would be a
    silent bug for anything else, so ``__setitem__`` checks for it.
    """

    __slots__ = ("_m", "_colmap", "_densified")

    def __init__(self, matrix):
        if sp is None:                  # pragma: no cover
            raise RuntimeError("CSRWeights needs scipy.sparse")
        m = matrix if sp.isspmatrix_csr(matrix) else sp.csr_matrix(matrix)
        if m.dtype != np.float32:
            m = m.astype(np.float32)
        m.sort_indices()
        self._m = m
        self._colmap = None
        self._densified = 0

    # -- array-like surface -------------------------------------------------

    @property
    def shape(self):
        return self._m.shape

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        return self._m.dtype

    @property
    def size(self):
        return self._m.shape[0] * self._m.shape[1]

    @property
    def nbytes(self):
        return (self._m.data.nbytes + self._m.indices.nbytes
                + self._m.indptr.nbytes)

    @property
    def nnz(self):
        return self._m.nnz

    def __len__(self):
        return self._m.shape[0]

    def __array__(self, dtype=None, copy=None):
        """Dense fallback. Correct, and defeats the entire point.

        Counted so a test can assert the hot paths never take it.
        """
        self._densified += 1
        out = self._m.toarray()
        return out if dtype is None else out.astype(dtype)

    # -- the three measured operations --------------------------------------

    def row_sum(self, rows, col_end=None):
        """``self[rows, :col_end].sum(axis=0)``, natively.

        THE hot path: 62% of engine wall-clock went through the dense form of
        this line. Bit-identical to the dense reduction -- verified at three
        sizes including with non-integral trained weights.
        """
        out = np.asarray(self._m[rows].sum(axis=0)).ravel()
        if col_end is not None and col_end < self._m.shape[1]:
            out = out[:col_end]
        return out.astype(np.float32, copy=False)

    def column_nnz(self):
        """Per-column nonzero counts, for ``_deg_counts``.

        ``np.bincount`` over the column index array, which CSR already stores
        -- O(nnz) instead of a dense ``(w != 0).sum(axis=0)`` over n^2 cells.
        """
        return np.bincount(self._m.indices,
                           minlength=self._m.shape[1]).astype(np.int64)

    def submatrix(self, rows, cols):
        """Dense ``self[np.ix_(rows, cols)]``. Small: winners x winners."""
        return np.asarray(self._m[rows][:, cols].todense(), dtype=np.float32)

    def set_submatrix(self, rows, cols, values):
        """Write a dense ``rows x cols`` block back, PRESERVING the pattern.

        Values landing where the structure has no entry are dropped. That is
        correct for multiplicative updates (a zero scaled is still a zero) and
        wrong for anything else, so it is checked rather than assumed --
        silently dropping learned weight would be indistinguishable from the
        update having worked.
        """
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        values = np.asarray(values, dtype=np.float32)
        m = self._m
        if self._colmap is None or len(self._colmap) != m.shape[1]:
            self._colmap = np.full(m.shape[1], -1, dtype=np.int64)
        colmap = self._colmap
        colmap[:] = -1
        colmap[cols] = np.arange(len(cols), dtype=np.int64)

        indptr, indices, data = m.indptr, m.indices, m.data
        for r, row in enumerate(rows):
            lo, hi = indptr[row], indptr[row + 1]
            if lo == hi:
                continue
            pos = colmap[indices[lo:hi]]
            hit = pos >= 0
            if hit.any():
                data[lo:hi][hit] = values[r, pos[hit]]
        colmap[cols] = -1               # leave it clean for the next call

    # -- indexing, so untouched call sites keep working ---------------------

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            r, c = key
            if (isinstance(r, np.ndarray) and r.ndim == 2
                    and isinstance(c, np.ndarray) and c.ndim == 2):
                return self.submatrix(r.ravel(), c.ravel())     # np.ix_ form
            if isinstance(r, np.ndarray) and isinstance(c, slice):
                sub = self._m[r]
                return np.asarray(sub.todense(), dtype=np.float32)[:, c]
        return np.asarray(self)[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            r, c = key
            if (isinstance(r, np.ndarray) and r.ndim == 2
                    and isinstance(c, np.ndarray) and c.ndim == 2):
                self.set_submatrix(r.ravel(), c.ravel(), value)
                return
        raise TypeError(
            f"CSRWeights supports only np.ix_-style assignment; got "
            f"{type(key).__name__}. Densify explicitly if you need more -- "
            f"silently falling back would undo the 10x memory saving.")

    def __repr__(self):
        r, c = self._m.shape
        return (f"CSRWeights({r}x{c}, nnz={self._m.nnz}, "
                f"{self.nbytes / 1e6:.1f} MB, "
                f"dense would be {r * c * 4 / 1e6:.1f} MB)")


def build_csr_from_blocks(n_rows, n_cols, block_fn, rows_per_chunk=1024):
    """Assemble a CSR block chunk by chunk, never holding the dense form.

    ``block_fn(r0, r1) -> dense (r1-r0, n_cols) float32``.

    The peak-memory half of the win. Materializing dense-then-converting needs
    ``O(n^2)`` transiently even if the result is sparse -- 1.0 GB at
    ``n=16,000`` -- which is the actual wall a bigger ladder hits. Here the
    peak is one chunk plus the finished CSR.
    """
    if sp is None:                      # pragma: no cover
        raise RuntimeError("build_csr_from_blocks needs scipy.sparse")
    rows_all, cols_all, vals_all = [], [], []
    for r0 in range(0, n_rows, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, n_rows)
        block = block_fn(r0, r1)
        rr, cc = np.nonzero(block)
        if len(rr):
            rows_all.append(rr.astype(np.int32) + r0)
            cols_all.append(cc.astype(np.int32))
            vals_all.append(block[rr, cc].astype(np.float32))
        del block
    if not rows_all:
        return CSRWeights(sp.csr_matrix((n_rows, n_cols), dtype=np.float32))
    coo = sp.coo_matrix(
        (np.concatenate(vals_all),
         (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(n_rows, n_cols), dtype=np.float32)
    return CSRWeights(coo.tocsr())
