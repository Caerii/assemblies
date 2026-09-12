"""The CSR drive mirror: read-only acceleration of dense-block row sums.

Split out of `_sparse.py` 2026-08-10 as a PURE MOVE (see `_growth.py` for the
protocol). The concern: a cached CSR copy of a dense block, consulted only
with plasticity off, that answers `w[rows].sum(axis=0)` 3-11x faster --
including the cached REJECTION for fibers too dense to benefit, whose absence
was 99%% of an evaluation's runtime (6f779ef).
"""

from __future__ import annotations

import numpy as np

from ._csr_weights import scipy_sparse
from ._virtual_weights import VirtualWeights
from ..backend import get_xp

def _csr_storage_available(xp=None) -> bool:
    """CSR storage needs scipy AND a numpy-backed engine.

    ``scipy.sparse`` is CPU-only, so a CuPy-backed engine must fall back to
    dense, or it raises a confusing "Implicit conversion to a NumPy array is
    not allowed" from deep inside a chunked build.

    Takes the CALLER'S array module. Reading the process-global instead was
    the bug: ``set_backend("cupy")`` is called by the CuPy and CUDA engine
    constructors and never restored, so the answer here used to depend on
    whichever engine had been built most recently anywhere in the process
    rather than on the engine actually asking. Passing ``self._xp`` makes it a
    question about one engine. The ``xp=None`` fallback is for the handful of
    module-level callers that have no engine to hand.

    Checking the backend FIRST also keeps the lazy scipy import off the CuPy
    path entirely: a GPU run never pays for a module it could not use.
    """
    if (xp if xp is not None else get_xp()) is not np:
        return False
    return scipy_sparse() is not None

# A CSR mirror costs one dense pass to build and is amortised over every
# settle round that follows. Below this many cells the gather it saves is
# smaller than the build, at any settle length.
_CSR_MIN_CELLS = 1_000_000
# Above this occupancy CSR stores more bytes than the dense block it mirrors
# (data + indices = 8 bytes per nonzero vs 4 per cell) and gathers no faster.
_CSR_MAX_DENSITY = 0.25



class DriveCacheMixin:
    """Drive-mirror methods of `NumpySparseEngine`; see module docstring."""


    def invalidate_csr_drive(self, src: str = None, tgt: str = None) -> None:
        """Drop CSR mirrors. Call from EVERY path that writes area weights.

        Over-invalidating costs one rebuild; under-invalidating silently
        computes drive from stale weights, which is the failure mode this
        engine is worst at surfacing. When in doubt, clear everything.
        """
        if src is None and tgt is None:
            self._csr_drive.clear()
            return
        for key in [k for k in self._csr_drive
                    if (src is None or k[0] == src)
                    and (tgt is None or k[1] == tgt)]:
            del self._csr_drive[key]

    def _csr_row_sum(self, src: str, tgt: str, w, rows, col_end: int):
        """``w[rows, :col_end].sum(axis=0)`` via a cached CSR mirror, or None.

        WHY.  A materialised area->area block is dense but ~``p`` occupied --
        measured 4.99% at ``p=0.05``, and the sparsity pattern is INVARIANT
        under training (``_apply_plasticity`` does ``w[ix] *= (1+beta)`` and
        normalisation does ``sub * scale``; both are multiplicative, so a zero
        never becomes nonzero). Gathering ``k`` random rows out of a dense
        ``n x n`` block is memory-bound on the 95% that are zero.

        Measured against the dense path, BIT-IDENTICAL at every size, including
        with non-integral trained weights (float32 in, float32 out)::

            n=2000 k=200   0.87 ms -> 0.28 ms    3.2x
            n=4000 k=400   3.18 ms -> 0.29 ms   10.9x
            n=8000 k=800  11.36 ms -> 2.52 ms    4.5x

        WHY IT IS ONLY A MIRROR, NOT THE STORAGE.  Writes here are
        multiplicative updates to a dense SUBMATRIX (``w[np.ix_(rows, cols)]``),
        which CSR does badly. Reads are the hot path and CSR does them well, so
        the representation is chosen per-direction rather than globally.

        SAFETY.  Populated and consulted ONLY while plasticity is off, and
        ``project_into`` clears the whole cache on any plasticity-enabled call.
        A mirror therefore cannot survive a write. The ``id``/``shape`` check
        additionally catches reallocation by growth.

        Returns None when caching is not worthwhile or not safe, in which case
        the caller must use the dense path.
        """
        if isinstance(w, VirtualWeights):
            return None                     # has a native row_sum
        if not _csr_storage_available(self._xp) or w.ndim != 2:
            return None
        # Below this the CSR build (one dense pass) is not amortised by the
        # gather it saves, whatever the settle length.
        if w.shape[0] * w.shape[1] < _CSR_MIN_CELLS:
            return None
        key = (src, tgt)
        entry = self._csr_drive.get(key)
        if entry is None or entry[1] != id(w) or entry[2] != w.shape:
            dens = float(np.count_nonzero(w)) / max(w.size, 1)
            if dens > _CSR_MAX_DENSITY:
                # THE REJECTION IS CACHED TOO, and forgetting to was expensive.
                # `count_nonzero` here scans the WHOLE block; returning None
                # without recording the answer meant the next projection found
                # no entry and scanned it again, re-deriving the same "no"
                # forever. On a word-problem organ at organ_p=0.4 -- above this
                # threshold, so always rejected -- against a fully materialised
                # 20000x4200 state area, that was 314 ms per projection and
                # 62.8s of a 63.4s evaluation: 99% of the run recomputing one
                # boolean. Every fiber denser than _CSR_MAX_DENSITY paid it.
                #
                # `None` in the slot means "measured, not worth it". The
                # id/shape guard re-tests after reallocation, and the existing
                # invalidation clears negative and positive entries alike, so
                # this needs no separate lifecycle.
                self._csr_drive[key] = (None, id(w), w.shape)
                return None
            entry = (scipy_sparse().csr_matrix(w), id(w), w.shape)
            self._csr_drive[key] = entry
        if entry[0] is None:
            return None
        csr = entry[0]
        out = np.asarray(csr[rows].sum(axis=0)).ravel()
        if col_end < w.shape[1]:
            out = out[:col_end]
        return out.astype(np.float32, copy=False)
