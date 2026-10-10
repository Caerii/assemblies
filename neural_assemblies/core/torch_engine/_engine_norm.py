"""norm_init for the torch engine: the in-degree each storage format realizes.

The pricing LAW is core/_pricing.py's, shared with the numpy engine; only extracting a
column's realized in-degree is backend-specific, because it depends on how the fiber is
stored (stimulus vector, CSR, dense).

A mixin of TorchSparseEngine (_engine.py), which owns the state these methods read;
the methods were moved out of _engine.py unchanged."""
from ._torch_ops import torch_ops

from .._pricing import (
    candidate_divisor, inverse_indegree,
)


class NormInitMixin:
    """norm_init's realized in-degree, per storage format."""

    # -- norm_init: read-time incoming-weight normalization -----------------
    # The LAW lives in core/_pricing.py and is shared with the numpy engine.
    # Only in-degree EXTRACTION is backend-specific and stays here, because how
    # you count a column's realized synapses depends on the storage format.
    #
    # This used to be a hand-written "on-device mirror" of the numpy code, and
    # it drifted: two fixes to the numpy copy never arrived here, so the two
    # engines priced k-WTA differently. See core/_pricing.py for the measured
    # divergence table.

    def _norm_candidate_divisor(self, tgt_n: int, input_sizes=None,
                                src_pops=None) -> float:
        """Scale for sampled (unmaterialized) candidate drive.

        See `core._pricing.candidate_divisor`. With ``src_pops`` omitted this
        falls back to ``tgt_n * p``, which is correct only when every source
        population equals the target's ``n``.
        """
        return candidate_divisor(self.p, tgt_n, input_sizes, src_pops)

    def _norm_scale_stim(self, conn, n_pre, stim_size, needed):
        """1/d_j for a 1-D stimulus fiber (numpy _norm_scale, 1-D branch)."""
        w = conn.weights
        if w is None or w.numel() == 0:
            return None
        cols = int(min(needed, int(w.numel())))
        if cols <= 0:
            return None
        # The stored value IS the observed in-degree from the stimulus, but
        # only until plasticity scales it -- snapshot each column the first
        # time it is seen (a fresh column is read before it is potentiated).
        base = getattr(conn, "_norm_deg_base", None)
        have = 0 if base is None else int(base.numel())
        if have < cols:
            add = w[have:cols].detach().float()
            base = add if (base is None or have == 0) else torch_ops.cat([base, add])
            conn._norm_deg_base = base
        assert base is not None
        deg = base[:cols]
        return inverse_indegree(deg, n_pre, stim_size, self.p)

    def _norm_scale_area(self, csr, n_pre, rows_known, needed):
        """1/d_j for a 2-D area fiber; see `core._pricing.inverse_indegree`."""
        cols = int(min(needed, int(csr._ncols)))
        if cols <= 0:
            return None
        rows = min(int(rows_known), int(csr._nrows))
        deg = csr.column_indegree(cols, nrows=rows)
        return inverse_indegree(deg, n_pre, rows, self.p)

    def _norm_scale_dense(self, weights, n_pre, needed):
        """1/d_j for a DENSE explicit-source fiber.

        Every row of a dense connectome exists, so there is no unknown-row term
        and the in-degree is just the column count of present synapses. This
        path had NO normalization at all until the law was unified: an explicit
        source's incumbents were delivered as raw counts while candidates were
        divided by n*p, leaving the two populations a factor of n*p apart, which
        sealed the target at k. See `_explicit_src_norm_enabled` in the numpy
        engine for the measured numbers.
        """
        cols = int(min(needed, int(weights.shape[1])))
        if cols <= 0:
            return None
        deg = (weights[:, :cols] > 0).sum(dim=0).float()
        return inverse_indegree(deg, n_pre, n_pre, self.p)
