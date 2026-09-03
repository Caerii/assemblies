"""CSR-format sparse connectivity matrix for area->area connections.

Stores weights as Compressed Sparse Row — three arrays (crow, col,
val) — using O(nnz) memory instead of O(rows x cols).  At typical
connection probability p=0.0005 this is ~2000x smaller than dense.
"""

import torch

from .._homeostasis import column_scale

from ._hash import WEIGHT_DTYPE, csr_flat_indices


class CSRConn:
    """CSR-format area->area connectivity on GPU."""

    sparse = True  # protocol flag: not a dense Connectome bridge

    def __init__(self, device='cuda'):
        self._device = device
        self._nrows = 0
        self._ncols = 0
        self._log_rows = 0   # hash-initialised row extent
        self._log_cols = 0   # hash-initialised col extent
        self._crow = torch.zeros(1, dtype=torch.int64, device=device)
        self._col = torch.empty(0, dtype=torch.int32, device=device)
        self._val = torch.empty(0, dtype=WEIGHT_DTYPE, device=device)

    @property
    def nnz(self):
        return len(self._col)

    # -- Input accumulation (project_into hot path) -------------------------

    def accumulate_rows(self, row_indices, out_size):
        """Sum selected rows -> dense float32 vector of *out_size*."""
        result = torch.zeros(out_size, dtype=torch.float32,
                             device=self._device)
        if self.nnz == 0 or len(row_indices) == 0:
            return result
        flat_idx = csr_flat_indices(
            self._crow, row_indices, self._nrows, self._device)
        if flat_idx is None:
            return result
        sel_cols = self._col[flat_idx].long()
        sel_vals = self._val[flat_idx].float()
        valid = sel_cols < out_size
        if not valid.all():
            sel_cols = sel_cols[valid]
            sel_vals = sel_vals[valid]
        result.scatter_add_(0, sel_cols, sel_vals)
        return result

    # -- Hebbian plasticity -------------------------------------------------

    def hebbian_update(self, src_winners, tgt_winners, beta, w_max):
        """Multiply entries at (src, tgt) intersections by (1+beta)."""
        if self.nnz == 0 or len(src_winners) == 0 or len(tgt_winners) == 0:
            return
        flat_idx = csr_flat_indices(
            self._crow, src_winners, self._nrows, self._device)
        if flat_idx is None:
            return
        sel_cols = self._col[flat_idx]
        col_mask = torch.isin(sel_cols.int(), tgt_winners.int())
        update_idx = flat_idx[col_mask]
        if len(update_idx) > 0:
            updated = self._val[update_idx].float() * (1 + beta)
            if w_max is not None and w_max > 0:
                updated = updated.clamp(max=w_max)
            self._val[update_idx] = updated.to(WEIGHT_DTYPE)

    # -- Expansion (add new rows / columns) ---------------------------------

    def expand(self, needed_rows, needed_cols, new_r, new_c, new_v):
        """Merge new COO entries into the CSR and rebuild."""
        # Convert existing CSR -> COO
        if self.nnz > 0:
            lengths = self._crow[1:] - self._crow[:-1]
            old_r = torch.repeat_interleave(
                torch.arange(self._nrows, dtype=torch.int32,
                             device=self._device),
                lengths.int())
            old_c = self._col
            old_v = self._val
        else:
            old_r = torch.empty(0, dtype=torch.int32, device=self._device)
            old_c = torch.empty(0, dtype=torch.int32, device=self._device)
            old_v = torch.empty(0, dtype=WEIGHT_DTYPE, device=self._device)

        all_r = torch.cat([old_r, new_r]) if len(new_r) > 0 else old_r
        all_c = torch.cat([old_c, new_c]) if len(new_c) > 0 else old_c
        all_v = torch.cat([old_v, new_v]) if len(new_v) > 0 else old_v

        self._rebuild_csr(needed_rows, needed_cols, all_r, all_c, all_v)

    def _rebuild_csr(self, nrows, ncols, rows, cols, vals):
        """Build CSR from COO, deduplicating (last value wins)."""
        if len(rows) > 0:
            nrows = max(nrows, int(rows.long().max().item()) + 1)
            ncols = max(ncols, int(cols.long().max().item()) + 1)
            valid = (
                (rows >= 0) & (rows < nrows)
                & (cols >= 0) & (cols < ncols)
            )
            if not bool(valid.all()):
                rows = rows[valid]
                cols = cols[valid]
                vals = vals[valid]
        self._nrows = nrows
        self._ncols = ncols
        if len(rows) == 0:
            self._crow = torch.zeros(
                nrows + 1, dtype=torch.int64, device=self._device)
            self._col = torch.empty(0, dtype=torch.int32, device=self._device)
            self._val = torch.empty(0, dtype=WEIGHT_DTYPE, device=self._device)
            return

        # Sort by (row, col); stable so last duplicate wins
        sort_key = rows.long() * ncols + cols.long()
        order = sort_key.argsort(stable=True)
        rows = rows[order]; cols = cols[order]; vals = vals[order]
        sk = sort_key[order]

        # Keep last occurrence of each (row, col) pair
        unique = torch.ones(len(sk), dtype=torch.bool, device=self._device)
        unique[:-1] = sk[:-1] != sk[1:]
        rows = rows[unique]; cols = cols[unique]; vals = vals[unique]

        # Build crow from row counts
        self._crow = torch.zeros(
            nrows + 1, dtype=torch.int64, device=self._device)
        if len(rows) > 0:
            counts = torch.zeros(
                nrows, dtype=torch.int64, device=self._device)
            counts.scatter_add_(
                0, rows.long(),
                torch.ones(len(rows), dtype=torch.int64,
                           device=self._device))
            self._crow[1:] = counts.cumsum(0)
        self._col = cols.int()
        self._val = vals

    # -- Column in-degree (norm_init) ---------------------------------------

    def column_indegree(self, ncols, nrows=None):
        """Per-column count of present synapses over the first ``nrows`` rows.

        This is the realized in-degree ``d_j`` each materialized target column
        has accumulated so far (``(weights[:nrows] != 0).sum(axis=0)`` in the
        dense picture). norm_init divides a column's summed drive by ``d_j`` to
        cancel its degree advantage; see `core._pricing.inverse_indegree`.
        Counts are potentiation-invariant (present synapses, not summed
        weights), matching the reference's take-it-once-at-init semantics.

        ``nrows`` IS NOT OPTIONAL IN EFFECT.  The caller pairs this count with
        an unknown-row term ``p * (n_pre - rows_known)`` covering the rows that
        have not materialized yet.  Counting rows beyond ``rows_known`` here
        charges those same rows twice, inflating ``d_j``, which shrinks ``1/d_j``
        and under-drives every incumbent -- so candidates win and the area
        recruits without bound.  Measured with this bound missing (k=100,
        p=0.05, n_src=1000 -> n_tgt=10000, 15 rounds): w=464 and stability 0.77,
        against w=154 and stability 1.000 for the numpy engine, which has always
        passed the bound.  Defaults to all stored rows only for callers that
        genuinely want the full count.
        """
        deg = torch.zeros(ncols, dtype=torch.float32, device=self._device)
        if self.nnz > 0:
            cols = self._col.long()
            valid = cols < ncols
            if nrows is not None and int(nrows) < self._nrows:
                # Row r owns the flat slice [crow[r], crow[r+1]); everything at
                # or past crow[nrows] belongs to a row we must not count.
                cutoff = int(self._crow[int(nrows)].item())
                bound = torch.zeros_like(valid)
                bound[:cutoff] = True
                valid = valid & bound
            if not bool(valid.all()):
                cols = cols[valid]
            deg.scatter_add_(
                0, cols, torch.ones(len(cols), dtype=torch.float32,
                                    device=self._device))
        return deg

    # -- Homeostatic synaptic scaling ---------------------------------------

    def scale_columns(self, cols, setpoint, nrows=None, eps=1e-12):
        """Renormalize the TOUCHED columns' stored mass to *setpoint*.

        The CSR half of `NumpySparseEngine._scale_columns_now`: after a
        Hebbian write, each just-touched column's total incoming weight over
        the first ``nrows`` (materialized) rows is rescaled to the fiber's
        setpoint, so learning redistributes a fixed budget instead of
        inflating the total. Zero entries do not exist here, so the stored
        sum IS the dense column sum over materialized rows.

        ``nrows`` carries the same bound as `column_indegree`: rows past the
        source's logical extent are other bookkeeping, not population, and
        counting them would inflate the mass and silently under-normalize.

        Sums and scales are computed in float32 and cast back to
        WEIGHT_DTYPE, matching `hebbian_update`'s precision discipline.
        """
        if self.nnz == 0 or self._ncols == 0 or len(cols) == 0:
            return
        cols = cols.long()
        cols = cols[(cols >= 0) & (cols < self._ncols)]
        if len(cols) == 0:
            return
        colmask = torch.zeros(self._ncols, dtype=torch.bool,
                              device=self._device)
        colmask[cols] = True
        entry_mask = colmask[self._col.long()]
        if nrows is not None and int(nrows) < self._nrows:
            # Row r owns the flat slice [crow[r], crow[r+1]); everything at
            # or past crow[nrows] belongs to a row we must not touch.
            cutoff = int(self._crow[int(nrows)].item())
            bound = torch.zeros_like(entry_mask)
            bound[:cutoff] = True
            entry_mask &= bound
        idx = entry_mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return
        ecols = self._col[idx].long()
        evals = self._val[idx].float()
        sums = torch.zeros(self._ncols, dtype=torch.float32,
                           device=self._device)
        sums.scatter_add_(0, ecols, evals)
        factors = column_scale(sums, setpoint, eps=eps)
        self._val[idx] = (evals * factors[ecols]).to(WEIGHT_DTYPE)

    # -- Column normalisation -----------------------------------------------

    def normalize_columns(self, eps=1e-8):
        """Column-normalize so each column sums to 1.0."""
        if len(self._val) == 0 or self._ncols == 0:
            return
        sums = torch.zeros(self._ncols, dtype=torch.float32,
                           device=self._device)
        sums.scatter_add_(0, self._col.long(), self._val.float())
        sums = sums.clamp(min=eps)
        factors = sums[self._col.long()]
        self._val = (self._val.float() / factors).to(WEIGHT_DTYPE)

    # -- Reset --------------------------------------------------------------

    def reset(self):
        """Clear all entries and dimensions."""
        self._nrows = 0
        self._ncols = 0
        self._log_rows = 0
        self._log_cols = 0
        self._crow = torch.zeros(1, dtype=torch.int64, device=self._device)
        self._col = torch.empty(0, dtype=torch.int32, device=self._device)
        self._val = torch.empty(0, dtype=WEIGHT_DTYPE, device=self._device)


#: Above this density a fiber stores dense. The numpy engine's CSR mirror
#: rejects at 0.25 for the same reason in the other direction
#: (`_drive_cache._CSR_MAX_DENSITY`): near half-occupancy, index arrays cost
#: as much as the zeros they avoid, and every structural change pays an
#: O(nnz log nnz) rebuild.
DENSE_MIN_P = 0.25

#: A CSR fiber whose entry count crosses this migrates to dense at its next
#: growth, provided the dense form fits DENSIFY_MAX_BYTES. Density is not
#: the only way CSR turns pathological: a LOW-density fiber that grows every
#: recruitment step (an area self-fiber during training) pays the full
#: O(nnz log nnz) rebuild each step. Measured on the Z60 word-problem organ:
#: the 20000x20000 p=0.05 arc self-fiber put training back inside
#: `_rebuild_csr` for minutes after the organ fibers had already gone dense.
DENSIFY_MIN_NNZ = 2_000_000

#: Never densify past this (bfloat16 bytes). An n=100k self-fiber stays CSR
#: and eats the rebuild cost rather than OOMing the device.
DENSIFY_MAX_BYTES = 1_500_000_000


def densify(csr, device='cuda', max_rows=None, max_cols=None):
    """Convert a CSRConn to a TorchDenseConn with identical logical content."""
    dense = TorchDenseConn(device=device, max_rows=max_rows,
                           max_cols=max_cols)
    dense._w = dense._new_buffer(csr._nrows, csr._ncols)
    if csr.nnz > 0:
        lengths = csr._crow[1:] - csr._crow[:-1]
        rows = torch.repeat_interleave(
            torch.arange(csr._nrows, dtype=torch.int64, device=device),
            lengths)
        # `.to(DTYPE)` like EVERY other write into `_w`. CSR stores
        # bfloat16 and dense stores float32 -- deliberately, because
        # bf16's 7-bit mantissa randomised the Z60 readout margin -- so
        # the conversion is the whole point of the boundary and this was
        # the one write site in the file that skipped it. Torch refuses
        # the mismatch outright (`Index put requires the source and
        # destination dtypes match`), so every densify on a trained CSR
        # fiber raised.
        dense._w[rows, csr._col.long()] = csr._val.to(dense.DTYPE)
    dense._log_rows = csr._log_rows
    dense._log_cols = csr._log_cols
    dense._ext_rows = csr._nrows
    dense._ext_cols = csr._ncols
    return dense


class TorchDenseConn:
    """Dense-tensor fiber for HIGH-DENSITY area->area connections.

    WHY THIS EXISTS. `CSRConn` was built for p ~ 0.0005, where three index
    arrays are ~2000x smaller than dense -- and its `expand` REBUILDS the
    whole matrix (COO concat + stable argsort + dedup) on every structural
    growth. At an organ fiber's p=0.5 that is a ~40M-entry rebuild per
    recruitment step: a Z60 word-problem training run sat 13+ minutes at
    100% CPU inside `_rebuild_csr` with the GPU idle (py-spy confirmed the
    stack). Dense at that density is 20000x4200 bfloat16 = 168 MB, growth
    is amortised-doubling buffer copy with NO index maintenance, and every
    op (row-gather drive, submatrix Hebbian, column scaling) is one native
    GPU kernel. The same per-direction representation choice the numpy
    engine makes, decided by the same density threshold.

    Selected in `add_connectivity` when the fiber's p >= DENSE_MIN_P --
    which is guaranteed to happen before any traffic (structural-precedence
    guard), so no CSR content ever needs migrating.

    Duck-types CSRConn's engine-facing surface: `sparse` flag, `nnz`,
    `_nrows/_ncols/_log_rows/_log_cols`, `accumulate_rows`,
    `hebbian_update`, `expand`, `column_indegree`, `scale_columns`,
    `normalize_columns`, `reset`.
    """

    sparse = True  # protocol flag: not a dense explicit Connectome bridge

    #: float32, NOT the CSR store's bfloat16. Dense fibers hold the trained
    #: organs, where the readout margin is a few units of drive out of ~60
    #: and training compounds multiply-then-renormalize for tens of rounds --
    #: bf16's 7-bit mantissa randomized those margins (Z60 trajectory acc
    #: 0.040 in bf16 vs 0.320 in f32, numpy f32 reference 0.560, same seed).
    #: CSR keeps bf16: its fibers are the low-density ambient ones where the
    #: memory halving is the point and per-entry precision is not marginal.
    DTYPE = torch.float32

    def __init__(self, device='cuda', max_rows=None, max_cols=None):
        self._device = device
        # LAYOUT FOLLOWS SHAPE, exactly as on the numpy engine (see
        # `_sparse._scale_columns_now`): a scaled fiber pays a column
        # gather/scatter over rows x k and a drive row-gather over k x cols,
        # and whichever slice is larger should own the contiguity. On GPU the
        # penalty is uncoalescing rather than cache misses -- `index_select(1,
        # cols)` on a row-major (20000, 4200) fiber reads each column with a
        # 16.8 KB stride, so every element is its own memory transaction.
        # Measured on this exact shape: 1195 us row-major vs 537 us
        # column-major (2.2x) for the scaling op.
        #
        # Layout preserves logical [i, j], so this is byte-identical -- the
        # same claim proven end-to-end on numpy (identical census and
        # connectome crc across layouts); it only moves time.
        self._col_major = (max_rows is not None and max_cols is not None
                           and int(max_rows) > int(max_cols))
        self._w = self._new_buffer(0, 0)
        # Physical-capacity caps: the areas' n. Amortised doubling MUST stop
        # here, like the numpy engine's `min(max(needed, phys*2, ...),
        # max(n, needed))` -- uncapped, per-projection growth doubled a
        # 168 MB fiber to a 35.9 GiB allocation request (44.9 GiB "allocated"
        # through CUDA unified-memory spill) before OOMing the device.
        self._max_rows = max_rows
        self._max_cols = max_cols
        self._log_rows = 0   # hash-initialised row extent
        self._log_cols = 0   # hash-initialised col extent
        # Logical CONTENT extent (rises with growth requests and scattered
        # entries), distinct from the physical buffer, which doubles for
        # amortisation. `_nrows/_ncols` MUST report this, not the buffer
        # shape: `_expand_connectomes` takes max(needed, conn._nrows), so a
        # physical-capacity report leaks padding into the needed extent and
        # `_hash_grow_parts` then hash-fills the ENTIRE padded rectangle
        # (~42M entries) on EVERY projection -- measured as a 20-minute
        # presentation with one CPU core pegged. CSRConn never had the
        # hazard because its matrix has no padding.
        self._ext_rows = 0
        self._ext_cols = 0

    @property
    def _nrows(self):
        return self._ext_rows

    @property
    def _ncols(self):
        return self._ext_cols

    @property
    def nnz(self):
        """Logical-content marker (cheap), not a stored-entry count.

        Every engine consumer uses ``nnz`` as an is-there-anything-here
        guard; a true nonzero count would scan 168 MB to answer a boolean.
        """
        return self._log_rows * self._log_cols

    def _new_buffer(self, rows, cols):
        """Zero buffer of the requested SHAPE in this fiber's memory order.

        A column-major (rows, cols) tensor is a contiguous (cols, rows)
        viewed transposed: stride becomes (1, rows), so a column is
        contiguous. Logical [i, j] is unchanged, so every caller and every
        stored value is identical either way.
        """
        if self._col_major:
            return torch.zeros((cols, rows), dtype=self.DTYPE,
                               device=self._device).t()
        return torch.zeros((rows, cols), dtype=self.DTYPE,
                           device=self._device)

    # -- Input accumulation (project_into hot path) -------------------------

    def accumulate_rows(self, row_indices, out_size):
        """Sum selected rows -> dense float32 vector of *out_size*."""
        result = torch.zeros(out_size, dtype=torch.float32,
                             device=self._device)
        if self._w.shape[1] == 0 or len(row_indices) == 0:
            return result
        valid = row_indices[row_indices < self._w.shape[0]].long()
        if valid.numel() == 0:
            return result
        cols = min(out_size, self._w.shape[1])
        result[:cols] = (self._w.index_select(0, valid)[:, :cols]
                         .float().sum(dim=0))
        return result

    # -- Hebbian plasticity -------------------------------------------------

    def hebbian_update(self, src_winners, tgt_winners, beta, w_max):
        """Multiply entries at (src, tgt) intersections by (1+beta).

        Absent synapses are exact zeros, and 0 * (1+beta) = 0, so the
        sparsity pattern is invariant -- same contract as the CSR form.
        """
        rows = src_winners[src_winners < self._w.shape[0]].long()
        cols = tgt_winners[tgt_winners < self._w.shape[1]].long()
        if rows.numel() == 0 or cols.numel() == 0:
            return
        sub = self._w[rows.unsqueeze(1), cols].float() * (1 + beta)
        if w_max is not None and w_max > 0:
            sub = sub.clamp(max=w_max)
        self._w[rows.unsqueeze(1), cols] = sub.to(self.DTYPE)

    # -- Expansion ----------------------------------------------------------

    def expand(self, needed_rows, needed_cols, new_r, new_c, new_v):
        """Grow physical extent (amortised doubling) and scatter new entries.

        Same signature and semantics as `CSRConn.expand` -- extents also
        rise to cover the largest incoming entry, later duplicates win --
        but O(new entries), never a rebuild of what exists.
        """
        nr, nc = int(needed_rows), int(needed_cols)
        if len(new_r) > 0:
            nr = max(nr, int(new_r.long().max().item()) + 1)
            nc = max(nc, int(new_c.long().max().item()) + 1)
        pr, pc = self._w.shape
        if nr > pr or nc > pc:
            cap_r = max(nr, 2 * pr)
            cap_c = max(nc, 2 * pc)
            if self._max_rows is not None:
                cap_r = min(cap_r, max(int(self._max_rows), nr))
            if self._max_cols is not None:
                cap_c = min(cap_c, max(int(self._max_cols), nc))
            buf = self._new_buffer(cap_r, cap_c)
            if pr > 0 and pc > 0:
                buf[:pr, :pc] = self._w
            self._w = buf
        self._ext_rows = max(self._ext_rows, nr)
        self._ext_cols = max(self._ext_cols, nc)
        if len(new_r) > 0:
            self._w[new_r.long(), new_c.long()] = new_v.to(self.DTYPE)

    # -- Column in-degree (norm_init) ---------------------------------------

    def column_indegree(self, ncols, nrows=None):
        """Per-column count of present synapses over the first ``nrows`` rows
        (see `CSRConn.column_indegree` for why the bound is not optional)."""
        deg = torch.zeros(ncols, dtype=torch.float32, device=self._device)
        r = self._w.shape[0] if nrows is None else min(int(nrows),
                                                       self._w.shape[0])
        c = min(ncols, self._w.shape[1])
        if r > 0 and c > 0:
            deg[:c] = (self._w[:r, :c] != 0).sum(dim=0).float()
        return deg

    # -- Homeostatic synaptic scaling ---------------------------------------

    def scale_columns(self, cols, setpoint, nrows=None, eps=1e-12):
        """Renormalize the TOUCHED columns' mass to *setpoint* (see
        `CSRConn.scale_columns`; this is the dense form -- one gather, one
        reduction, one scatter, all on device)."""
        if self._w.shape[1] == 0 or len(cols) == 0:
            return
        cols = cols.long()
        cols = cols[(cols >= 0) & (cols < self._w.shape[1])]
        if cols.numel() == 0:
            return
        rows = self._w.shape[0] if nrows is None else min(int(nrows),
                                                          self._w.shape[0])
        if rows <= 0:
            return
        sub = self._w[:rows].index_select(1, cols).float()
        sums = sub.sum(dim=0)
        factors = column_scale(sums, setpoint, eps=eps)
        self._w[:rows, cols] = (sub * factors).to(self.DTYPE)

    # -- Column normalisation -----------------------------------------------

    def normalize_columns(self, eps=1e-8):
        """Column-normalize so each column sums to 1.0 (CSR parity)."""
        if self._w.shape[0] == 0 or self._w.shape[1] == 0:
            return
        sums = self._w.float().sum(dim=0).clamp(min=eps)
        self._w = (self._w.float() / sums).to(self.DTYPE)

    # -- Reset --------------------------------------------------------------

    def reset(self):
        self._w = self._new_buffer(0, 0)
        self._log_rows = 0
        self._log_cols = 0
        self._ext_rows = 0
        self._ext_cols = 0
