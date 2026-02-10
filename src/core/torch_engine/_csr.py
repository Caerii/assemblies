"""CSR-format sparse connectivity matrix for area->area connections.

Stores weights as Compressed Sparse Row — three arrays (crow, col,
val) — using O(nnz) memory instead of O(rows x cols).  At typical
connection probability p=0.0005 this is ~2000x smaller than dense.
"""

import torch

from ._hash import WEIGHT_DTYPE, csr_flat_indices


class CSRConn:
    """CSR-format area->area connectivity on GPU."""

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
