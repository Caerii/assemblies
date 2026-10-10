"""The DENSE ORGAN fiber: a count matrix per brain, for the organ's regime (DESIGN_sequence_port.md).

Moved from _hashed.py unchanged; _hashed.py re-exports every name."""
from __future__ import annotations

from ._torch_ops import torch_ops
from typing import Any
from . import _fused_cuda
from .._pricing import (chain_table as _chain_table,
                        count_saturation_is_exact)
from ._hashed_common import per_brain, clip_count
from ._hashed_organ_counts import OrganCounts


class DenseOrganFiber(OrganCounts):
    """An area -> area fiber in the ORGAN's regime (DESIGN_sequence_port.md):
    organ_p ~ 0.2, k ~ 200, n to 50,000. Present-only lists do not fit a row
    of thousands of synapses; the int16 count MATRIX does, and at this
    density its predicated loads use their sectors. Absolute pricing by the
    engine's chain table (clip included), norm_init, no column scaling --
    the same numbers as `AreaFiber` there. Rows and winners of -1 are
    skipped: the dead-brain and per-brain-inhibit convention.
    """

    MAX_BYTES = 6 << 30
    #: int8 counts: the chain table saturates at the clip (count ~31 at
    #: beta 0.1, w_max 20), so a count never needs more than 7 bits, and
    #: the matrix is half the size it was at int16 -- twice the brains per
    #: launch at n = 10,000.
    MAX_COUNT = 127
    mod: Any
    pres: Any
    dj: Any
    invdj: Any

    def __init__(self, seeds, n_pre, n_post, p, *, beta=0.1,
                 w_max: float | None = 20.0,
                 norm_init=True, max_rounds=4096, device="cuda", count_dtype=None):
        self.mod = _fused_cuda.load()
        if self.mod is None:
            raise RuntimeError(f"fused kernels unavailable: "
                               f"{_fused_cuda.last_error()}")
        B = len(seeds)
        # COUNT WIDTH: int8 wherever every brain's weight clip binds by count
        # 127 (every rate from ~0.024 up: the tables and counts of every study
        # before Amendment 22); int16 below, where a weak write's counts run
        # on before the clip. The table then reaches the clip, so a count is
        # priced exactly however far it runs (a table shorter than the clip
        # would price every longer count at its last, unclipped, entry).
        clip = clip_count(beta, w_max, B)
        if count_dtype is None:
            count_dtype = "int8" if clip is not None and clip <= 127 else (
                "int16" if clip is not None else "int8")
        # PACKED 4-BIT COUNTS, opt-in ("int4"; kernels/06a_organ_packed_kernels.cu):
        # two counts a byte, half the bytes of int8 and half a drive's count
        # traffic. Writing and reading are bit-identical to int8 wherever the
        # clip binds by count 15 -- the drive depends only on min(count, clip).
        # UNLEARNING IS NOT: a count held at 15 falls below the clip after fewer
        # decrements than an int8 count that ran on toward 127, so a study that
        # sleeps or downscales with packed counts runs the "counts saturate at 15"
        # variant and must say so. A packed fiber's `C` is None (code reading it
        # as counts fails); `Cp` holds the bytes, read and edited through
        # OrganCounts (_hashed_organ_counts.py).
        if count_dtype not in ("int8", "int16", "int4"):
            raise ValueError("count_dtype must be int8, int16 or int4")
        self.count_dtype = count_dtype
        self.packed = count_dtype == "int4"
        self.MAX_COUNT = {"int8": 127, "int16": 32767, "int4": 15}[count_dtype]
        if self.packed and clip is None:
            raise ValueError("packed counts need a weight clip that binds by count 15")
        if clip is not None and clip > self.MAX_COUNT:
            raise ValueError(f"the weight clip binds at count {clip}, past the "
                             f"{count_dtype} range: counts would saturate inexactly")
        if clip is not None and clip + 1 > int(max_rounds):
            max_rounds = clip + 1                   # one entry past: the table reads saturated
        #: bytes per row of the packed layout: four columns in one aligned 16-bit load
        self.NB = ((n_post + 3) // 4) * 2
        row_bytes = (self.NB if self.packed
                     else n_post * (1 if count_dtype == "int8" else 2))
        need = B * n_pre * (row_bytes + ((n_post + 31) // 32) * 4)
        if need > self.MAX_BYTES:
            raise ValueError(f"organ count matrices would be {need / 2**30:.1f} "
                             "GiB; fewer brains per launch")
        self.B, self.n_pre, self.n, self.p = B, n_pre, n_post, float(p)
        #: one learning rate per brain, or None when they share `beta`
        self.betas = per_brain(beta, B)
        self.beta = None if self.betas else float(beta)
        self.w_max = w_max
        self.seeds = torch_ops.as_tensor(seeds, dtype=torch_ops.int32, device=device)
        self.threshold = _fused_cuda.threshold_for(p)
        self.device = device
        self.learns = any(self.betas) if self.betas else bool(beta)
        self.relative, self.absolute = False, True
        self.pres = self.mod.hashed_presence(self.seeds, n_pre, n_post, self.threshold)
        if self.packed:
            self.C = None
            self.Cp = torch_ops.zeros(B, n_pre, self.NB, dtype=torch_ops.uint8, device=device)
        else:
            self.C = torch_ops.zeros(B, n_pre, n_post, dtype=getattr(torch_ops, count_dtype),
                                     device=device)
            self.Cp = None
        self.err = torch_ops.zeros(1, dtype=torch_ops.int32, device=device)
        self.max_rounds = int(max_rounds)
        self._no_map = torch_ops.zeros(0, dtype=torch_ops.int32, device=device)
        self.tab = self._table(self.max_rounds)
        deg = self.mod.hashed_indegree(self.seeds, n_pre, n_post, self.threshold, 1.0)
        self.dj = deg if norm_init else None
        self.invdj = (1.0 / deg) if norm_init else torch_ops.zeros(
            0, dtype=torch_ops.float32, device=device)

    def _table(self, depth):
        """The chain table: [depth + 1] shared, or [B, depth + 1] per brain."""
        if self.betas is None:
            return torch_ops.from_numpy(_chain_table(self.beta, self.w_max, depth)).to(self.device)
        import numpy as np
        rows = {b: _chain_table(b, self.w_max, depth) for b in set(self.betas)}
        return torch_ops.from_numpy(np.stack([rows[b] for b in self.betas])).to(self.device)

    def counts(self):
        """the count matrix (an unpacked int16 copy for packed counts)"""
        return self.unpacked() if self.packed else self.C

    def select(self, keep):
        """Keep only the brains ``keep`` [B'] (indices), in that order: a
        batched sweep drops the rates whose stores have finished.

        The count matrices and connectome bits are COMPACTED IN PLACE when
        ``keep`` ascends (as a sweep's does): each kept brain's slice is copied
        forward over a dropped one, then the tensor is narrowed. index_select
        allocated the kept copy while the old one lived -- at n = 8000 a 3.6
        GiB second copy, which ran a full card out of memory."""
        order = [int(i) for i in keep]
        keep = torch_ops.as_tensor(order, dtype=torch_ops.int64, device=self.device)
        store = self.storage
        if all(a < b for a, b in zip(order, order[1:])):
            for j, src in enumerate(order):
                if src != j:                      # src > j: slot j is free by now
                    store[j].copy_(store[src])
                    self.pres[j].copy_(self.pres[src])
            store = store[:len(order)]
            self.pres = self.pres[:len(order)]
        else:
            store = store.index_select(0, keep)
            self.pres = self.pres.index_select(0, keep)
        if self.packed:
            self.Cp = store
        else:
            self.C = store
        self.seeds = self.seeds.index_select(0, keep)
        if self.dj is not None:
            self.dj = self.dj.index_select(0, keep)
            self.invdj = self.invdj.index_select(0, keep)
        if self.betas is not None:
            self.betas = tuple(self.betas[i] for i in keep.tolist())
            self.tab = self.tab.index_select(0, keep)
        self.B = int(keep.numel())

    @property
    def nnz(self):
        return self.nonzero()

    @property
    def store(self):
        self.check()
        top = int(self.unpacked().max()) if self.packed else int(self.C.max())

        class _S:
            max_count = top
        return _S()

    @property
    def count_saturation_is_exact(self) -> bool:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-organ-count-saturation"""
        tab = self.tab.cpu().numpy()
        return all(count_saturation_is_exact(row, self.MAX_COUNT)
                   for row in (tab if tab.ndim == 2 else [tab]))

    def check(self):
        code = int(self.err.item())
        if code == 1:
            if self.count_saturation_is_exact:
                # The write kernel leaves a count at MAX_COUNT when the next
                # potentiation would pass it, and the drive kernel prices
                # every count at or beyond the table's last index with that
                # entry, which is the clipped weight. A stored 127 and the true
                # count therefore give identical drives: saturation is exact,
                # and the flag is informational. Clear it and continue.
                self.err.zero_()
                return
            raise OverflowError(
                f"a count passed {self.MAX_COUNT} and the chain table is not "
                "saturated at its last entry (no weight clip, or a table longer "
                "than the count range), so the lost potentiations would have "
                "changed a weight")
        if code:
            raise RuntimeError(f"organ fiber kernel error {code}")

    def ensure_depth(self, depth):
        depth = int(depth)
        if self.tab.shape[-1] < depth + 1:
            self.tab = self._table(depth)

    def contribute(self, drive, rows, brains=None):
        """Add this fiber's drive from ``rows`` [V, K] into ``drive`` [V, n].
        ``brains`` [V] (int32) names the brain whose synapses each row reads;
        omitted, row b is brain b."""
        if rows.shape[1] == 0:
            return
        self.mod.organ_drive(rows.to(torch_ops.int32), self.storage, self.pres, self.invdj,
                             self.tab, self._no_map if brains is None else brains, drive)

    def begin_episode(self):
        pass

    def observe(self, prev, new):
        if not (self.learns and prev.shape[1] and new.shape[1]):
            return
        self.mod.organ_write(prev.to(torch_ops.int32), new.to(torch_ops.int32), self.storage,
                             self.pres, self.err)

    def end_episode(self):
        pass
