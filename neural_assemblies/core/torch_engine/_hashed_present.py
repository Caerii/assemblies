"""The PRESENT-ONLY fiber: per row, only the synapses that exist, with their counts.

Moved from _hashed.py unchanged; _hashed.py re-exports every name."""
from __future__ import annotations

from ._torch_ops import torch_ops
from typing import Any
from . import _fused_cuda
from .._homeostasis import scaling_setpoint
from .._pricing import (chain_table as _chain_table,
                        relative_table as _rel_table)


class PresentFiber:
    """An area -> area fiber that stores only what EXISTS (DESIGN_present_only.md).

    Same numbers as `AreaFiber` in its unclipped max-relative regime (and
    as the retired dense-matrix fiber, gated identical at atol 0 before its
    deletion -- DESIGN_present_only.md). Per (brain, row): the present columns with their
    int16 counts, one packed 32-bit entry (column low, count high), padded
    to ``DMAX`` -- the largest row degree -- with -1. Built once from the
    presence bitmask (the hash's stored form). A round reads K row lists
    instead of K x n_post counts, and the kernels give a brain ONE WARP
    that walks the rows in order, so every column's sum is the same float
    sequence as the dense kernel's.
    """

    MAX_BYTES = 4 << 30
    MAX_COUNT = 32767
    mod: Any
    _nnz_of: Any
    rel: Any
    cmax: Any
    dj: Any
    invdj: Any
    mass: Any
    scale: Any

    def __init__(self, seeds, n_pre, n_post, p, *, beta=0.1, norm_init=False,
                 synaptic_scaling=True, w_max=None, max_rounds=4096,
                 device="cuda"):
        self.mod = _fused_cuda.load()
        if self.mod is None:
            raise RuntimeError(f"fused kernels unavailable: "
                               f"{_fused_cuda.last_error()}")
        B = len(seeds)
        if n_post > 65535:
            raise ValueError("n_post must fit the 16-bit column field")
        # TWO REGIMES, one representation. With column scaling and no clip
        # the price is MAX-RELATIVE (`_rel_table`, index cmax_j - c); without
        # scaling it is ABSOLUTE by the engine's chain (`_chain_table`, clip
        # included, index c). Scaling with a finite clip is refused as in
        # `AreaFiber`: the two do not commute.
        if synaptic_scaling and w_max is not None:
            raise ValueError("synaptic_scaling with a finite w_max: column "
                             "scaling and the clip do not commute (see AreaFiber)")
        self.absolute = not synaptic_scaling
        self.B, self.n_pre, self.n, self.p = B, n_pre, n_post, float(p)
        self.beta, self.w_max = float(beta), w_max
        self.seeds = torch_ops.as_tensor(seeds, dtype=torch_ops.int32, device=device)
        self.threshold = _fused_cuda.threshold_for(p)
        self.device = device
        self.learns = bool(beta)
        self.relative = True
        pres = self.mod.hashed_presence(self.seeds, n_pre, n_post, self.threshold)
        self.degree = self.mod.present_degree(pres)                  # [B, n_pre]
        self.DMAX = max(int(self.degree.max()), 1)
        need = B * n_pre * self.DMAX * 4
        if need > self.MAX_BYTES:
            raise ValueError(f"present-only lists would be {need / 2**30:.1f} "
                             "GiB; use AreaFiber (the store) at this size")
        self.ent = self.mod.present_fill(pres, n_post, self.DMAX)    # [B, n_pre, DMAX]
        del pres
        self.err = torch_ops.zeros(1, dtype=torch_ops.int32, device=device)
        self.max_rounds = int(max_rounds)
        self.rel = self._table(self.max_rounds)
        self.tab = self.rel
        self._nnz_of = None
        self.cmax = torch_ops.zeros(B, n_post, dtype=torch_ops.int32, device=device)
        deg = self.mod.hashed_indegree(self.seeds, n_pre, n_post, self.threshold, 1.0)
        self.dj = deg if norm_init else None
        self.invdj = (1.0 / deg) if norm_init else torch_ops.zeros(
            0, dtype=torch_ops.float32, device=device)
        self.scaling = bool(synaptic_scaling)
        self.setpoint = scaling_setpoint(n_pre, self.p)
        self.mass = deg.to(torch_ops.float64).clone()
        self.scale = torch_ops.ones(B, n_post, dtype=torch_ops.float32, device=device)

    # -- the lists, unpacked --------------------------------------------------
    def columns(self):
        """[B, n_pre, DMAX] int64 column per entry, -1 where padded."""
        return torch_ops.where(self.ent == -1, torch_ops.full_like(self.ent, -1),
                           self.ent & 0xFFFF).to(torch_ops.int64)

    def counts(self):
        """[B, n_pre, DMAX] int16 count per entry, 0 where padded."""
        c = (self.ent >> 16) & 0xFFFF
        return torch_ops.where(self.ent == -1, torch_ops.zeros_like(c), c).to(torch_ops.int16)

    @property
    def nnz(self):
        return int((self.counts() > 0).sum())

    @property
    def store(self):
        self.check()
        class _S:                                   # the guard's interface
            max_count = int(self.counts().max())
        return _S()

    def check(self):
        """Raise if a kernel flagged an int16 count overflow (1) or a
        selection that did not return k winners (2). One sync."""
        code = int(self.err.item())
        if code == 1:
            raise OverflowError(f"a count passed {self.MAX_COUNT}: int16 counts "
                                "cannot hold this schedule")
        if code:
            raise RuntimeError(f"present fiber kernel error {code}")

    def _table(self, depth):
        if self.absolute:
            return torch_ops.from_numpy(_chain_table(self.beta, self.w_max, depth)).to(self.device)
        return torch_ops.from_numpy(_rel_table(self.beta, depth)).to(self.device)

    def ensure_depth(self, depth):
        depth = int(depth)
        if self.rel.numel() < depth + 1:
            self.rel = self._table(depth)
            self.tab = self.rel
            self._nnz_of = None

    def price_head(self):
        """(nnz, nsh): the table's nonzero head (past it the price IS zero,
        so no lookup) and the part the kernels stage in shared memory."""
        if self._nnz_of is None or self._nnz_of[0] is not self.rel:
            nnz = int((self.rel > 0).sum())
            self._nnz_of = (self.rel, nnz, min(nnz, 1024))
        return self._nnz_of[1], self._nnz_of[2]

    def contribute(self, drive, rows):
        if rows.shape[1] == 0:
            return
        nnz, _ = self.price_head()
        self.mod.present_drive(self.ent, rows.to(torch_ops.int32), self.cmax,
                               self.scale, self.invdj, self.rel, nnz, drive,
                               1 if self.absolute else 0)

    def begin_episode(self):
        pass

    def observe(self, prev, new):
        if not (self.learns and prev.shape[1] and new.shape[1]):
            return
        nnz, _ = self.price_head()
        self.mod.present_write(prev.to(torch_ops.int32), new.to(torch_ops.int32),
                               self.ent, self.cmax, self.mass, self.scale,
                               self.rel, nnz, float(self.setpoint),
                               1 if self.scaling else 0, self.err)

    def end_episode(self):
        pass
