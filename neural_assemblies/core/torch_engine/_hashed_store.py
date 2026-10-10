"""The STORE fiber: an area -> area fiber whose deviations live in a sorted (key, count)
store (RunStore, an LSM of runs), read by row (see _hashed.py for the algebra).

Moved from _hashed.py unchanged; _hashed.py re-exports every name."""
from __future__ import annotations

from ._torch_ops import torch_ops
from typing import Any
from . import _fused_cuda
from .._homeostasis import column_scale, scaling_setpoint
from .._pricing import (chain_table as _chain_table,
                        relative_table as _rel_table)
from ._hashed_common import _local_index


class RunStore:
    """Sorted (key, count) runs of geometrically increasing size -- an LSM.

    WHY NOT ONE SORTED ARRAY. Merging each episode into a single sorted store
    re-sorts the WHOLE store every time, which is O(M^2) over a study: 15e9
    sorted elements at M=255, B=16, against 0.94e9 for O(M log M). Keeping runs
    and merging only when the last two become comparable sorts each element
    about log2(M) times instead of M.

    Reads search every run, but there are only ~log2(M) of them and a run
    search is the same binary search the single array needed.

    Counts are exact integers and a merge is integer addition, so the store is
    exact however often it is reorganised ([[HEBB-OUTER-PRODUCT]]).
    """

    def __init__(self, device):
        self.device = device
        # Allocated lazily on the first episode; Any preserves the tensor
        # device/dtype while making the lifecycle explicit to static checks.
        self.keys: Any = None
        self.cnts: Any = None
        self.used = 0
        self.offs = [0]
        #: the largest per-cell count held anywhere in the store. A fiber's
        #: chain table must extend past it; see `AreaFiber.end_episode`.
        self.max_count = 0

    @property
    def nnz(self):
        return self.used

    @property
    def nruns(self):
        return len(self.offs) - 1

    def _reserve(self, extra):
        need = self.used + extra
        cap = 0 if self.keys is None else self.keys.numel()
        if need <= cap:
            return
        cap = max(need, 2 * cap, 1 << 16)
        k = torch_ops.empty(cap, dtype=torch_ops.int64, device=self.device)
        c = torch_ops.empty(cap, dtype=torch_ops.int32, device=self.device)
        if self.used:
            k[:self.used] = self.keys[:self.used]
            c[:self.used] = self.cnts[:self.used]
        self.keys, self.cnts = k, c

    def append(self, nkey, ncnt):
        """Add one episode's block, then collapse comparable tail runs."""
        if nkey is None or nkey.numel() == 0:
            return
        order = torch_ops.argsort(nkey)
        self.max_count = max(self.max_count, int(ncnt.max()))
        self._reserve(nkey.numel())
        a = self.used
        self.keys[a:a + nkey.numel()] = nkey[order]
        self.cnts[a:a + ncnt.numel()] = ncnt[order]
        self.used = a + nkey.numel()
        self.offs.append(self.used)
        while len(self.offs) >= 3:
            a, b, c = self.offs[-3], self.offs[-2], self.offs[-1]
            if (b - a) > (c - b):
                break
            self._merge_tail(a, c)

    def _merge_tail(self, a, c):
        """Merge the last two runs in place; only the TAIL moves, so nothing
        after them needs shifting."""
        seg_k, seg_c = self.keys[a:c], self.cnts[a:c]
        order = torch_ops.argsort(seg_k)
        sk, sc = seg_k[order], seg_c[order]
        uk, inv, _ = torch_ops.unique_consecutive(sk, return_inverse=True,
                                              return_counts=True)
        uv = torch_ops.zeros(uk.numel(), dtype=torch_ops.int32, device=self.device)
        uv.scatter_add_(0, inv, sc)
        self.max_count = max(self.max_count, int(uv.max()))
        m = uk.numel()
        self.keys[a:a + m] = uk
        self.cnts[a:a + m] = uv
        self.used = a + m
        self.offs = self.offs[:-2] + [self.used]

    def view(self):
        """(keys, counts, run offsets) for the reader."""
        return (self.keys[:self.used], self.cnts[:self.used],
                torch_ops.tensor(self.offs, dtype=torch_ops.int64, device=self.device))


class AreaFiber:
    """An area -> area projection. Deviations are keyed ``(i, j)``.

    Owns the column scale too, because substrate C scales THIS FIBER's stored
    weights (`_scale_columns_now` iterates the target's fibers), not the
    target's drive.
    """

    MAX_EPISODE_ROUNDS = 64

    # Runtime-owned CUDA tensors and the compiled module are established by
    # __init__ (some are intentionally lazy).  These declarations describe
    # that state machine without erasing the Tensor annotations at call sites.
    mod: Any
    dj: Any
    scale: Any
    rel: Any
    cmax: Any
    _rowmask: Any
    _colmask: Any
    _scratch: Any
    _cscratch: Any
    _colmap: Any

    def __init__(self, seeds, n_pre, n_post, p, *, beta=0.0, w_max=None,
                 norm_init=False, synaptic_scaling=False, max_rounds=64,
                 device="cuda", scaling_allows_clip=False):
        # COLUMN SCALING AND A WEIGHT CLIP DO NOT COMMUTE. The factored scale
        # S_j = setpoint / M_j is exact only while no cell has ever been
        # clipped: the engine clips per cell and rescales per column in an
        # interleaved order that the count-then-apply form cannot reproduce.
        # `_rescale` checks the SCALED value against w_max, which is not the
        # same thing -- measured on the aligner at w_max=20, 217k of 742k
        # cells were clipped while that check never fired, and the learner's
        # reconstructions inverted. So the pair is refused here. The capacity
        # protocol (T <= 8 rounds per item, counts far below the ~31 at which
        # a 1.1 gain reaches 20) opts in explicitly; its four-arm parity test
        # is what licenses that.
        if synaptic_scaling and w_max is not None and not scaling_allows_clip:
            raise ValueError(
                "synaptic_scaling with a finite w_max: column scaling and the "
                "clip do not commute, so the factored scale is exact only "
                "while no cell is ever clipped. Run with w_max=None (scaling "
                "bounds the weights itself), or pass scaling_allows_clip=True "
                "for a protocol whose counts provably stay below the clip.")
        self.mod = _fused_cuda.load()
        if self.mod is None:
            raise RuntimeError(f"fused kernels unavailable: "
                               f"{_fused_cuda.last_error()}")
        B = len(seeds)
        self.B, self.n_pre, self.n, self.p = B, n_pre, n_post, float(p)
        self.beta, self.w_max = float(beta), w_max
        self.seeds = torch_ops.as_tensor(seeds, dtype=torch_ops.int32, device=device)
        self.threshold = _fused_cuda.threshold_for(p)
        self.device = device
        self.learns = bool(beta)
        self.tab = torch_ops.from_numpy(
            _chain_table(beta, w_max, max_rounds)).to(device)
        self.colids = torch_ops.arange(n_post, dtype=torch_ops.int32,
                                   device=device).expand(B, n_post)
        # `_pricing.inverse_indegree` prices unknown rows at
        # `p * (n_pre - rows_known)`; a generated connectome HAS every row, so
        # that term is identically zero and d_j is the TRUE in-degree. The
        # defect class living on the estimate cannot occur here.
        self.dj = (self.mod.hashed_indegree(self.seeds, n_pre, n_post,
                                            self.threshold, 1.0)
                   if norm_init else None)
        self.scale = (torch_ops.ones(B, n_post, dtype=torch_ops.float32, device=device)
                      if synaptic_scaling else None)
        # MAX-RELATIVE PRICING (see `_rel_table`). With scaling and no clip
        # the absolute chain overflows float32 on long training; the relative
        # form is the same number, bounded. `cmax` is each column's leading
        # count, updated wherever the column is rescaled -- which is every
        # column whose counts changed, since scaling touches the winner
        # columns of every write.
        self.relative = bool(synaptic_scaling) and w_max is None
        if self.relative:
            self.rel = torch_ops.from_numpy(_rel_table(beta, max_rounds)).to(device)
            self.cmax = torch_ops.zeros(B, n_post, dtype=torch_ops.int32, device=device)
        else:
            self.rel = self.cmax = None
        self.setpoint = scaling_setpoint(n_pre, self.p)
        self.store = RunStore(device)
        self._rowmask = self._colmask = None    # this episode only
        self._scratch = None                    # count accumulator, cached
        self._cscratch = self._colmap = None    # column-mass accumulators
        self._prevs, self._news = [], []
        self._t = 0

    # -- reading ---------------------------------------------------------
    def contribute(self, drive, rows):
        """Add this fiber's drive for source winners ``rows`` [B, k_src].

        THE CORRECTION IS NOT ADDITIVE ACROSS A SPLIT COUNT. Potentiation is
        multiplicative, so a cell with c0 events in the store and c1 in the
        current episode needs `tab[c0+c1]-1`, and `(tab[c0]-1)+(tab[c1]-1)` is
        wrong by the cross term -- the same holds BETWEEN LSM runs. COUNTS are
        additive ([[HEBB-OUTER-PRODUCT]]), so integer counts from every source
        are accumulated into a scratch [B, k, n] first and `tab` is applied
        ONCE per cell. Caught by engine parity at rel 2e-3; the
        reference-based tests shared the flawed structure and passed.
        """
        if rows.shape[1] == 0:
            return
        r = rows.to(torch_ops.int32).contiguous()
        d = self.mod.hashed_drive(r, self.seeds, self.n, self.threshold)
        has_mask = self._rowmask is not None and self._t > 0
        if self.store.nnz or has_mask:
            k_src = int(r.shape[1])
            need = self.B * k_src * self.n
            if self._scratch is None or self._scratch.numel() < need:
                self._scratch = torch_ops.zeros(need, dtype=torch_ops.int32,
                                            device=self.device)
            else:
                self._scratch[:need].zero_()
            sk, sc, so = (self.store.view() if self.store.nnz else
                          (torch_ops.zeros(0, dtype=torch_ops.int64,
                                       device=self.device),
                           torch_ops.zeros(0, dtype=torch_ops.int32,
                                       device=self.device),
                           torch_ops.zeros(1, dtype=torch_ops.int64,
                                       device=self.device)))
            rm = (self._rowmask if has_mask else
                  torch_ops.zeros(0, dtype=torch_ops.int64, device=self.device))
            cm = (self._colmask if has_mask else
                  torch_ops.zeros(0, dtype=torch_ops.int64, device=self.device))
            if self.relative:
                # d <- rel[cmax] * base + SUM_touched base * (rel[cmax-c] - rel[cmax])
                d = d * self.rel[self.cmax.long()]
                self.mod.dev_correct_rel(r, sk, sc, so, rm, cm,
                                         self._scratch[:need].view(
                                             self.B, k_src, self.n),
                                         self.rel, self.cmax, self.seeds,
                                         self.threshold, d)
            else:
                self.mod.dev_correct_exact(r, sk, sc, so, rm, cm,
                                           self._scratch[:need].view(
                                               self.B, k_src, self.n),
                                           self.tab, self.seeds,
                                           self.threshold, d)
        elif self.relative:
            d = d * self.rel[self.cmax.long()]
        if self.scale is not None:
            d = d * self.scale
        if self.dj is not None:
            d = d / self.dj
        drive += d

    def ensure_depth(self, depth):
        """Extend the pricing tables to `depth` potentiations.

        A cell can be potentiated once per round it co-fires in, so the depth a
        training run needs is its TOTAL number of rounds -- a quantity the
        fiber cannot know at construction and a learner only knows once it
        sees its corpus. Measured on the aligner: a FEAT column in a frequent
        bundle passed 2,000 co-firings inside one pass over 198 sentences and
        walked off a 2,048-entry table. Cheap to extend (a few KB), and the
        relative table underflows to 0 past ~900 anyway.
        """
        depth = int(depth)
        if self.relative:
            if self.rel.numel() < depth + 1:
                self.rel = torch_ops.from_numpy(
                    _rel_table(self.beta, depth)).to(self.device)
                self.tab = self.rel          # the guard reads one table
        elif self.tab.numel() < depth + 1:
            self.tab = torch_ops.from_numpy(
                _chain_table(self.beta, self.w_max, depth)).to(self.device)

    # -- writing ---------------------------------------------------------
    def begin_episode(self):
        """One 64-bit round mask covers an episode; the store holds the rest."""
        if not (self.learns or self.scale is not None):
            return
        self._rowmask = torch_ops.zeros(self.B, 1, self.n_pre, dtype=torch_ops.int64,
                                    device=self.device)
        self._colmask = torch_ops.zeros(self.B, 1, self.n, dtype=torch_ops.int64,
                                    device=self.device)
        self._prevs, self._news, self._t = [], [], 0

    def observe(self, prev, new):
        """Record one round's co-firing: source winners x target winners.

        The ENGINE's pairing ([[HEBB-OUTER-PRODUCT]]) -- `hebbian_update` takes
        the source's winners, and for a recurrent fiber `tgt.winners` is
        assigned after, so it is prev x new.
        """
        if self._rowmask is None:
            return
        if self._t >= self.MAX_EPISODE_ROUNDS:
            raise ValueError(
                f"episode exceeds {self.MAX_EPISODE_ROUNDS} rounds: the "
                "per-episode round mask is one 64-bit word")
        bit = 1 << self._t
        if prev.shape[1] and new.shape[1]:
            rw, cw = self._rowmask[:, 0], self._colmask[:, 0]
            rw.scatter_(1, prev, rw.gather(1, prev) | bit)
            cw.scatter_(1, new, cw.gather(1, new) | bit)
            if self.learns:
                self._prevs.append(prev)
                self._news.append(new)
        self._t += 1
        # SCALING FIRES ONLY WHEN THIS FIBER FIRED. The engine filters
        # sourceless areas out of `from_areas` before plasticity, so on a
        # stimulus-only round (empty prev -- the first round after inhibition)
        # `_scale_columns_now` never runs. Rescaling there anyway sets the new
        # winners' columns to setpoint/in-degree (0.93-1.08 at p=0.1), an ~8%
        # drive divergence on exactly those columns -- caught by the four-arm
        # engine-parity capacity test at ep0 t1.
        if self.scale is not None and new.shape[1] and prev.shape[1]:
            self._rescale(new)

    def end_episode(self):
        """Fold the episode into the store via a GEMM, then drop its mask.

        THE TABLE MUST OUTRUN THE COUNTS. `max_rounds` sizes the chain table
        -- the most potentiations one cell can ever receive -- NOT an episode.
        `dev_correct_exact` clamps a count to the table's last entry, which is
        silent: a fiber built with `max_rounds=1` and trained in 1-round
        episodes read `tab[1]` for cells that had co-fired three times, and
        its drive simply stopped growing (found by the aligner parity gate).
        Refuse it here, where the store knows its largest count.
        """
        if self.learns and self._prevs:
            self.store.append(*self._emit())
            table = self.rel if self.relative else self.tab
            if self.store.max_count >= table.numel():
                raise ValueError(
                    f"a cell has co-fired {self.store.max_count} times but "
                    f"the chain table prices at most {table.numel() - 1}: "
                    "size max_rounds by the TOTAL potentiations a cell can "
                    "accumulate over training, not by the episode")
        self._rowmask = self._colmask = None
        self._prevs, self._news = [], []

    # -- substrate C -----------------------------------------------------
    def _rescale(self, cols):
        """`w[:, j] *= setpoint / mass_j` on winner columns.

        mass_j needs the TOTAL count per cell -- the correction's lesson
        applies verbatim: counts are additive, `tab` is not
        ([[HEBB-OUTER-PRODUCT]]). Counts from the store (one linear pass with
        a column -> slot map; the store is row-keyed so a column cannot be
        binary-searched) and the current mask are accumulated per cell, and
        `tab` applied once. Column scaling still commutes with per-cell
        potentiation, so the factored scale `S_j = setpoint / M_j` is exact --
        only while the `w_max` clip never binds, checked against the ACTUAL
        deepest cell.
        """
        c = cols.to(torch_ops.int32).contiguous()
        B, K = c.shape
        need = B * K * self.n
        if self._cscratch is None or self._cscratch.numel() < need:
            self._cscratch = torch_ops.zeros(need, dtype=torch_ops.int32,
                                         device=self.device)
        else:
            self._cscratch[:need].zero_()
        if self.store.nnz:
            if self._colmap is None:
                self._colmap = torch_ops.full((B, self.n), -1, dtype=torch_ops.int32,
                                          device=self.device)
            else:
                self._colmap.fill_(-1)
            self._colmap.scatter_(
                1, cols, torch_ops.arange(K, dtype=torch_ops.int32,
                                      device=self.device).expand(B, K))
            sk, sc, _ = self.store.view()
        else:
            sk = torch_ops.zeros(0, dtype=torch_ops.int64, device=self.device)
            sc = torch_ops.zeros(0, dtype=torch_ops.int32, device=self.device)
        if self.relative:
            mass, colmax = self.mod.column_mass_rel(
                c, sk, sc,
                self._colmap if self.store.nnz else sk.to(torch_ops.int32),
                self._rowmask, self._colmask,
                self._cscratch[:need].view(B, K, self.n),
                self.rel, self.seeds, self.n, self.threshold)
            self.cmax.scatter_(1, cols, colmax)
            new = column_scale(mass, self.setpoint)
            self.scale.scatter_(1, cols, new)
            return
        mass, cellmax = self.mod.column_mass_exact(
            c, sk, sc,
            self._colmap if self.store.nnz else sk.to(torch_ops.int32),
            self._rowmask, self._colmask,
            self._cscratch[:need].view(B, K, self.n),
            self.tab, self.seeds, self.n, self.threshold)
        new = column_scale(mass, self.setpoint)
        self.scale.scatter_(1, cols, new)
        if self.w_max is not None:
            bound = float((cellmax * new).max().item())
            if bound >= float(self.w_max):
                raise RuntimeError(
                    f"synaptic_scaling would cross w_max={self.w_max} "
                    f"(bound {bound:.4g}): column scaling and the clip do not "
                    "commute, so the factored form is no longer exact. Re-run "
                    "with w_max=None or fewer rounds.")

    def _emit(self):
        """This episode's nonzero cells as (key, count), by GEMM.

        Rounds with no source winners contribute nothing and are dropped -- an
        inhibited area's first round is stimulus-only -- so per-round widths
        differ and the round index is carried explicitly.
        """
        if len(self._prevs) != len(self._news):
            raise RuntimeError("hashed fiber round histories must have equal length")
        keep = [(a, b) for a, b in zip(self._prevs, self._news, strict=True)
                if a.shape[1] and b.shape[1]]
        if not keep:
            return None, None
        T, B, dev = len(keep), self.B, self.device

        def ind(per_round):
            cat = torch_ops.cat(per_round, dim=1)
            rid = torch_ops.cat([
                torch_ops.full((c.shape[1],), t, dtype=torch_ops.int64, device=dev)
                for t, c in enumerate(per_round)])
            loc, vals, W = _local_index(cat)
            flat = torch_ops.zeros(B, T * W, device=dev)
            flat.scatter_(1, rid.view(1, -1) * W + loc, 1.0)
            return flat.view(B, T, W), vals, W

        Rind, rows, _ = ind([a for a, _ in keep])
        Cind, cols, _ = ind([b for _, b in keep])
        counts = torch_ops.bmm(Rind.transpose(1, 2), Cind)      # [B, R, C]
        bi, ri, ci = (counts > 0).nonzero(as_tuple=True)
        if bi.numel() == 0:
            return None, None
        key = (bi.to(torch_ops.int64) * self.n_pre * self.n
               + rows[bi, ri] * self.n + cols[bi, ci])
        return key, counts[bi, ri, ci].to(torch_ops.int32)

    @property
    def nnz(self):
        return self.store.nnz
