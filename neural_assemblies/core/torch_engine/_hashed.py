"""Composable pieces for an area whose connectome is GENERATED, not stored.

WHY THIS SHAPE. The algebra composes, so the code should:

    drive[j] = SUM over afferent FIBERS f of   norm_f(j) * SUM_{i in rows_f} w_f[i,j]

and the engine already works this way -- `_norm_scale` is called per fiber,
`_scale_columns_now` iterates the fibers of a target. So a fiber owns three
things and an area owns none of them:

    its BASE        a hash, `w[i,j] != 0 iff float24(fmix32(...)) < p`
    its DEVIATIONS  what training changed, [[HEBB-OUTER-PRODUCT]]
    its PRICE       the norm divisor, and the column scale if any

Adding an input is then adding a fiber to a list, not adding a flag to a
projection -- which is what `batched_project_hashed` had started to become.

TWO KINDS OF FIBER, and the difference is not cosmetic. An area->area fiber is
2-D and its deviations are keyed `(i, j)`. A stimulus fiber stores PRE-SUMMED
input -- one weight per target neuron, equal to how many stimulus neurons wire
to it -- so its deviations are keyed by `j` alone, its `w_max` cap is scaled by
the initial magnitude `stim_size * p`, and its norm divisor is priced at the
TARGET's n rather than its own size. Getting either wrong is silent and has a
direction; see `AreaFiber.contribute` and `StimulusFiber.contribute`.

THE DEVIATION STORE. Within one episode the co-firing record is a single
64-bit round mask per neuron. At the end of the episode it is folded into a
sorted (key, count) store via a GEMM and never walked again:

    query cost   O(sum_{i in S} nnz_i)      -- the store, read BY ROW
    versus       O(k n W), W = rounds / 64  -- a mask, which cannot say WHICH
                                               cells are nonzero

a ratio `n^2 T / (64 k^2)` that is independent of M ([[DRIVE-SPLIT]]) -- 8894x
at n=16000, k=60, T=8. Keeping the mask across episodes made W grow with
training length and a study's cost quadratic in it.
"""
from __future__ import annotations

import torch

from . import _fused_cuda
from .._homeostasis import column_scale, scaling_setpoint


def _chain_table(beta, w_max, rounds):
    """``chain(1.0, c)`` for c = 0..rounds, by the ENGINE's own arithmetic.

    The engine potentiates with ``w *= (1 + beta)`` and clamps at ``w_max``
    every round, so a cell potentiated c times is a per-step
    multiply-and-clip -- NOT ``min((1+beta)**c, w_max)``, which differs once
    the clip binds.
    """
    import numpy as np
    g = np.float32(1.0 + beta)
    out = np.ones(rounds + 1, dtype=np.float32)
    v = np.float32(1.0)
    for c in range(1, rounds + 1):
        v = np.float32(v * g)
        if w_max is not None:
            v = min(v, np.float32(w_max))
        out[c] = v
    return out


def _rel_table(beta, depth):
    """``(1 + beta)**(-d)`` for d = 0..depth, by repeated float32 division.

    The MAX-RELATIVE price of a cell: with column scaling and no clip a
    weight is ``base * (1+beta)^c * s_j`` with ``s_j`` a per-column scalar, so
    every column is a SHARE distribution and only count DIFFERENCES within a
    column matter. ``(1+beta)^c`` overflows float32 at c ~ 900 -- a cell that
    co-fires 1325 times in a long training run is ordinary -- while
    ``(1+beta)^(c - cmax_j)`` lies in (0, 1] and underflows only for cells
    ~900 counts behind the column's leader, whose share is zero in the engine
    as well. Underflow to 0 is therefore the right limit, not an error.
    """
    import numpy as np
    g = np.float32(1.0 + beta)
    out = np.ones(depth + 1, dtype=np.float32)
    v = np.float32(1.0)
    for d in range(1, depth + 1):
        v = np.float32(v / g)
        out[d] = v
    return out


def _gain_table(beta, rounds):
    """``(1 + beta)**c``, for a weight that does NOT start at 1.0.

    SIZE THIS BY THE EPISODE, not by a whole study. Unlike `_chain_table` there
    is no clip to bound the recursion, so `1.1**2048` overflows float32 to inf;
    the stimulus drive then goes infinite, no assembly stabilises, and the
    deviation store explodes -- measured as 0.10 -> 82 ms/brain/round before
    the cause was found. A stimulus weight can be potentiated at most once per
    round of the episode it belongs to.
    """
    import numpy as np
    g = np.float32(1.0 + beta)
    out = np.ones(rounds + 1, dtype=np.float32)
    v = np.float32(1.0)
    for c in range(1, rounds + 1):
        v = np.float32(v * g)
        if not np.isfinite(v):
            raise ValueError(
                f"(1+beta)^{c} overflows float32 at beta={beta}; size the "
                "gain table by the EPISODE's rounds, not a whole study")
        out[c] = v
    return out


def _local_index(idx):
    """Map raw ids to per-brain local ids. ``idx`` [B, L] -> loc, values, W."""
    B, L = idx.shape
    srt, order = torch.sort(idx, dim=1)
    fresh = torch.ones_like(srt, dtype=torch.bool)
    fresh[:, 1:] = srt[:, 1:] != srt[:, :-1]
    loc_sorted = torch.cumsum(fresh, dim=1) - 1
    loc = torch.empty_like(loc_sorted)
    loc.scatter_(1, order, loc_sorted)
    W = int(fresh.sum(1).max())
    vals = torch.full((B, W), -1, dtype=torch.int64, device=idx.device)
    vals.scatter_(1, loc_sorted, srt)      # duplicates write the same value
    return loc, vals, W


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
        self.keys = self.cnts = None
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
        k = torch.empty(cap, dtype=torch.int64, device=self.device)
        c = torch.empty(cap, dtype=torch.int32, device=self.device)
        if self.used:
            k[:self.used] = self.keys[:self.used]
            c[:self.used] = self.cnts[:self.used]
        self.keys, self.cnts = k, c

    def append(self, nkey, ncnt):
        """Add one episode's block, then collapse comparable tail runs."""
        if nkey is None or nkey.numel() == 0:
            return
        order = torch.argsort(nkey)
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
        order = torch.argsort(seg_k)
        sk, sc = seg_k[order], seg_c[order]
        uk, inv, _ = torch.unique_consecutive(sk, return_inverse=True,
                                              return_counts=True)
        uv = torch.zeros(uk.numel(), dtype=torch.int32, device=self.device)
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
                torch.tensor(self.offs, dtype=torch.int64, device=self.device))


class AreaFiber:
    """An area -> area projection. Deviations are keyed ``(i, j)``.

    Owns the column scale too, because substrate C scales THIS FIBER's stored
    weights (`_scale_columns_now` iterates the target's fibers), not the
    target's drive.
    """

    MAX_EPISODE_ROUNDS = 64

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
        self.seeds = torch.as_tensor(seeds, dtype=torch.int32, device=device)
        self.threshold = _fused_cuda.threshold_for(p)
        self.device = device
        self.learns = bool(beta)
        self.tab = torch.from_numpy(
            _chain_table(beta, w_max, max_rounds)).to(device)
        self.colids = torch.arange(n_post, dtype=torch.int32,
                                   device=device).expand(B, n_post)
        # `_pricing.inverse_indegree` prices unknown rows at
        # `p * (n_pre - rows_known)`; a generated connectome HAS every row, so
        # that term is identically zero and d_j is the TRUE in-degree. The
        # defect class living on the estimate cannot occur here.
        self.dj = (self.mod.hashed_indegree(self.seeds, n_post,
                                            self.threshold, 1.0)
                   if norm_init else None)
        self.scale = (torch.ones(B, n_post, dtype=torch.float32, device=device)
                      if synaptic_scaling else None)
        # MAX-RELATIVE PRICING (see `_rel_table`). With scaling and no clip
        # the absolute chain overflows float32 on long training; the relative
        # form is the same number, bounded. `cmax` is each column's leading
        # count, updated wherever the column is rescaled -- which is every
        # column whose counts changed, since scaling touches the winner
        # columns of every write.
        self.relative = bool(synaptic_scaling) and w_max is None
        if self.relative:
            self.rel = torch.from_numpy(_rel_table(beta, max_rounds)).to(device)
            self.cmax = torch.zeros(B, n_post, dtype=torch.int32, device=device)
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
        r = rows.to(torch.int32).contiguous()
        d = self.mod.hashed_drive(r, self.seeds, self.n, self.threshold)
        has_mask = self._rowmask is not None and self._t > 0
        if self.store.nnz or has_mask:
            k_src = int(r.shape[1])
            need = self.B * k_src * self.n
            if self._scratch is None or self._scratch.numel() < need:
                self._scratch = torch.zeros(need, dtype=torch.int32,
                                            device=self.device)
            else:
                self._scratch[:need].zero_()
            sk, sc, so = (self.store.view() if self.store.nnz else
                          (torch.zeros(0, dtype=torch.int64,
                                       device=self.device),
                           torch.zeros(0, dtype=torch.int32,
                                       device=self.device),
                           torch.zeros(1, dtype=torch.int64,
                                       device=self.device)))
            rm = (self._rowmask if has_mask else
                  torch.zeros(0, dtype=torch.int64, device=self.device))
            cm = (self._colmask if has_mask else
                  torch.zeros(0, dtype=torch.int64, device=self.device))
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
                self.rel = torch.from_numpy(
                    _rel_table(self.beta, depth)).to(self.device)
        elif self.tab.numel() < depth + 1:
            self.tab = torch.from_numpy(
                _chain_table(self.beta, self.w_max, depth)).to(self.device)

    # -- writing ---------------------------------------------------------
    def begin_episode(self):
        """One 64-bit round mask covers an episode; the store holds the rest."""
        if not (self.learns or self.scale is not None):
            return
        self._rowmask = torch.zeros(self.B, 1, self.n_pre, dtype=torch.int64,
                                    device=self.device)
        self._colmask = torch.zeros(self.B, 1, self.n, dtype=torch.int64,
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
        c = cols.to(torch.int32).contiguous()
        B, K = c.shape
        need = B * K * self.n
        if self._cscratch is None or self._cscratch.numel() < need:
            self._cscratch = torch.zeros(need, dtype=torch.int32,
                                         device=self.device)
        else:
            self._cscratch[:need].zero_()
        if self.store.nnz:
            if self._colmap is None:
                self._colmap = torch.full((B, self.n), -1, dtype=torch.int32,
                                          device=self.device)
            else:
                self._colmap.fill_(-1)
            self._colmap.scatter_(
                1, cols, torch.arange(K, dtype=torch.int32,
                                      device=self.device).expand(B, K))
            sk, sc, _ = self.store.view()
        else:
            sk = torch.zeros(0, dtype=torch.int64, device=self.device)
            sc = torch.zeros(0, dtype=torch.int32, device=self.device)
        if self.relative:
            mass, colmax = self.mod.column_mass_rel(
                c, sk, sc,
                self._colmap if self.store.nnz else sk.to(torch.int32),
                self._rowmask, self._colmask,
                self._cscratch[:need].view(B, K, self.n),
                self.rel, self.seeds, self.n, self.threshold)
            self.cmax.scatter_(1, cols, colmax)
            new = column_scale(mass, self.setpoint)
            self.scale.scatter_(1, cols, new)
            return
        mass, cellmax = self.mod.column_mass_exact(
            c, sk, sc,
            self._colmap if self.store.nnz else sk.to(torch.int32),
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
        keep = [(a, b) for a, b in zip(self._prevs, self._news)
                if a.shape[1] and b.shape[1]]
        if not keep:
            return None, None
        T, B, dev = len(keep), self.B, self.device

        def ind(per_round):
            cat = torch.cat(per_round, dim=1)
            rid = torch.cat([
                torch.full((c.shape[1],), t, dtype=torch.int64, device=dev)
                for t, c in enumerate(per_round)])
            loc, vals, W = _local_index(cat)
            flat = torch.zeros(B, T * W, device=dev)
            flat.scatter_(1, rid.view(1, -1) * W + loc, 1.0)
            return flat.view(B, T, W), vals, W

        Rind, rows, _ = ind([a for a, _ in keep])
        Cind, cols, _ = ind([b for _, b in keep])
        counts = torch.bmm(Rind.transpose(1, 2), Cind)      # [B, R, C]
        bi, ri, ci = (counts > 0).nonzero(as_tuple=True)
        if bi.numel() == 0:
            return None, None
        key = (bi.to(torch.int64) * self.n_pre * self.n
               + rows[bi, ri] * self.n + cols[bi, ci])
        return key, counts[bi, ri, ci].to(torch.int32)

    @property
    def nnz(self):
        return self.store.nnz


class StimulusFiber:
    """A stimulus -> area projection. PRE-SUMMED: one weight per target.

    TWO PRICES THAT ARE EASY TO GET WRONG, both taken from the engine:

    * ``norm_init`` divides by ``d_j = deg_j + p * (tgt.n - stim_size)`` --
      note ``tgt.n``, NOT the stimulus size. That puts the stimulus on the same
      ``~n p`` divisor as the area fiber so the two are commensurable;
      dividing by its own in-degree would make an untrained stimulus
      contribute exactly 1.0 per touched neuron against an area contribution
      of ~0.06, and the stimulus would decide every winner.
    * ``w_max`` means "multiples of the INITIAL weight", and a stimulus weight
      starts near ``stim_size * p``, not at 1. The cap is
      ``w_max * max(1, stim_size * p)``.
    """

    def __init__(self, seeds, size, n_post, p, *, beta=0.0, w_max=None,
                 norm_init=False, max_rounds=64, device="cuda"):
        self.mod = _fused_cuda.load()
        B = len(seeds)
        self.B, self.size, self.n, self.p = B, size, n_post, float(p)
        self.seeds = torch.as_tensor(seeds, dtype=torch.int32, device=device)
        self.threshold = _fused_cuda.threshold_for(p)
        self.learns = bool(beta)
        rows = torch.arange(size, dtype=torch.int32,
                            device=device).expand(B, size).contiguous()
        self.base = self.mod.hashed_drive(rows, self.seeds, n_post,
                                          self.threshold)
        self.pot = torch.zeros(B, n_post, dtype=torch.int64, device=device)
        self.gain = torch.from_numpy(_gain_table(beta, max_rounds)).to(device)
        self.dj = ((self.base + self.p * (n_post - size)).clamp_min(1.0)
                   if norm_init else None)
        self.hi = (w_max * max(1.0, size * self.p)
                   if w_max is not None else float("inf"))

    #: DRIVE GAIN: a multiplier on this stimulus's contribution, 1.0 by
    #: default. It is the ANCHOR SHARE knob ([[capacity-is-an-anchor-ratio]],
    #: [[semantic-drive-share-is-the-lever]]): what an area forms as is decided
    #: by how much of its drive the stimulus supplies against learned fibers.
    #: The numpy engine's stimulus into a materialized area is 0-or-size, i.e.
    #: a full-size weight where this fiber generates a Binomial(size, p) count
    #: -- an accidental gain of ~1/p. Set explicitly rather than inherited.
    drive_gain = 1.0

    def contribute(self, drive, rows=None):
        d = self.base * self.gain[self.pot.clamp_max(self.gain.numel() - 1)]
        if self.hi != float("inf"):
            d = d.clamp_max(self.hi)
        if self.dj is not None:
            d = d / self.dj
        if self.drive_gain != 1.0:
            d = d * self.drive_gain
        drive += d

    def begin_episode(self):
        pass

    def observe(self, prev, new):
        if self.learns and new.shape[1]:
            self.pot.scatter_add_(1, new, torch.ones_like(new))

    def end_episode(self):
        pass


class HashedArea:
    """One area: its winners, its afferent fibers, and the k-WTA reading them.

    The projection loop is the algebra, once:

        for each round
            drive = SUM over fibers of fiber.contribute(...)
            winners = k-WTA(drive)
            each fiber observes (its source winners, the new winners)

    Nothing in here knows about `norm_init`, `w_max` or stimulus pricing --
    those live in the fiber that owns them, which is why adding an input is
    adding a fiber rather than a flag.
    """

    def __init__(self, n, k, seeds, device="cuda", refracted_strength=0.0,
                 tie_jitter=0.0):
        self.mod = _fused_cuda.load()
        if self.mod is None:
            raise RuntimeError(f"fused kernels unavailable: "
                               f"{_fused_cuda.last_error()}")
        self.n, self.k, self.B = n, k, len(seeds)
        self.device = device
        self.winners = torch.zeros(self.B, 0, dtype=torch.int64, device=device)
        #: neurons that have EVER fired -- what `rows/n` reads. Tracked here
        #: because the round masks are per-episode and get dropped.
        self.ever = torch.zeros(self.B, n, dtype=torch.bool, device=device)
        self.rounds_seen = 0
        #: REFRACTION, the engine's `refracted` mode (`core/_homeostasis.py`).
        #: A per-NEURON bias subtracted from drive before k-WTA and charged at
        #: the winners as `raw_drive * strength`, gated on plasticity exactly
        #: like the Hebbian write. It belongs to the area, not to a fiber: the
        #: reference charges against the TOTAL raw input of the winner, and
        #: the bias persists across `inhibit()` -- `RefractedArea.inhibit`
        #: does not touch it, only `reset()` does. At strength == beta the
        #: bias is the exact anti-Hebbian counterweight on a neuron's own
        #: repeated input (net drive stays at its base value), and a handicap
        #: on every other input; see PREREG_refraction_capacity.md.
        self.refracted_strength = float(refracted_strength or 0.0)
        self.bias = (torch.zeros(self.B, n, dtype=torch.float32, device=device)
                     if self.refracted_strength > 0 else None)
        #: TIE JITTER, opt-in. The selector breaks exact ties by smallest
        #: index -- canonical, and what every capacity result was measured
        #: with. But a STIMULUS-driven area under norm_init has a drive with a
        #: handful of distinct levels, so the k-WTA bar sits inside a tie
        #: class ([[KWTA-TIE-FRAGILE]]): at n=1000 k=50 p=0.05 the numpy
        #: engine's stimulus base leaves 41 neurons above the bar and 959 tied
        #: at zero, so 9 of 50 winners are tie-fill. A canonical rule then
        #: gives EVERY input the same low-index tie-fill, correlating assemblies
        #: that should be independent -- measured on the aligner as word
        #: assemblies that overlap across words and alignment that WORSENS with
        #: training. With `tie_jitter > 0` the ranking adds a deterministic
        #: per-(input pattern, column) offset below any real drive gap, so ties
        #: break pseudo-randomly, differently for different inputs, and
        #: reproducibly. The drive itself is untouched (parity replays still
        #: compare the true drive); only the ORDER among exact ties changes.
        self.tie_jitter = float(tie_jitter or 0.0)
        self._cols = torch.arange(n, dtype=torch.int64, device=device)

    def _jitter(self, fibers):
        """[B, n] offsets in [0, tie_jitter), keyed by the active fibers."""
        salt = torch.zeros(self.B, dtype=torch.int64, device=self.device)
        for f in fibers:
            salt = salt ^ (f.seeds.to(torch.int64) & 0xFFFFFFFF)
        h = (self._cols.view(1, -1) ^ salt.view(-1, 1)) * 0x9E3779B1
        h = (h ^ (h >> 15)) * 0x85EBCA6B
        h = (h ^ (h >> 13)) & 0xFFFFFFFF
        return h.to(torch.float32) * (self.tie_jitter / 4294967296.0)

    def apply_bias(self, raw):
        """Net drive the k-WTA ranks: raw minus the accumulated bias."""
        return raw if self.bias is None else raw - self.bias

    def charge(self, raw, new):
        """Charge the winners `raw * strength`, as `refraction_increment` does
        (it reconstructs raw as net + bias; here raw is still at hand)."""
        if self.bias is None:
            return
        self.bias.scatter_add_(
            1, new, torch.gather(raw, 1, new) * self.refracted_strength)

    def inhibit(self):
        """Clear the assembly. The next round is driven by afferents alone."""
        self.winners = torch.zeros(self.B, 0, dtype=torch.int64,
                                   device=self.device)

    def project(self, rounds, fibers, *, rows_for=None, freeze=False,
                stim_drive=None, return_drive=False, mask_bias=False,
                manage_episodes=True):
        """Run ``rounds`` rounds with ``fibers`` afferent.

        ``rows_for`` maps a fiber to its source winners; a fiber absent from it
        is driven by THIS area's winners, i.e. recurrently.

        ``mask_bias`` reads the SYNAPTIC memory alone: the refraction bias is
        neither subtracted nor charged. Only meaningful with ``freeze`` -- it
        is a probe of what the synapses hold with the intrinsic veto removed
        (PREREG_refraction_capacity P1), not a mode the reference has.

        ``manage_episodes=False`` leaves the fibers' episodes to the caller,
        so several one-round calls can share ONE episode (one mask, one GEMM
        fold into the store) -- a training step whose rounds interleave two
        areas cannot be a single multi-round call here, and per-round episodes
        cost a store append and merge each. Observe/charge still happen.
        """
        rows_for = rows_for or {}
        if mask_bias and not freeze:
            raise ValueError("mask_bias is a READOUT option; pass freeze=True")
        if not freeze and manage_episodes:
            for f in fibers:
                f.begin_episode()
        drive = None
        for _ in range(rounds):
            raw = torch.zeros(self.B, self.n, dtype=torch.float32,
                              device=self.device)
            for f in fibers:
                f.contribute(raw, rows_for.get(id(f), self.winners))
            if stim_drive is not None:
                raw = raw + stim_drive
            drive = raw if mask_bias else self.apply_bias(raw)
            ranked = (drive + self._jitter(fibers) if self.tie_jitter > 0
                      else drive)
            sel, ovf = self.mod.topk_select(ranked, min(self.k, self.n))
            bad = int(ovf.max())
            if bad:
                raise RuntimeError(
                    f"k-WTA candidate set overflowed ({bad} candidates) -- the "
                    "drive is too flat for the histogram to narrow. Refusing "
                    "to return a truncated winner set.")
            new = sel.to(torch.int64)
            if not freeze:
                for f in fibers:
                    f.observe(rows_for.get(id(f), self.winners), new)
                self.charge(raw, new)
                self.ever.scatter_(1, new, True)
                self.rounds_seen += 1
            self.winners = new
        if not freeze and manage_episodes:
            for f in fibers:
                f.end_episode()
        return (self.winners, drive) if return_drive else self.winners

    @property
    def fill(self):
        """``rows/n``: the fraction of the area that has ever fired."""
        return self.ever.sum(dim=1).float() / self.n
