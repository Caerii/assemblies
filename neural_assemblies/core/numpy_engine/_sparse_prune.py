"""k-WTA bound-and-prune for the numpy engine: the potentiated support of an area's columns,
and the set of columns a round must evaluate exactly (the rest are bounded out).

A mixin of NumpySparseEngine (_sparse.py), which owns the state these methods read."""
import os


import numpy as np


from typing import List


from ..backend import to_cpu


from ._growth import GrowthMixin, _self_fiber_deferred_init  # noqa: F401

from ._kwta_prune import (
    PotentiatedSupport, bound_outside, evaluate_set,
)

from ._sparse_switches import (  # noqa: F401  re-exported: the torch engine and _exact read them here
    _PRUNE_MAX_FRACTION, _explicit_src_norm_enabled, _fixed_target_plasticity_enabled,
    _strict_drive_enabled, _warn_fixed_target_enabled,
)


from ._drive_cache import (  # noqa: F401
    DriveCacheMixin, _csr_storage_available, _CSR_MIN_CELLS,
    _CSR_MAX_DENSITY,
)


class NumpyKwtaPrune:
    """The k-WTA prune's support and evaluate set (mixed into NumpySparseEngine)."""

    def _kwta_prune_on(self):
        if getattr(self, "kwta_prune", False):
            return True
        return os.environ.get(self._KWTA_PRUNE_ENV, "") == "1"

    def _support_for(self, src_name, target):
        """Per-fiber index of the cells plasticity has touched. Lazy, so an
        engine that never prunes never pays for one."""
        m = getattr(self, "_pot_support", None)
        if m is None:
            m = self._pot_support = {}
        key = (src_name, target)
        sup = m.get(key)
        if sup is None:
            sup = m[key] = PotentiatedSupport()
        return sup

    def drop_potentiated_support(self):
        """Forget every fiber's index.

        MUST be called by anything that renumbers compact indices -- the index
        is keyed on them, and a stale row->col map points at other neurons'
        columns ([[consolidation-resets-the-index-space]]). Dropping it only
        costs the prune; keeping a wrong one costs the science.
        """
        m = getattr(self, "_pot_support", None)
        if m:
            m.clear()

    def _prune_evaluate_set(self, target, tgt, from_stimuli, from_areas,
                            limit, record_activation=False):
        """Columns that must be gathered exactly, or None to gather all.

        THE DECISION IS MADE BEFORE ANY GATHER, which is what makes this safe
        to wire in without a fallback path. `drive[c] >= stim[c] + corr[c]`
        because the base term is non-negative, so the k-th largest of
        `stim + corr` over the evaluated set is a LOWER bound on the true tau.
        If that already clears `bound_outside`, pruning is provably valid and
        nothing has been computed twice. It declines more often than a
        gather-then-check would, and never guesses.
        """
        if not self._kwta_prune_on() or limit <= 0:
            return None
        # `record_activation` snapshots the FULL drive vector, so a pruned one
        # would hand the caller a partial vector that still looks like a
        # measurement. Checked HERE rather than at the call site so the hit
        # counter is not incremented for a projection that did not prune --
        # a counter that lies makes the guard untestable, which is how a
        # broken guard stays broken.
        if record_activation:
            return None
        k = int(tgt.k)
        if k <= 0:
            return None
        # GUARDS. Each breaks the bound; see `_kwta_prune`'s module docstring.
        if self.norm_init or self.synaptic_scaling:
            return None
        if float(getattr(tgt, "input_noise_std", 0.0) or 0.0) > 0.0:
            return None
        if getattr(tgt, "winner_policy", None) not in (None, "topk"):
            return None

        xp = self._xp
        stim_total = None
        if from_stimuli:
            stim_total = np.zeros(limit, dtype=np.float64)
            for stim in from_stimuli:
                sw = self._stim_conns[stim][target].weights
                end = min(limit, len(sw))
                if end > 0:
                    stim_total[:end] += np.asarray(to_cpu(sw[:end]),
                                                   dtype=np.float64)

        corr = np.zeros(limit, dtype=np.float64)
        touched: List[np.ndarray] = []
        total_active = 0
        for src_name in from_areas:
            if not self.fiber_learning_allowed(src_name, target):
                continue
            conn = self._area_conns[src_name][target]
            w = conn.weights
            # Only DENSE blocks carry a maintained index -- the sparse
            # representations answer `row_sum` natively and were never noted.
            if not isinstance(w, xp.ndarray) or getattr(w, "ndim", 0) != 2:
                return None
            src_w = xp.asarray(self._areas[src_name].winners)
            internal = [int(x) for x in to_cpu(src_w[src_w < w.shape[0]])]
            # COUNT EVERY ACTIVE ROW, not just the ones this block currently
            # covers. `eager_fiber_init` can materialise or widen a block
            # INSIDE the gather loop, after this decision has been taken, and
            # those fresh rows contribute base drive to columns whose bound was
            # computed without them. Counting `src.winners` is an upper bound
            # and therefore always safe; clipping to `w.shape[0]` undercounts
            # and makes the bound too low, which silently drops real winners.
            total_active += int(len(src_w))
            if not internal:
                continue
            cols = min(limit, int(w.shape[1]))
            c, t = self._support_for(src_name, target).correction(
                internal, w, cols)
            corr[:cols] += c
            if len(t):
                touched.append(t)
        if total_active == 0:
            return None

        touch = np.asarray(
            np.unique(np.concatenate(touched)) if touched
            else np.empty(0, dtype=np.int64), dtype=np.int64,
        )
        ev = np.asarray(evaluate_set(touch, stim_total, k, limit), dtype=np.int64)
        if len(ev) < k:
            return None

        def _tau(cand):
            lo = corr[cand] + (stim_total[cand]
                               if stim_total is not None else 0.0)
            return float(np.partition(lo, -k)[-k])

        tau_lo = _tau(ev)
        # WIDEN BY THE STIMULUS BEFORE GIVING UP. The base term is bounded by
        # |S| because it is Bernoulli 0/1; the STIMULUS term is not bounded by
        # anything, and on a real organ one symbol drives thousands of columns
        # hard. Evaluating only its top-k therefore leaves `bound_outside`
        # enormous and the prune declines on exactly the workload it was built
        # for -- measured on the Z60 arc: potentiated support 147 columns of
        # 19,999 (0.7%), and it still declined 1,798 times out of 1,798.
        #
        # So take every column whose stimulus alone could still reach tau, in
        # one pass. Widening can only help twice over: more candidates can only
        # raise the k-th largest, and every column moved inside lowers the max
        # left outside.
        if stim_total is not None and tau_lo > total_active:
            need = tau_lo - float(total_active)
            extra = np.nonzero(stim_total[:limit] >= need)[0]
            if len(extra):
                ev = np.asarray(np.union1d(ev, extra.astype(np.int64)), dtype=np.int64)
                if len(ev) >= k:
                    tau_lo = _tau(ev)

        # A prune that evaluates most of the area saves nothing and still pays
        # for the decision. Checked AFTER widening, since widening is what
        # decides how big the evaluated set really is.
        if len(ev) >= _PRUNE_MAX_FRACTION * limit or len(ev) < k:
            self._prune_misses = getattr(self, "_prune_misses", 0) + 1
            return None
        if tau_lo > bound_outside(stim_total, ev, limit, total_active):
            self._prune_hits = getattr(self, "_prune_hits", 0) + 1
            return ev.astype(np.int64)
        self._prune_misses = getattr(self, "_prune_misses", 0) + 1
        return None
