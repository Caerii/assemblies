"""Learning for the numpy engine: the Hebbian update w *= (1 + beta) clipped at w_max,
homeostatic synaptic scaling of area-to-area fibers (immediate or deferred and flushed), and
weight normalization.

A mixin of NumpySparseEngine (_sparse.py), which owns the state these methods read."""


import numpy as np


from typing import Optional


from ..backend import to_cpu


from .._homeostasis import (column_scale, scaling_applies, scaling_setpoint, check_area_homeostasis)


from ._growth import GrowthMixin, _self_fiber_deferred_init  # noqa: F401


from ._sparse_switches import (  # noqa: F401  re-exported: the torch engine and _exact read them here
    _PRUNE_MAX_FRACTION, _explicit_src_norm_enabled, _fixed_target_plasticity_enabled,
    _strict_drive_enabled, _warn_fixed_target_enabled,
)


from ._drive_cache import (  # noqa: F401
    DriveCacheMixin, _csr_storage_available, _CSR_MIN_CELLS,
    _CSR_MAX_DENSITY,
)


from ._csr_weights import CSRWeights

from ._virtual_weights import VirtualWeights


class NumpyPlasticity:
    """Hebbian update, synaptic scaling, normalization (mixed into NumpySparseEngine)."""

    # -- Plasticity ---------------------------------------------------------

    def _normalize_area_columns(self, target, from_areas, winners):
        """Homeostatic synaptic scaling on area->area fibers.

        Holds each postsynaptic neuron's TOTAL incoming weight on a fiber at a
        setpoint, so Hebbian learning redistributes a fixed budget across
        presynaptic sources instead of inflating the total. This is what stops
        the rich-get-richer runaway: belonging to an established assembly no
        longer buys extra total drive, only a different share of it.

        Two deliberate departures from the reference implementation
        (reference/nemo_numpy/areas.py:145-149), both forced by this engine's
        sparse representation:

        1. The setpoint is the initial expected column sum (``src.w * p``),
           NOT 1. Neurons that have never fired are not materialized here;
           their input is *sampled* as ~Binomial(active, p), which presumes
           weights of order 1. Normalizing materialized columns to sum to 1
           would leave them at total drive ~1 while fresh candidates sample
           ~k*p, so unmaterialized neurons would win every competition and no
           assembly could ever stabilize. Scaling to the initial expected sum
           keeps materialized and sampled neurons on the same scale, and is
           also the biologically accurate statement of synaptic scaling: a
           homeostatic setpoint, not unity.

        2. Stimulus fibers are excluded. The reference normalizes a 2-D
           (presynaptic x postsynaptic) matrix where only the *active* subset
           of presynaptic neurons delivers drive, so normalization redistributes
           across sources. This engine stores stimulus connectomes as 1-D
           pre-summed input, and a stimulus fires in full every step, so
           normalizing would drive every neuron to the identical value and
           erase the representation entirely. Stimulus weights stay bounded by
           w_max instead.

        Only the columns plasticity just touched are rescaled, so the cost is
        O(k) columns per step rather than the whole matrix.

        STATUS: opt-in and OFF by default, because this per-fiber formulation
        is not yet correct. Measured with recurrence enabled, it reduces the
        runaway substantially (independent-stimulus overlap 0.67 -> 0.22, and
        10-assembly overlap 3.0 against the literature's 4.0), but it breaks
        the two defining properties of an assembly: post-projection stability
        collapses to 0.01 (needs > 0.9) and pattern completion falls to 0.000
        (needs > 0.6).

        The reason is structural, not a tuning issue. Restoring each fiber to
        its own original total exactly cancels the net potentiation that makes
        an assembly self-sustaining -- an attractor requires the assembly's
        internal loop to end up stronger than its surroundings, and a per-fiber
        setpoint removes that gain by construction. The reference avoids this
        by normalizing EVERY fiber to a common scale, so a neuron's recurrent
        share can grow at the expense of its feedforward share.

        The fix is therefore joint normalization across all fibers into a
        neuron (stimulus + every area), holding the neuron's TOTAL drive
        constant while letting the recurrent/feedforward split shift. That
        requires the stimulus fibers to participate in the budget, which in
        turn needs their 1-D pre-summed representation reconciled with the
        2-D per-synapse form the split is defined over.
        """
        # The gate's spelling is interpreted ONCE, in the owner.
        if not scaling_applies(self.synaptic_scaling, target):
            return
        # SLOW HOMEOSTASIS (E9, #138): biological synaptic scaling operates
        # over hours-to-days, segregated from fast Hebbian plasticity --
        # and E8 measured why: per-update renormalization fights repeated
        # writes seed-bistably (variance +/-0.058 -> +/-0.120 under
        # repetition). Deferred mode ACCUMULATES touched columns and
        # normalizes only at flush_synaptic_scaling() (called by trainers
        # at phase boundaries): fast Hebbian inside a slowly renormalized
        # envelope. Default False = per-update, byte-identical.
        # RESEARCH KNOB, default "winners" = unchanged.
        #
        # THE ASYMMETRY. k-WTA selects among CANDIDATES. This rule only ever
        # touches columns that have ALREADY WON, so it cannot influence the
        # selection that produced them -- it arrives one step too late, every
        # step. `norm_init` divides EVERY candidate by its own degree at read
        # time, so a hub never gets its advantage in the first place; here a
        # hub keeps its full raw mass right up to the moment it wins and is cut
        # down only afterwards. Measured (`seq_scaling_merger_forensics`):
        # substrate C's multiply-shared neurons are the highest-degree columns
        # in the area (815 against a population 552) and the earliest recruited
        # (mean compact rank 2.0), yet their drive AFTER training is only
        # ~1.19x the population -- because normalization removed the advantage
        # after it had already been spent.
        #
        # "all" rescales every materialized column each round, normalizing the
        # candidates before the comparison instead of the winners after it.
        # MEASURED: cuts pairwise overlap 0.188 -> 0.142 and quadruples
        # half-cue completion 0.125 -> 0.500, on all three seeds. Real, and
        # only PART of the story -- the sampled (not yet materialized)
        # candidates cannot be rescaled at all, because they do not exist.
        # The full repair is `norm_init` alongside this, which cancels every
        # candidate's degree potentiation-invariantly at read time and takes
        # the overlap to 0.018-0.027, i.e. the chance floor.
        if getattr(self, "synaptic_scaling_scope", "winners") == "all":
            tgt = self._areas.get(target)
            if tgt is not None and int(tgt.w) > 0:
                winners = self._xp.arange(int(tgt.w))

        if getattr(self, "synaptic_scaling_deferred", False):
            pending = self._pending_scaling
            for src_name in from_areas:
                pending.setdefault((src_name, target), set()).update(
                    int(c) for c in winners)
            return
        self._scale_columns_now(target, from_areas, winners)

    def _scale_columns_now(self, target, from_areas, winners):
        check_area_homeostasis(target, refracted=self._areas[target].refracted,
                               synaptic_scaling=True)
        xp = self._xp
        cols = xp.asarray(winners, dtype=xp.int64)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            w = conn.weights
            if w is None or getattr(w, "ndim", 0) != 2 or w.shape[1] == 0:
                continue
            # LAYOUT FOLLOWS THE FIBER'S SHAPE. A scaled fiber pays two strided
            # ops per projection: this function's column gather/scatter over
            # src_rows x k (wants columns contiguous, F-order) and
            # `project_into`'s drive row-gather over k x tgt_cols (wants rows
            # contiguous, C-order). Whichever slice is LARGER should own the
            # contiguity: measured on the organ_p=0.5 pair, the tall 20000x4200
            # arc->state fiber scales at 34 ms in C vs 4 ms in F (scaling
            # dominates, 20000 > 4200), while the wide 4200x20000 state->arc
            # fiber's drive-read was 57% of project_into under blanket F-order
            # (drive dominates, 20000 > 4200 the other way). So: F-order iff
            # rows > cols. `asfortranarray` preserves logical [i,j], so either
            # layout is BYTE-IDENTICAL end-to-end -- proven by identical census
            # and connectome crc at PRES=8 and PRES=24; layout only moves time.
            # Converted lazily and persisted on the connection; growth
            # reallocates C-order (amortised doubling, front-loaded to early
            # recruitment ~log2(n/k) times), and this reconverts on the next
            # touch, so there is no per-step thrash.
            if (isinstance(w, xp.ndarray) and w.shape[0] > w.shape[1]
                    and not w.flags.f_contiguous):
                w = xp.asfortranarray(w)
                conn.weights = w
            valid = cols[cols < w.shape[1]]
            if len(valid) == 0:
                continue
            # Use the LOGICAL source size, not w.shape[0]: physical capacity is
            # amortised (it doubles), so the rows beyond src.w are unallocated
            # padding. Summing or setting the setpoint over them inflates it
            # and silently under-normalizes.
            rows = min(int(self._areas[src_name].w), int(w.shape[0]))
            if rows <= 0:
                continue
            sub = w[:rows, valid]
            sums = sub.sum(axis=0)
            # THE FIBER'S p, NOT THE BRAIN'S -- third member of the defect
            # class ([[pricing-law-implemented-twice]]; _norm_scale 79fba4f,
            # the stimulus w_max clamp). The setpoint is the initial expected
            # column sum OF THIS FIBER; pricing it at the global p renormalized
            # every trained column of a p=0.40 organ fiber inside a p=0.05
            # brain to 1/8 of its natural mass, while untouched columns kept
            # full mass -- inverting learning exactly like substrate B did.
            # Found by the substrate-C smoke run (every transition soft).
            setpoint = float(scaling_setpoint(rows, self._p_for(src_name, target)))
            # RESEARCH KNOB, default "population" = the line above, unchanged.
            # "degree" restores neuron j to the mass IT started with rather
            # than to the mass an average neuron started with.
            #
            # MEASURED AND REFUTED -- kept because a refuted arm is evidence,
            # and because the idea is the obvious one to re-try. It is not a
            # near-miss, it is the worst arm ever measured on this substrate:
            # `seq_scaling_merger_forensics` (n=2000 k=50 p=0.5 T=8 M=8, in
            # regime) gives pairwise overlap 0.667 = 26.7x chance against the
            # population setpoint's 0.188, and breaks retrieval from the FULL
            # cue (rank-1 0.625, where every other arm scores 1.000).
            #
            # The reason is that it is not a normalization at all. Restoring a
            # column to its own initial mass is a no-op on the structure and
            # cancels only the potentiation, so the raw in-degree competition
            # comes back undamped and the hubs win everything -- the collapse
            # `norm_init` exists to prevent ([[recurrence-needs-norm-init]]),
            # and the per-column form of the per-fiber failure this method's
            # own docstring already records.
            if getattr(self, "synaptic_scaling_setpoint", "population") == "degree":
                deg = xp.count_nonzero(sub, axis=0).astype(sums.dtype)
                setpoint = xp.where(deg > 0, deg, 1.0)
            scale = column_scale(sums, setpoint, xp=xp)
            w[:rows, valid] = sub * scale

    def flush_synaptic_scaling(self) -> int:
        """Apply deferred homeostatic scaling to every touched column.

        The slow half of the fast/slow separation (see
        _normalize_area_columns). Setpoints use the areas' CURRENT logical
        sizes -- slow homeostasis regulates the state as it stands at the
        boundary, which is the semantics the timescale argument wants.
        Returns the number of (src, tgt) fibers scaled.
        """
        pending = getattr(self, "_pending_scaling", None)
        if not pending:
            return 0
        n = 0
        for (src_name, target), cols in list(pending.items()):
            if not self.fiber_learning_allowed(src_name, target):
                continue
            self._scale_columns_now(target, [src_name], sorted(cols))
            del pending[(src_name, target)]
            n += 1
        return n

    def _apply_plasticity(self, target, from_stimuli, from_areas, winners):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-fiber-learning

        Hebbian learning and triggered scaling on permitted fibers only.
        """
        if not self._plasticity_enabled_global:
            return
        from_stimuli = [s for s in from_stimuli if self.fiber_learning_allowed(s, target)]
        from_areas = [s for s in from_areas if self.fiber_learning_allowed(s, target)]
        xp = self._xp
        tgt = self._areas[target]
        winners_arr = xp.asarray(winners, dtype=xp.int64)

        # Stimulus -> area (1-D weights)
        for stim_name in from_stimuli:
            conn = self._stim_conns[stim_name][target]
            beta = tgt.beta_by_source.get(stim_name, tgt.beta)
            if beta == 0:
                continue
            valid = winners_arr[winners_arr < len(conn.weights)]
            if len(valid) > 0:
                conn.weights[valid] *= (1 + beta)
            if self.w_max is not None:
                # w_max means "multiples of the initial weight". Area->area
                # weights start at 1, so the raw cap is correct there. A
                # stimulus connectome instead stores PRE-SUMMED input, which
                # starts at about stim_size * p -- with the defaults that is
                # ~20 for a cap of 20, so the very first update clipped every
                # winner to exactly w_max and pinned it there. Plasticity then
                # had no effect at any beta: assembly recovery measured a flat
                # 0.30 for beta = 0.001, 0.01 and 0.1 alike. Scaling the cap
                # by the initial magnitude restores the intended semantics.
                stim = self._stimuli.get(stim_name)
                # THIS FIBER's density, not the global one. A stimulus
                # connectome stores PRE-SUMMED input starting near
                # ``size * p``, so the clamp has to be scaled by the same p the
                # weights were DRAWN at. Using the global p clamped a dense
                # fiber inside a sparse brain at its sparse ceiling: drawn at
                # p=0.4 the weights start near 28 but were clipped at
                # w_max * 70 * 0.05 = 70, so 15 presentations of Hebbian growth
                # (1.1^15 = 4.18, reaching ~117) saturated instead. A saturated
                # conjunct cannot discriminate, and the mod-3 FSM fell from
                # 10/10 correct trajectories to 1/10.
                scale = max(1.0, float(getattr(stim, "size", 1))
                            * self._p_for(stim_name, target))
                lo, hi = self._weight_bounds(scale)
                xp.clip(conn.weights, lo, hi, out=conn.weights)

        # Area -> area (2-D weights)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            beta = tgt.beta_by_source.get(src_name, tgt.beta)
            if beta == 0:
                continue
            src = self._areas[src_name]
            src_w = xp.asarray(src.winners)
            if not conn.sparse and getattr(src, "explicit_source", False):
                valid_rows = src_w[src_w < conn.weights.shape[0]]
                post_ids = [
                    tgt.compact_to_neuron_id[int(c)]
                    for c in winners
                    if int(c) < len(tgt.compact_to_neuron_id)
                ]
                if len(valid_rows) > 0 and len(post_ids) > 0:
                    conn.update_weights(
                        to_cpu(valid_rows), post_ids, beta, w_max=self.w_max)
                continue
            if isinstance(conn.weights, VirtualWeights):
                vw = conn.weights
                valid_rows = src_w[src_w < vw.n_rows]
                valid_cols = winners_arr[winners_arr < vw.n_cols]
                if len(valid_rows) > 0 and len(valid_cols) > 0:
                    try:
                        vw.bump(np.asarray(to_cpu(valid_rows)),
                                np.asarray(to_cpu(valid_cols)), float(beta))
                    except ValueError:
                        # Beta changed mid-life: one exponent store cannot
                        # carry two growth factors, so DENSIFY -- correct
                        # under any beta history -- and apply this event on
                        # the dense block below.
                        conn.weights = self._to_xp(vw.todense())
                        conn._deg_counts_arr = None
                        conn._deg_rows = 0
                        ix = xp.ix_(valid_rows, valid_cols)
                        conn.weights[ix] *= (1 + beta)
                        if self.w_max is not None:
                            sub = conn.weights[ix]
                            _lo, _hi = self._weight_bounds()
                            xp.clip(sub, _lo, _hi, out=sub)
                            conn.weights[ix] = sub
                continue
            if conn.weights.ndim == 2:
                valid_rows = src_w[src_w < conn.weights.shape[0]]
                valid_cols = winners_arr[winners_arr < conn.weights.shape[1]]
                if len(valid_rows) > 0 and len(valid_cols) > 0:
                    ix = xp.ix_(valid_rows, valid_cols)
                    conn.weights[ix] *= (1 + beta)
                    # Record the touched support for the k-WTA prune. This is
                    # the ONLY place a dense fiber learns which cells are
                    # potentiated -- the block stores their values but not
                    # their index, and recovering it later would cost exactly
                    # the O(k*n) scan the prune exists to avoid. O(k^2) here
                    # against O(k*n) there.
                    if self._kwta_prune_on():
                        self._support_for(src_name, target).note(
                            np.asarray(to_cpu(valid_rows)),
                            np.asarray(to_cpu(valid_cols)))
                    if self.w_max is not None:
                        sub = conn.weights[ix]
                        _lo, _hi = self._weight_bounds()
                        xp.clip(sub, _lo, _hi, out=sub)
                        conn.weights[ix] = sub
            else:
                valid = winners_arr[winners_arr < len(conn.weights)]
                if len(valid) > 0:
                    conn.weights[valid] *= (1 + beta)
                if self.w_max is not None:
                    _lo, _hi = self._weight_bounds()
                    xp.clip(conn.weights, _lo, _hi, out=conn.weights)

        # Homeostatic scaling closes the loop on the update just applied.
        # (Runs BEFORE connectome expansion, so a first-time winner's
        # freshly-expanded column can sit above the setpoint until its next
        # update -- pinned in test_scoped_synaptic_scaling.py.)
        self._normalize_area_columns(target, from_areas, winners)

    # -- Weight normalization -----------------------------------------------

    def normalize_weights(self, target: str, source: Optional[str] = None) -> None:
        """Column-normalize weights into *target* so each neuron sums to 1.0."""
        check_area_homeostasis(target, refracted=self._areas[target].refracted,
                               synaptic_scaling=True)
        self.invalidate_csr_drive()
        xp = self._xp
        eps = 1e-8

        def _norm_conn(conn):
            w = conn.weights
            if isinstance(w, CSRWeights):
                w.normalize_columns(eps)
                return
            if w.ndim == 2 and w.size > 0:
                col_sums = w.sum(axis=0, keepdims=True)
                col_sums = xp.maximum(col_sums, eps)
                conn.weights = w / col_sums
            elif w.ndim == 1 and w.size > 0:
                total = float(xp.sum(w))
                if total > eps:
                    conn.weights = w / total

        if source is not None:
            if source in self._stim_conns and target in self._stim_conns[source]:
                _norm_conn(self._stim_conns[source][target])
            if source in self._area_conns and target in self._area_conns[source]:
                _norm_conn(self._area_conns[source][target])
            return

        for stim_name in self._stim_conns:
            if target in self._stim_conns[stim_name]:
                _norm_conn(self._stim_conns[stim_name][target])
        for src_name in self._area_conns:
            if target in self._area_conns[src_name]:
                _norm_conn(self._area_conns[src_name][target])
