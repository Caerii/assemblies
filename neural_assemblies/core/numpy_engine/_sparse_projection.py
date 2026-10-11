"""Projection for the numpy engine: one round of drive, k-WTA and learning into a target
area, as a sequence of phases over a _ProjectionRound (the same phases as the torch engine's).

A mixin of NumpySparseEngine (_sparse.py), which owns the state these methods read."""


import numpy as np


from typing import Any, List, Optional, cast


from ..backend import to_cpu

from .._pricing import (
    area_fiber_activity,
)

from .._homeostasis import (refraction_increment)

from ..engine import (
    ProjectionResult,
)


from ..semantics import (
    SampledRecurrencePolicy,
)


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


class _ProjectionRound:
    """What the phases of one NumpySparseEngine.project_into hand each other.

    The call's arguments, then, in the order the phases set them:

        tgt                       the target area's state
        rng                       the round's generator, drawn from the engine's
        _pre_kwta_snapshot, _raw_prev, _pre_kwta_total_val, _pre_kwta_count_val
                                  the activation snapshots a recording round takes
        _deferred_init_srcs       area fibers still empty: initialised after the winners
        _silent_area_srcs         area fibers named but delivering no drive this round
        prev_winner_inputs        the drive of every materialized neuron
        input_sizes, input_ps     per fiber: the active inputs it prices, its density
        norm_div                  norm_init's candidate divisor (None without norm_init)
        draw_key, fiber_sig, fiber_cur, eff
                                  the stable candidate stream's key and position
        all_inputs                the drive of the materialized neurons and the sampled
                                  candidates: the vector k-WTA ranks
        new_winner_indices        the winners
        num_first, first_winner_inputs, new_w
                                  the recruits, their sampled inputs, the area's new size

    A field read before any phase set it raises AttributeError, where the single
    function the phases were cut from raised UnboundLocalError."""

    __slots__ = ("target", "from_stimuli", "from_areas", "plasticity_enabled",
                 "record_activation", "tgt", "rng", "_pre_kwta_snapshot", "_raw_prev",
                 "_pre_kwta_total_val", "_pre_kwta_count_val", "_deferred_init_srcs",
                 "_silent_area_srcs", "prev_winner_inputs", "input_sizes", "input_ps",
                 "norm_div", "draw_key", "fiber_sig", "fiber_cur", "eff", "all_inputs",
                 "new_winner_indices", "num_first", "first_winner_inputs", "new_w")

    def __init__(self, *, target, from_stimuli, from_areas, plasticity_enabled,
                 record_activation):
        self.target = target
        self.from_stimuli = from_stimuli
        self.from_areas = from_areas
        self.plasticity_enabled = plasticity_enabled
        self.record_activation = record_activation


class NumpyProjection:
    """project_into and its phases (mixed into NumpySparseEngine)."""

    def project_into(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        record_activation: bool = False,
    ) -> ProjectionResult:
        """One round of projection into ``target``: drive, k-WTA, learning.

        The phases, in order (each a method below; the state they hand each other is
        a _ProjectionRound; the torch engine's project_into has the same phases):

            _project_admit        the policy check, the round's generator, live sources
            _project_fixed        a fixed target: winners kept, afferents learn  -> result
            _project_no_inputs    nothing projects: the assembly is kept         -> result
            _project_drive        the drive of every materialized neuron
                                  (an explicit dense source bootstraps instead)  -> result
            _project_zero_signal  no drive: kept, or noise picks the winners     -> result
            _project_compiled     a compiled area: top-k on pregrown columns     -> result
            _project_candidates   the never-fired neurons' best sampled inputs
            _project_penalties    LRI and refracted bias; recording snapshots
            _project_select       k-WTA
            _project_recruit      first-time winners become neurons
            _project_learn        the Hebbian update; synapses for the recruits
            _project_commit       winners, LRI history and bias become the area's state
            _project_result       total drive, deferred fiber initialisation, the result

        A phase marked -> result may end the round early with its result.
        """
        proj = _ProjectionRound(target=target, from_stimuli=from_stimuli, from_areas=from_areas,
                                plasticity_enabled=plasticity_enabled,
                                record_activation=record_activation)
        self._project_admit(proj)
        if (out := self._project_fixed(proj)) is not None:
            return out
        if (out := self._project_no_inputs(proj)) is not None:
            return out
        if (out := self._project_drive(proj)) is not None:
            return out
        if (out := self._project_zero_signal(proj)) is not None:
            return out
        if (out := self._project_compiled(proj)) is not None:
            return out
        self._project_candidates(proj)
        self._project_penalties(proj)
        self._project_select(proj)
        self._project_recruit(proj)
        self._project_learn(proj)
        self._project_commit(proj)
        return self._project_result(proj)

    def _project_admit(self, proj):
        """Admit the round: the target may be projected into (the sampled-recurrence policy), the
        round's generator is drawn, the CSR drive mirror is dropped before a learning
        round, and source areas without an assembly are left out."""
        proj.tgt = self._areas[proj.target]
        self.validate_probe_target(proj.target)
        if (
            proj.target in proj.from_areas
            and not proj.tgt.fixed_assembly
            and not self._no_recruitment
            and proj.tgt.winners.size > 0
            and proj.tgt.w > 0
            and proj.tgt.w < proj.tgt.n
        ):
            policy = getattr(
                self,
                "sampled_recurrence_policy",
                SampledRecurrencePolicy.WARN,
            )
            if policy is SampledRecurrencePolicy.FORBID:
                raise RuntimeError(
                    f"Recurrent projection into {proj.target!r} is forbidden because its "
                    "numpy connectome is still sampled. Materialize the area or use "
                    "a fixed-connectome engine. See "
                    "research/notes/sequence/PREREG_sampler_audit.md."
                )
            if (
                policy is SampledRecurrencePolicy.WARN
                and not getattr(self, "_sampled_recurrence_warned", False)
            ):
                import warnings

                warnings.warn(
                    f"Recurrent projection into {proj.target!r} uses the sampled numpy "
                    "connectome. Sequence-dynamics numbers are void until rerun "
                    "materialized or on a fixed-connectome engine (numpy_exact or "
                    "the hashed substrate). See "
                    "research/notes/sequence/PREREG_sampler_audit.md. Use "
                    "Brain.materialize_area before training if materialized "
                    "semantics are intended, or explicitly select "
                    "sampled_recurrence_policy='acknowledged' for a deliberate "
                    "sampled-engine comparison.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._sampled_recurrence_warned = True
        proj.rng = np.random.default_rng(self._rng.integers(0, 2**32))
        # Optional trace fields have stable defaults even when recording is
        # disabled, so the result assembly below never depends on branch-local
        # variables.
        proj._pre_kwta_snapshot: Optional[np.ndarray] = None
        proj._raw_prev: Optional[np.ndarray] = None
        proj._pre_kwta_total_val = 0.0
        proj._pre_kwta_count_val = 0

        # A learning round may rewrite any block, so no CSR mirror survives it.
        # This is the PRIMARY guarantee that `_csr_row_sum` cannot read stale
        # weights; the per-site invalidations below are belt and braces.
        if proj.plasticity_enabled:
            self._csr_drive.clear()

        # Filter out source areas with no assembly
        proj.from_areas = [
            a for a in proj.from_areas
            if self._areas[a].winners.size > 0
            and (
                self._areas[a].w > 0
                or getattr(self._areas[a], "explicit_source", False)
            )
        ]

    def _project_fixed(self, proj):
        """A FIXED target keeps its winners; its afferents still learn onto them (the reference's
        semantics). Returns the round's result, or None for an area that is not fixed."""
        # Fixed assembly — the winners do not move. Whether the AFFERENTS still
        # learn is the question, and we used to answer it differently from the
        # reference: inputs were discarded and no plasticity was applied, so a
        # projection into a fixed area looked like training but wrote nothing.
        # That footgun produced three separate "the mechanism doesn't work"
        # investigations (merge, associate, direct binding) -- and a fourth,
        # since it also made `reciprocal_project` unable to restore its source.
        #
        # The reference potentiates the afferents onto the frozen winners; see
        # `_fixed_target_plasticity_enabled` for the code, the idiom it enables
        # and the numbers. So the default now learns, and only the WINNERS are
        # held. The old short-circuit remains available for A/B.
        if proj.tgt.fixed_assembly:
            learn = (proj.plasticity_enabled and (proj.from_stimuli or proj.from_areas)
                     and _fixed_target_plasticity_enabled())
            if learn:
                # Size any never-used source block FIRST -- a multiplicative
                # `w *= 1 + beta` cannot grow an unmaterialised (0, 0) block,
                # so without this the potentiation would be a no-op of exactly
                # the kind this branch is being fixed for.
                self._init_deferred_area_srcs(proj.target, proj.from_areas, int(proj.tgt.w))
                self._apply_plasticity(
                    proj.target, proj.from_stimuli, proj.from_areas, proj.tgt.winners)
            elif (proj.plasticity_enabled and (proj.from_stimuli or proj.from_areas)
                    and _warn_fixed_target_enabled()):
                import warnings
                warnings.warn(
                    f"projection into FIXED area {proj.target!r} with inputs "
                    f"{list(proj.from_stimuli) + list(proj.from_areas)} is a no-op: the "
                    f"inputs are discarded and no plasticity is applied. Fix "
                    f"the SOURCE, or drive the target with a stimulus instead "
                    f"of fixing it (see ops.merge).",
                    RuntimeWarning, stacklevel=3,
                )
            return ProjectionResult(
                winners=np.array(to_cpu(proj.tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=proj.tgt.w,
            )

    def _project_no_inputs(self, proj):
        """With no inputs the assembly is kept. Returns the result, or None when there are inputs."""
        # No inputs -> keep assembly unchanged
        if len(proj.from_stimuli) == 0 and len(proj.from_areas) == 0:
            return ProjectionResult(
                winners=np.array(to_cpu(proj.tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=proj.tgt.w,
            )

    def _project_drive(self, proj):
        """The drive every materialized neuron receives from the inputs' current winners: stimulus
        fibers, then area fibers (deferring the initialisation of fibers that are still
        empty). An explicit dense source into an empty target bootstraps instead, and
        returns its result."""
        xp = self._xp
        # --- Accumulate inputs from previous winners ---
        proj.prev_winner_inputs = xp.zeros(proj.tgt.w, dtype=xp.float32)
        explicit_dense_act = None

        # Stimulus inputs (1-D slice up to w)
        limit = proj.tgt.w
        for stim in proj.from_stimuli:
            stim_conn = self._stim_conns[stim][proj.target]
            stim_w = stim_conn.weights
            end = min(limit, len(stim_w))
            if end > 0:
                nscale = self._norm_scale(
                    stim_conn, proj.tgt.n, self._stimuli[stim].size, end,
                    p=self._p_for(stim, proj.target))
                if nscale is None:
                    proj.prev_winner_inputs[:end] += stim_w[:end]
                else:
                    proj.prev_winner_inputs[:end] += stim_w[:end] * nscale[:end]

        # k-WTA BOUND-AND-PRUNE: which columns actually have to be gathered.
        # Decided from the potentiated support alone, before any gather, so a
        # declined prune costs one cheap pass and never a redundant one.
        _cand_cols = self._prune_evaluate_set(
            proj.target, proj.tgt, proj.from_stimuli, proj.from_areas, limit,
            record_activation=proj.record_activation)

        # Area inputs (2-D, vectorised fancy-index)
        # Track sources whose connectomes need deferred initialisation.
        proj._deferred_init_srcs = []
        # Area sources that were NAMED but delivered no drive this round --
        # their weight block is not materialised yet. Priced out below.
        proj._silent_area_srcs: set = set()
        for src_name in proj.from_areas:
            conn = self._area_conns[src_name][proj.target]
            src = self._areas[src_name]
            if not conn.sparse and getattr(src, "explicit_source", False):
                src_w = xp.asarray(src.winners)
                valid = src_w[src_w < conn.weights.shape[0]]
                if len(valid) == 0:
                    continue
                # norm_init applies to THIS fiber too.  The connectome is dense
                # and full-width, so its columns are NEURON IDs rather than
                # compact indices -- ask for the whole width and index the
                # result by neuron id.  All rows exist, so rows_known is the
                # full row count and `_norm_scale`'s unknown-row term is zero.
                # See `_explicit_src_norm_enabled` for what omitting this did.
                enorm = None
                if _explicit_src_norm_enabled():
                    enorm = self._norm_scale(
                        conn, src.n, conn.weights.shape[0],
                        int(conn.weights.shape[1]),
                        p=self._p_for(src_name, proj.target),
                    )
                if proj.tgt.w == 0:
                    contrib = conn.weights[valid].sum(axis=0)
                    if enorm is not None:
                        contrib = contrib * enorm[:len(contrib)]
                    if explicit_dense_act is None:
                        explicit_dense_act = contrib.astype(xp.float32, copy=True)
                    else:
                        explicit_dense_act += contrib
                    continue
                neuron_ids = xp.asarray(
                    proj.tgt.compact_to_neuron_id, dtype=xp.int64,
                )
                valid_cols = neuron_ids[neuron_ids < conn.weights.shape[1]]
                if len(valid_cols) > 0:
                    contrib = conn.weights[valid][:, valid_cols].sum(axis=0)
                    if enorm is not None:
                        contrib = contrib * enorm[valid_cols]
                    end = min(limit, len(contrib))
                    if end > 0:
                        proj.prev_winner_inputs[:end] += contrib[:end]
                continue
            if conn.weights.shape[1] == 0:
                # EAGER INIT: size the block NOW, so the fiber delivers on the
                # round it is first named rather than the one after.
                #
                # Deferring it is what kept PNAS Fig. 2 B1-B3 out of reach. The
                # paper's ~50% overlap(y1,y2) comes from y1's neurons getting
                # potentiated afferent input PLUS recurrent input from y1,
                # which together are comparable to a fresh candidate's
                # unpotentiated afferent plus the same recurrent term. Deferred,
                # the recurrent half simply is not there on the round that
                # decides y2, so the comparison is not the paper's.
                if (self.eager_fiber_init and not self._no_recruitment
                        and conn.sparse
                        and (src_name != proj.target
                             or _self_fiber_deferred_init())
                        and self._areas[src_name].w > 0 and proj.tgt.w > 0):
                    conn.weights = self._init_area_block(
                        src_name, proj.target, 0, int(self._areas[src_name].w),
                        0, int(proj.tgt.w))
                if conn.weights.shape[1] == 0:
                    # Still empty -- either eager init is off, or there is
                    # genuinely nothing to connect yet (w == 0 on one side).
                    # Mark for deferred init so it works on the NEXT round. See
                    # `_self_fiber_deferred_init` for why SELF fibers are
                    # excluded by default and what that costs.
                    #
                    # THIS ROUND THE FIBER DELIVERS NOTHING, and it must not be
                    # charged into the candidate price either -- see
                    # `_silent_area_srcs` below.
                    if (conn.sparse
                            and (src_name != proj.target
                                 or _self_fiber_deferred_init())
                            and self._areas[src_name].w > 0 and proj.tgt.w > 0):
                        proj._deferred_init_srcs.append(src_name)
                    proj._silent_area_srcs.add(src_name)
                    continue
            # STALE COVERAGE (#151 dead fiber): growth is recruitment-gated
            # (`_expand_connectomes` returns early with no first-time winner),
            # so a no-recruitment episode after the source has grown freezes
            # this block forever -- out-of-range rows are dropped from the
            # slice below, zero drive recruits nobody, and no recruitment
            # means no expansion. Mark for the deferred repair (same
            # next-round semantics as the empty-block path above). Self
            # fibers keep the `_self_fiber_deferred_init` gate, and a
            # read_only() probe must not repair -- growth is exactly the
            # channel that contract closes, and a stale fiber read under it
            # honestly reports the trained brain as it is.
            if (conn.sparse and not self._no_recruitment
                    and not isinstance(conn.weights, CSRWeights)
                    and getattr(conn.weights, "ndim", 0) == 2
                    and (src_name != proj.target or _self_fiber_deferred_init())):
                _cov_r = min(int(getattr(conn, "_log_rows",
                                         conn.weights.shape[0])),
                             int(conn.weights.shape[0]))
                _cov_c = min(int(getattr(conn, "_log_cols",
                                         conn.weights.shape[1])),
                             int(conn.weights.shape[1]))
                if int(src.w) > _cov_r or int(proj.tgt.w) > _cov_c:
                    proj._deferred_init_srcs.append(src_name)
            src_w = xp.asarray(src.winners)
            internal = src_w[src_w < conn.weights.shape[0]]
            if len(internal) > 0 and limit > 0:
                col_end = min(limit, conn.weights.shape[1])
                # A materialised block is ~p occupied; CSR gathers k rows out
                # of it 3-11x faster and bit-identically. Only consulted with
                # plasticity off (see `_csr_row_sum`), so it cannot go stale.
                contrib = None
                if isinstance(conn.weights, VirtualWeights):
                    # The drive kernel regenerates the k winner rows
                    # (~k x n_cols hashed cells) instead of traversing a
                    # stored block; deviations come from per-row dicts.
                    contrib = self._to_xp(conn.weights.row_sum(
                        np.asarray(to_cpu(internal)), col_end))
                elif isinstance(conn.weights, CSRWeights):
                    # Stored sparse: answer natively, no mirror needed, and
                    # safe with plasticity ON because there is nothing cached.
                    contrib = conn.weights.row_sum(internal, col_end)
                elif not proj.plasticity_enabled and _cand_cols is None:
                    contrib = self._csr_row_sum(
                        src_name, proj.target, conn.weights, internal, col_end)
                if contrib is None and _cand_cols is not None:
                    # Gather ONLY the columns whose bound has not already lost.
                    # Values stay bit-identical: taking a column subset never
                    # reorders any column's row sum. Pruned slots keep 0, which
                    # is below their true drive and far below tau, so they
                    # cannot enter the top-k either way -- and leaving them at
                    # 0 rather than -inf keeps every other consumer of this
                    # vector (the zero-signal check, total_activation) honest.
                    sub = _cand_cols[_cand_cols < col_end]
                    if len(sub) > 0:
                        proj.prev_winner_inputs[sub] += cast(Any, conn.weights)[
                            xp.ix_(internal, sub)].sum(axis=0)
                    continue
                if contrib is None:
                    contrib = cast(Any, conn.weights)[internal, :col_end].sum(axis=0)
                nscale = self._norm_scale(
                    conn, self._areas[src_name].n,
                    self._areas[src_name].w, col_end,
                    p=self._p_for(src_name, proj.target))
                if nscale is not None:
                    contrib = contrib * nscale[:col_end]
                proj.prev_winner_inputs[:col_end] += contrib

        if explicit_dense_act is not None and proj.tgt.w == 0:
            return self._bootstrap_from_explicit_dense(
                proj.target,
                explicit_dense_act,
                proj.from_stimuli,
                proj.from_areas,
                plasticity_enabled=proj.plasticity_enabled,
                rng=proj.rng,
                record_activation=proj.record_activation,
            )

    def _project_zero_signal(self, proj):
        """No drive at all: the assembly is preserved (or, with input noise, noise alone picks the
        winners). Returns the result, or None when there is signal."""
        xp = self._xp
        # Zero signal -> preserve current assembly
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-noise-only-observation
        zero_signal = len(proj.prev_winner_inputs) > 0 and not bool(xp.any(proj.prev_winner_inputs))
        if zero_signal and proj.tgt.input_noise_std > 0 and proj.tgt.w < proj.tgt.n:
            raise ValueError('noise-only projection requires a fully materialized population')
        if zero_signal and proj.tgt.input_noise_std == 0:
            # RUN THE DEFERRED INIT BEFORE RETURNING. Without this the
            # mechanism is UNREACHABLE in the one case it exists for: a source
            # area projecting into an already-grown target for the first time.
            # Its connectome block is empty, so it contributes nothing, so the
            # total is zero, so we return here -- before the consumption site
            # at the end of this function. Next round is identical, forever.
            #
            # The observable symptom is not a zero assembly but a STALE one:
            # this branch preserves `tgt.winners`, so every item driven through
            # the dead fiber "stores" whatever the target last held. Measured
            # on a role area fed by LEX_NOUN then LEX_VERB: all 20 verbs
            # returned the 40th noun's assembly, and swapping the training
            # order swapped which category collapsed (40 nouns onto the last
            # verb). Whichever source happens to go first is the only one that
            # ever works, and nothing warns.
            self._init_deferred_area_srcs(
                proj.target, proj._deferred_init_srcs, int(proj.tgt.w))
            if _strict_drive_enabled() and (proj.from_stimuli or proj.from_areas):
                import warnings
                empty = [s for s in proj.from_areas
                         if getattr(self._area_conns.get(s, {}).get(proj.target),
                                    "weights", None) is None
                         or tuple(getattr(
                             self._area_conns[s][proj.target].weights,
                             "shape", (0, 0)) or (0, 0))[1:2] == (0,)]
                warnings.warn(
                    f"projection into {proj.target!r} from "
                    f"{list(proj.from_stimuli) + list(proj.from_areas)} delivered ZERO "
                    f"drive; {proj.target!r} keeps its previous assembly, so this "
                    f"looks like it worked and stored nothing new"
                    + (f". Empty weight blocks: "
                       f"{', '.join(f'{s}->{proj.target}' for s in empty)}"
                       if empty else "")
                    + ". Common causes: reset_area_connections zeroed the "
                      "connectome (k-WTA then returns the same index "
                      "tie-break winners for every input), or a source is "
                      "projecting into this area for the first time.",
                    RuntimeWarning, stacklevel=3,
                )
            result = ProjectionResult(
                winners=np.array(to_cpu(proj.tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=proj.tgt.w,
            )
            if proj.record_activation:
                # RECORD THE OBSERVATION THIS BRANCH MADE: every existing
                # candidate summed to exactly zero. Leaving the fields at their
                # defaults reported ZERO CANDIDATES, which the typed
                # observation rightly rejects -- and the ERP adapter turned that
                # into its legacy 0.0 deficit, a PERFECT parse. Every category
                # violation routes through an untrained core->role fiber and
                # lands here, so the violation arm read as flawless and the
                # P600 AUC was exactly 0.000 at both curriculum depths.
                result.record_zero_signal(len(proj.prev_winner_inputs))
            return result

    def _project_compiled(self, proj):
        """A compiled area takes top-k over its pregrown columns only, with no sampling and no
        growth. Returns the result, or None for an area that is not compiled."""
        xp = self._xp
        # --- Compiled topology: top-k on pregrown columns only ---
        # Skips truncated-normal sampling and connectome expansion when
        # bridge pathways were pregrown (freeze + ring).  Used during
        # EmergentParser bridge training; parity-checked via predict_next.
        if self._use_compiled_projection(proj.tgt):
            limit = int(proj.tgt.w)
            inputs_slice = proj.prev_winner_inputs[:limit]
            # Same noise and competition contract as ordinary selection; only
            # the candidate population is restricted by compiled topology.
            proj.new_winner_indices = self._select_winner_indices(proj.tgt, inputs_slice, proj.rng)
            proj.new_winner_indices = xp.asarray(proj.new_winner_indices, dtype=xp.uint32)
            if proj.plasticity_enabled and self._plasticity_enabled_global:
                self._apply_plasticity(
                    proj.target, proj.from_stimuli, proj.from_areas, proj.new_winner_indices,
                )
            proj.tgt.winners = proj.new_winner_indices
            total_act = float(xp.sum(inputs_slice[proj.new_winner_indices]))
            result = ProjectionResult(
                winners=np.array(to_cpu(proj.new_winner_indices), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=proj.tgt.w,
                total_activation=total_act,
            )
            if proj.record_activation:
                snap = np.array(to_cpu(inputs_slice), dtype=np.float32, copy=True)
                result.pre_kwta_inputs = snap
                result.pre_kwta_prev_only = snap
                result.pre_kwta_total = float(xp.sum(inputs_slice))
                result.pre_kwta_count = int(len(inputs_slice))
            return result

    def _project_candidates(self, proj):
        """The best inputs the never-fired neurons could receive, sampled as order statistics of
        each fiber's truncated-normal tail and appended to the drive: all_inputs, the
        vector k-WTA ranks."""
        xp = self._xp
        # --- Sample new winner candidates via truncated normal ---
        # A source area contributes drive from the neurons that are ACTUALLY
        # firing, so that -- not the area's nominal cap -- is the size the
        # candidate sampler must price Binomial(size, p) against.  The two
        # agree whenever an assembly is complete, which is why the nominal `k`
        # was harmless in ordinary projection; they diverge exactly when a
        # PARTIAL assembly is presented, i.e. during pattern completion.
        # Measured: cueing 25 of 50 neurons, candidates were sampled as if 50
        # were active and came in at 0.060-0.090 normalized drive while the
        # genuine missing assembly members sat at 0.043-0.064, so sampled
        # candidates outbid real completions and completion stalled at 0.427.
        # Materialized non-assembly neurons were correctly below the assembly
        # (max 0.049), confirming the fault was in the sampler, not the
        # dynamics.  Gated on norm_init only to keep default results
        # bit-identical; it is a no-op whenever len(winners) == k.
        # A fiber whose block is not materialised yet contributed ZERO to
        # `prev_winner_inputs` above. Charging it into the candidate price
        # anyway is what made recurrence catastrophic: naming B->B on the round
        # it first appears doubled total_k (100 -> 200), lifting candidates from
        # [17,18] to [30,31], while the incumbents' drive was byte-identical
        # with and without it (min 18.00 max 22.00 mean 18.77 either way). Every
        # candidate then beat every incumbent -- 0 of 100 survived -- and
        # overlap(y1,y2) went 1.000 -> 0.000 with y2 an entirely fresh cohort.
        # PNAS 2020 predicts ~0.50 here.
        # Zeroed, NOT dropped. `input_sizes` is parallel to
        # from_stimuli + from_areas and `_expand_connectomes` indexes the split
        # it produces by that position; shortening the list raised IndexError
        # on the first multi-source projection. A zero entry costs nothing in
        # `total_k = sum(input_sizes)` and allocates no synapses in the split.
        def _priced(a: str) -> int:
            if self.stable_candidates and a in proj._silent_area_srcs:
                return 0
            return area_fiber_activity(self._areas[a].winners.size,
                                       self._areas[a].k, self.norm_init)

        proj.input_sizes = (
            [self._stimuli[s].size for s in proj.from_stimuli]
            + [_priced(a) for a in proj.from_areas]
        )
        if sum(proj.input_sizes) == 0:
            # EVERY source is silent, so this is a bootstrap round: nothing is
            # materialised yet and deferred init has not run. Pricing at zero
            # would leave `compute_input_splits` with total_k == 0, which
            # returns empty split vectors and makes `_expand_connectomes` raise
            # IndexError on `split[j]`. Fall back to the nominal sizes: with no
            # incumbents to protect there is nothing for the silent-fiber
            # correction to fix, and the area still needs to recruit.
            proj.input_sizes = (
                [self._stimuli[s].size for s in proj.from_stimuli]
                + [area_fiber_activity(self._areas[a].winners.size,
                                       self._areas[a].k, self.norm_init)
                   for a in proj.from_areas]
            )
        # Presynaptic POPULATION per fiber, parallel to input_sizes. Used only
        # to price candidates on the incumbent scale -- see
        # `_norm_candidate_divisor`. Stimulus fibers use the target's own n,
        # matching `_norm_scale`'s convention for them.
        src_pops = (
            [proj.tgt.n for _ in proj.from_stimuli]
            + [self._areas[a].n for a in proj.from_areas]
        )
        # Per-fiber densities, parallel to input_sizes. None while no fiber
        # overrides `p`, which keeps the pooled draw and the divisor on their
        # original scalar code paths -- see `add_connectivity`.
        proj.input_ps = ([self._p_for(s, proj.target) for s in proj.from_stimuli]
                    + [self._p_for(a, proj.target) for a in proj.from_areas]
                    ) if self.heterogeneous() else None

        proj.draw_key = None
        # These describe the stable candidate stream when recruitment is
        # enabled.  Initialise them for the read-only branch as well so the
        # state update below has one explicit, total control path.
        proj.fiber_sig = None
        proj.fiber_cur = {}
        proj.eff = 0.0
        if self._no_recruitment and proj.tgt.w >= proj.tgt.k:
            # A READ-ONLY probe answers "which of the neurons you already have
            # respond best?", so no candidates are offered and the area cannot
            # grow. Recruitment is the last channel by which measuring changes
            # the thing measured: frozen() stops weights changing but not w,
            # and two probe orders that recruit different numbers of neurons
            # are structurally different brains no matter how init is seeded.
            #
            # Gated on w >= k because below it there is nothing to select from,
            # and a silently short assembly would be worse than growing.
            potential_new = np.empty(0, dtype=np.float32)
            old_rng = self._sparse_sim.rng
            proj.draw_key = None          # read-only probe: recruits nothing
        else:
            old_rng = self._sparse_sim.rng
            self._sparse_sim.rng = proj.rng
            # Content key: what this projection IS, so the same projection
            # draws the same candidates. See sample_new_winner_inputs.
            proj.draw_key = (self._candidate_draw_key(proj.target, proj.tgt, proj.from_stimuli,
                                                 proj.from_areas)
                        if self.stable_candidates else None)
            # How far THIS key has already eaten into its own tail. Not the
            # area's `w` -- see _order_statistic_candidates for what that
            # sealed.
            # Two accumulators, and the offset is the larger:
            #   * the exact-repeat count, correct to the neuron for an input
            #     repeated byte-for-byte, and correct THROUGH interleaving
            #     with other inputs, which the fiber term is not;
            #   * the correlation-discounted fiber count, which is what covers
            #     a slowly drifting input -- see _fiber_draw_offset.
            draw_offset = None
            if proj.draw_key is not None:
                proj.fiber_sig, proj.fiber_cur, rho, proj.eff = self._fiber_draw_offset(
                    proj.target, proj.tgt, proj.from_stimuli, proj.from_areas, proj.input_sizes)
                draw_offset = max(self._key_recruited.get(proj.draw_key, 0),
                                  int(round(rho * proj.eff)))
            if self._deterministic:
                potential_new = self._sparse_sim.sample_new_winner_inputs_legacy(
                    proj.input_sizes, proj.tgt.n, proj.tgt.w, proj.tgt.k,
                    self.p if proj.input_ps is None else proj.input_ps, key=proj.draw_key,
                )
            else:
                potential_new = self._sparse_sim.sample_new_winner_inputs(
                    proj.input_sizes, proj.tgt.n, proj.tgt.w, proj.tgt.k,
                    self.p if proj.input_ps is None else proj.input_ps, key=proj.draw_key,
                    offset=draw_offset,
                )
        self._sparse_sim.rng = old_rng

        potential_new = self._to_xp(potential_new)
        # norm_init: bring sampled candidates onto the normalized scale (see
        # _norm_candidate_divisor).  Stored weights and the sampler stay on the
        # unit scale; only the drive comparison is rescaled.
        proj.norm_div = (self._norm_candidate_divisor(proj.tgt, proj.input_sizes, src_pops,
                                                 proj.input_ps)
                    if self.norm_init else None)
        if proj.norm_div is not None:
            potential_new = potential_new / proj.norm_div
        if len(proj.prev_winner_inputs) > 0:
            proj.all_inputs = xp.concatenate([proj.prev_winner_inputs, potential_new])
        else:
            proj.all_inputs = potential_new

    def _project_penalties(self, proj):
        """What the drive owes the area's history before ranking: the LRI penalty on recently fired
        neurons and the refracted cumulative bias (skipped by a masked read); the
        activation snapshots a recording round takes."""
        xp = self._xp
        # --- Snapshot raw prev_winner_inputs before penalties ---
        if proj.record_activation:
            proj._raw_prev = np.array(to_cpu(proj.prev_winner_inputs),
                                 dtype=np.float32, copy=True)

        # --- LRI: penalise recently-fired neurons ---
        if (proj.tgt.refractory_period > 0
                and proj.tgt.inhibition_strength > 0
                and len(proj.tgt._refractory_history) > 0):
            n_inputs = len(proj.all_inputs)
            for steps_ago_idx, winner_set in enumerate(
                    reversed(list(proj.tgt._refractory_history))):
                steps_ago = steps_ago_idx + 1
                decay = 1.0 - (steps_ago - 1) / proj.tgt.refractory_period
                penalty = proj.tgt.inhibition_strength * decay
                for cidx in winner_set:
                    if cidx < n_inputs:
                        proj.all_inputs[cidx] -= penalty

        # --- Refracted mode: cumulative bias penalty ---
        # A MASKED READ ranks the raw drive: the bias is skipped on a read
        # (no plasticity) when the area asks for it. Never on a write.
        masked_read = bool(getattr(proj.tgt, "masked_readout", False)) and not proj.plasticity_enabled
        if proj.tgt.refracted and proj.tgt._cumulative_bias is not None and not masked_read:
            bias = proj.tgt._cumulative_bias
            end = min(len(bias), len(proj.all_inputs))
            if end > 0:
                proj.all_inputs[:end] -= bias[:end]

        # --- Snapshot full all_inputs before top-k ---
        if proj.record_activation:
            proj._pre_kwta_snapshot = np.array(to_cpu(proj.all_inputs),
                                          dtype=np.float32, copy=True)
            proj._pre_kwta_total_val = float(xp.sum(proj.all_inputs))
            proj._pre_kwta_count_val = int(len(proj.all_inputs))

    def _project_select(self, proj):
        """k-WTA: the winners, by top-k or the area's competition policy, against the population
        input spread."""
        # --- Select winners (top-k or area policy) ---
        # Analytic sigma of the population input distribution: each source
        # contributes Binomial(active_count, p), so variances add.
        # This assumes UNIT weights, which is right: the population is the whole
        # area, and all but the |assembly| neurons that have actually fired are
        # still unpotentiated. Working the mixture variance out confirms it --
        # with f_p = w/n on the order of 60/20000, the potentiated group's
        # within- and between-group contributions are both negligible and
        # sigma_mixture ~= sigma_unit.
        #
        # Do NOT try to rescale this by the observed mean of `all_inputs` to
        # account for potentiation: that vector is top-biased on both halves
        # (potentiated incumbents plus SAMPLED TOP ORDER STATISTICS for the
        # unmaterialized neurons), so its mean cannot separate potentiation
        # from selection bias, and the correction overshoots badly.
        #
        # Known limitation, and the reason window="sigma" is not yet the
        # default for trained language areas: once an assembly is potentiated
        # its drive sits many population-sigma above the bulk, so the window
        # h_max - sigma_c*sigma admits only a couple of neurons and selection
        # falls to min_winners. Using the empirical std of `all_inputs`
        # instead swings the other way and admits every candidate. Neither is
        # emergent; a faithful fix needs the window referenced to an
        # inhibitory pool driven by RECENT ACTIVITY rather than by the silent
        # bulk, which is not modelled here yet.
        # Per-fiber densities where they are set: this is a population spread
        # over the SAME per-fiber binomials the sampler prices.
        _sigma_ps = proj.input_ps if proj.input_ps is not None else [self.p] * len(proj.input_sizes)
        if len(_sigma_ps) != len(proj.input_sizes):
            raise RuntimeError("projection density metadata must align with input sizes")
        pop_sigma = float(np.sqrt(sum(sz * pp * (1.0 - pp)
                                      for sz, pp in zip(proj.input_sizes, _sigma_ps, strict=True)))) or None
        if pop_sigma is not None and proj.norm_div is not None:
            pop_sigma = pop_sigma / proj.norm_div
        proj.new_winner_indices = self._select_winner_indices(
            proj.tgt, proj.all_inputs, proj.rng, population_sigma=pop_sigma)

    def _project_recruit(self, proj):
        """Winners that never fired before become neurons: each takes the next index (or a ring
        slot), and the input stream's position in its tail is advanced by what it took."""
        # --- Process first-time winners ---
        proj.num_first = 0
        proj.first_winner_inputs = []
        ring_mode = getattr(proj.tgt, '_ring_mode', False)
        ring_capacity = int(getattr(proj.tgt, '_ring_capacity_cols', 0))
        ring_slot = 0
        for i in range(len(proj.new_winner_indices)):
            if proj.new_winner_indices[i] >= proj.tgt.w:
                if ring_mode and ring_capacity > 0:
                    slot_idx = proj.tgt.w + ring_slot
                    if slot_idx < ring_capacity:
                        proj.new_winner_indices[i] = slot_idx
                        while len(proj.tgt.compact_to_neuron_id) <= slot_idx:
                            # Ring slots still need stable, globally unique
                            # neuron IDs.  Using ``len(mapping)`` here can
                            # collide with the randomized pool IDs already
                            # assigned to earlier slots, producing duplicate
                            # Assembly IDs and invalid snapshots.
                            if proj.tgt.neuron_id_pool is not None:
                                pid = proj.tgt.neuron_id_pool_ptr
                                if pid >= len(proj.tgt.neuron_id_pool):
                                    raise RuntimeError(
                                        f"Neuron id pool exhausted for area {proj.tgt.name}")
                                actual_id = int(proj.tgt.neuron_id_pool[pid])
                                proj.tgt.neuron_id_pool_ptr += 1
                            else:
                                actual_id = len(proj.tgt.compact_to_neuron_id)
                            proj.tgt.compact_to_neuron_id.append(actual_id)
                        ring_slot += 1
                        continue
                # Un-normalize: connectome expansion splits an INTEGER synapse
                # count across the input fibers, so it needs the unit-scale
                # drive, not the norm_init-scaled one.
                _fwi = float(proj.all_inputs[proj.new_winner_indices[i]])
                if proj.norm_div is not None:
                    _fwi *= proj.norm_div
                proj.first_winner_inputs.append(int(_fwi))
                if proj.tgt.neuron_id_pool is not None:
                    pid = proj.tgt.neuron_id_pool_ptr
                    if pid >= len(proj.tgt.neuron_id_pool):
                        raise RuntimeError(f"Neuron id pool exhausted for area {proj.tgt.name}")
                    actual_id = int(proj.tgt.neuron_id_pool[pid])
                    proj.tgt.neuron_id_pool_ptr += 1
                else:
                    actual_id = proj.tgt.w + proj.num_first
                proj.tgt.compact_to_neuron_id.append(actual_id)
                proj.new_winner_indices[i] = proj.tgt.w + proj.num_first
                proj.num_first += 1

        if ring_mode and ring_slot > 0:
            proj.new_w = proj.tgt.w + ring_slot
        else:
            proj.new_w = proj.tgt.w + proj.num_first

        # getattr, not attribute access: Brains are pickled to the disk backbone
        # cache and an engine restored from an entry written before this flag
        # existed has no such attribute.
        if (proj.num_first and not proj.plasticity_enabled
                and getattr(self, "_strict_probes", False)):
            # RECRUITMENT WHILE PLASTICITY IS OFF is the signature of a probe
            # written against `frozen()` that meant `read_only()`. frozen()
            # stops weights changing; it does not stop the area GROWING, and
            # growth is what makes a measurement change the measured -- two
            # probe orders that recruit different numbers of neurons are
            # structurally different brains however init is seeded.
            #
            # Off by default and opt-in via NEURAL_ASSEMBLIES_STRICT_PROBES=1,
            # same discipline as _VERIFY_NNZ: it converts "which of these 50
            # frozen() sites is a contaminating probe?" from an argument into a
            # measurement you can run over the whole suite. It is NOT on by
            # default because switching a site to read_only() CHANGES ITS
            # NUMBERS -- suppressing recruitment moves what the probe reads
            # (stored/probe overlap 0.038 -> 0.180 on one protocol) -- so each
            # site is a measured decision, not a mechanical rename.
            raise RuntimeError(
                f"STRICT PROBES: projection into {proj.target!r} recruited "
                f"{proj.num_first} neurons while plasticity was disabled. A read "
                f"that grows the area contaminates what it measures; use "
                f"brain.read_only() rather than brain.frozen() for probes, or "
                f"unset NEURAL_ASSEMBLIES_STRICT_PROBES if this projection is "
                f"meant to build structure without learning.")

        # Advance this input's position in its own tail by what it just took.
        # A repeat of the same input now resumes below the neurons it already
        # holds (idempotent); a novel input still starts near rank 0; a
        # drifting one resumes at the correlation-discounted position.
        # NOT gated on num_first: the fiber entry must be rewritten even when
        # nothing was recruited, because `fiber_cur` is what the NEXT round
        # measures rho against, and leaving it stale prices that round as if
        # this round's drift had not happened.
        if proj.draw_key is not None and proj.fiber_sig is not None:
            if proj.num_first > 0:
                self._key_recruited[proj.draw_key] = (
                    self._key_recruited.get(proj.draw_key, 0) + proj.num_first)
                self._key_recruited.move_to_end(proj.draw_key)
                while len(self._key_recruited) > self._key_recruited_max:
                    self._key_recruited.popitem(last=False)
            # Cumulative, and rewritten every round so `fiber_cur` tracks the
            # sources rho is measured against. See _fiber_draw_offset for why
            # this must NOT decay by rho.
            self._fiber_draw[proj.fiber_sig] = (proj.eff + proj.num_first, proj.fiber_cur)

    def _project_learn(self, proj):
        """Learning: the Hebbian update onto the winners, then synapses for the recruits."""
        # --- Apply plasticity ---
        if proj.plasticity_enabled and self._plasticity_enabled_global:
            self._apply_plasticity(proj.target, proj.from_stimuli, proj.from_areas, proj.new_winner_indices)

        # --- Expand connectomes for new winners ---
        if proj.num_first > 0:
            self._expand_connectomes(
                proj.target, proj.from_stimuli, proj.from_areas,
                proj.input_sizes, proj.new_winner_indices,
                proj.first_winner_inputs, proj.new_w,
            )

    def _project_commit(self, proj):
        """The round becomes the area's state: its winners, its LRI history, its refracted bias."""
        xp = self._xp
        # --- Commit state ---
        proj.tgt.winners = xp.asarray(proj.new_winner_indices, dtype=xp.uint32)
        proj.tgt.w = proj.new_w

        # --- Update LRI refractory history ---
        if proj.tgt.refractory_period > 0:
            proj.tgt._refractory_history.append(
                set(int(i) for i in proj.new_winner_indices))

        # --- Update refracted cumulative bias ---
        # Gated on plasticity, and on the SAME condition as the Hebbian update
        # above. The reference charges the bias inside `RefractedArea.update`,
        # so `update=False` stops learning and charging together; ours did not,
        # which meant a no-learn readout kept charging and altered the very
        # trajectory it was meant to observe -- one step of a test sequence
        # changing the next.
        if (proj.tgt.refracted and proj.tgt.refracted_strength > 0
                and proj.plasticity_enabled and self._plasticity_enabled_global):
            if len(proj.tgt._cumulative_bias) < proj.new_w:
                old = proj.tgt._cumulative_bias
                proj.tgt._cumulative_bias = xp.zeros(proj.new_w, dtype=xp.float32)
                if len(old) > 0:
                    proj.tgt._cumulative_bias[:len(old)] = old
            bias = proj.tgt._cumulative_bias
            widx = xp.asarray(proj.new_winner_indices)
            widx = widx[widx < len(bias)]
            if len(widx) > 0:
                bias[widx] += refraction_increment(
                    proj.all_inputs[widx], bias[widx], proj.tgt.refracted_strength)

    def _project_result(self, proj):
        """The winners' total drive; fibers that were empty this round initialised now, against the
        final winners; the round's result."""
        xp = self._xp
        total_act = float(xp.sum(proj.all_inputs[proj.new_winner_indices]))

        # --- Deferred connectome initialisation --------------------------------
        # Sources whose connectomes were empty this round get initialised now
        # so they can contribute signal on the NEXT projection round.  Uses a
        # deterministic per-pair seed to avoid disturbing the main RNG.
        self._init_deferred_area_srcs(proj.target, proj._deferred_init_srcs, proj.new_w)

        result = ProjectionResult(
            winners=np.array(proj.new_winner_indices, dtype=np.uint32),
            num_first_winners=proj.num_first,
            num_ever_fired=proj.new_w,
            total_activation=total_act,
        )
        if proj.record_activation:
            result.pre_kwta_inputs = proj._pre_kwta_snapshot
            result.pre_kwta_prev_only = proj._raw_prev
            result.pre_kwta_total = proj._pre_kwta_total_val
            result.pre_kwta_count = proj._pre_kwta_count_val
        return result
