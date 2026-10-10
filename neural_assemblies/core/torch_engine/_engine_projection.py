"""Projection for the torch engine: one round of drive, k-WTA and learning into a target
area (project_into), the Hebbian update w *= (1 + beta) clipped at w_max, and homeostatic
synaptic scaling of area-to-area fibers.

A mixin of TorchSparseEngine (_engine.py), which owns the state these methods read;
the methods were moved out of _engine.py unchanged."""
import numpy as np
from typing import Any, cast

from ._torch_ops import torch_ops

from .._pricing import area_fiber_activity
# The fixed-target learning semantics and its A/B escape hatch have ONE
# owner; both engines must read the same switch or an A/B on one engine
# silently means something else on the other.
from ..numpy_engine._sparse import (
    _fixed_target_plasticity_enabled as _np_fixed_target_plasticity_enabled,
)
from .._homeostasis import (check_area_homeostasis, refraction_increment, scaling_applies,
                            scaling_setpoint)
from ..engine import ProjectionResult

try:
    from ...compute.winner_policies import TopKPolicy
except ImportError:
    from compute.winner_policies import TopKPolicy


class ProjectionMixin:
    """One projection round, its Hebbian update and synaptic scaling."""

    # -- Projection (core operation) ----------------------------------------

    def project_into(self, target, from_stimuli, from_areas,
                     plasticity_enabled=True, record_activation=False):
        tgt = self._areas[target]
        self.validate_probe_target(target)
        rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # Filter sourceless areas
        from_areas = [
            a for a in from_areas
            if self._areas[a].winners.numel() > 0
            and (
                self._areas[a].w > 0
                or getattr(self._areas[a], "explicit_source", False)
            )
        ]

        # Fixed assembly -- the winners do not move, but the AFFERENTS learn.
        # This used to be a bare short-circuit (inputs discarded, no
        # plasticity), the exact footgun the numpy engine already fixed: a
        # projection into a fixed area looked like training and wrote nothing.
        # On this engine it presented as an EMPTY arc->state fiber after a
        # full FSM training run -- the fiber is only materialized on demand,
        # and the demand never came -- so every step read state 0. See
        # NumpySparseEngine.project_into's fixed-target branch for the
        # history and `_fixed_target_plasticity_enabled` for the A/B gate.
        if tgt.fixed_assembly:
            learn = (plasticity_enabled and (from_stimuli or from_areas)
                     and _np_fixed_target_plasticity_enabled())
            if learn:
                # Size any never-used source block FIRST -- a multiplicative
                # `w *= 1 + beta` cannot touch entries that do not exist, and
                # `hebbian_update` no-ops on an empty CSR.
                for src_name in from_areas:
                    csr = self._maybe_densify(src_name, target)
                    src = self._areas[src_name]
                    if int(src.w) > 0 and int(tgt.w) > 0:
                        r, c, v = self._hash_grow_parts(
                            csr, self._get_pair_seed(src_name, target),
                            self._p_for(src_name, target),
                            max(int(src.w), csr._log_rows),
                            max(int(tgt.w), csr._log_cols))
                        if r:
                            csr.expand(csr._log_rows, csr._log_cols,
                                       torch_ops.cat(r), torch_ops.cat(c),
                                       torch_ops.cat(v))
                self._apply_plasticity(
                    target, from_stimuli, from_areas, tgt.winners)
            return ProjectionResult(
                winners=tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w)

        # No inputs
        if not from_stimuli and not from_areas:
            return ProjectionResult(
                winners=tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w)

        # --- Accumulate inputs from previous winners ---
        prev_winner_inputs = torch_ops.zeros(
            tgt.w, dtype=torch_ops.float32, device=self._device)
        explicit_dense_act = None
        empty_fibers = []

        limit = tgt.w
        for stim in from_stimuli:
            stim_conn = self._stim_conns[stim][target]
            stim_w = stim_conn.weights
            end = min(limit, len(stim_w))
            if end > 0:
                contrib = stim_w[:end].float()
                if self.norm_init:
                    nscale = self._norm_scale_stim(
                        stim_conn, tgt.n, self._stimuli[stim].size, end)
                    if nscale is not None:
                        contrib = contrib * nscale[:end]
                prev_winner_inputs[:end] += contrib

        for src_name in from_areas:
            src = self._areas[src_name]
            dense_conn = self._dense_area_conns.get(src_name, {}).get(target)
            if dense_conn is not None and getattr(src, "explicit_source", False):
                w = self._dense_weights(dense_conn)
                valid = src.winners.long()
                valid = valid[valid < w.shape[0]]
                if len(valid) == 0:
                    continue
                # norm_init applies to THIS fiber too. The connectome is dense
                # and full-width, so its columns are NEURON IDs rather than
                # compact indices -- scale the whole width and index the result
                # by neuron id. Omitting this left explicit incumbents on the
                # raw-count scale while candidates were divided by n*p, so the
                # target sealed at k and every readout read exactly 1.0000.
                enorm = None
                if self.norm_init:
                    enorm = self._norm_scale_dense(w, src.n, int(w.shape[1]))
                if tgt.w == 0:
                    contrib = w[valid].sum(dim=0)
                    if enorm is not None:
                        contrib = contrib * enorm[:len(contrib)]
                    if explicit_dense_act is None:
                        explicit_dense_act = contrib
                    else:
                        explicit_dense_act += contrib
                    continue
                if tgt.compact_to_neuron_id:
                    id_t = torch_ops.tensor(
                        tgt.compact_to_neuron_id,
                        dtype=torch_ops.long,
                        device=self._device,
                    )
                    valid_cols = id_t[id_t < w.shape[1]]
                    if len(valid_cols) > 0:
                        contrib = w[valid][:, valid_cols].sum(dim=0)
                        if enorm is not None:
                            contrib = contrib * enorm[valid_cols]
                        end = min(limit, len(contrib))
                        if end > 0:
                            prev_winner_inputs[:end] += contrib[:end]
                continue

            csr = self._area_conns[src_name][target]
            if csr.nnz == 0:
                # An unmaterialised fiber delivers zero drive. That is FINE as
                # a transient -- a self-fiber is empty for the one round before
                # recruitment builds it -- but it DEADLOCKS when nothing else
                # drives the target: zero drive, so nothing is recruited, so
                # `_expand_connectomes` never runs, so the fiber stays empty
                # forever. Handled at the zero-signal branch below, which is
                # exactly the condition that separates the two.
                empty_fibers.append((src_name, csr))
                continue
            contrib = csr.accumulate_rows(src.winners.long(), limit)
            end = min(limit, len(contrib))
            if end > 0:
                contrib = contrib[:end]
                if self.norm_init:
                    nscale = self._norm_scale_area(csr, src.n, src.w, end)
                    if nscale is not None:
                        # nscale only covers the CSR's materialized columns
                        # (_ncols may be < end); columns beyond it carry no
                        # synapses, so their drive is 0 and left unscaled.
                        m = min(end, int(nscale.numel()))
                        contrib = contrib.clone()
                        contrib[:m] = contrib[:m] * nscale[:m]
                prev_winner_inputs[:end] += contrib

        if explicit_dense_act is not None and tgt.w == 0:
            return self._bootstrap_from_explicit_dense(
                target,
                explicit_dense_act,
                from_stimuli,
                from_areas,
                plasticity_enabled=plasticity_enabled,
                rng=rng,
                record_activation=record_activation,
            )

        # Zero signal — preserve current assembly
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-noise-only-observation
        zero_signal = prev_winner_inputs.numel() > 0 and not prev_winner_inputs.any()
        if zero_signal and tgt.input_noise_std > 0 and tgt.w < tgt.n:
            raise ValueError('noise-only projection requires a fully materialized population')
        if zero_signal and tgt.input_noise_std == 0:
            # UNLESS the silence is a DEAD FIBER rather than a quiet source.
            # Driving a converged target from a SECOND source left that
            # fiber at nrows=0 ncols=0 nnz=0 for every round while numpy grew
            # the same fiber to (514, 218) / 3837 entries and recruited
            # 109 -> 218; this engine's `w` never moved off 239. It presents
            # as a SEALED area, because a zero-drive projection still returns
            # k winners ([[silent-no-op-dead-fibers]]) and this branch then
            # freezes them.
            #
            # Seeding HERE rather than in the drive loop is what separates the
            # deadlock from the harmless transient: a self-fiber that is empty
            # for one round still has the stimulus driving recruitment, so it
            # never reaches this branch and its construction order is left
            # alone. Same defect the fixed-assembly branch above already fixes.
            grew = False
            for src_name, csr in empty_fibers:
                src = self._areas[src_name]
                if int(src.w) <= 0 or int(tgt.w) <= 0:
                    continue
                if src_name == target:
                    # A SELF-fiber that is silent means the area has nothing
                    # to say to itself yet; seeding it mid-run replaces the
                    # assembly with a fresh random draw, measured at stability
                    # 0.010 == chance (k/n). Preserving the assembly is the
                    # correct answer there. The deadlock is a fiber from
                    # ANOTHER area, which no other input will ever build.
                    continue
                r, c, v = self._hash_grow_parts(
                    csr, self._get_pair_seed(src_name, target),
                    self._p_for(src_name, target),
                    max(int(src.w), csr._log_rows),
                    max(int(tgt.w), csr._log_cols))
                if r:
                    csr.expand(csr._log_rows, csr._log_cols,
                               torch_ops.cat(r), torch_ops.cat(c), torch_ops.cat(v))
                    grew = True
            if grew:
                # Once: the fibers are non-empty now, so this cannot recur.
                return self.project_into(
                    target, from_stimuli, from_areas,
                    plasticity_enabled=plasticity_enabled,
                    record_activation=record_activation)
            result = ProjectionResult(
                winners=tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w)
            if record_activation:
                # Every existing candidate summed to exactly zero: a measured
                # zero over those candidates, not a missing observation.
                result.record_zero_signal(int(prev_winner_inputs.numel()))
            return result

        # --- Sample new winner candidates via truncated normal ---
        input_sizes = (
            [self._stimuli[s].size for s in from_stimuli]
            + [area_fiber_activity(int(self._areas[a].winners.numel()),
                                   self._areas[a].k, self.norm_init)
               for a in from_areas])
        # Presynaptic POPULATION per fiber, parallel to input_sizes. Used only
        # to price candidates on the incumbent scale -- see
        # `core._pricing.candidate_divisor`. Stimulus fibers use the target's
        # own n, matching the convention in `inverse_indegree`.
        src_pops = (
            [tgt.n for _ in from_stimuli]
            + [self._areas[a].n for a in from_areas])

        if self.readonly or (self._no_recruitment and tgt.w >= tgt.k):
            # No new candidates -> topk selects only among materialized neurons,
            # so the projection never grows the area (deterministic inference).
            # The second arm is `brain.read_only()` / `probe()`: same
            # semantics, scoped to the context manager instead of the
            # engine's lifetime. Gated on w >= k exactly like the numpy
            # engine -- below k there is nothing to select from, and a
            # silently short assembly would be worse than growing.
            potential_new = torch_ops.empty(0, dtype=torch_ops.float32,
                                        device=self._device)
        elif self.dense_drive:
            # Score EVERY unmaterialized neuron, not just k order statistics, so
            # the subsequent topk over n is exact (Lever A). See
            # _sample_dense_candidates and docs/gpu_scale_design.md.
            if self.heterogeneous():
                raise NotImplementedError(
                    "dense_drive prices candidates at the global p; this "
                    "brain has per-fiber densities (add_connectivity). "
                    "Refusing rather than sampling with the wrong statistics.")
            potential_new = self._sample_dense_candidates(
                input_sizes, tgt.n - tgt.w, rng)
        elif self._gpu_sampling and not self.heterogeneous():
            potential_new = self._sample_truncated_normal_gpu(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p, rng)
        else:
            # Heterogeneous brains take the SHARED CPU sampler: it already
            # carries the per-fiber law (Poisson-binomial moment-matched by
            # `_pricing.effective_binomial` -- see NumpySparseEngine.
            # add_connectivity), and candidates are O(k), so there is no law
            # duplicated on the GPU and nothing material lost off it.
            input_ps = ([self._p_for(s, target) for s in from_stimuli]
                        + [self._p_for(a, target) for a in from_areas]
                        ) if self.heterogeneous() else self.p
            old_rng = self._sparse_sim.rng
            self._sparse_sim.rng = rng
            if self._deterministic:
                potential_new_np = self._sparse_sim.sample_new_winner_inputs_legacy(
                    input_sizes, tgt.n, tgt.w, tgt.k, input_ps)
            else:
                potential_new_np = self._sparse_sim.sample_new_winner_inputs(
                    input_sizes, tgt.n, tgt.w, tgt.k, input_ps)
            self._sparse_sim.rng = old_rng
            if hasattr(potential_new_np, 'get'):
                potential_new_np = cast(Any, potential_new_np).get()
            potential_new_np = np.asarray(potential_new_np, dtype=np.float32)
            potential_new = torch_ops.from_numpy(potential_new_np).to(self._device)

        # norm_init: candidates are sampled on the unit-weight scale; bring them
        # onto the normalized scale by dividing by the mean in-degree (n*p), so
        # they compete with the 1/d_j-scaled materialized drive above. Stored
        # weights and the sampler stay unit-scale (see _norm_candidate_divisor).
        if self.norm_init:
            potential_new = potential_new / self._norm_candidate_divisor(
                tgt.n, input_sizes, src_pops)

        if prev_winner_inputs.numel() > 0:
            all_inputs = torch_ops.cat([prev_winner_inputs, potential_new])
        else:
            all_inputs = potential_new

        # --- Snapshot raw prev_winner_inputs before penalties ---
        _raw_prev_t = None
        _pre_kwta_snapshot = None
        _pre_kwta_total_val = None
        if record_activation:
            _raw_prev_t = prev_winner_inputs.clone()

        # --- LRI: penalise recently-fired neurons ---
        if (tgt.refractory_period > 0
                and tgt.inhibition_strength > 0
                and len(tgt._refractory_history) > 0):
            pen_indices = []
            pen_values = []
            n_inputs = len(all_inputs)
            for steps_ago_idx, winner_set in enumerate(
                    reversed(list(tgt._refractory_history))):
                steps_ago = steps_ago_idx + 1
                decay = 1.0 - (steps_ago - 1) / tgt.refractory_period
                penalty = tgt.inhibition_strength * decay
                for cidx in winner_set:
                    if cidx < n_inputs:
                        pen_indices.append(cidx)
                        pen_values.append(penalty)
            if pen_indices:
                idx_t = torch_ops.tensor(pen_indices, dtype=torch_ops.long,
                                     device=self._device)
                val_t = torch_ops.tensor(pen_values, dtype=torch_ops.float32,
                                     device=self._device)
                all_inputs.scatter_add_(
                    0, idx_t, -val_t)

        # --- Refracted mode: cumulative bias penalty ---
        if tgt.refracted and tgt._cumulative_bias is not None:
            bias = tgt._cumulative_bias
            end = min(len(bias), len(all_inputs))
            if end > 0:
                all_inputs[:end] -= bias[:end]

        # --- Snapshot full all_inputs before top-k ---
        if record_activation:
            _pre_kwta_snapshot = all_inputs.detach().cpu().numpy().astype(
                np.float32).copy()
            _pre_kwta_total_val = float(all_inputs.sum().item())

        # --- Select winners (policy-aware or default top-k) ---
        policy = tgt.winner_policy or TopKPolicy(k=tgt.k)
        # On-device fast path for the default top-k policy: run torch_ops.topk on
        # the GPU-resident drive vector so it never crosses to host, then bring
        # back only the k selected indices for compact-id bookkeeping. The CPU
        # path below copies the whole W-sized drive vector and runs numpy
        # argpartition every round -- measured 55-159x slower at large W. Custom
        # policies (threshold / e-percent / slotted) and additive input noise
        # keep the CPU path, which owns those semantics.
        if isinstance(policy, TopKPolicy) and tgt.input_noise_std == 0.0:
            k_sel = min(int(policy.k), int(all_inputs.numel()))
            _, sel = torch_ops.topk(all_inputs, k_sel, sorted=True)
            winners_gpu = sel.to(torch_ops.int32)
        else:
            inputs_cpu = all_inputs.detach().cpu().numpy().astype(np.float64)
            if tgt.input_noise_std > 0:
                inputs_cpu = inputs_cpu + rng.normal(
                    0.0, tgt.input_noise_std, size=inputs_cpu.shape,
                )
            winner_indices = self._winner_sel.select_with_policy(
                inputs_cpu, policy)
            winners_gpu = torch_ops.tensor(
                [int(i) for i in winner_indices],
                dtype=torch_ops.int32,
                device=self._device,
            )
        k = int(winners_gpu.numel())

        # --- Process first-time winners ---
        first_mask = winners_gpu.long() >= tgt.w
        first_input_vals = all_inputs[winners_gpu[first_mask].long()]
        if self.norm_init and first_input_vals.numel() > 0:
            # New winners are sampled candidates, so their drive was divided by
            # the candidate divisor above. Connectome expansion splits an INTEGER
            # synapse count across fibers, so un-normalize back to unit scale
            # first (mirror of NumpySparseEngine before _expand_connectomes).
            # Must be the SAME divisor that was applied, which now depends on
            # the fibers, not on tgt.n alone -- otherwise the round trip is
            # asymmetric and the recovered synapse count is wrong.
            first_input_vals = (
                first_input_vals * self._norm_candidate_divisor(
                    tgt.n, input_sizes, src_pops))

        winners_cpu = winners_gpu.cpu().tolist()
        first_inputs_cpu = (first_input_vals.cpu().tolist()
                            if first_input_vals.numel() > 0 else [])

        num_first = 0
        first_winner_inputs_cpu = []
        new_winner_indices = list(winners_cpu)
        first_idx = 0

        for i in range(k):
            if new_winner_indices[i] >= tgt.w:
                first_winner_inputs_cpu.append(
                    int(first_inputs_cpu[first_idx]))
                first_idx += 1
                actual_id = tgt.next_neuron_id()
                tgt.compact_to_neuron_id.append(actual_id)
                new_winner_indices[i] = tgt.w + num_first
                num_first += 1

        new_w = tgt.w + num_first
        remapped_gpu = torch_ops.tensor(
            new_winner_indices, dtype=torch_ops.int32, device=self._device)

        # --- Apply plasticity ---
        if plasticity_enabled and self._plasticity_enabled_global:
            self._apply_plasticity(
                target, from_stimuli, from_areas, remapped_gpu)

        # --- Expand connectomes for new winners ---
        if num_first > 0:
            self._expand_connectomes(
                target, from_stimuli, from_areas,
                input_sizes, new_winner_indices,
                first_winner_inputs_cpu, new_w)

        # --- Commit state ---
        tgt.winners = remapped_gpu
        tgt.w = new_w

        # --- Update LRI refractory history ---
        if tgt.refractory_period > 0:
            tgt._refractory_history.append(
                set(int(i) for i in new_winner_indices))

        # --- Update refracted cumulative bias ---
        # Rule and gating live in `core._homeostasis`; see that module for why
        # the increment is proportional to raw drive and why charging is tied
        # to the same condition as the Hebbian update.
        if (tgt.refracted and tgt.refracted_strength > 0
                and plasticity_enabled and self._plasticity_enabled_global):
            if len(tgt._cumulative_bias) < new_w:
                old = tgt._cumulative_bias
                tgt._cumulative_bias = torch_ops.zeros(
                    new_w, dtype=torch_ops.float32, device=self._device)
                if len(old) > 0:
                    tgt._cumulative_bias[:len(old)] = old
            bias = tgt._cumulative_bias
            widx = torch_ops.as_tensor(
                np.asarray(new_winner_indices, dtype=np.int64),
                device=bias.device)
            widx = widx[widx < len(bias)]
            if widx.numel() > 0:
                bias[widx] += refraction_increment(
                    all_inputs[widx], bias[widx], tgt.refracted_strength)

        total_act = float(all_inputs[new_winner_indices].sum().item())

        result = ProjectionResult(
            winners=np.array(new_winner_indices, dtype=np.uint32),
            num_first_winners=num_first,
            num_ever_fired=new_w,
            total_activation=total_act)
        if record_activation:
            assert (_pre_kwta_snapshot is not None
                    and _raw_prev_t is not None
                    and _pre_kwta_total_val is not None)
            result.pre_kwta_inputs = _pre_kwta_snapshot
            result.pre_kwta_prev_only = _raw_prev_t.cpu().numpy().astype(
                np.float32)
            result.pre_kwta_total = _pre_kwta_total_val
            result.pre_kwta_count = int(len(_pre_kwta_snapshot))
        return result

    # -- Plasticity ---------------------------------------------------------

    def _apply_plasticity(self, target, from_stimuli, from_areas,
                          winners_gpu):
        """Hebbian learning: w *= (1 + beta), clamped at w_max."""
        tgt = self._areas[target]
        winners_long = winners_gpu.long()

        for stim_name in from_stimuli:
            conn = self._stim_conns[stim_name][target]
            beta = tgt.beta_by_source.get(stim_name, tgt.beta)
            if beta == 0:
                continue
            valid = winners_long[winners_long < len(conn.weights)]
            if len(valid) > 0:
                conn.weights[valid] *= (1 + beta)
            if self.w_max is not None:
                conn.weights.clamp_(0, self.w_max)

        for src_name in from_areas:
            dense_conn = self._dense_area_conns.get(src_name, {}).get(target)
            if dense_conn is not None and getattr(
                self._areas[src_name], "explicit_source", False,
            ):
                beta = tgt.beta_by_source.get(src_name, tgt.beta)
                if beta > 0:
                    src = self._areas[src_name]
                    valid_rows = src.winners.long()
                    valid_rows = valid_rows[
                        valid_rows < dense_conn.weights.shape[0]
                    ]
                    neuron_ids = [
                        tgt.compact_to_neuron_id[int(w)]
                        for w in winners_gpu.cpu().tolist()
                        if int(w) < tgt.w
                    ]
                    if len(valid_rows) > 0 and len(neuron_ids) > 0:
                        dense_conn.update_weights(
                            valid_rows.cpu().numpy(),
                            neuron_ids,
                            beta,
                        )
                continue

            csr = self._area_conns[src_name][target]
            beta = tgt.beta_by_source.get(src_name, tgt.beta)
            if beta == 0:
                continue
            src = self._areas[src_name]
            csr.hebbian_update(
                src.winners.long(), winners_long, beta, self.w_max)

        self._normalize_area_columns(target, from_areas, winners_long)

    def _normalize_area_columns(self, target, from_areas, winners):
        """Homeostatic synaptic scaling on area->area fibers (torch port).

        The LAW and its measured failure modes live on
        `NumpySparseEngine._normalize_area_columns`; this mirrors its
        per-update form on CSR storage (`CSRConn.scale_columns`). The three
        semantics that MUST match, each a past defect on the numpy side:

        * setpoint priced at the FIBER's own density
          ([[pricing-law-implemented-twice]] -- the global-p setpoint
          renormalized a p=0.40 organ fiber to 1/8 of its natural mass and
          inverted learning);
        * mass summed over the source's LOGICAL rows only (rows past
          ``src.w`` are unallocated bookkeeping);
        * stimulus fibers excluded (1-D pre-summed: normalizing them drives
          every neuron to the same value and erases the representation).

        Explicit-dense bridges (`_dense_area_conns`) are NOT scaled -- the
        numpy engine scales any 2-D block, so a brain using explicit sources
        under scaling differs across engines; none of the scaling organs use
        them, and this note is the tripwire if one ever does.
        """
        if not scaling_applies(self.synaptic_scaling, target):
            return
        check_area_homeostasis(target, refracted=self._areas[target].refracted,
                               synaptic_scaling=True)
        for src_name in from_areas:
            csr = self._area_conns.get(src_name, {}).get(target)
            if csr is None or csr.nnz == 0:
                continue
            rows = min(int(self._areas[src_name].w), int(csr._nrows))
            if rows <= 0:
                continue
            setpoint = scaling_setpoint(rows, self._p_for(src_name, target))
            csr.scale_columns(winners, setpoint, nrows=rows)
