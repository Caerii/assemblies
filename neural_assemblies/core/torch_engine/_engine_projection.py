"""Projection for the torch engine: one round of drive, k-WTA and learning into a target
area (project_into), the Hebbian update w *= (1 + beta) clipped at w_max, and homeostatic
synaptic scaling of area-to-area fibers.

A mixin of TorchSparseEngine (_engine.py), which owns the state these methods read.
project_into is a sequence of phases over a _ProjectionRound, the same phases as the
numpy engine's."""
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


class _ProjectionRound:
    """What the phases of one TorchSparseEngine.project_into hand each other.

    The call's arguments, then, in the order the phases set them:

        tgt                       the target area's state
        rng                       the round's generator, drawn from the engine's
        empty_fibers              area fibers with no synapses yet: grown after the round
        prev_winner_inputs        the drive of every materialized neuron
        input_sizes, src_pops     per fiber: the active inputs it prices, its population
        all_inputs                the drive of the materialized neurons and the sampled
                                  candidates: the vector k-WTA ranks
        _pre_kwta_snapshot, _pre_kwta_total_val, _raw_prev_t
                                  the activation snapshots a recording round takes
        winners_gpu, k            the winners (positions in all_inputs) and their number
        num_first, first_winner_inputs_cpu, new_w
                                  the recruits, their sampled inputs, the area's new size
        new_winner_indices, remapped_gpu
                                  the winners as neuron indices (host and device)

    A field read before any phase set it raises AttributeError, where the single
    function the phases were cut from raised UnboundLocalError."""

    __slots__ = ("target", "from_stimuli", "from_areas", "plasticity_enabled",
                 "record_activation", "tgt", "rng", "empty_fibers", "prev_winner_inputs",
                 "input_sizes", "src_pops", "all_inputs", "_pre_kwta_snapshot",
                 "_pre_kwta_total_val", "_raw_prev_t", "winners_gpu", "k", "num_first",
                 "first_winner_inputs_cpu", "new_w", "new_winner_indices", "remapped_gpu")

    def __init__(self, *, target, from_stimuli, from_areas, plasticity_enabled,
                 record_activation):
        self.target = target
        self.from_stimuli = from_stimuli
        self.from_areas = from_areas
        self.plasticity_enabled = plasticity_enabled
        self.record_activation = record_activation


class ProjectionMixin:
    """One projection round, its Hebbian update and synaptic scaling."""

    # -- Projection (core operation) ----------------------------------------

    def project_into(self, target, from_stimuli, from_areas,
                     plasticity_enabled=True, record_activation=False):
        """One round of projection into ``target``: drive, k-WTA, learning.

        The phases, in order (each a method below; the state they hand each other is
        a _ProjectionRound; the numpy engine's project_into has the same phases, and a
        compiled-topology one this engine lacks):

            _project_admit        the round's generator, live sources
            _project_fixed        a fixed target: winners kept, afferents learn  -> result
            _project_no_inputs    nothing projects: the assembly is kept         -> result
            _project_drive        the drive of every materialized neuron
                                  (an explicit dense source bootstraps instead)  -> result
            _project_zero_signal  no drive: kept, or noise picks the winners     -> result
            _project_candidates   the never-fired neurons' best sampled inputs
            _project_penalties    LRI and refracted bias; recording snapshots
            _project_select       k-WTA
            _project_recruit      first-time winners become neurons
            _project_learn        the Hebbian update; synapses for the recruits
            _project_commit       winners, LRI history and bias become the area's state
            _project_result       total drive, the result

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
        self._project_candidates(proj)
        self._project_penalties(proj)
        self._project_select(proj)
        self._project_recruit(proj)
        self._project_learn(proj)
        self._project_commit(proj)
        return self._project_result(proj)

    def _project_admit(self, proj):
        """Admit the round: the target's state, the round's generator, and the source areas that
        have an assembly."""
        proj.tgt = self._areas[proj.target]
        self.validate_probe_target(proj.target)
        proj.rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # Filter sourceless areas
        proj.from_areas = [
            a for a in proj.from_areas
            if self._areas[a].winners.numel() > 0
            and (
                self._areas[a].w > 0
                or getattr(self._areas[a], "explicit_source", False)
            )
        ]

    def _project_fixed(self, proj):
        """A FIXED target keeps its winners; its afferents still learn onto them (the reference's
        semantics), after the fibers grow to the current sizes. Returns the round's
        result, or None for an area that is not fixed."""
        # Fixed assembly -- the winners do not move, but the AFFERENTS learn.
        # This used to be a bare short-circuit (inputs discarded, no
        # plasticity), the exact footgun the numpy engine already fixed: a
        # projection into a fixed area looked like training and wrote nothing.
        # On this engine it presented as an EMPTY arc->state fiber after a
        # full FSM training run -- the fiber is only materialized on demand,
        # and the demand never came -- so every step read state 0. See
        # NumpySparseEngine.project_into's fixed-target branch for the
        # history and `_fixed_target_plasticity_enabled` for the A/B gate.
        if proj.tgt.fixed_assembly:
            learn = (proj.plasticity_enabled and (proj.from_stimuli or proj.from_areas)
                     and _np_fixed_target_plasticity_enabled())
            if learn:
                # Size any never-used source block FIRST -- a multiplicative
                # `w *= 1 + beta` cannot touch entries that do not exist, and
                # `hebbian_update` no-ops on an empty CSR.
                for src_name in proj.from_areas:
                    csr = self._maybe_densify(src_name, proj.target)
                    src = self._areas[src_name]
                    if int(src.w) > 0 and int(proj.tgt.w) > 0:
                        r, c, v = self._hash_grow_parts(
                            csr, self._get_pair_seed(src_name, proj.target),
                            self._p_for(src_name, proj.target),
                            max(int(src.w), csr._log_rows),
                            max(int(proj.tgt.w), csr._log_cols))
                        if r:
                            csr.expand(csr._log_rows, csr._log_cols,
                                       torch_ops.cat(r), torch_ops.cat(c),
                                       torch_ops.cat(v))
                self._apply_plasticity(
                    proj.target, proj.from_stimuli, proj.from_areas, proj.tgt.winners)
            return ProjectionResult(
                winners=proj.tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=proj.tgt.w)

    def _project_no_inputs(self, proj):
        """With no inputs the assembly is kept. Returns the result, or None when there are inputs."""
        # No inputs
        if not proj.from_stimuli and not proj.from_areas:
            return ProjectionResult(
                winners=proj.tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=proj.tgt.w)

    def _project_drive(self, proj):
        """The drive every materialized neuron receives from the inputs' current winners: stimulus
        fibers, then area fibers (explicit dense sources, CSR fibers; empty fibers
        noted). An explicit dense source into an empty target bootstraps instead, and
        returns its result."""
        # --- Accumulate inputs from previous winners ---
        proj.prev_winner_inputs = torch_ops.zeros(
            proj.tgt.w, dtype=torch_ops.float32, device=self._device)
        explicit_dense_act = None
        proj.empty_fibers = []

        limit = proj.tgt.w
        for stim in proj.from_stimuli:
            stim_conn = self._stim_conns[stim][proj.target]
            stim_w = stim_conn.weights
            end = min(limit, len(stim_w))
            if end > 0:
                contrib = stim_w[:end].float()
                if self.norm_init:
                    nscale = self._norm_scale_stim(
                        stim_conn, proj.tgt.n, self._stimuli[stim].size, end)
                    if nscale is not None:
                        contrib = contrib * nscale[:end]
                proj.prev_winner_inputs[:end] += contrib

        for src_name in proj.from_areas:
            src = self._areas[src_name]
            dense_conn = self._dense_area_conns.get(src_name, {}).get(proj.target)
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
                if proj.tgt.w == 0:
                    contrib = w[valid].sum(dim=0)
                    if enorm is not None:
                        contrib = contrib * enorm[:len(contrib)]
                    if explicit_dense_act is None:
                        explicit_dense_act = contrib
                    else:
                        explicit_dense_act += contrib
                    continue
                if proj.tgt.compact_to_neuron_id:
                    id_t = torch_ops.tensor(
                        proj.tgt.compact_to_neuron_id,
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
                            proj.prev_winner_inputs[:end] += contrib[:end]
                continue

            csr = self._area_conns[src_name][proj.target]
            if csr.nnz == 0:
                # An unmaterialised fiber delivers zero drive. That is FINE as
                # a transient -- a self-fiber is empty for the one round before
                # recruitment builds it -- but it DEADLOCKS when nothing else
                # drives the target: zero drive, so nothing is recruited, so
                # `_expand_connectomes` never runs, so the fiber stays empty
                # forever. Handled at the zero-signal branch below, which is
                # exactly the condition that separates the two.
                proj.empty_fibers.append((src_name, csr))
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
                proj.prev_winner_inputs[:end] += contrib

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
        """No drive at all: the empty fibers are grown and the assembly is preserved (or, with
        input noise, noise alone picks the winners). Returns the result, or None when
        there is signal."""
        # Zero signal — preserve current assembly
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-noise-only-observation
        zero_signal = proj.prev_winner_inputs.numel() > 0 and not proj.prev_winner_inputs.any()
        if zero_signal and proj.tgt.input_noise_std > 0 and proj.tgt.w < proj.tgt.n:
            raise ValueError('noise-only projection requires a fully materialized population')
        if zero_signal and proj.tgt.input_noise_std == 0:
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
            for src_name, csr in proj.empty_fibers:
                src = self._areas[src_name]
                if int(src.w) <= 0 or int(proj.tgt.w) <= 0:
                    continue
                if src_name == proj.target:
                    # A SELF-fiber that is silent means the area has nothing
                    # to say to itself yet; seeding it mid-run replaces the
                    # assembly with a fresh random draw, measured at stability
                    # 0.010 == chance (k/n). Preserving the assembly is the
                    # correct answer there. The deadlock is a fiber from
                    # ANOTHER area, which no other input will ever build.
                    continue
                r, c, v = self._hash_grow_parts(
                    csr, self._get_pair_seed(src_name, proj.target),
                    self._p_for(src_name, proj.target),
                    max(int(src.w), csr._log_rows),
                    max(int(proj.tgt.w), csr._log_cols))
                if r:
                    csr.expand(csr._log_rows, csr._log_cols,
                               torch_ops.cat(r), torch_ops.cat(c), torch_ops.cat(v))
                    grew = True
            if grew:
                # Once: the fibers are non-empty now, so this cannot recur.
                return self.project_into(
                    proj.target, proj.from_stimuli, proj.from_areas,
                    plasticity_enabled=proj.plasticity_enabled,
                    record_activation=proj.record_activation)
            result = ProjectionResult(
                winners=proj.tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=proj.tgt.w)
            if proj.record_activation:
                # Every existing candidate summed to exactly zero: a measured
                # zero over those candidates, not a missing observation.
                result.record_zero_signal(int(proj.prev_winner_inputs.numel()))
            return result

    def _project_candidates(self, proj):
        """The best inputs the never-fired neurons could receive, sampled as order statistics of
        each fiber's truncated-normal tail (or i.i.d. under dense drive) and appended to
        the drive: all_inputs, the vector k-WTA ranks."""
        # --- Sample new winner candidates via truncated normal ---
        proj.input_sizes = (
            [self._stimuli[s].size for s in proj.from_stimuli]
            + [area_fiber_activity(int(self._areas[a].winners.numel()),
                                   self._areas[a].k, self.norm_init)
               for a in proj.from_areas])
        # Presynaptic POPULATION per fiber, parallel to input_sizes. Used only
        # to price candidates on the incumbent scale -- see
        # `core._pricing.candidate_divisor`. Stimulus fibers use the target's
        # own n, matching the convention in `inverse_indegree`.
        proj.src_pops = (
            [proj.tgt.n for _ in proj.from_stimuli]
            + [self._areas[a].n for a in proj.from_areas])

        if self.readonly or (self._no_recruitment and proj.tgt.w >= proj.tgt.k):
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
                proj.input_sizes, proj.tgt.n - proj.tgt.w, proj.rng)
        elif self._gpu_sampling and not self.heterogeneous():
            potential_new = self._sample_truncated_normal_gpu(
                proj.input_sizes, proj.tgt.n, proj.tgt.w, proj.tgt.k, self.p, proj.rng)
        else:
            # Heterogeneous brains take the SHARED CPU sampler: it already
            # carries the per-fiber law (Poisson-binomial moment-matched by
            # `_pricing.effective_binomial` -- see NumpySparseEngine.
            # add_connectivity), and candidates are O(k), so there is no law
            # duplicated on the GPU and nothing material lost off it.
            input_ps = ([self._p_for(s, proj.target) for s in proj.from_stimuli]
                        + [self._p_for(a, proj.target) for a in proj.from_areas]
                        ) if self.heterogeneous() else self.p
            old_rng = self._sparse_sim.rng
            self._sparse_sim.rng = proj.rng
            if self._deterministic:
                potential_new_np = self._sparse_sim.sample_new_winner_inputs_legacy(
                    proj.input_sizes, proj.tgt.n, proj.tgt.w, proj.tgt.k, input_ps)
            else:
                potential_new_np = self._sparse_sim.sample_new_winner_inputs(
                    proj.input_sizes, proj.tgt.n, proj.tgt.w, proj.tgt.k, input_ps)
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
                proj.tgt.n, proj.input_sizes, proj.src_pops)

        if proj.prev_winner_inputs.numel() > 0:
            proj.all_inputs = torch_ops.cat([proj.prev_winner_inputs, potential_new])
        else:
            proj.all_inputs = potential_new

    def _project_penalties(self, proj):
        """What the drive owes the area's history before ranking: the LRI penalty on recently fired
        neurons and the refracted cumulative bias; the activation snapshots a recording
        round takes."""
        # --- Snapshot raw prev_winner_inputs before penalties ---
        proj._raw_prev_t = None
        proj._pre_kwta_snapshot = None
        proj._pre_kwta_total_val = None
        if proj.record_activation:
            proj._raw_prev_t = proj.prev_winner_inputs.clone()

        # --- LRI: penalise recently-fired neurons ---
        if (proj.tgt.refractory_period > 0
                and proj.tgt.inhibition_strength > 0
                and len(proj.tgt._refractory_history) > 0):
            pen_indices = []
            pen_values = []
            n_inputs = len(proj.all_inputs)
            for steps_ago_idx, winner_set in enumerate(
                    reversed(list(proj.tgt._refractory_history))):
                steps_ago = steps_ago_idx + 1
                decay = 1.0 - (steps_ago - 1) / proj.tgt.refractory_period
                penalty = proj.tgt.inhibition_strength * decay
                for cidx in winner_set:
                    if cidx < n_inputs:
                        pen_indices.append(cidx)
                        pen_values.append(penalty)
            if pen_indices:
                idx_t = torch_ops.tensor(pen_indices, dtype=torch_ops.long,
                                     device=self._device)
                val_t = torch_ops.tensor(pen_values, dtype=torch_ops.float32,
                                     device=self._device)
                proj.all_inputs.scatter_add_(
                    0, idx_t, -val_t)

        # --- Refracted mode: cumulative bias penalty ---
        if proj.tgt.refracted and proj.tgt._cumulative_bias is not None:
            bias = proj.tgt._cumulative_bias
            end = min(len(bias), len(proj.all_inputs))
            if end > 0:
                proj.all_inputs[:end] -= bias[:end]

        # --- Snapshot full all_inputs before top-k ---
        if proj.record_activation:
            proj._pre_kwta_snapshot = proj.all_inputs.detach().cpu().numpy().astype(
                np.float32).copy()
            proj._pre_kwta_total_val = float(proj.all_inputs.sum().item())

    def _project_select(self, proj):
        """k-WTA: the winners, by the area's competition policy or the default top-k."""
        # --- Select winners (policy-aware or default top-k) ---
        policy = proj.tgt.winner_policy or TopKPolicy(k=proj.tgt.k)
        # On-device fast path for the default top-k policy: run torch_ops.topk on
        # the GPU-resident drive vector so it never crosses to host, then bring
        # back only the k selected indices for compact-id bookkeeping. The CPU
        # path below copies the whole W-sized drive vector and runs numpy
        # argpartition every round -- measured 55-159x slower at large W. Custom
        # policies (threshold / e-percent / slotted) and additive input noise
        # keep the CPU path, which owns those semantics.
        if isinstance(policy, TopKPolicy) and proj.tgt.input_noise_std == 0.0:
            k_sel = min(int(policy.k), int(proj.all_inputs.numel()))
            _, sel = torch_ops.topk(proj.all_inputs, k_sel, sorted=True)
            proj.winners_gpu = sel.to(torch_ops.int32)
        else:
            inputs_cpu = proj.all_inputs.detach().cpu().numpy().astype(np.float64)
            if proj.tgt.input_noise_std > 0:
                inputs_cpu = inputs_cpu + proj.rng.normal(
                    0.0, proj.tgt.input_noise_std, size=inputs_cpu.shape,
                )
            winner_indices = self._winner_sel.select_with_policy(
                inputs_cpu, policy)
            proj.winners_gpu = torch_ops.tensor(
                [int(i) for i in winner_indices],
                dtype=torch_ops.int32,
                device=self._device,
            )
        proj.k = int(proj.winners_gpu.numel())

    def _project_recruit(self, proj):
        """Winners that never fired before become neurons: each takes the next index, and the
        winners are remapped to their indices."""
        # --- Process first-time winners ---
        first_mask = proj.winners_gpu.long() >= proj.tgt.w
        first_input_vals = proj.all_inputs[proj.winners_gpu[first_mask].long()]
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
                    proj.tgt.n, proj.input_sizes, proj.src_pops))

        winners_cpu = proj.winners_gpu.cpu().tolist()
        first_inputs_cpu = (first_input_vals.cpu().tolist()
                            if first_input_vals.numel() > 0 else [])

        proj.num_first = 0
        proj.first_winner_inputs_cpu = []
        proj.new_winner_indices = list(winners_cpu)
        first_idx = 0

        for i in range(proj.k):
            if proj.new_winner_indices[i] >= proj.tgt.w:
                proj.first_winner_inputs_cpu.append(
                    int(first_inputs_cpu[first_idx]))
                first_idx += 1
                actual_id = proj.tgt.next_neuron_id()
                proj.tgt.compact_to_neuron_id.append(actual_id)
                proj.new_winner_indices[i] = proj.tgt.w + proj.num_first
                proj.num_first += 1

        proj.new_w = proj.tgt.w + proj.num_first
        proj.remapped_gpu = torch_ops.tensor(
            proj.new_winner_indices, dtype=torch_ops.int32, device=self._device)

    def _project_learn(self, proj):
        """Learning: the Hebbian update onto the winners, then synapses for the recruits."""
        # --- Apply plasticity ---
        if proj.plasticity_enabled and self._plasticity_enabled_global:
            self._apply_plasticity(
                proj.target, proj.from_stimuli, proj.from_areas, proj.remapped_gpu)

        # --- Expand connectomes for new winners ---
        if proj.num_first > 0:
            self._expand_connectomes(
                proj.target, proj.from_stimuli, proj.from_areas,
                proj.input_sizes, proj.new_winner_indices,
                proj.first_winner_inputs_cpu, proj.new_w)

    def _project_commit(self, proj):
        """The round becomes the area's state: its winners, its LRI history, its refracted bias."""
        # --- Commit state ---
        proj.tgt.winners = proj.remapped_gpu
        proj.tgt.w = proj.new_w

        # --- Update LRI refractory history ---
        if proj.tgt.refractory_period > 0:
            proj.tgt._refractory_history.append(
                set(int(i) for i in proj.new_winner_indices))

        # --- Update refracted cumulative bias ---
        # Rule and gating live in `core._homeostasis`; see that module for why
        # the increment is proportional to raw drive and why charging is tied
        # to the same condition as the Hebbian update.
        if (proj.tgt.refracted and proj.tgt.refracted_strength > 0
                and proj.plasticity_enabled and self._plasticity_enabled_global):
            if len(proj.tgt._cumulative_bias) < proj.new_w:
                old = proj.tgt._cumulative_bias
                proj.tgt._cumulative_bias = torch_ops.zeros(
                    proj.new_w, dtype=torch_ops.float32, device=self._device)
                if len(old) > 0:
                    proj.tgt._cumulative_bias[:len(old)] = old
            bias = proj.tgt._cumulative_bias
            widx = torch_ops.as_tensor(
                np.asarray(proj.new_winner_indices, dtype=np.int64),
                device=bias.device)
            widx = widx[widx < len(bias)]
            if widx.numel() > 0:
                bias[widx] += refraction_increment(
                    proj.all_inputs[widx], bias[widx], proj.tgt.refracted_strength)

    def _project_result(self, proj):
        """The winners' total drive and the round's result (with the activation snapshots of a
        recording round)."""
        total_act = float(proj.all_inputs[proj.new_winner_indices].sum().item())

        result = ProjectionResult(
            winners=np.array(proj.new_winner_indices, dtype=np.uint32),
            num_first_winners=proj.num_first,
            num_ever_fired=proj.new_w,
            total_activation=total_act)
        if proj.record_activation:
            assert (proj._pre_kwta_snapshot is not None
                    and proj._raw_prev_t is not None
                    and proj._pre_kwta_total_val is not None)
            result.pre_kwta_inputs = proj._pre_kwta_snapshot
            result.pre_kwta_prev_only = proj._raw_prev_t.cpu().numpy().astype(
                np.float32)
            result.pre_kwta_total = proj._pre_kwta_total_val
            result.pre_kwta_count = int(len(proj._pre_kwta_snapshot))
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
