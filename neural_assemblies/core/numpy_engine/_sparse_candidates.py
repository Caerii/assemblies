"""Candidates and selection for the numpy engine: the stable candidate stream (its key and
its position per fiber), the winner selection under the area's policy, and the bootstrap of an
empty target from an explicit dense source.

A mixin of NumpySparseEngine (_sparse.py), which owns the state these methods read."""

import zlib

import numpy as np

from ..index_spaces import reserve_initial_neuron_ids

from typing import Any, List, cast


from ..backend import to_cpu


from ..engine import (
    ProjectionResult,
)


try:
    from ...compute.winner_policies import TopKPolicy
except ImportError:
    from compute.winner_policies import TopKPolicy

from ._growth import GrowthMixin, _self_fiber_deferred_init  # noqa: F401


from ._sparse_switches import (  # noqa: F401  re-exported: the torch engine and _exact read them here
    _PRUNE_MAX_FRACTION, _explicit_src_norm_enabled, _fixed_target_plasticity_enabled,
    _strict_drive_enabled, _warn_fixed_target_enabled,
)


from ._drive_cache import (  # noqa: F401
    DriveCacheMixin, _csr_storage_available, _CSR_MIN_CELLS,
    _CSR_MAX_DENSITY,
)


class NumpyCandidates:
    """Candidate streams, winner selection, the dense bootstrap (mixed into NumpySparseEngine)."""

    def _candidate_draw_key(self, target, tgt, from_stimuli, from_areas):
        """Identify a projection by its CONTENT, for the candidate sampler.

        Two projections that should elect the same winners must produce the
        same key, and two that should not must not. So the key carries:

          * the target area and the size of its never-fired pool (`w`), which
            is what the truncated-tail draw is a statement about;
          * which stimuli fired -- by name, since a stimulus is a fixed
            pattern;
          * which areas fired AND the assembly each one fired with, because a
            source area's drive is determined by its current winners, not by
            its name. Keying on the name alone would hand the same candidates
            to two different assemblies projecting along the same fiber.

        Winners are digested with crc32 rather than embedded, to keep the key
        small and hashable; `stable_seed` then crc32s the whole tuple.
        """
        # NOTE: `w` is deliberately NOT in the key. It is the OFFSET into the
        # order-statistic sequence this key names; including it would mint a
        # new sequence on every recruitment and defeat the whole fix.
        parts = [target, int(tgt.n), int(tgt.k)]
        parts.extend(sorted(from_stimuli))
        for a in sorted(from_areas):
            w = np.asarray(to_cpu(self._areas[a].winners), dtype=np.uint32)
            parts.append((a, int(zlib.crc32(np.sort(w).tobytes()))))
        return tuple(parts)

    def _fiber_draw_offset(self, target, tgt, from_stimuli, from_areas,
                           input_sizes):
        """How far a DRIFTING input has already eaten into its own tail.

        `_key_recruited` answers this exactly for an input that repeats
        byte-for-byte and gives 0 for everything else. That binary reading is
        wrong in the middle, and the middle is where the interesting protocols
        live: in `merge_sim` the source assemblies shift by a few neurons per
        round, so every round mints a fresh key, every round is priced as a
        brand-new input, and every round is handed candidates from rank ~0 --
        the extreme top of a pool of n. Measured over 50 rounds of the merge
        protocol, support per assembly in units of k (explicit engine = ground
        truth, which does no candidate sampling at all):

            n     k     explicit    old sampler   keyed, no discount
            1000  32       6.1          10.0          18.4
            2000  45       6.1          14.2          30.9
            4000  63       6.9          12.2          25.9

        The model says why. Drive is |{j in x : synapse j->i}|, so two inputs
        overlapping 90% give drives correlated 0.9 -- the neurons already taken
        are still near the top for the shifted input, and only the 10% that is
        genuinely new can reach past them. Treating that as a fresh draw
        recruits a full k every round forever.

        So the offset is discounted by that correlation. `rho` is the fraction
        of this projection's drive that also drove the previous projection
        along the same fibers (stimuli are fixed patterns and count as fully
        shared), and the offset is

            offset = rho * (neurons this fiber has ever recruited)

        rho = 1 recovers the exact-repeat count; rho = 0 (a genuinely
        independent input on the same fibers) gives rank 0, which is what an
        independent input deserves.

        THE COUNT IS CUMULATIVE, NOT DISCOUNTED PER ROUND.  An earlier version
        carried `eff <- rho * eff + recruited`, treating the correlation
        between round t and round t-d as rho^d. That is wrong whenever any part
        of the drive PERSISTS, and here some always does: a stimulus fires the
        identical pattern every round, so a neuron wired to it is favoured in
        all of them. Traced at n=1e5 k=317, rho sat at exactly 1/3 =
        stim/(stim + A + C) for 25 consecutive rounds with winner stability
        0.000 -- the geometric sum pinned the offset near 155 while `w` ran past
        8000, so candidates kept outbidding settled incumbents and the assembly
        never converged at all. Summing without the decay lets the offset track
        `w`; the transient then terminates and w_A at paper scale falls
        8725 -> 4273.

        THIS IS A DAMPER, NOT A DERIVATION, AND THE ALTERNATIVE WAS MEASURED.
        The offset is a rank, and a rank IS directly computable: the incumbents'
        exact drives are in `prev_winner_inputs`, so one can solve
        j* = |{i : d_i > v_j*}| and keep no state at all. That was implemented
        and rejected on evidence -- it saturates at k_eff for any trained area,
        which SEALS it (`test_materialize_area` caught a weight block with zero
        rows, and association overlap fell to 0.020 against a 0.030 floor),
        while merge at paper scale went the wrong way, 4273 -> 10698. The
        reason is that merge's over-recruitment lives in the TRANSIENT, where
        the incumbents genuinely are weak and a correct offset therefore says
        "recruit". No offset policy can fix that: the offset says where in THIS
        input's tail to start and cannot express that this tail is 90% of last
        round's. Charging recruitment history regardless of incumbent strength
        is exactly the crude thing that does damp it.

        Keyed on the FIBER, not the content, so it is bounded by the number of
        distinct projection shapes rather than growing without limit.

        Returns ``(signature, current source winners, rho, cumulative count)``;
        the caller applies `rho * cumulative` and writes the count back.
        """
        sig = (target, int(tgt.n), int(tgt.k),
               tuple(sorted(from_stimuli)), tuple(sorted(from_areas)))
        cur = {}
        for a in sorted(from_areas):
            w = np.asarray(to_cpu(self._areas[a].winners), dtype=np.uint32)
            cur[a] = frozenset(int(x) for x in w)

        prev = self._fiber_draw.get(sig)
        if prev is None:
            return sig, cur, 0.0, 0.0

        eff, last = prev
        # Weight each fiber by how much drive it carries, so a big stimulus
        # does not get the same say as a k-neuron area. `input_sizes` is
        # positionally parallel to from_stimuli + from_areas.
        sizes = list(input_sizes)
        stims = sorted(from_stimuli)
        shared = total = 0.0
        for i, _s in enumerate(stims):
            sz = float(sizes[i]) if i < len(sizes) else 0.0
            shared += sz          # a stimulus is the same pattern every time
            total += sz
        for j, a in enumerate(sorted(from_areas)):
            idx = len(stims) + j
            sz = float(sizes[idx]) if idx < len(sizes) else 0.0
            if sz <= 0:
                continue          # silent fiber: carries no drive, no say
            prev_w = last.get(a)
            if prev_w:
                shared += sz * len(cur[a] & prev_w) / max(1, len(cur[a]))
            total += sz
        rho = (shared / total) if total > 0 else 0.0
        return sig, cur, rho, float(eff)

    def _select_winner_indices(self, tgt, all_inputs, rng, population_sigma=None):
        """Select winner indices using area policy (default top-k)."""
        from ...compute.winner_policies import TopKPolicy

        inputs = all_inputs
        if getattr(tgt, "input_noise_std", 0.0) > 0:
            noise = rng.normal(0, tgt.input_noise_std, size=len(inputs))
            inputs = inputs + self._to_xp(noise.astype(np.float32))

        policy = getattr(tgt, "winner_policy", None) or TopKPolicy(k=tgt.k)
        if (
            isinstance(policy, TopKPolicy)
            and policy.k == tgt.k
            and getattr(tgt, "input_noise_std", 0.0) == 0.0
        ):
            return self._winner_sel.heapq_select_top_k(inputs, tgt.k).tolist()

        # The selector requires a sigma only for sigma-window policies.  Keep
        # the optional public parameter at this boundary, then narrow it
        # explicitly instead of allowing an Unknown/None value to leak into
        # the mathematical selection operation.
        if population_sigma is None:
            selected = self._winner_sel.select_with_policy(inputs, cast(Any, policy))
        else:
            selected = self._winner_sel.select_with_policy(
                inputs, cast(Any, policy), population_sigma=float(population_sigma))
        return [int(i) for i in to_cpu(selected)]

    def _bootstrap_from_explicit_dense(
        self,
        target: str,
        dense_act,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        rng=None,
        record_activation: bool = False,
    ) -> ProjectionResult:
        """First assembly in a sparse area driven by explicit-source dense input."""
        xp = self._xp
        tgt = self._areas[target]
        if rng is None:
            rng = np.random.default_rng(self._rng.integers(0, 2**32))

        act = dense_act.astype(xp.float32, copy=True)
        for stim in from_stimuli:
            stim_conn = self._stim_conns[stim][target]
            if not stim_conn.sparse and len(stim_conn.weights) > 0:
                act += stim_conn.weights.sum(axis=0).astype(xp.float32, copy=False)

        if tgt.input_noise_std > 0:
            act = act + rng.normal(
                0.0, tgt.input_noise_std, size=act.shape,
            ).astype(xp.float32)

        neuron_ids = self._winner_sel.select_with_policy(
            act, cast(Any, tgt.winner_policy or TopKPolicy(k=tgt.k)),
        )
        neuron_ids = [int(i) for i in to_cpu(neuron_ids)]
        compact = list(range(len(neuron_ids)))
        pool = reserve_initial_neuron_ids(tgt.neuron_id_pool, neuron_ids, n=tgt.n)
        tgt.compact_to_neuron_id = list(neuron_ids)
        tgt.neuron_id_pool = pool
        tgt.neuron_id_pool_ptr = len(neuron_ids)

        if plasticity_enabled and self._plasticity_enabled_global:
            for src_name in from_areas:
                if not self.fiber_learning_allowed(src_name, target):
                    continue
                conn = self._area_conns[src_name][target]
                src = self._areas[src_name]
                if not conn.sparse and getattr(src, "explicit_source", False):
                    beta = tgt.beta_by_source.get(src_name, tgt.beta)
                    if beta > 0:
                        valid_rows = xp.asarray(src.winners)
                        valid_rows = valid_rows[valid_rows < conn.weights.shape[0]]
                        if len(valid_rows) > 0 and len(neuron_ids) > 0:
                            conn.update_weights(
                                to_cpu(valid_rows),
                                neuron_ids,
                                beta,
                                w_max=self.w_max,
                            )

        tgt.winners = xp.asarray(compact, dtype=xp.uint32)
        tgt.w = len(compact)
        total_act = float(xp.sum(act[neuron_ids])) if neuron_ids else 0.0

        result = ProjectionResult(
            winners=np.array(compact, dtype=np.uint32),
            num_first_winners=len(compact),
            num_ever_fired=len(compact),
            total_activation=total_act,
        )
        if record_activation:
            result.pre_kwta_inputs = np.array(to_cpu(act), dtype=np.float32, copy=True)
            result.pre_kwta_prev_only = np.zeros(0, dtype=np.float32)
            result.pre_kwta_total = float(xp.sum(act))
            result.pre_kwta_count = int(len(act))
        return result
