"""Sampling for the torch engine: the seeded device generator, i.i.d. drive for
unmaterialized candidates (dense-drive mode), the GPU truncated-normal order statistics of
the sparse algorithm, and the bootstrap from an explicit dense connectome.

A mixin of TorchSparseEngine (_engine.py), which owns the state these methods read;
the methods were moved out of _engine.py unchanged."""
import math
import numpy as np
from typing import Any, List, cast

import torch

from ._torch_ops import torch_ops

from ..engine import ProjectionResult
from ..index_spaces import reserve_initial_neuron_ids

try:
    from ...compute.winner_policies import TopKPolicy
except ImportError:
    from compute.winner_policies import TopKPolicy


class SamplingMixin:
    """Seeded device sampling: candidates, order statistics, bootstrap."""

    # -- seeded device RNG --------------------------------------------------

    def _device_rng(self, rng):
        """A torch generator whose stream is derived from the seeded ``rng``.

        WHY THIS EXISTS.  Both candidate samplers below took an
        ``rng: np.random.Generator`` argument and then drew from torch's
        PROCESS-GLOBAL stream instead -- ``torch_ops.rand`` and ``torch_ops.normal``
        with no ``generator=``.  ``_sample_truncated_normal_gpu`` declared the
        parameter and never referenced it at all; ``_sample_dense_candidates``
        used it only on the ``_deterministic`` branch.

        The consequence is invisible within one process and fatal across two:
        ``Brain(seed=1)`` reproduces perfectly if you run it twice in the same
        interpreter, because the global stream advances the same way -- and
        diverges between processes, because the global stream's starting point
        is not ours to control.  Measured on one cell (k=100, p=0.05, beta=0.05,
        15 rounds, norm_init, n_src=1000 -> n_tgt=10000): w = 484, 514, 506 on
        three separate invocations at a fixed seed.  Candidate draws land right
        at the k-WTA cut, so a different draw flips borderline winners and the
        difference compounds over rounds.

        This is the fourth appearance of this class in this codebase -- global
        RNG leaking between Brain constructions, ``hash()``-derived seeds
        differing across processes, CUDA's non-Bernoulli init, and now this.
        The pattern each time: **an in-process test cannot catch it**, because
        within one process the global stream is perfectly repeatable.  Any test
        for this must spawn a subprocess and compare.

        Seeding per call off ``rng`` keeps the device stream slaved to the
        numpy stream that the caller already threads through, so one seed still
        determines the whole run.
        """
        gen = self._torch_gen
        if gen is None:
            gen = torch_ops.Generator(device=self._device)
            self._torch_gen = gen
        if rng is not None:
            gen.manual_seed(int(rng.integers(0, 2 ** 63 - 1)))
        return gen

    # -- dense-drive candidate sampling (Lever A) ---------------------------

    def _sample_dense_candidates(self, input_sizes, n_unmat, rng):
        """i.i.d. drive for every unmaterialized neuron (dense-drive mode).

        The drive from ``M = sum(input_sizes)`` active presynaptic units onto an
        unmaterialized neuron is ``Binomial(M, p)``; we approximate it by a
        clamped normal with the matching mean/variance, drawn for all ``n_unmat``
        neurons at once (O(n) GPU work, deliberately). This is the same model the
        sparse sampler draws only the top-k order statistics of -- here we draw
        the full population and let topk over n choose, so selection is exact.
        """
        if n_unmat <= 0:
            return torch_ops.empty(0, dtype=torch_ops.float32, device=self._device)
        M = float(sum(input_sizes))
        mu = M * self.p
        sigma = math.sqrt(max(M * self.p * (1.0 - self.p), 0.0))
        if self._deterministic:
            draw = rng.normal(mu, sigma or 1e-6, size=int(n_unmat))
            cand = torch_ops.from_numpy(
                np.asarray(draw, dtype=np.float32)).to(self._device)
        else:
            # Same defect as `_sample_truncated_normal_gpu` had: without an
            # explicit generator this reads torch's process-global stream while
            # `rng` -- already threaded in, and used on the branch above -- is
            # ignored. See `_device_rng`.
            cand = torch_ops.normal(
                mu, sigma or 1e-6, size=(int(n_unmat),),
                device=self._device, dtype=torch_ops.float32,
                generator=self._device_rng(rng))
        return cand.clamp_(min=0.0)

    # -- GPU truncated normal sampling --------------------------------------

    def _sample_truncated_normal_gpu(
        self,
        input_sizes: list,
        n: int,
        w: int,
        k: int,
        p: float,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Sample new-winner input strengths entirely on GPU.

        Uses torch_ops.erfinv for the inverse-CDF transform, avoiding the
        scipy dependency and CPU-to-GPU transfer.  The binom.ppf
        threshold (alpha) is still computed on CPU via the cached
        scipy call -- it's O(1) per unique parameter set.

        Mathematically equivalent to SparseSimulationEngine
        .sample_new_winner_inputs():
            alpha = binom.ppf((effective_n - k)/effective_n, total_k, p)
            a = (alpha - mu) / std
            u ~ Uniform(Phi(a), 1)
            x = mu + std * Phi_inv(u)

        Where Phi_inv(u) = sqrt(2) * erfinv(2*u - 1).
        """
        from ...compute.sparse_simulation import _binom_ppf_cached

        total_k = sum(input_sizes)
        effective_n = n - w

        # GRACEFUL SATURATION, ported from `sparse_simulation` where it has
        # lived since the numpy engine hit this. `effective_n = n - w` counts
        # the neurons that have never fired -- the only source of brand-new
        # winners. When it drops to k or below the area cannot recruit a full
        # k of fresh ones, so recruit as many as remain and let the caller
        # complete the winner set from already-materialised incumbents (it
        # top-k selects over `prev_winner_inputs` plus these).
        #
        # A biological area at capacity simply stops recruiting; it must not
        # crash mid-training. This engine RAISED instead, which made every
        # saturating workload unrunnable on GPU -- and saturation is not an
        # edge case here, it is where the capacity studies live (an area at
        # rows/n = 1.0 is the normal end state of storing many assemblies).
        #
        # `k_eff == k` whenever `effective_n > k`, so no non-saturated run
        # changes.
        k_eff = min(k, max(0, effective_n - 1))
        if k_eff <= 0:
            return torch_ops.empty(0, dtype=torch_ops.float32, device=self._device)

        alpha = _binom_ppf_cached(effective_n - k_eff, effective_n, total_k, p)

        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))
        if std == 0:
            return torch_ops.full((k_eff,), mu, dtype=torch_ops.float32,
                              device=self._device)

        a = (alpha - mu) / std

        _SQRT2 = math.sqrt(2.0)
        phi_a = 0.5 * (1.0 + math.erf(a / _SQRT2))

        u = torch_ops.rand(k_eff, dtype=torch_ops.float32, device=self._device,
                       generator=self._device_rng(rng))
        u = phi_a + (1.0 - phi_a) * u
        u.clamp_(phi_a + 1e-12, 1.0 - 1e-12)

        samples = mu + std * _SQRT2 * torch_ops.erfinv(2.0 * u - 1.0)
        samples.round_()
        samples.clamp_(0, total_k)

        return samples

    def _bootstrap_from_explicit_dense(
        self,
        target: str,
        dense_act: torch.Tensor,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        rng=None,
        record_activation: bool = False,
    ) -> ProjectionResult:
        """First assembly in a sparse area driven by explicit-source dense input."""
        tgt = self._areas[target]
        if rng is None:
            rng = np.random.default_rng(self._rng.integers(0, 2**32))

        act = dense_act.float().clone()
        for stim in from_stimuli:
            stim_conn = self._stim_conns[stim][target]
            if not stim_conn.sparse and len(stim_conn.weights) > 0:
                act += stim_conn.weights.float().sum()

        policy = tgt.winner_policy or TopKPolicy(k=tgt.k)
        act_cpu = act.detach().cpu().numpy().astype(np.float64)
        if tgt.input_noise_std > 0:
            act_cpu = act_cpu + rng.normal(
                0.0, tgt.input_noise_std, size=act_cpu.shape,
            )
        neuron_ids = self._winner_sel.select_with_policy(
            act_cpu, cast(Any, policy))
        neuron_ids = [int(i) for i in neuron_ids]
        compact = list(range(len(neuron_ids)))
        pool = reserve_initial_neuron_ids(tgt.neuron_id_pool, neuron_ids, n=tgt.n)
        tgt.compact_to_neuron_id = list(neuron_ids)
        tgt.neuron_id_pool = pool
        tgt.neuron_id_pool_ptr = len(neuron_ids)

        if plasticity_enabled and self._plasticity_enabled_global:
            for src_name in from_areas:
                dense_conn = self._dense_area_conns.get(src_name, {}).get(target)
                if dense_conn is None:
                    continue
                src = self._areas[src_name]
                if not getattr(src, "explicit_source", False):
                    continue
                beta = tgt.beta_by_source.get(src_name, tgt.beta)
                if beta > 0:
                    valid_rows = src.winners.long()
                    valid_rows = valid_rows[valid_rows < dense_conn.weights.shape[0]]
                    if len(valid_rows) > 0 and len(neuron_ids) > 0:
                        dense_conn.update_weights(
                            valid_rows.cpu().numpy(),
                            neuron_ids,
                            beta,
                        )

        tgt.winners = torch_ops.tensor(
            compact, dtype=torch_ops.int32, device=self._device,
        )
        tgt.w = len(compact)
        total_act = float(act[neuron_ids].sum().item()) if neuron_ids else 0.0

        result = ProjectionResult(
            winners=np.array(compact, dtype=np.uint32),
            num_first_winners=len(compact),
            num_ever_fired=len(compact),
            total_activation=total_act,
        )
        if record_activation:
            result.pre_kwta_inputs = act.detach().cpu().numpy().astype(
                np.float32, copy=True,
            )
            result.pre_kwta_prev_only = np.zeros(0, dtype=np.float32)
            result.pre_kwta_total = float(act.sum().item())
            result.pre_kwta_count = int(act.numel())
        return result
