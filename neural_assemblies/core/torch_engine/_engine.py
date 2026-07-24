"""TorchSparseEngine: PyTorch-native GPU engine for assembly calculus.

Uses PyTorch CUDA tensors for all state and computation.  Eliminates
CuPy entirely from the hot path, gaining:
- Lower per-op dispatch overhead (~50us vs ~200us for CuPy)
- torch.topk: single fused CUDA kernel for winner selection
- torch advanced indexing for Hebbian updates
- Zero CuPy<->torch conversion overhead
- Hash-based deterministic initialization (ported from cuda_engine.py)

The same statistical sparse algorithm as NumpySparseEngine: truncated
normal sampling, Hebbian w *= (1+beta), amortised buffer growth, lazy
expansion.  Truncated normal sampling defaults to GPU-native
(torch.erfinv) but can fall back to CPU (scipy) for deterministic mode
or via gpu_sampling=False.

Requires: torch with CUDA support.
"""

import math
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List

import torch

from ..connectome import Connectome
from ..engine import ComputeEngine, ProjectionResult

try:
    from ...compute.sparse_simulation import SparseSimulationEngine
    from ...compute.winner_selection import WinnerSelector
    from ...compute.winner_policies import TopKPolicy
except ImportError:
    from compute.sparse_simulation import SparseSimulationEngine
    from compute.winner_selection import WinnerSelector
    from compute.winner_policies import TopKPolicy

from ._hash import (
    WEIGHT_DTYPE, fnv1a_pair_seed, hash_stim_counts,
    hash_bernoulli_coo,
)
from ._csr import CSRConn
from ._state import (
    LAZY_ID_THRESHOLD, TorchAreaState, StimulusState, TorchConn,
)


class TorchSparseEngine(ComputeEngine):
    """PyTorch-native GPU engine with hash-based connectivity.

    Same statistical sparse algorithm as NumpySparseEngine but all arrays
    are torch.cuda tensors.  Key performance advantages:
    - torch.topk: single fused kernel (vs argpartition + argsort)
    - Lower per-op dispatch overhead than CuPy
    - No CuPy<->torch conversion for operations
    - Hash-based deterministic initialization
    - Optional GPU-native truncated normal sampling via torch.erfinv

    Parameters:
        p:             Connection probability.
        seed:          Global random seed.
        w_max:         Hebbian weight ceiling.
        deterministic: If True, use legacy exact-fit expansion.
        gpu_sampling:  If True (default), sample truncated normal on GPU
                       using torch.erfinv instead of CPU scipy.  Falls
                       back to CPU path when deterministic=True.
    """

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False, gpu_sampling: bool = True,
                 **kwargs):
        self.p = p
        self.w_max = w_max
        self._deterministic = deterministic
        self._gpu_sampling = gpu_sampling and not deterministic
        # One-time incoming-weight normalization (reference `norm_init`): a
        # read-time per-postsynaptic 1/d_j scale, ported from NumpySparseEngine.
        # Previously this kwarg was silently swallowed by **kwargs and ignored,
        # so a Brain(norm_init=True, engine="torch_sparse") got NO normalization.
        self.norm_init = bool(kwargs.get("norm_init", False))
        self._rng = np.random.default_rng(seed)
        self._plasticity_enabled_global = True
        self._global_seed = seed
        self._pair_seeds: Dict[tuple, int] = {}

        self._device = torch.device('cuda')

        # Internal state
        self._areas: Dict[str, TorchAreaState] = {}
        self._stimuli: Dict[str, StimulusState] = {}

        # Connectivity: stim_name -> area_name -> TorchConn (1-D weights)
        self._stim_conns: Dict[str, Dict[str, TorchConn]] = defaultdict(dict)
        # Connectivity: src_area -> tgt_area -> CSRConn (2-D weights)
        self._area_conns: Dict[str, Dict[str, CSRConn]] = defaultdict(dict)
        # Dense explicit→sparse edges (Connectome objects, not CSR)
        self._dense_area_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)

        # Reusable CPU math primitives (truncated normal, input splits)
        self._sparse_sim = SparseSimulationEngine(
            np.random.default_rng(seed))
        self._winner_sel = WinnerSelector(self._rng)

    # -- Pair seed derivation -----------------------------------------------

    def _get_pair_seed(self, source: str, target: str) -> int:
        key = (source, target)
        if key not in self._pair_seeds:
            self._pair_seeds[key] = fnv1a_pair_seed(
                self._global_seed, source, target)
        return self._pair_seeds[key]

    # -- norm_init: read-time incoming-weight normalization -----------------
    # Ports NumpySparseEngine._norm_scale / _norm_candidate_divisor. The math
    # and rationale are documented there; this is the on-device mirror. Because
    # plasticity is multiplicative (w *= 1+beta), dividing a neuron's summed
    # drive by its in-degree d_j at read time is identical to having
    # initialized its incoming weights to 1/d_j, so storage stays unit-scale.

    def _norm_candidate_divisor(self, tgt_n: int) -> float:
        """Scale for sampled (unmaterialized) candidate drive: mean in-degree."""
        return max(float(tgt_n) * self.p, 1e-12)

    def _norm_scale_stim(self, conn, n_pre, stim_size, needed):
        """1/d_j for a 1-D stimulus fiber (numpy _norm_scale, 1-D branch)."""
        w = conn.weights
        if w is None or w.numel() == 0:
            return None
        cols = int(min(needed, int(w.numel())))
        if cols <= 0:
            return None
        # The stored value IS the observed in-degree from the stimulus, but
        # only until plasticity scales it -- snapshot each column the first
        # time it is seen (a fresh column is read before it is potentiated).
        base = getattr(conn, "_norm_deg_base", None)
        have = 0 if base is None else int(base.numel())
        if have < cols:
            add = w[have:cols].detach().float()
            base = add if (base is None or have == 0) else torch.cat([base, add])
            conn._norm_deg_base = base
        deg = base[:cols]
        unknown = max(int(n_pre) - int(stim_size), 0)
        d = deg + unknown * self.p
        return 1.0 / torch.clamp(d, min=1.0)

    def _norm_scale_area(self, csr, n_pre, rows_known, needed):
        """1/d_j for a 2-D area fiber (numpy _norm_scale, 2-D branch)."""
        cols = int(min(needed, int(csr._ncols)))
        if cols <= 0:
            return None
        deg = csr.column_indegree(cols)
        rows = min(int(rows_known), int(csr._nrows))
        unknown = max(int(n_pre) - rows, 0)
        d = deg + unknown * self.p
        return 1.0 / torch.clamp(d, min=1.0)

    def set_dense_area_conn(self, src: str, tgt: str, conn: Connectome) -> None:
        """Install a dense connectome for explicit→sparse cross-engine edges."""
        self._dense_area_conns[src][tgt] = conn

    def _dense_weights(self, conn: Connectome) -> torch.Tensor:
        w = conn.weights
        if isinstance(w, torch.Tensor):
            return w.float()
        return torch.from_numpy(np.asarray(w, dtype=np.float32)).to(self._device)

    # -- Registration -------------------------------------------------------

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0,
                 winner_policy=None,
                 input_noise_std: float = 0.0) -> None:
        area = TorchAreaState(
            name=name, n=n, k=k, beta=beta,
            refractory_period=refractory_period,
            inhibition_strength=inhibition_strength,
            winner_policy=winner_policy,
            input_noise_std=input_noise_std,
        )
        if n > LAZY_ID_THRESHOLD:
            area._lazy_ids = True
            area._used_ids = set()
            area._id_rng = np.random.default_rng(
                self._rng.integers(0, 2**32))
        else:
            area.neuron_id_pool = self._rng.permutation(
                np.arange(n, dtype=np.uint32))
            area.neuron_id_pool_ptr = 0
        self._areas[name] = area

        for stim_name in self._stimuli:
            conn = TorchConn(
                torch.empty(0, dtype=WEIGHT_DTYPE, device=self._device),
                sparse=True)
            self._stim_conns[stim_name][name] = conn
            area.beta_by_source[stim_name] = beta

        for other_name, other in self._areas.items():
            if other_name == name:
                self._area_conns[name][name] = CSRConn(
                    device=self._device)
            else:
                self._area_conns[other_name][name] = CSRConn(
                    device=self._device)
                self._area_conns[name][other_name] = CSRConn(
                    device=self._device)
                area.beta_by_source[other_name] = beta
                other.beta_by_source[name] = beta

    def add_stimulus(self, name: str, size: int) -> None:
        self._stimuli[name] = StimulusState(name=name, size=size)
        for area_name, area in self._areas.items():
            conn = TorchConn(
                torch.empty(0, dtype=WEIGHT_DTYPE, device=self._device),
                sparse=True)
            self._stim_conns[name][area_name] = conn
            area.beta_by_source[name] = area.beta

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        pass  # connectivity created in add_area / add_stimulus

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

        Uses torch.erfinv for the inverse-CDF transform, avoiding the
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

        if effective_n <= k:
            raise RuntimeError(
                f"Remaining size of area too small to sample k new winners "
                f"(effective_n={effective_n}, k={k}).")

        alpha = _binom_ppf_cached(effective_n - k, effective_n, total_k, p)

        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))
        if std == 0:
            return torch.full((k,), mu, dtype=torch.float32,
                              device=self._device)

        a = (alpha - mu) / std

        _SQRT2 = math.sqrt(2.0)
        phi_a = 0.5 * (1.0 + math.erf(a / _SQRT2))

        u = torch.rand(k, dtype=torch.float32, device=self._device)
        u = phi_a + (1.0 - phi_a) * u
        u.clamp_(phi_a + 1e-12, 1.0 - 1e-12)

        samples = mu + std * _SQRT2 * torch.erfinv(2.0 * u - 1.0)
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
        neuron_ids = self._winner_sel.select_with_policy(act_cpu, policy)
        neuron_ids = [int(i) for i in neuron_ids]
        compact = list(range(len(neuron_ids)))
        tgt.compact_to_neuron_id = list(neuron_ids)
        if tgt.neuron_id_pool is not None:
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

        tgt.winners = torch.tensor(
            compact, dtype=torch.int32, device=self._device,
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
        return result

    # -- Projection (core operation) ----------------------------------------

    def project_into(self, target, from_stimuli, from_areas,
                     plasticity_enabled=True, record_activation=False):
        tgt = self._areas[target]
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

        # Fixed assembly — short-circuit
        if tgt.fixed_assembly:
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
        prev_winner_inputs = torch.zeros(
            tgt.w, dtype=torch.float32, device=self._device)
        explicit_dense_act = None

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
                if tgt.w == 0:
                    contrib = w[valid].sum(dim=0)
                    if explicit_dense_act is None:
                        explicit_dense_act = contrib
                    else:
                        explicit_dense_act += contrib
                    continue
                if tgt.compact_to_neuron_id:
                    id_t = torch.tensor(
                        tgt.compact_to_neuron_id,
                        dtype=torch.long,
                        device=self._device,
                    )
                    valid_cols = id_t[id_t < w.shape[1]]
                    if len(valid_cols) > 0:
                        contrib = w[valid][:, valid_cols].sum(dim=0)
                        end = min(limit, len(contrib))
                        if end > 0:
                            prev_winner_inputs[:end] += contrib[:end]
                continue

            csr = self._area_conns[src_name][target]
            if csr.nnz == 0:
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
        if prev_winner_inputs.numel() > 0 and not prev_winner_inputs.any():
            return ProjectionResult(
                winners=tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w)

        # --- Sample new winner candidates via truncated normal ---
        input_sizes = (
            [self._stimuli[s].size for s in from_stimuli]
            + [self._areas[a].k for a in from_areas])

        if self._gpu_sampling:
            potential_new = self._sample_truncated_normal_gpu(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p, rng)
        else:
            old_rng = self._sparse_sim.rng
            self._sparse_sim.rng = rng
            if self._deterministic:
                potential_new_np = self._sparse_sim.sample_new_winner_inputs_legacy(
                    input_sizes, tgt.n, tgt.w, tgt.k, self.p)
            else:
                potential_new_np = self._sparse_sim.sample_new_winner_inputs(
                    input_sizes, tgt.n, tgt.w, tgt.k, self.p)
            self._sparse_sim.rng = old_rng
            if hasattr(potential_new_np, 'get'):
                potential_new_np = potential_new_np.get()
            potential_new_np = np.asarray(potential_new_np, dtype=np.float32)
            potential_new = torch.from_numpy(potential_new_np).to(self._device)

        # norm_init: candidates are sampled on the unit-weight scale; bring them
        # onto the normalized scale by dividing by the mean in-degree (n*p), so
        # they compete with the 1/d_j-scaled materialized drive above. Stored
        # weights and the sampler stay unit-scale (see _norm_candidate_divisor).
        if self.norm_init:
            potential_new = potential_new / self._norm_candidate_divisor(tgt.n)

        if prev_winner_inputs.numel() > 0:
            all_inputs = torch.cat([prev_winner_inputs, potential_new])
        else:
            all_inputs = potential_new

        # --- Snapshot raw prev_winner_inputs before penalties ---
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
                idx_t = torch.tensor(pen_indices, dtype=torch.long,
                                     device=self._device)
                val_t = torch.tensor(pen_values, dtype=torch.float32,
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
        # On-device fast path for the default top-k policy: run torch.topk on
        # the GPU-resident drive vector so it never crosses to host, then bring
        # back only the k selected indices for compact-id bookkeeping. The CPU
        # path below copies the whole W-sized drive vector and runs numpy
        # argpartition every round -- measured 55-159x slower at large W. Custom
        # policies (threshold / e-percent / slotted) and additive input noise
        # keep the CPU path, which owns those semantics.
        if isinstance(policy, TopKPolicy) and tgt.input_noise_std == 0.0:
            k_sel = min(int(policy.k), int(all_inputs.numel()))
            _, sel = torch.topk(all_inputs, k_sel, sorted=True)
            winners_gpu = sel.to(torch.int32)
        else:
            inputs_cpu = all_inputs.detach().cpu().numpy().astype(np.float64)
            if tgt.input_noise_std > 0:
                inputs_cpu = inputs_cpu + rng.normal(
                    0.0, tgt.input_noise_std, size=inputs_cpu.shape,
                )
            winner_indices = self._winner_sel.select_with_policy(
                inputs_cpu, policy)
            winners_gpu = torch.tensor(
                [int(i) for i in winner_indices],
                dtype=torch.int32,
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
            first_input_vals = (
                first_input_vals * self._norm_candidate_divisor(tgt.n))

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
        remapped_gpu = torch.tensor(
            new_winner_indices, dtype=torch.int32, device=self._device)

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
        if tgt.refracted and tgt.refracted_strength > 0:
            if len(tgt._cumulative_bias) < new_w:
                old = tgt._cumulative_bias
                tgt._cumulative_bias = torch.zeros(
                    new_w, dtype=torch.float32, device=self._device)
                if len(old) > 0:
                    tgt._cumulative_bias[:len(old)] = old
            for cidx in new_winner_indices:
                if cidx < len(tgt._cumulative_bias):
                    tgt._cumulative_bias[cidx] += tgt.refracted_strength

        total_act = float(all_inputs[new_winner_indices].sum().item())

        result = ProjectionResult(
            winners=np.array(new_winner_indices, dtype=np.uint32),
            num_first_winners=num_first,
            num_ever_fired=new_w,
            total_activation=total_act)
        if record_activation:
            result.pre_kwta_inputs = _pre_kwta_snapshot
            result.pre_kwta_prev_only = _raw_prev_t.cpu().numpy().astype(
                np.float32)
            result.pre_kwta_total = _pre_kwta_total_val
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

    # -- Connectome expansion -----------------------------------------------

    def _expand_connectomes(self, target, from_stimuli, from_areas,
                            input_sizes, winners, first_winner_inputs,
                            new_w):
        """Expand connectivity for first-time winners using hash-based init."""
        tgt = self._areas[target]
        inputs_names = list(from_stimuli) + list(from_areas)

        splits_per_new = self._sparse_sim.compute_input_splits(
            input_sizes, first_winner_inputs)

        prior_w = tgt.w
        new_indices = [int(w) for w in winners if int(w) >= prior_w]

        stim_names = [n for n in inputs_names if n in self._stimuli]
        area_names = [n for n in inputs_names if n in self._areas]

        # --- Expand stim->area 1-D vectors ---
        for stim_name in self._stimuli.keys():
            conn = self._stim_conns[stim_name][target]
            if conn.sparse:
                old = len(conn.weights)
                if new_w > old:
                    add_len = new_w - old
                    if stim_name not in stim_names:
                        pair_seed = self._get_pair_seed(stim_name, target)
                        add = hash_stim_counts(
                            self._stimuli[stim_name].size, old, new_w,
                            pair_seed, self.p, device=self._device)
                    else:
                        add = torch.zeros(add_len, dtype=WEIGHT_DTYPE,
                                          device=self._device)
                    conn.weights = torch.cat([conn.weights, add])

        # Write allocations for firing stimuli
        for idx, win in enumerate(new_indices):
            if win >= new_w:
                continue
            split = (splits_per_new[idx]
                     if idx < len(splits_per_new) else None)
            if split is None:
                continue
            for j, name in enumerate(inputs_names):
                alloc = int(split[j])
                if name in self._stimuli:
                    conn = self._stim_conns[name][target]
                    if conn.sparse and win < len(conn.weights):
                        conn.weights[win] = alloc

        # --- Expand area->area CSR matrices ---
        for src_name in area_names:
            csr = self._area_conns[src_name][target]
            src = self._areas[src_name]
            pair_seed = self._get_pair_seed(src_name, target)

            src_w_arr = src.winners.long()
            max_src_idx = (
                (int(src_w_arr.max()) + 1)
                if src_w_arr.numel() > 0 else 0)
            # Never shrink below pregrown CSR extent (CONTEXT w can reset to 0
            # while connectome topology is preserved — matches numpy dense path).
            needed_rows = max(
                max_src_idx,
                new_w if src_name == target else src.w,
                csr._nrows,
            )
            needed_cols = max(new_w, csr._ncols)

            log_rows = csr._log_rows
            log_cols = csr._log_cols
            coo_r_parts, coo_c_parts, coo_v_parts = [], [], []

            if needed_rows <= log_rows and needed_cols <= log_cols:
                pass  # no hash expansion needed
            else:
                # Block A: new rows x existing cols
                if needed_rows > log_rows and log_cols > 0:
                    r, c, v = hash_bernoulli_coo(
                        log_rows, needed_rows, 0, log_cols,
                        pair_seed, self.p, device=self._device)
                    if len(r) > 0:
                        coo_r_parts.append(r)
                        coo_c_parts.append(c)
                        coo_v_parts.append(v)

                # Block B: existing rows x new cols
                if needed_cols > log_cols and log_rows > 0:
                    r, c, v = hash_bernoulli_coo(
                        0, log_rows, log_cols, needed_cols,
                        pair_seed, self.p, device=self._device)
                    if len(r) > 0:
                        coo_r_parts.append(r)
                        coo_c_parts.append(c)
                        coo_v_parts.append(v)

                # Block C: new rows x new cols
                if needed_rows > log_rows and needed_cols > log_cols:
                    r, c, v = hash_bernoulli_coo(
                        log_rows, needed_rows, log_cols, needed_cols,
                        pair_seed, self.p, device=self._device)
                    if len(r) > 0:
                        coo_r_parts.append(r)
                        coo_c_parts.append(c)
                        coo_v_parts.append(v)

                csr._log_rows = max(getattr(csr, '_log_rows', 0), needed_rows)
                csr._log_cols = max(getattr(csr, '_log_cols', 0), needed_cols)

            # Explicit entries from first-timer allocations
            from_index = inputs_names.index(src_name)
            local_rng = np.random.default_rng(
                self._rng.integers(0, 2**32))
            src_winners_cpu = src.winners.cpu().numpy().astype(np.int64)

            exp_rows, exp_cols = [], []
            for idx, win in enumerate(new_indices):
                alloc = (
                    int(splits_per_new[idx][from_index])
                    if idx < len(splits_per_new) else 0)
                if alloc <= 0 or src.w == 0:
                    continue
                sample_size = min(alloc, len(src.winners))
                if sample_size <= 0:
                    continue
                chosen = local_rng.choice(
                    src_winners_cpu, size=sample_size, replace=False)
                col_idx = win - prior_w
                if col_idx < 0 or col_idx >= needed_cols:
                    continue
                for r in chosen:
                    r_int = int(r)
                    if r_int < 0 or r_int >= needed_rows:
                        continue
                    exp_rows.append(r_int)
                    exp_cols.append(col_idx)

            if exp_rows:
                coo_r_parts.append(torch.tensor(
                    exp_rows, dtype=torch.int32, device=self._device))
                coo_c_parts.append(torch.tensor(
                    exp_cols, dtype=torch.int32, device=self._device))
                coo_v_parts.append(torch.ones(
                    len(exp_rows), dtype=WEIGHT_DTYPE,
                    device=self._device))

            # Merge into CSR
            if coo_r_parts:
                new_r = torch.cat(coo_r_parts)
                new_c = torch.cat(coo_c_parts)
                new_v = torch.cat(coo_v_parts)
                csr.expand(needed_rows, needed_cols, new_r, new_c, new_v)
            elif needed_rows > csr._nrows or needed_cols > csr._ncols:
                e = torch.empty(0, dtype=torch.int32, device=self._device)
                csr.expand(needed_rows, needed_cols,
                           e, e.clone(),
                           torch.empty(0, dtype=WEIGHT_DTYPE,
                                       device=self._device))

    # -- State accessors ----------------------------------------------------

    def get_winners(self, area: str) -> np.ndarray:
        st = self._areas[area]
        return st.winners.cpu().numpy().astype(np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        st = self._areas[area]
        # torch.tensor(uint32_array, dtype=int32, device=cuda) hits a slow
        # element-wise path -- uint32 is not a native torch dtype, so at large k
        # this dominated the whole projection (measured 32ms/round at k=100k).
        # Route through int64 (torch-native) so from_numpy is zero-copy, then a
        # single fused H2D + cast kernel.
        arr = np.ascontiguousarray(winners, dtype=np.int64)
        st.winners = torch.from_numpy(arr).to(
            self._device, dtype=torch.int32, non_blocking=True)

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].w

    def get_neuron_id_mapping(self, area: str) -> list:
        return self._areas[area].compact_to_neuron_id

    # -- Plasticity control -------------------------------------------------

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = beta

    def get_beta(self, target: str, source: str) -> float:
        tgt = self._areas[target]
        return tgt.beta_by_source.get(source, tgt.beta)

    # -- Assembly fixation --------------------------------------------------

    def fix_assembly(self, area: str) -> None:
        st = self._areas[area]
        if st.winners is None or st.winners.numel() == 0:
            raise ValueError(f"Area {area} has no winners to fix.")
        st.fixed_assembly = True

    def unfix_assembly(self, area: str) -> None:
        self._areas[area].fixed_assembly = False

    def is_fixed(self, area: str) -> bool:
        return self._areas[area].fixed_assembly

    # -- Connection reset ---------------------------------------------------

    def reset_area_connections(self, area: str) -> None:
        for src_name in list(self._area_conns.keys()):
            if area not in self._area_conns[src_name]:
                continue
            self._area_conns[src_name][area].reset()

    # -- LRI control --------------------------------------------------------

    def clear_refractory(self, area: str) -> None:
        self._areas[area]._refractory_history.clear()

    def set_lri(self, area: str, refractory_period: int,
                inhibition_strength: float) -> None:
        st = self._areas[area]
        st.refractory_period = refractory_period
        st.inhibition_strength = inhibition_strength
        st._refractory_history = deque(
            maxlen=max(refractory_period, 1))

    # -- Refracted mode control ---------------------------------------------

    def set_refracted(self, area: str, enabled: bool,
                      strength: float = 0.0) -> None:
        st = self._areas[area]
        st.refracted = enabled
        st.refracted_strength = strength
        if enabled and st._cumulative_bias.numel() == 0:
            st._cumulative_bias = torch.zeros(
                max(st.w, 0), dtype=torch.float32, device=self._device)

    def clear_refracted_bias(self, area: str) -> None:
        st = self._areas[area]
        st._cumulative_bias = torch.zeros(
            max(st.w, 0), dtype=torch.float32, device=self._device)

    # -- Weight normalization -----------------------------------------------

    def normalize_weights(self, target: str, source: str = None) -> None:
        eps = 1e-8

        def _norm_stim(conn):
            w = conn.weights
            if w.ndim == 1 and w.numel() > 0:
                total = w.sum().item()
                if total > eps:
                    conn.weights = w / total

        if source is not None:
            if (source in self._stim_conns
                    and target in self._stim_conns[source]):
                _norm_stim(self._stim_conns[source][target])
            if (source in self._area_conns
                    and target in self._area_conns[source]):
                self._area_conns[source][target].normalize_columns(eps)
            return

        for stim_name in self._stim_conns:
            if target in self._stim_conns[stim_name]:
                _norm_stim(self._stim_conns[stim_name][target])
        for src_name in self._area_conns:
            if target in self._area_conns[src_name]:
                self._area_conns[src_name][target].normalize_columns(eps)

    # -- Tight projection loop ----------------------------------------------

    def project_rounds(self, target, from_stimuli, from_areas,
                       rounds, plasticity_enabled=True,
                       record_activation=False):
        result = None
        for _ in range(rounds):
            result = self.project_into(
                target, from_stimuli, from_areas, plasticity_enabled,
                record_activation=record_activation)
        return result

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "torch_sparse"
