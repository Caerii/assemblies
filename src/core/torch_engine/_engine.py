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

from ..engine import ComputeEngine, ProjectionResult

try:
    from ...compute.sparse_simulation import SparseSimulationEngine
except ImportError:
    from compute.sparse_simulation import SparseSimulationEngine

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

        # Reusable CPU math primitives (truncated normal, input splits)
        self._sparse_sim = SparseSimulationEngine(
            np.random.default_rng(seed))

    # -- Pair seed derivation -----------------------------------------------

    def _get_pair_seed(self, source: str, target: str) -> int:
        key = (source, target)
        if key not in self._pair_seeds:
            self._pair_seeds[key] = fnv1a_pair_seed(
                self._global_seed, source, target)
        return self._pair_seeds[key]

    # -- Registration -------------------------------------------------------

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0) -> None:
        area = TorchAreaState(
            name=name, n=n, k=k, beta=beta,
            refractory_period=refractory_period,
            inhibition_strength=inhibition_strength)
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

    # -- Projection (core operation) ----------------------------------------

    def project_into(self, target, from_stimuli, from_areas,
                     plasticity_enabled=True):
        tgt = self._areas[target]
        rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # Filter sourceless areas
        from_areas = [a for a in from_areas
                      if self._areas[a].winners.numel() > 0
                      and self._areas[a].w > 0]

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

        limit = tgt.w
        for stim in from_stimuli:
            stim_w = self._stim_conns[stim][target].weights
            end = min(limit, len(stim_w))
            if end > 0:
                prev_winner_inputs[:end] += stim_w[:end].float()

        for src_name in from_areas:
            csr = self._area_conns[src_name][target]
            if csr.nnz == 0:
                continue
            src = self._areas[src_name]
            contrib = csr.accumulate_rows(src.winners.long(), limit)
            end = min(limit, len(contrib))
            if end > 0:
                prev_winner_inputs[:end] += contrib[:end]

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

        if prev_winner_inputs.numel() > 0:
            all_inputs = torch.cat([prev_winner_inputs, potential_new])
        else:
            all_inputs = potential_new

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

        # --- Select top-k winners (torch.topk — single fused kernel) ---
        k = tgt.k
        if k >= len(all_inputs):
            winners_gpu = torch.arange(len(all_inputs), dtype=torch.int32,
                                       device=self._device)
        else:
            _, top_idx = torch.topk(all_inputs, k, sorted=True)
            winners_gpu = top_idx.int()

        # --- Process first-time winners ---
        first_mask = winners_gpu.long() >= tgt.w
        first_input_vals = all_inputs[winners_gpu[first_mask].long()]

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

        return ProjectionResult(
            winners=np.array(new_winner_indices, dtype=np.uint32),
            num_first_winners=num_first,
            num_ever_fired=new_w)

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
            needed_rows = max(
                max_src_idx,
                new_w if src_name == target else src.w)
            needed_cols = new_w

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

                csr._log_rows = needed_rows
                csr._log_cols = needed_cols

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
                for r in chosen:
                    exp_rows.append(int(r))
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
        st.winners = torch.tensor(
            winners, dtype=torch.int32, device=self._device)

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
                       rounds, plasticity_enabled=True):
        result = None
        for _ in range(rounds):
            result = self.project_into(
                target, from_stimuli, from_areas, plasticity_enabled)
        return result

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "torch_sparse"
