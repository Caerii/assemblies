"""
CUDA compute engine with hash-based connectivity initialization.

CudaImplicitEngine: Subclasses NumpySparseEngine to use the same
statistical sparse algorithm (truncated normal, Hebbian, amortized
buffer growth) but replaces random Bernoulli initialization with
deterministic hash-based initialization.

Key properties:
- Correct assembly dynamics (same algorithm as numpy_sparse)
- Deterministic initialization (same area names -> same connectivity)
- GPU-parallelizable (each hash eval is independent)
- No RNG state consumed for connectivity initialization
- Hash function matches CUDA kernels in kernels/implicit.py
- Custom CUDA kernels for fused projection hot paths

Requires: cupy (for GPU arrays; kernels in kernels/implicit.py are optional).
"""

import numpy as np
from typing import Dict, List

from .engine import ProjectionResult, register_engine
from .backend import get_xp, to_cpu, to_xp

# Guard — this module is only loaded if cupy is available
import cupy as cp

from .numpy_engine import NumpySparseEngine
from .kernels.sparse_ops import (
    apply_lri_penalties,
    apply_hebbian_1d,
    apply_hebbian_2d,
)

# PyTorch top-k: single kernel call, faster than CuPy argpartition+argsort
try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def _fnv1a_pair_seed(global_seed: int, source: str, target: str) -> int:
    """Deterministic FNV-1a 32-bit seed for a (source, target) pair.

    Produces the same seed regardless of Python process or hash
    randomization.  Same algorithm as the original CudaImplicitEngine.
    """
    h = 0x811c9dc5
    for byte in global_seed.to_bytes(4, 'little'):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    for byte in source.encode('utf-8'):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    for byte in b'\x00':
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    for byte in target.encode('utf-8'):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h


def _hash_bernoulli_2d(row_start, row_end, col_start, col_end,
                       pair_seed, p):
    """Vectorized hash-based Bernoulli(p) matrix on GPU.

    Matches the CUDA kernel hash function:
        hash = (src * 2654435761) ^ (dst * 2246822519) ^ seed
        connected = (hash & 0xFFFFFF) < (p * 2^24)

    Returns a CuPy float32 matrix of shape (row_end-row_start, col_end-col_start).
    """
    nr = row_end - row_start
    nc = col_end - col_start
    if nr == 0 or nc == 0:
        return cp.empty((nr, nc), dtype=cp.float32)

    rows = cp.arange(row_start, row_end, dtype=cp.uint32)
    cols = cp.arange(col_start, col_end, dtype=cp.uint32)
    r, c = cp.meshgrid(rows, cols, indexing='ij')

    # Same hash as implicit_projection_kernel in kernels/implicit.py
    h = (r * cp.uint32(2654435761)) ^ (c * cp.uint32(2246822519))
    h ^= cp.uint32(pair_seed)
    threshold = cp.uint32(int(p * 16777216.0))
    return ((h & cp.uint32(0xFFFFFF)) < threshold).astype(cp.float32)


def _hash_stim_counts(stim_size, neuron_start, neuron_end,
                      pair_seed, p):
    """Hash-based stim->area connectivity: count of connected stim neurons.

    For each target neuron j in [neuron_start, neuron_end), counts how
    many stimulus neurons i in [0, stim_size) are connected via hash.
    Replaces ``rng.binomial(stim_size, p, size=add_len)``.

    Returns a CuPy float32 1-D array of length (neuron_end - neuron_start).
    """
    n_neurons = neuron_end - neuron_start
    if n_neurons == 0:
        return cp.empty(0, dtype=cp.float32)

    # For small stim_size (typical: 100), full meshgrid is fine
    # For large stim_size, use batched approach to limit memory
    if stim_size <= 1024:
        stim_ids = cp.arange(stim_size, dtype=cp.uint32)
        neuron_ids = cp.arange(neuron_start, neuron_end, dtype=cp.uint32)
        s, n = cp.meshgrid(stim_ids, neuron_ids, indexing='ij')
        h = (s * cp.uint32(2654435761)) ^ (n * cp.uint32(2246822519))
        h ^= cp.uint32(pair_seed)
        threshold = cp.uint32(int(p * 16777216.0))
        connected = (h & cp.uint32(0xFFFFFF)) < threshold
        return connected.sum(axis=0).astype(cp.float32)
    else:
        # Batched: process stim_size in chunks of 1024
        result = cp.zeros(n_neurons, dtype=cp.float32)
        neuron_ids = cp.arange(neuron_start, neuron_end, dtype=cp.uint32)
        threshold = cp.uint32(int(p * 16777216.0))
        for batch_start in range(0, stim_size, 1024):
            batch_end = min(batch_start + 1024, stim_size)
            stim_ids = cp.arange(batch_start, batch_end, dtype=cp.uint32)
            s, n = cp.meshgrid(stim_ids, neuron_ids, indexing='ij')
            h = (s * cp.uint32(2654435761)) ^ (n * cp.uint32(2246822519))
            h ^= cp.uint32(pair_seed)
            connected = (h & cp.uint32(0xFFFFFF)) < threshold
            result += connected.sum(axis=0).astype(cp.float32)
        return result


# ---------------------------------------------------------------------------
# CudaImplicitEngine
# ---------------------------------------------------------------------------

class CudaImplicitEngine(NumpySparseEngine):
    """GPU engine with hash-based initialization and correct sparse dynamics.

    Subclasses NumpySparseEngine for the full projection algorithm
    (truncated normal sampling, Hebbian learning ``w *= (1+β)``,
    amortised buffer growth, lazy expansion, winner selection via
    argpartition).

    The ONLY difference from NumpySparseEngine: instead of using
    ``rng.random(shape) < p`` to initialize new connectome entries,
    this engine evaluates a deterministic hash function that matches
    the CUDA kernels in ``kernels/implicit.py``.

    This gives:
    - Deterministic init: same area names always produce same connectivity
    - No RNG consumed: truncated-normal and other RNG-dependent paths
      retain the exact same sequence as numpy_sparse with the same seed
    - GPU-parallel init: hash evaluation is embarrassingly parallel
    - Correct dynamics: same assembly formation as numpy_sparse

    Parameters:
        p:             Connection probability.
        seed:          Global random seed (for RNG and hash).
        w_max:         Hebbian weight ceiling.
        deterministic: If True, use legacy exact-fit expansion.
    """

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False, **kwargs):
        # Activate CuPy backend before parent creates arrays
        from .backend import set_backend
        set_backend("cupy")

        # Parent init (NumpySparseEngine) — all arrays will be CuPy
        super().__init__(p=p, seed=seed, w_max=w_max,
                         deterministic=deterministic)

        self._global_seed = seed
        # Cache pair seeds: (source_name, target_name) -> uint32
        self._pair_seeds: Dict[tuple, int] = {}

    # -- Pair seed derivation -----------------------------------------------

    def _get_pair_seed(self, source: str, target: str) -> int:
        """Get or compute deterministic pair seed."""
        key = (source, target)
        if key not in self._pair_seeds:
            self._pair_seeds[key] = _fnv1a_pair_seed(
                self._global_seed, source, target)
        return self._pair_seeds[key]

    # -- Override: hash-based connectome expansion --------------------------

    def _expand_connectomes(self, target, from_stimuli, from_areas,
                            input_sizes, winners, first_winner_inputs,
                            new_w):
        """Expand connectivity for first-time winners using hash-based init.

        Same algorithm as NumpySparseEngine._expand_connectomes but
        replaces ``rng.random(shape) < p`` with deterministic hash
        evaluation matching the CUDA kernels.
        """
        tgt = self._areas[target]
        inputs_names = list(from_stimuli) + list(from_areas)

        splits_per_new = self._sparse_sim.compute_input_splits(
            input_sizes, first_winner_inputs,
        )

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
                        stim_size = self._stimuli[stim_name].size
                        pair_seed = self._get_pair_seed(stim_name, target)
                        # Hash-based stim connectivity (replaces rng.binomial)
                        add = _hash_stim_counts(
                            stim_size, old, new_w, pair_seed, self.p)
                    else:
                        add = cp.zeros(add_len, dtype=cp.float32)
                    conn.weights = cp.concatenate([conn.weights, add])

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

        # --- Expand area->area 2-D matrices ---
        for src_name in area_names:
            conn = self._area_conns[src_name][target]
            if not conn.sparse:
                continue
            src = self._areas[src_name]
            if conn.weights.ndim != 2:
                conn.weights = cp.empty((0, 0), dtype=cp.float32)

            pair_seed = self._get_pair_seed(src_name, target)
            phys_rows, phys_cols = conn.weights.shape
            src_w_arr = cp.asarray(src.winners)
            max_src_idx = (
                (int(cp.max(src_w_arr)) + 1)
                if src_w_arr.size > 0 else 0
            )
            needed_rows = max(
                max_src_idx,
                new_w if src_name == target else src.w,
            )
            needed_cols = new_w

            if self._deterministic:
                # Legacy exact-fit with hash-based init
                if needed_rows > phys_rows:
                    nr = needed_rows - phys_rows
                    new_rows = _hash_bernoulli_2d(
                        phys_rows, needed_rows, 0, phys_cols,
                        pair_seed, self.p)
                    conn.weights = (
                        cp.vstack([conn.weights, new_rows])
                        if phys_cols > 0
                        else cp.zeros((needed_rows, 0), dtype=cp.float32)
                    )
                    phys_rows = needed_rows
                if needed_cols > phys_cols:
                    nc = needed_cols - phys_cols
                    new_cols = _hash_bernoulli_2d(
                        0, phys_rows, phys_cols, needed_cols,
                        pair_seed, self.p)
                    conn.weights = (
                        cp.hstack([conn.weights, new_cols])
                        if phys_rows > 0
                        else cp.zeros((0, needed_cols), dtype=cp.float32)
                    )
                    phys_cols = needed_cols
            else:
                # Amortised buffer growth with hash-based Bernoulli init
                log_rows = getattr(conn, '_log_rows', phys_rows)
                log_cols = getattr(conn, '_log_cols', phys_cols)
                if log_rows > phys_rows or log_cols > phys_cols:
                    log_rows = min(log_rows, phys_rows)
                    log_cols = min(log_cols, phys_cols)

                new_pr, new_pc = phys_rows, phys_cols
                need_realloc = False
                if needed_rows > phys_rows:
                    new_pr = max(needed_rows, phys_rows * 2, 2 * src.k)
                    need_realloc = True
                if needed_cols > phys_cols:
                    new_pc = max(needed_cols, phys_cols * 2, 2 * tgt.k)
                    need_realloc = True

                if need_realloc:
                    buf = cp.zeros((new_pr, new_pc), dtype=cp.float32)
                    if phys_rows > 0 and phys_cols > 0:
                        buf[:phys_rows, :phys_cols] = conn.weights
                    conn.weights = buf
                    phys_rows, phys_cols = new_pr, new_pc

                # Hash-based Bernoulli initialization
                nr = needed_rows - log_rows
                nc = needed_cols - log_cols
                # Region A: new rows x old cols
                if nr > 0 and log_cols > 0:
                    conn.weights[log_rows:needed_rows, :log_cols] = (
                        _hash_bernoulli_2d(
                            log_rows, needed_rows, 0, log_cols,
                            pair_seed, self.p))
                # Region B: old rows x new cols
                if nc > 0 and log_rows > 0:
                    conn.weights[:log_rows, log_cols:needed_cols] = (
                        _hash_bernoulli_2d(
                            0, log_rows, log_cols, needed_cols,
                            pair_seed, self.p))
                # Region C: new rows x new cols
                if nr > 0 and nc > 0:
                    conn.weights[
                        log_rows:needed_rows, log_cols:needed_cols
                    ] = _hash_bernoulli_2d(
                        log_rows, needed_rows, log_cols, needed_cols,
                        pair_seed, self.p)

                conn._log_rows = needed_rows
                conn._log_cols = needed_cols

            # Write specific allocations for first-time winners
            from_index = inputs_names.index(src_name)
            local_rng = np.random.default_rng(
                self._rng.integers(0, 2**32))
            src_winners_cpu = np.asarray(
                src.winners.get()
                if hasattr(src.winners, 'get')
                else src.winners
            )
            for idx, win in enumerate(new_indices):
                alloc = (
                    int(splits_per_new[idx][from_index])
                    if idx < len(splits_per_new) else 0
                )
                if alloc <= 0 or src.w == 0:
                    continue
                sample_size = min(alloc, len(src.winners))
                if sample_size <= 0:
                    continue
                chosen = local_rng.choice(
                    src_winners_cpu, size=sample_size, replace=False)
                col_idx = win - prior_w
                if 0 <= col_idx < phys_cols:
                    conn.weights[chosen, col_idx] = 1.0

    # -- Override: GPU-optimized project_into --------------------------------

    def project_into(self, target, from_stimuli, from_areas,
                     plasticity_enabled=True):
        """GPU-optimized projection with custom CUDA kernels.

        Same algorithm as NumpySparseEngine.project_into but with:
        - Fewer GPU-to-CPU sync points (2 instead of ~12)
        - Fused CUDA kernels for LRI, plasticity, first-timer detection
        - Pre-allocated GPU buffers where possible
        """
        tgt = self._areas[target]
        rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # Filter out source areas with no assembly
        from_areas = [a for a in from_areas
                      if self._areas[a].winners.size > 0 and self._areas[a].w > 0]

        # Fixed assembly — short-circuit
        if tgt.fixed_assembly:
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # No inputs — keep assembly unchanged
        if len(from_stimuli) == 0 and len(from_areas) == 0:
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # --- Accumulate inputs from previous winners ---
        prev_winner_inputs = cp.zeros(tgt.w, dtype=cp.float32)

        # Stimulus inputs (1-D slice up to w)
        limit = tgt.w
        for stim in from_stimuli:
            stim_w = self._stim_conns[stim][target].weights
            end = min(limit, len(stim_w))
            if end > 0:
                prev_winner_inputs[:end] += stim_w[:end]

        # Area inputs (2-D, vectorised fancy-index)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            if conn.weights.shape[1] == 0:
                continue
            src = self._areas[src_name]
            src_w = cp.asarray(src.winners)
            internal = src_w[src_w < conn.weights.shape[0]]
            if len(internal) > 0 and limit > 0:
                col_end = min(limit, conn.weights.shape[1])
                prev_winner_inputs[:col_end] += conn.weights[internal, :col_end].sum(axis=0)

        # Zero signal — preserve current assembly
        # Use cp.any() to avoid GPU->CPU scalar transfer
        if len(prev_winner_inputs) > 0 and not bool(cp.any(prev_winner_inputs)):
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # --- Sample new winner candidates via truncated normal (CPU) ---
        input_sizes = (
            [self._stimuli[s].size for s in from_stimuli]
            + [self._areas[a].k for a in from_areas]
        )

        old_rng = self._sparse_sim.rng
        self._sparse_sim.rng = rng
        if self._deterministic:
            potential_new = self._sparse_sim.sample_new_winner_inputs_legacy(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p,
            )
        else:
            potential_new = self._sparse_sim.sample_new_winner_inputs(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p,
            )
        self._sparse_sim.rng = old_rng

        potential_new = to_xp(potential_new)
        if len(prev_winner_inputs) > 0:
            all_inputs = cp.concatenate([prev_winner_inputs, potential_new])
        else:
            all_inputs = potential_new

        # --- LRI: penalise recently-fired neurons (vectorised) ---
        if (tgt.refractory_period > 0
                and tgt.inhibition_strength > 0
                and len(tgt._refractory_history) > 0):
            # Build penalty pairs on CPU (small), transfer once
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
                pen_idx_gpu = cp.array(pen_indices, dtype=cp.uint32)
                pen_val_gpu = cp.array(pen_values, dtype=cp.float32)
                apply_lri_penalties(all_inputs, pen_idx_gpu, pen_val_gpu)

        # --- Refracted mode: cumulative bias penalty ---
        if tgt.refracted and tgt._cumulative_bias is not None:
            bias = tgt._cumulative_bias
            end = min(len(bias), len(all_inputs))
            if end > 0:
                all_inputs[:end] -= bias[:end]

        # --- Select top-k winners (stay on GPU) ---
        k = tgt.k
        if k >= len(all_inputs):
            winners_gpu = cp.arange(len(all_inputs), dtype=cp.uint32)
        elif _HAS_TORCH:
            # PyTorch top-k: single fused kernel, ~2x faster than
            # CuPy argpartition + argsort (avoids 2 separate launches)
            torch_inputs = torch.as_tensor(all_inputs, device='cuda')
            _, top_idx = torch.topk(torch_inputs, k, sorted=True)
            winners_gpu = cp.asarray(top_idx).astype(cp.uint32)
        else:
            part_idx = cp.argpartition(-all_inputs, k)[:k]
            sorted_order = cp.argsort(-all_inputs[part_idx])
            winners_gpu = part_idx[sorted_order].astype(cp.uint32)

        # --- Process first-time winners (batch GPU gather) ---
        # Batch-gather first-timer inputs on GPU, then single transfer to CPU
        first_mask_gpu = winners_gpu >= tgt.w
        # Gather all_inputs at winner positions for first-timers (GPU-side)
        first_inputs_gathered = all_inputs[winners_gpu[first_mask_gpu]]

        # Single sync: transfer winners + first-timer inputs to CPU
        winners_cpu = winners_gpu.get().tolist()
        first_inputs_cpu = first_inputs_gathered.get().tolist() if first_inputs_gathered.size > 0 else []

        # Remap first-timers on CPU (tiny: k iterations)
        num_first = 0
        first_winner_inputs_cpu = []
        new_winner_indices = list(winners_cpu)
        first_idx = 0

        for i in range(k):
            if new_winner_indices[i] >= tgt.w:
                first_winner_inputs_cpu.append(int(first_inputs_cpu[first_idx]))
                first_idx += 1
                # Handle neuron_id_pool
                if tgt.neuron_id_pool is not None:
                    pid = tgt.neuron_id_pool_ptr
                    if pid >= len(tgt.neuron_id_pool):
                        raise RuntimeError(
                            f"Neuron id pool exhausted for area {tgt.name}")
                    actual_id = int(tgt.neuron_id_pool[pid])
                    tgt.neuron_id_pool_ptr += 1
                    tgt.compact_to_neuron_id.append(actual_id)
                else:
                    tgt.compact_to_neuron_id.append(tgt.w + num_first)
                new_winner_indices[i] = tgt.w + num_first
                num_first += 1

        new_w = tgt.w + num_first
        remapped_gpu = cp.array(new_winner_indices, dtype=cp.uint32)

        # --- Apply plasticity (fused kernels) ---
        if plasticity_enabled and self._plasticity_enabled_global:
            self._apply_plasticity_fused(
                target, from_stimuli, from_areas, remapped_gpu)

        # --- Expand connectomes for new winners ---
        if num_first > 0:
            self._expand_connectomes(
                target, from_stimuli, from_areas,
                input_sizes, new_winner_indices,
                first_winner_inputs_cpu, new_w,
            )

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
                tgt._cumulative_bias = cp.zeros(new_w, dtype=cp.float32)
                if len(old) > 0:
                    tgt._cumulative_bias[:len(old)] = old
            for cidx in new_winner_indices:
                if cidx < len(tgt._cumulative_bias):
                    tgt._cumulative_bias[cidx] += tgt.refracted_strength

        return ProjectionResult(
            winners=np.array(new_winner_indices, dtype=np.uint32),
            num_first_winners=num_first,
            num_ever_fired=new_w,
        )

    # -- Override: fused GPU plasticity --------------------------------------

    def _apply_plasticity_fused(self, target, from_stimuli, from_areas,
                                winners_gpu):
        """Hebbian learning using fused CUDA kernels.

        Uses hebbian_1d_kernel for stim->area and hebbian_2d_kernel
        for area->area, replacing multi-step CuPy fancy indexing.
        """
        tgt = self._areas[target]

        # Stimulus -> area (1-D weights)
        for stim_name in from_stimuli:
            conn = self._stim_conns[stim_name][target]
            beta = tgt.beta_by_source.get(stim_name, tgt.beta)
            if beta == 0:
                continue
            valid = winners_gpu[winners_gpu < len(conn.weights)]
            if len(valid) > 0:
                apply_hebbian_1d(conn.weights, valid, beta, self.w_max)

        # Area -> area (2-D weights)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            beta = tgt.beta_by_source.get(src_name, tgt.beta)
            if beta == 0:
                continue
            src = self._areas[src_name]
            src_w = cp.asarray(src.winners)
            if conn.weights.ndim == 2:
                valid_rows = src_w[src_w < conn.weights.shape[0]]
                valid_cols = winners_gpu[winners_gpu < conn.weights.shape[1]]
                if len(valid_rows) > 0 and len(valid_cols) > 0:
                    apply_hebbian_2d(
                        conn.weights,
                        valid_rows.astype(cp.uint32),
                        valid_cols.astype(cp.uint32),
                        beta, self.w_max)
            else:
                valid = winners_gpu[winners_gpu < len(conn.weights)]
                if len(valid) > 0:
                    apply_hebbian_1d(conn.weights, valid, beta, self.w_max)

    # -- Override: tight project_rounds loop ---------------------------------

    def project_rounds(self, target, from_stimuli, from_areas,
                       rounds, plasticity_enabled=True):
        """Execute multiple projection rounds with minimal overhead.

        Pre-resolves references, runs optimized project_into per round,
        only returns final ProjectionResult.
        """
        result = None
        for _ in range(rounds):
            result = self.project_into(
                target, from_stimuli, from_areas, plasticity_enabled)
        return result

    # -- Override: hash-based connection reset ------------------------------

    def reset_area_connections(self, area: str) -> None:
        """Reset area->area connections involving *area*.

        For sparse connectomes, resets to empty (will be re-expanded
        with hash-based init on next projection).  For dense connectomes,
        re-initializes with hash-based Bernoulli.
        """
        for src_name in list(self._area_conns.keys()):
            if area not in self._area_conns[src_name]:
                continue
            conn = self._area_conns[src_name][area]
            if conn.sparse:
                conn.weights = cp.empty((0, 0), dtype=cp.float32)
                if hasattr(conn, '_log_rows'):
                    del conn._log_rows
                if hasattr(conn, '_log_cols'):
                    del conn._log_cols
            else:
                pair_seed = self._get_pair_seed(src_name, area)
                rows, cols = conn.weights.shape
                conn.weights = _hash_bernoulli_2d(
                    0, rows, 0, cols, pair_seed, self.p)

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "cuda_implicit"


# Register engine (only succeeds if cupy imported successfully above)
register_engine("cuda_implicit", CudaImplicitEngine)
