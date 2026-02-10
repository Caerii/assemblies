"""
PyTorch-native GPU engine for assembly calculus.

TorchSparseEngine: Uses PyTorch CUDA tensors for all state and
computation.  Eliminates CuPy entirely from the hot path, gaining:
- Lower per-op dispatch overhead (~50us vs ~200us for CuPy)
- torch.topk: single fused CUDA kernel for winner selection
- torch advanced indexing for Hebbian updates
- Zero CuPy<->torch conversion overhead
- Hash-based deterministic initialization (ported from cuda_engine.py)

The same statistical sparse algorithm as NumpySparseEngine: truncated
normal sampling, Hebbian w *= (1+beta), amortised buffer growth, lazy
expansion.  Truncated normal stays on CPU (scipy dependency) with a
single transfer per projection.

Requires: torch with CUDA support.
"""

import math
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .engine import ComputeEngine, ProjectionResult, register_engine

try:
    from ..compute.sparse_simulation import SparseSimulationEngine
except ImportError:
    from compute.sparse_simulation import SparseSimulationEngine


# ---------------------------------------------------------------------------
# Hash utilities (ported from cuda_engine.py — CuPy -> torch)
# ---------------------------------------------------------------------------

# Multiplicative hash constants as signed int32
# 2654435761 unsigned = -1640531535 signed int32
# 2246822519 unsigned = -2048144777 signed int32
# Weight dtype: bfloat16 halves memory and bandwidth for connectivity
# matrices.  bfloat16 (1 sign + 8 exponent + 7 mantissa) shares the same
# dynamic range as float32, avoiding overflow/underflow that float16 can
# hit, while still halving memory footprint.  Weights are binary 0/1 with
# Hebbian updates up to w_max (~20), well within bfloat16 precision.
_WEIGHT_DTYPE = torch.bfloat16

_HASH_A = torch.tensor(-1640531535, dtype=torch.int32)
_HASH_B = torch.tensor(-2048144777, dtype=torch.int32)


def _fnv1a_pair_seed(global_seed: int, source: str, target: str) -> int:
    """Deterministic FNV-1a 32-bit seed for a (source, target) pair.

    Same algorithm as cuda_engine._fnv1a_pair_seed.
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


def _to_signed32(val):
    """Convert unsigned 32-bit value to signed int32 for torch."""
    val = val & 0xFFFFFFFF
    if val >= 0x80000000:
        return val - 0x100000000
    return val


def _hash_bernoulli_2d_torch(row_start, row_end, col_start, col_end,
                              pair_seed, p, device='cuda'):
    """Vectorized hash-based Bernoulli(p) matrix on GPU using PyTorch.

    Same hash function as cuda_engine._hash_bernoulli_2d but using
    torch ops instead of CuPy.  Returns torch float32 tensor.
    """
    nr = row_end - row_start
    nc = col_end - col_start
    if nr == 0 or nc == 0:
        return torch.empty((nr, nc), dtype=_WEIGHT_DTYPE, device=device)

    rows = torch.arange(row_start, row_end, dtype=torch.int32, device=device)
    cols = torch.arange(col_start, col_end, dtype=torch.int32, device=device)
    r, c = torch.meshgrid(rows, cols, indexing='ij')

    ha = _HASH_A.to(device)
    hb = _HASH_B.to(device)
    h = (r * ha) ^ (c * hb)
    seed_s32 = _to_signed32(pair_seed)
    h = h ^ torch.tensor(seed_s32, dtype=torch.int32, device=device)
    threshold = int(p * 16777216.0)
    return ((h & 0xFFFFFF) < threshold).to(_WEIGHT_DTYPE)


def _hash_stim_counts_torch(stim_size, neuron_start, neuron_end,
                             pair_seed, p, device='cuda'):
    """Hash-based stim->area connectivity count using PyTorch on GPU.

    For each target neuron j, counts how many stimulus neurons are
    connected via hash.  Returns 1D torch float32 tensor.
    """
    n_neurons = neuron_end - neuron_start
    if n_neurons == 0:
        return torch.empty(0, dtype=_WEIGHT_DTYPE, device=device)

    ha = _HASH_A.to(device)
    hb = _HASH_B.to(device)
    seed_t = torch.tensor(_to_signed32(pair_seed), dtype=torch.int32,
                           device=device)
    threshold = int(p * 16777216.0)

    if stim_size <= 1024:
        stim_ids = torch.arange(stim_size, dtype=torch.int32, device=device)
        neuron_ids = torch.arange(neuron_start, neuron_end,
                                  dtype=torch.int32, device=device)
        s, n = torch.meshgrid(stim_ids, neuron_ids, indexing='ij')
        h = (s * ha) ^ (n * hb)
        h = h ^ seed_t
        connected = (h & 0xFFFFFF) < threshold
        return connected.sum(dim=0).to(_WEIGHT_DTYPE)
    else:
        result = torch.zeros(n_neurons, dtype=_WEIGHT_DTYPE, device=device)
        neuron_ids = torch.arange(neuron_start, neuron_end,
                                  dtype=torch.int32, device=device)
        for batch_start in range(0, stim_size, 1024):
            batch_end = min(batch_start + 1024, stim_size)
            stim_ids = torch.arange(batch_start, batch_end,
                                    dtype=torch.int32, device=device)
            s, n = torch.meshgrid(stim_ids, neuron_ids, indexing='ij')
            h = (s * ha) ^ (n * hb)
            h = h ^ seed_t
            connected = (h & 0xFFFFFF) < threshold
            result += connected.sum(dim=0).to(_WEIGHT_DTYPE)
        return result


# ---------------------------------------------------------------------------
# Internal state containers
# ---------------------------------------------------------------------------

# Threshold above which we use lazy ID generation instead of
# pre-computing a full permutation of n neuron IDs.
_LAZY_ID_THRESHOLD = 1_000_000


@dataclass
class _TorchAreaState:
    """Per-area state for TorchSparseEngine."""
    name: str
    n: int
    k: int
    beta: float
    w: int = 0
    winners: Optional[torch.Tensor] = None  # int32 on CUDA
    compact_to_neuron_id: list = field(default_factory=list)
    neuron_id_pool: Optional[np.ndarray] = None  # pre-computed for small n
    neuron_id_pool_ptr: int = 0
    # Lazy ID generation for large n (avoids O(n) permutation)
    _lazy_ids: bool = False
    _used_ids: Optional[set] = None
    _id_rng: Optional[np.random.Generator] = None
    fixed_assembly: bool = False
    beta_by_source: dict = field(default_factory=dict)
    # LRI
    refractory_period: int = 0
    inhibition_strength: float = 0.0
    _refractory_history: Optional[deque] = None
    # Refracted mode
    refracted: bool = False
    refracted_strength: float = 0.0
    _cumulative_bias: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.winners is None:
            self.winners = torch.empty(0, dtype=torch.int32, device='cuda')
        if self._refractory_history is None:
            self._refractory_history = deque(
                maxlen=max(self.refractory_period, 1))
        if self._cumulative_bias is None:
            self._cumulative_bias = torch.zeros(0, dtype=torch.float32,
                                                device='cuda')

    def next_neuron_id(self) -> int:
        """Get next unique neuron ID, either from pool or lazy sampling."""
        if self._lazy_ids:
            # Sample random unique ID — for n >> w, almost never retries
            while True:
                candidate = int(self._id_rng.integers(0, self.n))
                if candidate not in self._used_ids:
                    self._used_ids.add(candidate)
                    return candidate
        else:
            pid = self.neuron_id_pool_ptr
            if pid >= len(self.neuron_id_pool):
                raise RuntimeError(
                    f"Neuron id pool exhausted for area {self.name}")
            self.neuron_id_pool_ptr += 1
            return int(self.neuron_id_pool[pid])


@dataclass
class _StimulusState:
    """Stimulus descriptor."""
    name: str
    size: int


class _TorchConn:
    """Lightweight connectivity wrapper for stim→area (1-D weights)."""
    __slots__ = ('weights', 'sparse')

    def __init__(self, weights, sparse=True):
        self.weights = weights
        self.sparse = sparse


# ---------------------------------------------------------------------------
# CSR sparse connectivity for area→area connections
# ---------------------------------------------------------------------------

def _csr_flat_indices(crow, row_indices, nrows, device):
    """Flat indices into CSR col/val arrays for selected rows.

    Given CSR row-pointer array *crow* and a tensor of *row_indices*,
    return a 1-D int64 tensor of positions into the col/val arrays
    that belong to those rows, or ``None`` if empty.
    """
    valid = row_indices[row_indices < nrows].long()
    if len(valid) == 0:
        return None
    starts = crow[valid]
    ends = crow[valid + 1]
    lengths = (ends - starts)
    total = lengths.sum().item()
    if total == 0:
        return None
    row_starts = torch.repeat_interleave(starts, lengths)
    cum = lengths.cumsum(0)
    bases = torch.cat([torch.zeros(1, dtype=torch.int64, device=device),
                       cum[:-1]])
    offsets = torch.arange(total, dtype=torch.int64, device=device)
    offsets -= torch.repeat_interleave(bases, lengths)
    return (row_starts + offsets).long()


def _hash_bernoulli_coo_torch(row_start, row_end, col_start, col_end,
                               pair_seed, p, device='cuda', tile_size=4096):
    """Hash-based Bernoulli(p) connectivity as COO entries.

    Generates tiles of up to *tile_size* × *tile_size* to bound peak
    memory, then extracts non-zero positions.

    Returns ``(rows_int32, cols_int32, vals_weight_dtype)`` on *device*.
    """
    nr = row_end - row_start
    nc = col_end - col_start
    if nr == 0 or nc == 0:
        e = torch.empty(0, dtype=torch.int32, device=device)
        return e, e.clone(), torch.empty(0, dtype=_WEIGHT_DTYPE, device=device)

    all_r, all_c, all_v = [], [], []
    for rb in range(row_start, row_end, tile_size):
        re = min(rb + tile_size, row_end)
        for cb in range(col_start, col_end, tile_size):
            ce = min(cb + tile_size, col_end)
            tile = _hash_bernoulli_2d_torch(rb, re, cb, ce,
                                            pair_seed, p, device)
            nz = tile.nonzero(as_tuple=True)
            if len(nz[0]) > 0:
                all_r.append((nz[0] + rb).int())
                all_c.append((nz[1] + cb).int())
                all_v.append(tile[nz[0], nz[1]])
    if all_r:
        return torch.cat(all_r), torch.cat(all_c), torch.cat(all_v)
    e = torch.empty(0, dtype=torch.int32, device=device)
    return e, e.clone(), torch.empty(0, dtype=_WEIGHT_DTYPE, device=device)


class _CSRConn:
    """CSR-format area→area connectivity on GPU.

    Stores weights as Compressed Sparse Row — three arrays (crow, col,
    val) — using O(nnz) memory instead of O(rows × cols).  At typical
    connection probability p=0.0005 this is ~2000× smaller than dense.
    """

    def __init__(self, device='cuda'):
        self._device = device
        self._nrows = 0
        self._ncols = 0
        self._log_rows = 0   # hash-initialised row extent
        self._log_cols = 0   # hash-initialised col extent
        self._crow = torch.zeros(1, dtype=torch.int64, device=device)
        self._col = torch.empty(0, dtype=torch.int32, device=device)
        self._val = torch.empty(0, dtype=_WEIGHT_DTYPE, device=device)

    @property
    def nnz(self):
        return len(self._col)

    # -- Input accumulation (project_into hot path) -------------------------

    def accumulate_rows(self, row_indices, out_size):
        """Sum selected rows → dense float32 vector of *out_size*."""
        result = torch.zeros(out_size, dtype=torch.float32,
                             device=self._device)
        if self.nnz == 0 or len(row_indices) == 0:
            return result
        flat_idx = _csr_flat_indices(
            self._crow, row_indices, self._nrows, self._device)
        if flat_idx is None:
            return result
        sel_cols = self._col[flat_idx].long()
        sel_vals = self._val[flat_idx].float()
        valid = sel_cols < out_size
        if not valid.all():
            sel_cols = sel_cols[valid]
            sel_vals = sel_vals[valid]
        result.scatter_add_(0, sel_cols, sel_vals)
        return result

    # -- Hebbian plasticity -------------------------------------------------

    def hebbian_update(self, src_winners, tgt_winners, beta, w_max):
        """Multiply entries at (src, tgt) intersections by (1+β)."""
        if self.nnz == 0 or len(src_winners) == 0 or len(tgt_winners) == 0:
            return
        flat_idx = _csr_flat_indices(
            self._crow, src_winners, self._nrows, self._device)
        if flat_idx is None:
            return
        sel_cols = self._col[flat_idx]
        col_mask = torch.isin(sel_cols.int(), tgt_winners.int())
        update_idx = flat_idx[col_mask]
        if len(update_idx) > 0:
            updated = self._val[update_idx].float() * (1 + beta)
            if w_max is not None and w_max > 0:
                updated = updated.clamp(max=w_max)
            self._val[update_idx] = updated.to(_WEIGHT_DTYPE)

    # -- Expansion (add new rows / columns) ---------------------------------

    def expand(self, needed_rows, needed_cols, new_r, new_c, new_v):
        """Merge new COO entries into the CSR and rebuild."""
        # Convert existing CSR → COO
        if self.nnz > 0:
            lengths = self._crow[1:] - self._crow[:-1]
            old_r = torch.repeat_interleave(
                torch.arange(self._nrows, dtype=torch.int32,
                             device=self._device),
                lengths.int())
            old_c = self._col
            old_v = self._val
        else:
            old_r = torch.empty(0, dtype=torch.int32, device=self._device)
            old_c = torch.empty(0, dtype=torch.int32, device=self._device)
            old_v = torch.empty(0, dtype=_WEIGHT_DTYPE, device=self._device)

        all_r = torch.cat([old_r, new_r]) if len(new_r) > 0 else old_r
        all_c = torch.cat([old_c, new_c]) if len(new_c) > 0 else old_c
        all_v = torch.cat([old_v, new_v]) if len(new_v) > 0 else old_v

        self._rebuild_csr(needed_rows, needed_cols, all_r, all_c, all_v)

    def _rebuild_csr(self, nrows, ncols, rows, cols, vals):
        """Build CSR from COO, deduplicating (last value wins)."""
        self._nrows = nrows
        self._ncols = ncols
        if len(rows) == 0:
            self._crow = torch.zeros(
                nrows + 1, dtype=torch.int64, device=self._device)
            self._col = torch.empty(0, dtype=torch.int32, device=self._device)
            self._val = torch.empty(0, dtype=_WEIGHT_DTYPE, device=self._device)
            return

        # Sort by (row, col); stable so last duplicate wins
        sort_key = rows.long() * ncols + cols.long()
        order = sort_key.argsort(stable=True)
        rows = rows[order]; cols = cols[order]; vals = vals[order]
        sk = sort_key[order]

        # Keep last occurrence of each (row, col) pair
        unique = torch.ones(len(sk), dtype=torch.bool, device=self._device)
        unique[:-1] = sk[:-1] != sk[1:]
        rows = rows[unique]; cols = cols[unique]; vals = vals[unique]

        # Build crow from row counts
        self._crow = torch.zeros(
            nrows + 1, dtype=torch.int64, device=self._device)
        if len(rows) > 0:
            counts = torch.zeros(
                nrows, dtype=torch.int64, device=self._device)
            counts.scatter_add_(
                0, rows.long(),
                torch.ones(len(rows), dtype=torch.int64,
                           device=self._device))
            self._crow[1:] = counts.cumsum(0)
        self._col = cols.int()
        self._val = vals

    # -- Column normalisation -----------------------------------------------

    def normalize_columns(self, eps=1e-8):
        """Column-normalize so each column sums to 1.0."""
        if len(self._val) == 0 or self._ncols == 0:
            return
        sums = torch.zeros(self._ncols, dtype=torch.float32,
                           device=self._device)
        sums.scatter_add_(0, self._col.long(), self._val.float())
        sums = sums.clamp(min=eps)
        factors = sums[self._col.long()]
        self._val = (self._val.float() / factors).to(_WEIGHT_DTYPE)

    # -- Reset --------------------------------------------------------------

    def reset(self):
        """Clear all entries and dimensions."""
        self._nrows = 0
        self._ncols = 0
        self._log_rows = 0
        self._log_cols = 0
        self._crow = torch.zeros(1, dtype=torch.int64, device=self._device)
        self._col = torch.empty(0, dtype=torch.int32, device=self._device)
        self._val = torch.empty(0, dtype=_WEIGHT_DTYPE, device=self._device)


# ---------------------------------------------------------------------------
# TorchSparseEngine
# ---------------------------------------------------------------------------

class TorchSparseEngine(ComputeEngine):
    """PyTorch-native GPU engine with hash-based connectivity.

    Same statistical sparse algorithm as NumpySparseEngine but all arrays
    are torch.cuda tensors.  Key performance advantages:
    - torch.topk: single fused kernel (vs argpartition + argsort)
    - Lower per-op dispatch overhead than CuPy
    - No CuPy<->torch conversion for operations
    - Hash-based deterministic initialization

    Parameters:
        p:             Connection probability.
        seed:          Global random seed.
        w_max:         Hebbian weight ceiling.
        deterministic: If True, use legacy exact-fit expansion.
    """

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False, **kwargs):
        self.p = p
        self.w_max = w_max
        self._deterministic = deterministic
        self._rng = np.random.default_rng(seed)
        self._plasticity_enabled_global = True
        self._global_seed = seed
        self._pair_seeds: Dict[tuple, int] = {}

        self._device = torch.device('cuda')

        # Internal state
        self._areas: Dict[str, _TorchAreaState] = {}
        self._stimuli: Dict[str, _StimulusState] = {}

        # Connectivity: stim_name -> area_name -> _TorchConn (1-D weights)
        self._stim_conns: Dict[str, Dict[str, _TorchConn]] = defaultdict(dict)
        # Connectivity: src_area -> tgt_area -> _TorchConn (2-D weights)
        self._area_conns: Dict[str, Dict[str, _TorchConn]] = defaultdict(dict)

        # Reusable CPU math primitives (truncated normal, input splits)
        self._sparse_sim = SparseSimulationEngine(
            np.random.default_rng(seed))

    # -- Pair seed derivation -----------------------------------------------

    def _get_pair_seed(self, source: str, target: str) -> int:
        key = (source, target)
        if key not in self._pair_seeds:
            self._pair_seeds[key] = _fnv1a_pair_seed(
                self._global_seed, source, target)
        return self._pair_seeds[key]

    # -- Registration -------------------------------------------------------

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0) -> None:
        area = _TorchAreaState(
            name=name, n=n, k=k, beta=beta,
            refractory_period=refractory_period,
            inhibition_strength=inhibition_strength)
        if n > _LAZY_ID_THRESHOLD:
            # Lazy ID generation: O(1) init, O(1) per new neuron
            area._lazy_ids = True
            area._used_ids = set()
            area._id_rng = np.random.default_rng(
                self._rng.integers(0, 2**32))
        else:
            area.neuron_id_pool = self._rng.permutation(
                np.arange(n, dtype=np.uint32))
            area.neuron_id_pool_ptr = 0
        self._areas[name] = area

        # stim->area for every already-registered stimulus
        for stim_name in self._stimuli:
            conn = _TorchConn(
                torch.empty(0, dtype=_WEIGHT_DTYPE, device=self._device),
                sparse=True)
            self._stim_conns[stim_name][name] = conn
            area.beta_by_source[stim_name] = beta

        # area->area for every existing area (both directions) — CSR format
        for other_name, other in self._areas.items():
            if other_name == name:
                self._area_conns[name][name] = _CSRConn(
                    device=self._device)
            else:
                self._area_conns[other_name][name] = _CSRConn(
                    device=self._device)
                self._area_conns[name][other_name] = _CSRConn(
                    device=self._device)
                area.beta_by_source[other_name] = beta
                other.beta_by_source[name] = beta

    def add_stimulus(self, name: str, size: int) -> None:
        self._stimuli[name] = _StimulusState(name=name, size=size)
        for area_name, area in self._areas.items():
            conn = _TorchConn(
                torch.empty(0, dtype=_WEIGHT_DTYPE, device=self._device),
                sparse=True)
            self._stim_conns[name][area_name] = conn
            area.beta_by_source[name] = area.beta

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        pass  # connectivity created in add_area / add_stimulus

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

        # --- Sample new winner candidates via truncated normal (CPU) ---
        input_sizes = (
            [self._stimuli[s].size for s in from_stimuli]
            + [self._areas[a].k for a in from_areas])

        old_rng = self._sparse_sim.rng
        self._sparse_sim.rng = rng
        if self._deterministic:
            potential_new_np = self._sparse_sim.sample_new_winner_inputs_legacy(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p)
        else:
            potential_new_np = self._sparse_sim.sample_new_winner_inputs(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p)
        self._sparse_sim.rng = old_rng

        # Transfer to GPU (single copy)
        # sample_new_winner_inputs may return numpy or cupy depending on
        # the global backend — normalize to numpy first.
        if hasattr(potential_new_np, 'get'):
            potential_new_np = potential_new_np.get()  # cupy -> numpy
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
        # Batch-gather first-timer inputs on GPU
        first_mask = winners_gpu.long() >= tgt.w
        first_input_vals = all_inputs[winners_gpu[first_mask].long()]

        # Single sync: transfer to CPU
        winners_cpu = winners_gpu.cpu().tolist()
        first_inputs_cpu = (first_input_vals.cpu().tolist()
                            if first_input_vals.numel() > 0 else [])

        # Remap first-timers on CPU (k iterations, tiny)
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

        # Stimulus -> area (1-D weights)
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

        # Area -> area (CSR weights)
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
                        add = _hash_stim_counts_torch(
                            self._stimuli[stim_name].size, old, new_w,
                            pair_seed, self.p, device=self._device)
                    else:
                        add = torch.zeros(add_len, dtype=_WEIGHT_DTYPE,
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
                pass  # no hash expansion needed, but explicit entries below
            else:
                # Generate hash-based Bernoulli blocks as COO (tiled)
                # Block A: new rows × existing cols
                if needed_rows > log_rows and log_cols > 0:
                    r, c, v = _hash_bernoulli_coo_torch(
                        log_rows, needed_rows, 0, log_cols,
                        pair_seed, self.p, device=self._device)
                    if len(r) > 0:
                        coo_r_parts.append(r)
                        coo_c_parts.append(c)
                        coo_v_parts.append(v)

                # Block B: existing rows × new cols
                if needed_cols > log_cols and log_rows > 0:
                    r, c, v = _hash_bernoulli_coo_torch(
                        0, log_rows, log_cols, needed_cols,
                        pair_seed, self.p, device=self._device)
                    if len(r) > 0:
                        coo_r_parts.append(r)
                        coo_c_parts.append(c)
                        coo_v_parts.append(v)

                # Block C: new rows × new cols
                if needed_rows > log_rows and needed_cols > log_cols:
                    r, c, v = _hash_bernoulli_coo_torch(
                        log_rows, needed_rows, log_cols, needed_cols,
                        pair_seed, self.p, device=self._device)
                    if len(r) > 0:
                        coo_r_parts.append(r)
                        coo_c_parts.append(c)
                        coo_v_parts.append(v)

                csr._log_rows = needed_rows
                csr._log_cols = needed_cols

            # Collect explicit entries from first-timer allocations
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
                    len(exp_rows), dtype=_WEIGHT_DTYPE,
                    device=self._device))

            # Merge into CSR
            if coo_r_parts:
                new_r = torch.cat(coo_r_parts)
                new_c = torch.cat(coo_c_parts)
                new_v = torch.cat(coo_v_parts)
                csr.expand(needed_rows, needed_cols, new_r, new_c, new_v)
            elif needed_rows > csr._nrows or needed_cols > csr._ncols:
                # Dimensions grew but no new entries — just update dims
                e = torch.empty(0, dtype=torch.int32, device=self._device)
                csr.expand(needed_rows, needed_cols,
                           e, e.clone(),
                           torch.empty(0, dtype=_WEIGHT_DTYPE,
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


# Register engine (only succeeds if torch+CUDA available)
if torch.cuda.is_available():
    register_engine("torch_sparse", TorchSparseEngine)
