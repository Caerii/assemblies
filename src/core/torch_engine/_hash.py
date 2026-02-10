"""Deterministic hash-based connectivity utilities for TorchSparseEngine.

Provides hash-based Bernoulli matrix generation, stimulus connectivity
counts, and CSR index helpers — all using PyTorch GPU tensors.
Ported from cuda_engine.py (CuPy -> torch).
"""

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weight dtype: bfloat16 halves memory and bandwidth for connectivity
# matrices.  bfloat16 (1 sign + 8 exponent + 7 mantissa) shares the same
# dynamic range as float32, avoiding overflow/underflow that float16 can
# hit, while still halving memory footprint.  Weights are binary 0/1 with
# Hebbian updates up to w_max (~20), well within bfloat16 precision.
WEIGHT_DTYPE = torch.bfloat16

# Multiplicative hash constants as signed int32
# 2654435761 unsigned = -1640531535 signed int32
# 2246822519 unsigned = -2048144777 signed int32
_HASH_A = torch.tensor(-1640531535, dtype=torch.int32)
_HASH_B = torch.tensor(-2048144777, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Hash seed derivation
# ---------------------------------------------------------------------------

def fnv1a_pair_seed(global_seed: int, source: str, target: str) -> int:
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


# ---------------------------------------------------------------------------
# Hash-based Bernoulli matrices
# ---------------------------------------------------------------------------

def hash_bernoulli_2d(row_start, row_end, col_start, col_end,
                      pair_seed, p, device='cuda'):
    """Vectorized hash-based Bernoulli(p) matrix on GPU using PyTorch.

    Same hash function as cuda_engine._hash_bernoulli_2d but using
    torch ops instead of CuPy.  Returns torch tensor in WEIGHT_DTYPE.
    """
    nr = row_end - row_start
    nc = col_end - col_start
    if nr == 0 or nc == 0:
        return torch.empty((nr, nc), dtype=WEIGHT_DTYPE, device=device)

    rows = torch.arange(row_start, row_end, dtype=torch.int32, device=device)
    cols = torch.arange(col_start, col_end, dtype=torch.int32, device=device)
    r, c = torch.meshgrid(rows, cols, indexing='ij')

    ha = _HASH_A.to(device)
    hb = _HASH_B.to(device)
    h = (r * ha) ^ (c * hb)
    seed_s32 = _to_signed32(pair_seed)
    h = h ^ torch.tensor(seed_s32, dtype=torch.int32, device=device)
    threshold = int(p * 16777216.0)
    return ((h & 0xFFFFFF) < threshold).to(WEIGHT_DTYPE)


def hash_stim_counts(stim_size, neuron_start, neuron_end,
                     pair_seed, p, device='cuda'):
    """Hash-based stim->area connectivity count using PyTorch on GPU.

    For each target neuron j, counts how many stimulus neurons are
    connected via hash.  Returns 1D torch tensor in WEIGHT_DTYPE.
    """
    n_neurons = neuron_end - neuron_start
    if n_neurons == 0:
        return torch.empty(0, dtype=WEIGHT_DTYPE, device=device)

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
        return connected.sum(dim=0).to(WEIGHT_DTYPE)
    else:
        result = torch.zeros(n_neurons, dtype=WEIGHT_DTYPE, device=device)
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
            result += connected.sum(dim=0).to(WEIGHT_DTYPE)
        return result


def hash_bernoulli_coo(row_start, row_end, col_start, col_end,
                       pair_seed, p, device='cuda', tile_size=4096):
    """Hash-based Bernoulli(p) connectivity as COO entries.

    Generates tiles of up to *tile_size* x *tile_size* to bound peak
    memory, then extracts non-zero positions.

    Returns ``(rows_int32, cols_int32, vals_weight_dtype)`` on *device*.
    """
    nr = row_end - row_start
    nc = col_end - col_start
    if nr == 0 or nc == 0:
        e = torch.empty(0, dtype=torch.int32, device=device)
        return e, e.clone(), torch.empty(0, dtype=WEIGHT_DTYPE, device=device)

    all_r, all_c, all_v = [], [], []
    for rb in range(row_start, row_end, tile_size):
        re = min(rb + tile_size, row_end)
        for cb in range(col_start, col_end, tile_size):
            ce = min(cb + tile_size, col_end)
            tile = hash_bernoulli_2d(rb, re, cb, ce,
                                     pair_seed, p, device)
            nz = tile.nonzero(as_tuple=True)
            if len(nz[0]) > 0:
                all_r.append((nz[0] + rb).int())
                all_c.append((nz[1] + cb).int())
                all_v.append(tile[nz[0], nz[1]])
    if all_r:
        return torch.cat(all_r), torch.cat(all_c), torch.cat(all_v)
    e = torch.empty(0, dtype=torch.int32, device=device)
    return e, e.clone(), torch.empty(0, dtype=WEIGHT_DTYPE, device=device)


# ---------------------------------------------------------------------------
# CSR index helper
# ---------------------------------------------------------------------------

def csr_flat_indices(crow, row_indices, nrows, device):
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
