"""Deterministic hash-based connectivity utilities for TorchSparseEngine.

Provides hash-based Bernoulli matrix generation, stimulus connectivity
counts, and CSR index helpers — all using PyTorch GPU tensors.
Ported from cuda_engine.py (CuPy -> torch).
"""

from ._torch_ops import torch_ops

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weight dtype: bfloat16 halves memory and bandwidth for connectivity
# matrices.  bfloat16 (1 sign + 8 exponent + 7 mantissa) shares the same
# dynamic range as float32, avoiding overflow/underflow that float16 can
# hit, while still halving memory footprint.  Weights are binary 0/1 with
# Hebbian updates up to w_max (~20), well within bfloat16 precision.
WEIGHT_DTYPE = torch_ops.bfloat16

# Multiplicative hash constants as signed int32
# 2654435761 unsigned = -1640531535 signed int32
# 2246822519 unsigned = -2048144777 signed int32
_HASH_A = torch_ops.tensor(-1640531535, dtype=torch_ops.int32)
_HASH_B = torch_ops.tensor(-2048144777, dtype=torch_ops.int32)


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
    """Convert unsigned 32-bit value to signed int32 for torch_ops."""
    val = val & 0xFFFFFFFF
    if val >= 0x80000000:
        return val - 0x100000000
    return val


# Murmur3 fmix32 constants, as signed int32 for torch_ops.
_FMIX_M1 = _to_signed32(0x85EBCA6B)
_FMIX_M2 = _to_signed32(0xC2B2AE35)


def _fmix32(h):
    """Murmur3's avalanche finalizer, on an int32 tensor of uint32 patterns.

    THIS WAS MISSING, and its absence is not cosmetic. ``(r*A) ^ (c*B) ^ seed``
    has low bits that are close to a function of the low bits of r and c alone,
    and the Bernoulli test reads exactly those low 24 bits. Density comes out
    right; the dependence structure does not. Measured here at 2048x2048,
    p=0.05, reproducing the table in ``kernels/implicit.py``:

        source                   density  row disp  col disp   corr(i)
        numpy Generator (ref)    0.04998     0.969     0.926  -0.00009
        raw hash (this file)     0.04999     3.794     0.015  -0.05262
        + fmix32                 0.04995     0.930     0.925   0.00047

    Column dispersion 0.015 means IN-DEGREE WAS NEARLY CONSTANT -- and
    ``norm_init`` scales each neuron's incoming weights by its in-degree, so
    the raw hash quietly degenerates that mechanism while every density check
    stays green. ``_seeding.mix32`` (numpy), ``na-kernels`` (rust),
    ``cuda_engine._fmix32`` and all four sites in ``kernels/implicit.py``
    finalize; this file claimed to be the "same hash function as
    cuda_engine._hash_bernoulli_2d" and was not, so the two engines disagreed
    about 9.5% of cells -- see ``tests/test_torch_hash_finalizer.py``.

    Torch's ``>>`` on int32 is ARITHMETIC, so each logical shift is masked back
    to the bits a uint32 shift would keep. The multiplies wrap, which is
    exactly uint32 arithmetic mod 2**32.
    """
    h = h ^ ((h >> 16) & 0xFFFF)
    h = h * _FMIX_M1
    h = h ^ ((h >> 13) & 0x7FFFF)
    h = h * _FMIX_M2
    h = h ^ ((h >> 16) & 0xFFFF)
    return h


# ---------------------------------------------------------------------------
# Hash-based Bernoulli matrices
# ---------------------------------------------------------------------------

def hash_bernoulli_2d(row_start, row_end, col_start, col_end,
                      pair_seed, p, device='cuda'):
    """Vectorized hash-based Bernoulli(p) matrix on GPU using PyTorch.

    Same hash function as cuda_engine._hash_bernoulli_2d -- INCLUDING the
    fmix32 finalizer, whose absence here made that claim false and the two
    engines disagree about 9.5% of cells. See `_fmix32`.
    """
    nr = row_end - row_start
    nc = col_end - col_start
    if nr == 0 or nc == 0:
        return torch_ops.empty((nr, nc), dtype=WEIGHT_DTYPE, device=device)

    rows = torch_ops.arange(row_start, row_end, dtype=torch_ops.int32, device=device)
    cols = torch_ops.arange(col_start, col_end, dtype=torch_ops.int32, device=device)
    r, c = torch_ops.meshgrid(rows, cols, indexing='ij')

    ha = _HASH_A.to(device)
    hb = _HASH_B.to(device)
    h = (r * ha) ^ (c * hb)
    seed_s32 = _to_signed32(pair_seed)
    h = h ^ torch_ops.tensor(seed_s32, dtype=torch_ops.int32, device=device)
    h = _fmix32(h)
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
        return torch_ops.empty(0, dtype=WEIGHT_DTYPE, device=device)

    ha = _HASH_A.to(device)
    hb = _HASH_B.to(device)
    seed_t = torch_ops.tensor(_to_signed32(pair_seed), dtype=torch_ops.int32,
                           device=device)
    threshold = int(p * 16777216.0)

    if stim_size <= 1024:
        stim_ids = torch_ops.arange(stim_size, dtype=torch_ops.int32, device=device)
        neuron_ids = torch_ops.arange(neuron_start, neuron_end,
                                  dtype=torch_ops.int32, device=device)
        s, n = torch_ops.meshgrid(stim_ids, neuron_ids, indexing='ij')
        h = (s * ha) ^ (n * hb)
        h = _fmix32(h ^ seed_t)
        connected = (h & 0xFFFFFF) < threshold
        return connected.sum(dim=0).to(WEIGHT_DTYPE)
    else:
        result = torch_ops.zeros(n_neurons, dtype=WEIGHT_DTYPE, device=device)
        neuron_ids = torch_ops.arange(neuron_start, neuron_end,
                                  dtype=torch_ops.int32, device=device)
        for batch_start in range(0, stim_size, 1024):
            batch_end = min(batch_start + 1024, stim_size)
            stim_ids = torch_ops.arange(batch_start, batch_end,
                                    dtype=torch_ops.int32, device=device)
            s, n = torch_ops.meshgrid(stim_ids, neuron_ids, indexing='ij')
            h = (s * ha) ^ (n * hb)
            h = _fmix32(h ^ seed_t)
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
        e = torch_ops.empty(0, dtype=torch_ops.int32, device=device)
        return e, e.clone(), torch_ops.empty(0, dtype=WEIGHT_DTYPE, device=device)

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
        return torch_ops.cat(all_r), torch_ops.cat(all_c), torch_ops.cat(all_v)
    e = torch_ops.empty(0, dtype=torch_ops.int32, device=device)
    return e, e.clone(), torch_ops.empty(0, dtype=WEIGHT_DTYPE, device=device)


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
    row_starts = torch_ops.repeat_interleave(starts, lengths)
    cum = lengths.cumsum(0)
    bases = torch_ops.cat([torch_ops.zeros(1, dtype=torch_ops.int64, device=device),
                       cum[:-1]])
    offsets = torch_ops.arange(total, dtype=torch_ops.int64, device=device)
    offsets -= torch_ops.repeat_interleave(bases, lengths)
    return (row_starts + offsets).long()
