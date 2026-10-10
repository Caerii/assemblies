"""Shared helpers of the hashed fibers and area: per-brain values, the stimulus gain
tables, the count at which the weight clip binds, local indices, the overflow error.

Moved from _hashed.py unchanged; _hashed.py re-exports every name."""
from __future__ import annotations

from ._torch_ops import torch_ops
from .._pricing import (chain_table as _chain_table,
                        gain_table as _gain_table)


def per_brain(value, B):
    """``value`` as a tuple of B floats when it is a sequence (one per brain),
    else None: the scalar is shared and every caller keeps its scalar path.

    A learning-rate sweep runs its rates as brains of ONE launch
    (DESIGN_memory_throughput.md); each brain's arithmetic is the arithmetic
    it would do alone, so a swept brain's trajectory equals its solo run's."""
    if value is None or isinstance(value, (int, float)):
        return None
    values = tuple(float(v) for v in value)
    if len(values) != B:
        raise ValueError(f"{len(values)} per-brain values for {B} brains")
    return values


_GAIN_TABLES: dict = {}


def _device_gain(betas, rounds, device):
    """The stimulus gain table on the device: [rounds + 1] for one rate,
    [B, rounds + 1] for one rate per brain. Read-only and cached: a stimulus
    fiber is built per stored item, and copying its table from host memory
    each time made every item wait for the GPU to drain."""
    key = (betas, rounds, device)
    table = _GAIN_TABLES.get(key)
    if table is None:
        if isinstance(betas, tuple):
            import numpy as np
            rows = {b: _gain_table(b, rounds) for b in set(betas)}
            host = np.stack([rows[b] for b in betas])
        else:
            host = _gain_table(betas, rounds)
        if len(_GAIN_TABLES) > 256:
            _GAIN_TABLES.clear()
        table = _GAIN_TABLES[key] = torch_ops.from_numpy(host).to(device)
    return table


def clip_count(beta, w_max, B=1):
    """The count at which the weight clip binds -- the first c with
    chain(1, c) == w_max under the engine's per-step multiply-and-clip --
    for the largest such c over the brains' rates; None without a clip or
    for a rate that never learns."""
    if w_max is None:
        return None
    betas = per_brain(beta, B) or (float(beta),)
    import math
    worst = 0
    for b in set(betas):
        if b <= 0:
            continue
        guess = int(math.log(w_max) / math.log1p(b)) + 4
        table = _chain_table(b, w_max, guess)
        hits = [c for c in range(len(table)) if table[c] == table[-1]]
        if table[-1] != table[-2]:
            table = _chain_table(b, w_max, 2 * guess + 64)
            hits = [c for c in range(len(table)) if table[c] == table[-1]]
        worst = max(worst, hits[0])
    return worst


def _local_index(idx):
    """Map raw ids to per-brain local ids. ``idx`` [B, L] -> loc, values, W."""
    B, L = idx.shape
    srt, order = torch_ops.sort(idx, dim=1)
    fresh = torch_ops.ones_like(srt, dtype=torch_ops.bool)
    fresh[:, 1:] = srt[:, 1:] != srt[:, :-1]
    loc_sorted = torch_ops.cumsum(fresh, dim=1) - 1
    loc = torch_ops.empty_like(loc_sorted)
    loc.scatter_(1, order, loc_sorted)
    W = int(fresh.sum(1).max())
    vals = torch_ops.full((B, W), -1, dtype=torch_ops.int64, device=idx.device)
    vals.scatter_(1, loc_sorted, srt)      # duplicates write the same value
    return loc, vals, W


def _raise_overflow(bad):
    if bad:
        raise RuntimeError(
            f"k-WTA candidate set overflowed ({bad} candidates) -- the "
            "drive is too flat for the histogram to narrow. Refusing "
            "to return a truncated winner set.")
