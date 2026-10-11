"""How a memory study reads a memory, and the thresholds it reads against.

    ROUNDS            rounds of recurrence per write and per frozen recall (8)
    RECALL_SAMPLE     items a capacity reading recalls per brain (32), drawn by
    MEASUREMENT_SEED  the seed of that draw (1234), fixed so every condition reads the same items
    sample_for(M)     the items read at store size M
    HALF_BAR          the capacity criterion: M* is where the ensemble-mean rank-1 crosses 0.5
    COMPLETE          completion: a recall recovers at least 0.8 of its item (Amendment 10)
    MATCH             a replayed element matches when its own overlap is >= 0.3 (Amendment 37)
    overlap(a, b)     per brain, the fraction of a's winners found in b: [B, k] x [B, k] -> [B]

Owned here since 2026-10-10. Before, ROUNDS (as T), RECALL_SAMPLE, MEASUREMENT_SEED, sample_for
and HALF_BAR lived in memory_pattern_efficiency (Amendment 9), COMPLETE in memory_write_strength,
MATCH in memory_load_law, and overlap twice, as the private _overlap of memory_sequences and of
memory_write_rules; those modules re-export them.
"""
from __future__ import annotations

import numpy as np

ROUNDS = 8
HALF_BAR = 0.5
RECALL_SAMPLE = 32
MEASUREMENT_SEED = 1234
COMPLETE = 0.8
MATCH = 0.3


def sample_for(M):
    return np.random.default_rng([MEASUREMENT_SEED, M]).choice(
        M, min(RECALL_SAMPLE, M), replace=False)


def overlap(a, b):
    """[B, k] x [B, k] -> [B]: the fraction of a's winners in b."""
    return (a.unsqueeze(2) == b.unsqueeze(1)).any(2).float().mean(dim=1)
