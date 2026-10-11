"""Brain seeds to the seeds the device stores.

    seeds_for(seeds)  each brain seed's recurrent fiber seed, fnv1a_pair_seed(seed, "A", "A"),
                      as a signed 32-bit integer
    to_i32(v)         an integer's low 32 bits, read as signed

Owned here since 2026-10-10. Before, both lived in seq_capacity_scaling, which re-exports them.
"""
from __future__ import annotations

from neural_assemblies.core.numpy_engine import _seeding


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def seeds_for(seeds):
    return [to_i32(_seeding.fnv1a_pair_seed(seed, "A", "A")) for seed in seeds]
