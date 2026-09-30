"""The schedule instrument's single-episode write IS the capacity protocol's write.

`PREREG_presentation_schedule.md` bar PS-7 (and PS-9 after it) asked whether
the instrument reproduces the capacity line's published control cell, and both
failed. This test answers the question the bars were reaching for directly:
one episode of `TOTAL_ROUNDS` rounds through `ScheduledMemory.visit` returns
the same winners as `AssemblyMemory.store`, bit for bit, on the same seeds.

It does, so the bars failed on their premise rather than on the instrument.
The premise was that the published `M* = 8` means recall near 0.5 at M = 8. It
does not: `ceiling_from_curve` returns the grid's FIRST point when no point
clears the threshold (research/experiments/_substrate.py), so `M* = 8` on a
grid starting at 8 means the control was already below the bar there, which is
what the instrument measures too.

Marked `gpu`, and skipped when the fused kernels cannot build: the hashed
area compiles them on construction, which needs the documented developer
shell, so the maintained gate skips these rather than failing them.
"""
from __future__ import annotations

import numpy as np
import pytest
from neural_assemblies.tests import _devices



@pytest.fixture
def fused():
    """Skip rather than fail when the fused kernels cannot build.

    `HashedArea` compiles them on construction, and that needs the documented
    developer shell (`scripts/cuda-dev.cmd`: vcvars64, ninja, CUDA_HOME). The
    maintained gate runs without it, so a bare `gpu` mark is not enough: the
    two tests that construct a hashed area request this one. The third needs
    no device and runs everywhere.
    """
    return _devices.fused_kernels()


SEEDS = list(range(42, 46))
ITEMS = 8


def _derived(seeds):
    from neural_assemblies.core.numpy_engine import _seeding
    from research.experiments.presentation_schedule import to_i32
    return [to_i32(_seeding.fnv1a_pair_seed(s, "A", "A")) for s in seeds]


def _stim_seeds(seeds, i):
    from neural_assemblies.core.numpy_engine import _seeding
    from research.experiments.presentation_schedule import to_i32
    return [to_i32(_seeding.fnv1a_pair_seed(s, f"s{i}", "A")) for s in seeds]


def _protocol(items):
    from neural_assemblies.core.torch_engine._memory import AssemblyMemory
    from research.experiments.presentation_schedule import (
        BETA, K, N, P, TOTAL_ROUNDS, W_MAX)
    mem = AssemblyMemory(_derived(SEEDS), N, K, P, beta=BETA, w_max=W_MAX,
                         norm_init=True, synaptic_scaling=False,
                         rounds=TOTAL_ROUNDS, strength=0.0, gate=False,
                         max_items=items, device="cuda")
    return mem, [mem.store(_stim_seeds(SEEDS, i), stim_size=K) for i in range(items)]


def _instrument(items, *, strength=0.0):
    from research.experiments.presentation_schedule import (
        TOTAL_ROUNDS, ScheduledMemory)
    mem = ScheduledMemory(SEEDS, strength=strength, items=items, visits=1,
                          device="cuda", episode_rounds=TOTAL_ROUNDS)
    return mem, [mem.visit(i) for i in range(items)]


def _sorted(winners):
    return np.sort(winners.cpu().numpy(), axis=1)


def test_one_episode_reproduces_the_protocols_write_bit_for_bit(fused):
    protocol_mem, protocol_winners = _protocol(ITEMS)
    instrument_mem, instrument_winners = _instrument(ITEMS)
    for i, (a, b) in enumerate(zip(protocol_winners, instrument_winners)):
        np.testing.assert_array_equal(_sorted(a), _sorted(b), err_msg=f"item {i}")
    np.testing.assert_array_equal(protocol_mem.fill.cpu().numpy(),
                                  instrument_mem.fill.cpu().numpy())


def test_splitting_the_episode_does_change_the_write(fused):
    """The true negative: if the instrument returned the protocol's winners
    whatever the schedule, the whole study would be measuring nothing."""
    from research.experiments.presentation_schedule import (
        EPISODE_ROUNDS, EPISODES, ScheduledMemory)
    _, single = _instrument(ITEMS)
    split = ScheduledMemory(SEEDS, strength=0.0, items=ITEMS, visits=EPISODES,
                            device="cuda", episode_rounds=EPISODE_ROUNDS)
    last = [None] * ITEMS
    for i in range(ITEMS):
        for _ in range(EPISODES):
            last[i] = split.visit(i)
    differ = sum(not np.array_equal(_sorted(a), _sorted(b))
                 for a, b in zip(single, last) if b is not None)
    assert differ > 0, "splitting a round budget must change what is written"


def test_the_grid_bottom_ceiling_is_a_sentinel_not_a_recall_of_one_half():
    """Why PS-7 and PS-9 failed: `M* = 8` on a grid starting at 8 means NO
    point cleared the threshold, not that recall at 8 was near the bar."""
    from research.experiments._substrate import ceiling_from_curve
    collapsed = ceiling_from_curve([(8, 0.15), (16, 0.07), (32, 0.03)], threshold=0.5)
    assert collapsed.m_star == 8.0
    healthy = ceiling_from_curve([(8, 1.0), (16, 0.9), (32, 0.1)], threshold=0.5)
    assert healthy.m_star > 16.0
