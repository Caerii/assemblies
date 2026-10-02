"""Organ count saturation is exact only when the chain table has clipped.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-organ-count-saturation

The organ fiber stores int8 counts (at most 127) and prices every count at
or beyond its chain table's last index with that entry. When the weight
clip has bound by the last entry, a count held at 127 gives the same drive
as the true count and the kernel's overflow flag is informational; without
a clip, or with a table longer than the count range, it is a real loss and
must stay an error. The gap-3 temporal-position run of 2026-09-13 is where
this bit.
"""
import numpy as np
import pytest

from neural_assemblies.core._pricing import (
    chain_table as _chain_table, count_saturation_is_exact)


def test_clipped_table_within_the_count_range_saturates_exactly():
    table = _chain_table(0.1, 20.0, 64)          # the registered organ schedule
    assert table[-1] == table[-2] == np.float32(20.0)
    assert count_saturation_is_exact(table, 127)


def test_unclipped_table_is_not_exact():
    table = _chain_table(0.1, None, 64)
    assert table[-1] > table[-2]
    assert not count_saturation_is_exact(table, 127)


def test_a_long_table_clipped_by_the_cap_is_exact():
    table = _chain_table(0.1, 20.0, 200)          # index 200 > 127, clipped from ~32
    assert table[-1] == table[-2] == table[127]
    assert count_saturation_is_exact(table, 127)


def test_a_long_table_clipped_only_past_the_cap_is_not_exact():
    table = _chain_table(0.01, 20.0, 400)         # (1.01)^c reaches 20 near c = 301
    assert table[-1] == table[-2]
    assert table[127] < table[-1]
    assert not count_saturation_is_exact(table, 127)


@pytest.mark.parametrize("table", [np.ones(1, dtype=np.float32), np.ones((2, 2))])
def test_degenerate_tables_are_not_exact(table):
    assert not count_saturation_is_exact(table, 127)


def test_a_clip_that_binds_exactly_at_the_last_entry_counts_as_saturated():
    # beta 0.1, w_max 1.1**3 -> clipped from count 3 on; table to count 4
    table = _chain_table(0.1, float(np.float32(1.1) ** 3), 4)
    assert table[3] == table[4]
    assert count_saturation_is_exact(table, 127)
