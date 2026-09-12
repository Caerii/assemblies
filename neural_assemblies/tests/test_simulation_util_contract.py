"""Simulation utility boundaries fail explicitly on undefined ratios."""

import pytest

from neural_assemblies.simulation._util import (
    get_overlaps,
    intersection_count,
    overlap,
    reference_fraction,
)


def test_named_simulation_overlap_operations_preserve_legacy_wrapper():
    a, b = [1, 2, 2], [2, 3]
    assert intersection_count(a, b) == overlap(a, b) == 1
    assert reference_fraction(a, b) == overlap(a, b, percentage=True) == 0.5


def test_percentage_overlap_rejects_empty_base():
    with pytest.raises(ValueError, match="non-empty base"):
        get_overlaps([[], [1, 2]], 0, percentage=True)


def test_count_overlap_allows_empty_base():
    assert get_overlaps([[], [1, 2]], 0) == [0, 0]


@pytest.mark.parametrize("base", [-1, 2, True, "0"])
def test_invalid_base_index_fails_explicitly(base):
    with pytest.raises(ValueError, match="base"):
        get_overlaps([[1]], base)


@pytest.mark.parametrize("percentage", [1, 0, "yes", None])
def test_percentage_option_must_be_boolean(percentage):
    with pytest.raises(ValueError, match="percentage must be boolean"):
        get_overlaps([[1]], 0, percentage=percentage)
