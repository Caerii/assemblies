"""Winner-set overlap helpers for the simulation modules.

These two functions came from the pre-package ``brain_util.py`` (now
``legacy/root_modules/brain_util.py``); the simulation modules imported
that root shim, which meant package code depended on the archive. They
live here now, with the Python 2 ``xrange`` fixed.
"""
from __future__ import annotations

import pickle
from collections.abc import Collection, Hashable
from typing import Sequence


def sim_save(file_name, obj):
    """Pickle ``obj`` (a Brain, a list of saved winners, ...) to ``file_name``."""
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


def sim_load(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def intersection_count(a: Collection[Hashable], b: Collection[Hashable]) -> int:
    """Number of distinct shared items in two winner collections."""
    return len(set(a) & set(b))


def reference_fraction(
    observed: Collection[Hashable], reference: Collection[Hashable]
) -> float:
    """Shared-item fraction relative to the second (reference) collection."""
    if not len(reference):
        raise ValueError("reference fraction requires a non-empty reference collection")
    return float(intersection_count(observed, reference)) / float(len(reference))


def overlap(
    a: Collection[Hashable], b: Collection[Hashable], percentage: bool = False
) -> int | float:
    """Legacy wrapper for ``intersection_count`` or ``reference_fraction``."""
    if type(percentage) is not bool:
        raise ValueError("percentage must be boolean")
    return reference_fraction(a, b) if percentage else intersection_count(a, b)


def get_overlaps(
    winners_list: Sequence[Collection[Hashable]],
    base: int,
    percentage: bool = False,
) -> list[int | float]:
    """Overlap of every winner list in ``winners_list`` with
    ``winners_list[base]``."""
    if type(percentage) is not bool:
        raise ValueError("percentage must be boolean")
    if isinstance(base, bool) or not isinstance(base, int):
        raise ValueError("base must be an integer winner-list index")
    if base < 0 or base >= len(winners_list):
        raise ValueError(
            f"base index {base} is outside winner-list range [0, {len(winners_list)})"
        )
    base_winners = winners_list[base]
    k = len(base_winners)
    if percentage and k == 0:
        raise ValueError("percentage overlap requires a non-empty base winner set")
    out = []
    for w in winners_list:
        out.append(
            reference_fraction(w, base_winners)
            if percentage else intersection_count(w, base_winners)
        )
    return out
