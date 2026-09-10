"""Winner-set overlap helpers for the simulation modules.

These two functions came from the pre-package ``brain_util.py`` (now
``legacy/root_modules/brain_util.py``); the simulation modules imported
that root shim, which meant package code depended on the archive. They
live here now, with the Python 2 ``xrange`` fixed.
"""
from __future__ import annotations

import pickle
from typing import Sequence


def sim_save(file_name, obj):
    """Pickle ``obj`` (a Brain, a list of saved winners, ...) to ``file_name``."""
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


def sim_load(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def overlap(a, b, percentage: bool = False):
    """Item overlap of two winner lists viewed as sets; a fraction of ``b``
    when ``percentage`` is set."""
    o = len(set(a) & set(b))
    return (float(o) / float(len(b))) if percentage else o


def get_overlaps(winners_list: Sequence, base: int, percentage: bool = False):
    """Overlap of every winner list in ``winners_list`` with
    ``winners_list[base]``."""
    base_winners = winners_list[base]
    k = len(base_winners)
    out = []
    for w in winners_list:
        o = overlap(w, base_winners)
        out.append(float(o) / float(k) if percentage else o)
    return out
