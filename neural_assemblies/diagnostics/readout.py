"""The one sanctioned readout of an area's assembly, and the overlap and spread of
assemblies read that way.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations

import itertools
import statistics
from typing import (Iterable)

import numpy as np

from neural_assemblies.core.index_spaces import NeuronIds
from neural_assemblies.core.measurement import Measured


# --------------------------------------------------------------------------
# The one sanctioned readout
# --------------------------------------------------------------------------

def read_assembly(brain, area: str) -> NeuronIds:
    """Current assembly in *area* as STABLE NEURON IDS.

    Always use this rather than ``brain.areas[area].winners``, which is a
    different coordinate system (compact engine indices, renumbered as the area
    recruits). Comparing the two returns chance -- silently, looking exactly
    like a negative result. See failure shape 3 in the module docstring.
    """
    from neural_assemblies.assembly_calculus.ops import _snap
    return NeuronIds(np.asarray(_snap(brain, area).winners, dtype=np.int64))


def assembly_overlap(a: NeuronIds, b: NeuronIds) -> float:
    """Overlap between two assemblies of NEURON IDS. Order-insensitive.

    Both operands are asserted into the neuron-ID space, which is what makes
    this the sanctioned pairing for `read_assembly`. Handing it compact engine
    indices is the defect in `core/index_spaces` -- it will return a plausible
    number that reads as chance -- so read through `read_assembly`, never off
    `area.winners` directly.
    """
    from neural_assemblies.assembly_calculus.assembly import neuron_overlap
    return float(neuron_overlap(NeuronIds(np.asarray(a, dtype=np.int64)),
                                NeuronIds(np.asarray(b, dtype=np.int64))))


def _spread(assemblies: Iterable) -> Measured:
    """Mean pairwise overlap. UNDEFINED with fewer than two assemblies.

    One assembly has no pair to overlap with, so there is no spread -- as
    distinct from a spread of zero, which is what a bare 0.0 here would claim
    and which is the *healthiest* possible reading.
    """
    items = list(assemblies)
    pairs = list(itertools.combinations(items, 2))
    if not pairs:
        return Measured.undefined(
            "fewer than two assemblies, so there is no pair to overlap",
            n_assemblies=len(items))
    return Measured.of(statistics.mean(assembly_overlap(x, y) for x, y in pairs))
