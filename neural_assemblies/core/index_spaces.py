"""The two neuron index spaces, as types the checker can tell apart.

THE DEFECT THIS EXISTS TO MAKE UNSPELLABLE. Two different quantities in this
codebase are both called ``winners``:

  * ``Area.winners``     -- COMPACT ENGINE INDICES, ``0..w-1``, dense over the
                            neurons materialized so far.
  * ``Assembly.winners`` -- STABLE NEURON IDS in ``0..n-1``, produced by mapping
                            compact indices through ``compact_to_neuron_id``.

Both retain NumPy array behavior and ``uint32`` storage, but their branded
runtime types and static annotations distinguish the spaces at public
boundaries. Mixing unbranded arrays remains possible, so callers must annotate
or convert raw values explicitly.
(the conclusion "merge recall does not hold" was withdrawn only after the index
spaces were found to differ). The spaces were measured to be COMPLETELY
DISJOINT, so a cross-space comparison is not merely noisy -- it is meaningless.

WHY TYPES AND NOT MORE DOCUMENTATION. The hazard was already documented on
`Assembly`, and a `neuron_ids` alias already existed, and the bug still
happened. Prose cannot fail a build. These branded ndarray subclasses preserve
NumPy behavior and make the mistake both a checker error and a runtime error
at boundaries that can observe the brand.

HOW TO WRITE A FUNCTION OVER EITHER SPACE. Use explicit overloads for each
legal pair::

    @overload
    def overlap(a: CompactIdx, b: CompactIdx) -> float: ...
    @overload
    def overlap(a: NeuronIds, b: NeuronIds) -> float: ...

A union parameter would wrongly accept one of each, and `np.ndarray` on both
accepts everything, which is where we started.

CONVERTING. `to_neuron_ids` is the one direction that is ever correct. There is
deliberately no `to_compact`: compact indices are engine-internal and change
whenever an area grows, so a stored one is a bug waiting to be dereferenced.
"""
from typing import List

import numpy as np

class _BrandedIndices(np.ndarray):
    """Zero-copy runtime brand for one semantic index space."""

    def __new__(cls, values):
        # CuPy arrays cannot be passed through ``np.asarray`` without an
        # explicit host transfer. Preserve them on their native backend; the
        # runtime brand is available for NumPy values, where ndarray subclassing
        # is zero-copy.
        if hasattr(values, "__cuda_array_interface__"):
            return values
        return np.asarray(values).view(cls)

    def __array_finalize__(self, _obj):
        pass


class CompactIdx(_BrandedIndices):
    """Engine-internal compact positions; unstable across materialization."""


class NeuronIds(_BrandedIndices):
    """Stable neuron identities in the area population index space."""

def validated_indices(values, *, upper: int | None = None, label: str = 'indices', xp=np, unique: bool = False) -> np.ndarray:
    """Validate before uint32 conversion on the supplied array backend.

    Never truncate floats or wrap negatives. NumPy is the default; an Area uses
    its own array module so validation need not copy a device array to the CPU.
    """
    arr = xp.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f'{label} must be a one-dimensional index array')
    if arr.size and arr.dtype.kind not in 'iu':
        raise ValueError(f'{label} must contain integer indices')
    limit = 2 ** 32 if upper is None else min(upper, 2 ** 32)
    if arr.size and (xp.any(arr < 0) or xp.any(arr >= limit)):
        raise ValueError(f'{label} outside valid range [0, {limit})')
    if unique and xp.unique(arr).size != arr.size:
        raise ValueError(f'{label} must not contain duplicates')
    result = arr.astype(xp.uint32, copy=False)
    if isinstance(values, _BrandedIndices):
        result = result.view(type(values))
    return result


def to_neuron_ids(
    compact: CompactIdx,
    compact_to_neuron_id: List[int],
) -> NeuronIds:
    """Map engine-compact indices to stable neuron IDs.

    An EMPTY ``compact_to_neuron_id`` means the area is explicit -- its index
    already IS the neuron id -- so the input passes through unchanged. That is
    why this cannot simply index the list: on an explicit area there is nothing
    to index into, and raising there would break every explicit-engine caller.

    Invalid indices raise. Dropping or passing them through would silently change
    assembly membership and turn an invalid readout into a plausible number.
    """
    if isinstance(compact, NeuronIds):
        raise TypeError(
            "to_neuron_ids requires CompactIdx; neuron IDs are already in "
            "the stable space"
        )
    arr = validated_indices(compact, label='compact indices')
    if len(compact_to_neuron_id) == 0:
        return NeuronIds(arr)
    table = validated_indices(compact_to_neuron_id, label='neuron ID mapping')
    arr = validated_indices(arr, upper=len(table), label='compact indices')
    return NeuronIds(table[arr])


def reserve_initial_neuron_ids(pool, selected, *, n: int) -> np.ndarray:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-initial-recruitment

    Return selected IDs followed by the remaining permutation in its old order.
    This is initialization, not a reset of an already recruited population.
    """
    chosen = validated_indices(selected, upper=n, label='initial neuron IDs', unique=True)
    pending = np.arange(n, dtype=np.uint32) if pool is None else validated_indices(
        pool, upper=n, label='recruitment pool', unique=True)
    if len(pending) != n:
        raise ValueError('initial recruitment pool must cover the complete population')
    return np.concatenate((chosen, pending[~np.isin(pending, chosen)]))
