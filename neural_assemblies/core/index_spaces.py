"""The two neuron index spaces, as types the checker can tell apart.

THE DEFECT THIS EXISTS TO MAKE UNSPELLABLE. Two different quantities in this
codebase are both called ``winners``:

  * ``Area.winners``     -- COMPACT ENGINE INDICES, ``0..w-1``, dense over the
                            neurons materialized so far.
  * ``Assembly.winners`` -- STABLE NEURON IDS in ``0..n-1``, produced by mapping
                            compact indices through ``compact_to_neuron_id``.

Both are ``np.ndarray`` of ``uint32``, so mixing them is silently accepted,
returns a number, and reads as chance. It voided a merge-recall result once
(the conclusion "merge recall does not hold" was withdrawn only after the index
spaces were found to differ). The spaces were measured to be COMPLETELY
DISJOINT, so a cross-space comparison is not merely noisy -- it is meaningless.

WHY TYPES AND NOT MORE DOCUMENTATION. The hazard was already documented on
`Assembly`, and a `neuron_ids` alias already existed, and the bug still
happened. Prose cannot fail a build. These NewTypes cost nothing at runtime
(`NeuronIds(x) is x`) and make the mistake a checker error.

HOW TO WRITE A FUNCTION OVER EITHER SPACE. Use `SameSpace`, not a union::

    def overlap(a: SameSpace, b: SameSpace) -> float: ...

A value-restricted TypeVar binds to ONE member per call, so `overlap` accepts
two NeuronIds or two CompactIdx and REJECTS one of each -- which is the actual
rule. A union parameter would wrongly accept the mixed call, and `np.ndarray`
on both accepts everything, which is where we started.

CONVERTING. `to_neuron_ids` is the one direction that is ever correct. There is
deliberately no `to_compact`: compact indices are engine-internal and change
whenever an area grows, so a stored one is a bug waiting to be dereferenced.
"""
from typing import List, NewType, TypeVar

import numpy as np

#: Engine-internal positions, ``0..w-1``. NOT stable: they are reassigned as an
#: area materializes more neurons, so never persist one across a projection.
CompactIdx = NewType("CompactIdx", np.ndarray)

#: Stable identities, ``0..n-1``. Safe to store, compare across time, and
#: compare across areas (their ID spaces genuinely overlap).
NeuronIds = NewType("NeuronIds", np.ndarray)

#: Binds to ONE of the two per call site. Use for functions valid within either
#: space but never across them -- see the module docstring.
SameSpace = TypeVar("SameSpace", CompactIdx, NeuronIds)


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
    return arr.astype(xp.uint32, copy=False)


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
