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


def to_neuron_ids(
    compact: CompactIdx,
    compact_to_neuron_id: List[int],
) -> NeuronIds:
    """Map engine-compact indices to stable neuron IDs.

    An EMPTY ``compact_to_neuron_id`` means the area is explicit -- its index
    already IS the neuron id -- so the input passes through unchanged. That is
    why this cannot simply index the list: on an explicit area there is nothing
    to index into, and raising there would break every explicit-engine caller.

    Out-of-range compact indices are dropped rather than clamped. Clamping would
    invent membership in a neuron that never fired, which is precisely the
    silent-plausible-number failure this module exists to prevent.
    """
    arr = np.asarray(compact, dtype=np.uint32)
    if not compact_to_neuron_id:
        return NeuronIds(arr)
    table = np.asarray(compact_to_neuron_id, dtype=np.uint32)
    in_range = arr < len(table)
    return NeuronIds(table[arr[in_range]])


def same_space(a: np.ndarray, b: np.ndarray) -> bool:
    """Best-effort RUNTIME companion to the static check, for probe code.

    Returns False only when the two arrays provably cannot be in the same
    space. It is a smoke alarm, not a proof: equal-looking ranges do not
    establish sameness, so a True here means "not obviously wrong", nothing
    more. Prefer the types; reach for this only where a value crosses a
    dynamically-typed boundary (a dict, JSON, a saved golden).
    """
    if a.size == 0 or b.size == 0:
        return True
    return not (int(a.max()) < b.min() or int(b.max()) < a.min())
