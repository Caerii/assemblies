"""
Assembly handle and overlap measurement.

An Assembly is a lightweight, immutable snapshot of a neural assembly —
a set of k neurons in a specific brain area at a specific moment in time.

WHY A SNAPSHOT RATHER THAN A HANDLE.  In the theory an assembly is a
persistent object: "the assembly for dog" survives across operations.  In the
simulation there is no such object -- an area has exactly one live winner set,
and the next projection overwrites it.  Anything an operation returns must
therefore be a copy taken at an instant, and the immutability here is what
enforces that reading: the array is copied on construction and marked
non-writeable, so a snapshot cannot be silently aliased to a live winner
array or edited after the fact.

What persists across operations is not this object but the CONNECTOME.  The
assembly is recoverable because the synapses that select those neurons were
potentiated; the snapshot is a record of a recovery, not the thing recovered.
That is why two snapshots of "the same" assembly taken at different times can
differ slightly and still be the same assembly in the theory's sense -- and
why overlap, not equality, is the working comparison throughout this package.

Note that ``winners`` holds STABLE NEURON IDs, not the engine's compact
indices; ``ops._snap`` is the only correct way to build one from a live area,
and it documents why.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Assembly:
    """A snapshot of a neural assembly in a brain area.

    THE INDEX SPACE IS NOT THE SAME AS ``Area.winners``.  There are two, and
    they are both called "winners" in this codebase:

      * ``Area.winners`` -- COMPACT ENGINE INDICES, ``0..w-1``, dense over the
        neurons that have been materialized so far;
      * ``Assembly.winners`` (this) -- NEURON IDS in ``0..n-1``, produced by
        ``_snap`` mapping through ``compact_to_neuron_id``.

    Comparing one to the other is silently accepted, returns a number, and
    reads as chance -- it voided a merge-recall result once. Prefer the alias
    ``neuron_ids`` in new code so the space is stated at the call site.

    Attributes:
        area: Name of the brain area this assembly lives in.
        winners: Neuron IDs (uint32) forming the assembly. See ``neuron_ids``.
    """

    area: str
    winners: np.ndarray

    @property
    def neuron_ids(self) -> np.ndarray:
        """``winners``, named for the index space it is actually in.

        Same array, no copy. Exists so a reader does not have to know which of
        the two "winners" they are holding -- see
        [[two-index-spaces-compact-vs-neuron-id]].
        """
        return self.winners

    def __post_init__(self):
        # Store an immutable copy so the snapshot can't be mutated
        # through the original array. frozen=True prevents attribute
        # reassignment but ndarray contents are still mutable, so we
        # copy on construction.
        object.__setattr__(
            self, "winners", np.array(self.winners, dtype=np.uint32, copy=True)
        )
        self.winners.flags.writeable = False

    def overlap(self, other: "Assembly") -> float:
        """Fraction of shared neurons: |A ∩ B| / min(|A|, |B|).

        Works for assemblies in the same area or different areas
        (cross-area overlap is meaningful when neuron ID spaces overlap).
        """
        return overlap(self.winners, other.winners)

    @classmethod
    def from_area(cls, brain, area_name: str) -> "Assembly":
        """Canonical snapshot of the live assembly in *area_name*.

        Always routes through ``_snap`` so sparse compact indices map to
        real neuron IDs and explicit areas are left unmapped.
        """
        from neural_assemblies.assembly_calculus.ops import _snap

        return _snap(brain, area_name)

    def __len__(self) -> int:
        return len(self.winners)

    def __repr__(self) -> str:
        return f"Assembly(area={self.area!r}, size={len(self)})"

    def __eq__(self, other):
        if not isinstance(other, Assembly):
            return NotImplemented
        return self.area == other.area and np.array_equal(self.winners, other.winners)

    def __hash__(self):
        return hash((self.area, tuple(self.winners)))


def overlap(a, b) -> float:
    """Overlap ratio between two winner arrays or Assemblies.

    Returns |A ∩ B| / min(|A|, |B|), or 0.0 if either is empty.

    THE NORMALISATION IS A CHOICE, and it is the min rather than the union
    (Jaccard) for a specific reason: assemblies here are not always the same
    size.  E%-WTA produces variable-size assemblies, sparse areas recruit
    neurons over time, and a stored lexicon entry is routinely smaller than
    the live activity it is being matched against.  Min-normalisation asks
    "is the smaller one contained in the larger?", which is the right question
    for recognition -- a full assembly that has additionally recruited noise
    still counts as recognised.

    The consequence to keep in mind is that a strict SUBSET scores 1.0.  That
    is intended for readout, but it makes this function unsuitable for asking
    "did the winner set change?", where growth must count as change.  Use the
    Jaccard measures in ``metrics.instability`` for that; the two disagree
    exactly on the growth case.

    Baseline: two unrelated k-subsets of n neurons overlap by about ``k/n``,
    not 0 -- see :func:`chance_overlap`.  Compare against that, not against
    zero.

    Args:
        a: numpy array of neuron indices, list, or Assembly.
        b: numpy array of neuron indices, list, or Assembly.
    """
    winners_a = a.winners if isinstance(a, Assembly) else np.asarray(a)
    winners_b = b.winners if isinstance(b, Assembly) else np.asarray(b)

    if len(winners_a) == 0 or len(winners_b) == 0:
        return 0.0

    set_a = set(winners_a.tolist())
    set_b = set(winners_b.tolist())
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))

    return intersection / min_size if min_size > 0 else 0.0


def chance_overlap(k: int, n: int) -> float:
    """Expected overlap between two random k-subsets of [n].

    If A and B are independent uniform random k-subsets of {0, ..., n-1},
    then E[|A ∩ B|] / k = k / n  (hypergeometric mean / k).
    """
    return k / n


def overlap_from_binary(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Normalized overlap of two binary ``n``-vectors with support size *k*.

    Returns ``dot(a, b) / k``.  For equal-size assemblies this matches
    :func:`overlap` on :meth:`Assembly.from_area` snapshots.
    """
    return float(np.dot(a, b)) / max(int(k), 1)
