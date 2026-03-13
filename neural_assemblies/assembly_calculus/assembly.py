"""
Assembly handle and overlap measurement.

An Assembly is a lightweight, immutable snapshot of a neural assembly —
a set of k neurons in a specific brain area at a specific moment in time.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Assembly:
    """A snapshot of a neural assembly in a brain area.

    Attributes:
        area: Name of the brain area this assembly lives in.
        winners: Neuron indices (uint32) forming the assembly.
    """

    area: str
    winners: np.ndarray

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
