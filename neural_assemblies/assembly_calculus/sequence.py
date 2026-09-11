"""
Sequence -- ordered list of assembly snapshots.

Represents a temporal sequence of assemblies in a single brain area,
as produced by ``sequence_memorize`` or ``ordered_recall``.

WHAT THE OVERLAP HELPERS ARE FOR.  A memorised sequence should satisfy two
conditions that pull in opposite directions, and the two accessors here
measure them separately:

    pairwise_overlaps()  Consecutive items must be DISTINCT.  If adjacent
        assemblies overlap heavily, the area collapsed the sequence into one
        attractor and recall cannot step -- there is nothing to step to.
    overlap_matrix()     Non-adjacent items must also be distinct.  The
        diagnostic failure this catches is a sequence that wraps: item 5
        overlapping item 1 means recall will cycle rather than terminate,
        which is exactly the condition ``ordered_recall`` breaks on.

A healthy sequence therefore shows a near-diagonal overlap matrix, with
off-diagonal values near chance (``k/n``, see ``assembly.chance_overlap``) --
not zero.  Elevated values anywhere off the diagonal are the interference that
the scaffold construction in ``scaffold.py`` exists to reduce.

This object is a record of a measurement, not a live handle: the assemblies it
holds are snapshots, so it stays valid while the brain moves on.

Reference:
    Dabagia, Papadimitriou, Vempala.
    "Computation with Sequences of Assemblies in a Model of the Brain."
    Neural Computation (2025) / ALT 2024.  arXiv:2306.03812.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .assembly import Assembly, overlap


@dataclass(frozen=True)
class Sequence:
    """An ordered sequence of Assembly snapshots in a single area.

    Attributes:
        area: Name of the brain area.
        assemblies: Ordered tuple of Assembly snapshots.
    """
    area: str
    assemblies: tuple  # tuple[Assembly, ...]

    def __post_init__(self):
        if isinstance(self.assemblies, list):
            object.__setattr__(self, 'assemblies', tuple(self.assemblies))
        if not isinstance(self.area, str) or not self.area:
            raise ValueError("sequence area must be a non-empty string")
        for index, assembly in enumerate(self.assemblies):
            if not isinstance(assembly, Assembly):
                raise TypeError(f"sequence item {index} must be an Assembly")
            if assembly.area != self.area:
                raise ValueError(
                    f"sequence item {index} belongs to {assembly.area!r}, "
                    f"not sequence area {self.area!r}"
                )

    def __len__(self) -> int:
        return len(self.assemblies)

    def __getitem__(self, idx):
        return self.assemblies[idx]

    def __iter__(self):
        return iter(self.assemblies)

    def pairwise_overlaps(self) -> List[float]:
        """Overlap between consecutive assemblies: [ovlp(0,1), ovlp(1,2), ...]."""
        return [overlap(self.assemblies[i], self.assemblies[i + 1])
                for i in range(len(self.assemblies) - 1)]

    def overlap_matrix(self) -> np.ndarray:
        """Full pairwise overlap matrix (n x n)."""
        n = len(self.assemblies)
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                mat[i, j] = overlap(self.assemblies[i], self.assemblies[j])
        return mat

    def mean_consecutive_overlap(self) -> float:
        """Mean overlap between adjacent assemblies in the sequence."""
        pw = self.pairwise_overlaps()
        return float(np.mean(pw)) if pw else 0.0
