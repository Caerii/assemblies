"""
Sequence -- ordered list of assembly snapshots.

Represents a temporal sequence of assemblies in a single brain area,
as produced by ``sequence_memorize`` or ``ordered_recall``.

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
