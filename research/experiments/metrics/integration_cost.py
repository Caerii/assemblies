"""
Integration Cost Metrics

VP assembly distance and competition margin utilities for P600 analysis.
These are paired metrics computed across conditions (e.g., grammatical
vs. agreement violation) rather than within a single measurement.
"""

from typing import Set


def compute_vp_distance(winners_a: Set[int], winners_b: Set[int]) -> float:
    """Jaccard distance between two VP assemblies.

    Measures how much the VP representation shifts between two conditions
    (e.g., grammatical vs. agreement violation). Higher distance means
    the violation produced a more different VP assembly.

    Args:
        winners_a: VP assembly neurons from condition A.
        winners_b: VP assembly neurons from condition B.

    Returns:
        Jaccard distance (1 - Jaccard similarity), in [0, 1].
        Returns 0.0 if both sets are empty.
    """
    union = winners_a | winners_b
    if len(union) == 0:
        return 0.0
    return 1.0 - len(winners_a & winners_b) / len(union)
