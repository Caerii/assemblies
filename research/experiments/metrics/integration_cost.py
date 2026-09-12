"""
Integration Cost Metrics

VP assembly distance and competition margin utilities for P600 analysis.
These are paired metrics computed across conditions (e.g., grammatical
vs. agreement violation) rather than within a single measurement.
"""

from collections.abc import Hashable, Iterable

from neural_assemblies.assembly_calculus.metrics import jaccard_similarity


def compute_vp_distance(
    winners_a: Iterable[Hashable], winners_b: Iterable[Hashable]
) -> float:
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
    return 1.0 - jaccard_similarity(winners_a, winners_b)
