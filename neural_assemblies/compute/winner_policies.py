"""
Winner policy definitions for neural assembly selection.

These policy objects provide a stable API for experimenting with
different competition rules without forcing engine code to hard-code
one inhibition mechanism. The existing engines still default to fixed
top-k selection; this module establishes the contract for future
pluggable winner policies such as thresholded or E%-style rules.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TopKPolicy:
    """Select exactly ``k`` highest-scoring winners."""

    k: int
    tie_policy: str = "value_then_index"


@dataclass(frozen=True)
class ThresholdPolicy:
    """Select winners above an absolute threshold, capped at ``k``."""

    k: int
    threshold: float
    tie_policy: str = "value_then_index"


@dataclass(frozen=True)
class RelativeThresholdPolicy:
    """Select winners whose score is a fraction of the max input.

    This is a lightweight foundation for variable-sized competition:
    the winner count depends on the input distribution rather than a
    fixed ``k`` alone. ``min_winners`` prevents empty selections when
    at least one candidate exists, and ``max_winners`` can cap growth.
    """

    fraction_of_max: float
    min_winners: int = 1
    max_winners: int | None = None
    tie_policy: str = "value_then_index"


WinnerPolicy = TopKPolicy | ThresholdPolicy | RelativeThresholdPolicy
