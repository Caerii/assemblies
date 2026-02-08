# neural_computation.py

"""
Core neural computation algorithms for neural assembly simulations.

This module will contain the fundamental neural computation algorithms
extracted from the root brain.py projection logic.
"""

import numpy as np

try:
    from ..core.backend import get_xp
except ImportError:
    from core.backend import get_xp


class NeuralComputationEngine:
    """
    Engine for core neural computation algorithms.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def compute_inputs(self, connectome, winners):
        """Compute inputs to target neurons from active pre-synaptic neurons."""
        return connectome[winners].sum(axis=0)

    def select_winners(self, inputs, k: int, method: str = "top_k"):
        """Select k winners based on input strengths."""
        xp = get_xp()
        inputs = xp.asarray(inputs)
        if method == "top_k":
            if k >= len(inputs):
                return xp.arange(len(inputs))
            part_idx = xp.argpartition(-inputs, k)[:k]
            return part_idx[xp.argsort(-inputs[part_idx])]
        elif method == "heapq":
            return self._heapq_select_top_k(inputs, k)
        else:
            raise ValueError(f"Unknown winner selection method: {method}")

    def _heapq_select_top_k(self, inputs, k: int):
        """Select top-k using argpartition (O(n) average, GPU-native)."""
        xp = get_xp()
        inputs = xp.asarray(inputs)
        if k >= len(inputs):
            return xp.arange(len(inputs))
        part_idx = xp.argpartition(-inputs, k)[:k]
        return part_idx[xp.argsort(-inputs[part_idx])]

    def update_plasticity(self, connectome, pre_winners, post_winners, beta: float) -> None:
        """Update synaptic weights based on Hebbian plasticity."""
        xp = get_xp()
        pre_winners = xp.asarray(pre_winners)
        post_winners = xp.asarray(post_winners)
        if len(pre_winners) > 0 and len(post_winners) > 0:
            ix = xp.ix_(pre_winners, post_winners)
            connectome[ix] *= (1 + beta)
