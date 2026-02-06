# neural_computation.py

"""
Core neural computation algorithms for neural assembly simulations.

This module will contain the fundamental neural computation algorithms
extracted from the root brain.py projection logic.
"""

import numpy as np

class NeuralComputationEngine:
    """
    Engine for core neural computation algorithms.
    """
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    
    def compute_inputs(self, connectome: np.ndarray, winners: np.ndarray) -> np.ndarray:
        """Compute inputs to target neurons from active pre-synaptic neurons."""
        return connectome[winners].sum(axis=0)
    
    def select_winners(self, inputs: np.ndarray, k: int, method: str = "top_k") -> np.ndarray:
        """Select k winners based on input strengths."""
        if method == "top_k":
            return np.argsort(-inputs)[:k]
        elif method == "heapq":
            return self._heapq_select_top_k(inputs, k)
        else:
            raise ValueError(f"Unknown winner selection method: {method}")
    
    def _heapq_select_top_k(self, inputs: np.ndarray, k: int) -> np.ndarray:
        """Select top-k using heap for efficiency."""
        import heapq
        if k >= len(inputs):
            return np.arange(len(inputs))
        return np.array(heapq.nlargest(k, range(len(inputs)), inputs.take))
    
    def update_plasticity(self, connectome: np.ndarray, pre_winners: np.ndarray, 
                         post_winners: np.ndarray, beta: float) -> None:
        """Update synaptic weights based on Hebbian plasticity."""
        for pre in pre_winners:
            connectome[pre, post_winners] *= (1 + beta)
