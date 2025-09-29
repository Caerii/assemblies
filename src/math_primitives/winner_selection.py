# winner_selection.py

"""
Winner selection algorithms for neural assembly simulations.

Implements the top-k selection used in the root projection logic,
including first-time winner remapping and optional inhibition masks.
"""

import numpy as np
import heapq
from typing import List, Optional, Tuple
from .utils import validate_finite, normalize_index_list

class WinnerSelector:
    """
    Handles winner selection algorithms for neural assemblies.
    """
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    
    def select_top_k_indices(self, features: np.ndarray, k: int) -> np.ndarray:
        """Select indices of top-k values using argsort."""
        return np.argsort(-features)[:k]
    
    def heapq_select_top_k(self, features: np.ndarray, k: int) -> np.ndarray:
        """Select indices of top-k values using heap for efficiency."""
        if k >= len(features):
            return np.arange(len(features))
        return np.array(heapq.nlargest(k, range(len(features)), features.take))
    
    def select_winners_with_threshold(self, inputs: np.ndarray, k: int, 
                                    threshold: float = None) -> np.ndarray:
        """Select winners above a threshold, up to k winners."""
        if threshold is None:
            return self.select_top_k_indices(inputs, k)
        
        above_threshold = inputs >= threshold
        if np.sum(above_threshold) <= k:
            return np.where(above_threshold)[0]
        else:
            # Select top k from those above threshold
            above_threshold_indices = np.where(above_threshold)[0]
            above_threshold_inputs = inputs[above_threshold_indices]
            top_k_indices = np.argsort(-above_threshold_inputs)[:k]
            return above_threshold_indices[top_k_indices]

    def select_combined_winners(self,
                                all_inputs: np.ndarray,
                                target_area_w: int,
                                target_area_k: int,
                                method: str = "heapq",
                                mask_inhibit: Optional[np.ndarray] = None,
                                tie_policy: str = "value_then_index") -> Tuple[List[int], List[float], int, List[int]]:
        """
        Select k winners from combined previous winners (0..w-1) and potentials (w..),
        then remap first-time winners to contiguous indices and return their inputs.

        Args:
            all_inputs: 1D array of inputs of length w + k_potential
            target_area_w: number of ever-fired neurons (previous winners length)
            target_area_k: number of winners to select
            method: 'heapq' or 'argsort'
            mask_inhibit: optional boolean mask same length as all_inputs; True = inhibit
            tie_policy: 'value_then_index' ensures deterministic ordering on ties

        Returns:
            (new_winner_indices, first_winner_inputs, num_first)
        """
        if all_inputs.ndim != 1:
            raise ValueError("all_inputs must be 1D")
        validate_finite(all_inputs, "all_inputs")
        n = len(all_inputs)
        if mask_inhibit is not None:
            if mask_inhibit.shape != all_inputs.shape:
                raise ValueError("mask_inhibit must be same shape as all_inputs")
            # Coerce to boolean mask robustly
            inhibit_mask = mask_inhibit.astype(bool, copy=False)
            # Build candidate set excluding inhibited indices entirely
            candidate_indices = [i for i in range(n) if not inhibit_mask[i]]
            safe_inputs = all_inputs  # no need to mutate values when we prefilter
        else:
            inhibit_mask = None
            candidate_indices = list(range(n))
            safe_inputs = all_inputs

        k = min(target_area_k, n)
        if method == "heapq":
            if tie_policy == "value_then_index":
                # Select by descending value, break ties by smaller index deterministically
                if k >= len(candidate_indices):
                    new_indices = list(candidate_indices)
                else:
                    new_indices = heapq.nsmallest(
                        k, candidate_indices, key=lambda i: (-(safe_inputs[i]), i)
                    )
            else:
                if k >= len(candidate_indices):
                    new_indices = list(candidate_indices)
                else:
                    new_indices = heapq.nlargest(
                        k, candidate_indices, key=lambda i: safe_inputs[i]
                    )
        elif method == "argsort":
            if tie_policy == "value_then_index":
                # stable tie by index: sort by (-value, index)
                order = np.lexsort((np.array(candidate_indices), -safe_inputs[candidate_indices]))
                # order gives positions within candidate_indices; map back to absolute indices
                ordered_abs = [candidate_indices[pos] for pos in order]
                new_indices = list(ordered_abs[:k])
            else:
                order = np.argsort(-safe_inputs[candidate_indices])
                new_indices = [candidate_indices[pos] for pos in order[:k]]
        else:
            raise ValueError("Unknown method")

        # Remap first-time winners and collect their inputs
        first_winner_inputs: List[float] = []
        num_first = 0
        original_indices = list(new_indices)
        for i in range(len(new_indices)):
            if new_indices[i] >= target_area_w:
                first_winner_inputs.append(all_inputs[new_indices[i]])
                new_indices[i] = target_area_w + num_first
                num_first += 1

        return new_indices, first_winner_inputs, num_first, original_indices
