# winner_selection.py

"""
Winner selection algorithms for neural assembly simulations.

Implements the top-k selection used in the root projection logic,
including first-time winner remapping and optional inhibition masks.
"""

import numpy as np
from typing import List, Optional, Tuple
from .utils import validate_finite

try:
    from ..core.backend import get_xp, to_cpu
except ImportError:
    from core.backend import get_xp, to_cpu


class WinnerSelector:
    """
    Handles winner selection algorithms for neural assemblies.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_top_k_indices(self, features, k: int):
        """Select indices of top-k values using argpartition (O(n) average)."""
        xp = get_xp()
        features = xp.asarray(features)
        if k >= len(features):
            return xp.arange(len(features))
        part_idx = xp.argpartition(-features, k)[:k]
        sorted_order = xp.argsort(-features[part_idx])
        return part_idx[sorted_order]

    def heapq_select_top_k(self, features, k: int):
        """Select indices of top-k values using argpartition (O(n) average, GPU-native)."""
        xp = get_xp()
        features = xp.asarray(features)
        if k >= len(features):
            return xp.arange(len(features))
        part_idx = xp.argpartition(-features, k)[:k]
        sorted_order = xp.argsort(-features[part_idx])
        return part_idx[sorted_order]

    def select_winners_with_threshold(self, inputs, k: int,
                                    threshold: float = None):
        """Select winners above a threshold, up to k winners."""
        xp = get_xp()
        inputs = xp.asarray(inputs)
        if threshold is None:
            return self.select_top_k_indices(inputs, k)

        above_threshold = inputs >= threshold
        if int(xp.sum(above_threshold)) <= k:
            return xp.where(above_threshold)[0]
        else:
            # Select top k from those above threshold
            above_threshold_indices = xp.where(above_threshold)[0]
            above_threshold_inputs = inputs[above_threshold_indices]
            top_k_indices = xp.argsort(-above_threshold_inputs)[:k]
            return above_threshold_indices[top_k_indices]

    def select_combined_winners(self,
                                all_inputs,
                                target_area_w: int,
                                target_area_k: int,
                                method: str = "heapq",
                                mask_inhibit=None,
                                tie_policy: str = "value_then_index") -> Tuple[List[int], List[float], int, List[int]]:
        """
        Select k winners from combined previous winners (0..w-1) and potentials (w..),
        then remap first-time winners to contiguous indices and return their inputs.

        Args:
            all_inputs: 1D array of inputs of length w + k_potential
            target_area_w: number of ever-fired neurons (previous winners length)
            target_area_k: number of winners to select
            method: 'heapq' or 'argsort' (both use argpartition internally now)
            mask_inhibit: optional boolean mask same length as all_inputs; True = inhibit
            tie_policy: 'value_then_index' ensures deterministic ordering on ties

        Returns:
            (new_winner_indices, first_winner_inputs, num_first, original_indices)
        """
        # Move to CPU for tie-breaking logic (small arrays)
        all_inputs_cpu = np.asarray(to_cpu(all_inputs), dtype=np.float64)
        if all_inputs_cpu.ndim != 1:
            raise ValueError("all_inputs must be 1D")
        validate_finite(all_inputs_cpu, "all_inputs")
        n = len(all_inputs_cpu)

        if mask_inhibit is not None:
            mask_cpu = np.asarray(to_cpu(mask_inhibit))
            if mask_cpu.shape != all_inputs_cpu.shape:
                raise ValueError("mask_inhibit must be same shape as all_inputs")
            inhibit_mask = mask_cpu.astype(bool, copy=False)
            candidate_indices = np.where(~inhibit_mask)[0]
        else:
            candidate_indices = np.arange(n)

        k = min(target_area_k, n)

        if tie_policy == "value_then_index":
            if k >= len(candidate_indices):
                new_indices = list(candidate_indices)
            else:
                # lexsort: primary key = -value (descending), secondary = index (ascending)
                cand_vals = all_inputs_cpu[candidate_indices]
                order = np.lexsort((candidate_indices, -cand_vals))
                new_indices = [int(candidate_indices[pos]) for pos in order[:k]]
        else:
            if k >= len(candidate_indices):
                new_indices = list(candidate_indices)
            else:
                cand_vals = all_inputs_cpu[candidate_indices]
                order = np.argsort(-cand_vals)
                new_indices = [int(candidate_indices[pos]) for pos in order[:k]]

        # Remap first-time winners and collect their inputs
        first_winner_inputs: List[float] = []
        num_first = 0
        original_indices = list(new_indices)
        for i in range(len(new_indices)):
            if new_indices[i] >= target_area_w:
                first_winner_inputs.append(float(all_inputs_cpu[new_indices[i]]))
                new_indices[i] = target_area_w + num_first
                num_first += 1

        return new_indices, first_winner_inputs, num_first, original_indices
