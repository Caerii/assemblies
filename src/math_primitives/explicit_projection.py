# explicit_projection.py

"""
Utilities for explicit-area projection behavior extracted from brain.py.

This module provides helpers for:
- Validating source winners vs connectome shapes
- Accumulating previous winner inputs for explicit targets
- Applying area->area plasticity scaling
"""

import numpy as np
from typing import List


class ExplicitProjectionEngine:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def validate_source_winners(self, connectome: np.ndarray, source_winners: np.ndarray) -> None:
        """
        Ensure that the maximum winner index does not exceed the source rows of the connectome.
        Mirrors brain.py index-bounds checks for explicit projection.
        """
        if source_winners.size == 0:
            raise ValueError("Projecting from area with no assembly")
        max_idx = int(np.max(source_winners))
        if max_idx >= connectome.shape[0]:
            raise IndexError(
                f"Winner index {max_idx} exceeds connectome source size {connectome.shape[0]}"
            )

    def accumulate_prev_inputs_explicit(
        self,
        target_n: int,
        stimuli_vectors: List[np.ndarray],
        area_connectomes_and_winners: List[tuple],
    ) -> np.ndarray:
        """
        Sum inputs from stimuli and area winners into a vector sized target_n.
        Matches brain.py behavior for explicit targets (prev_winner_inputs).
        """
        prev_winner_inputs = np.zeros(target_n, dtype=np.float32)
        for vec in stimuli_vectors:
            prev_winner_inputs += vec.astype(np.float32, copy=False)
        for connectome, winners in area_connectomes_and_winners:
            for w in winners:
                prev_winner_inputs += connectome[int(w)]
        return prev_winner_inputs

    def apply_area_to_area_plasticity(
        self,
        connectome: np.ndarray,
        pre_winner_rows: np.ndarray,
        post_winner_cols: np.ndarray,
        beta: float,
        disable: bool = False,
    ) -> None:
        """
        In-place plasticity scaling for explicit area connectome: connectome[j, i] *= 1+beta.
        """
        if disable or beta == 0.0:
            return
        factor = 1.0 + beta
        for i in post_winner_cols:
            ii = int(i)
            connectome[pre_winner_rows, ii] *= factor


# Standalone functions for direct use in brain.py
def validate_source_winners(from_areas, connectomes, area_by_name, target_area_name):
    """Validate source winners against connectome bounds."""
    for from_area_name in from_areas:
        connectome = connectomes[from_area_name][target_area_name]
        from_area = area_by_name[from_area_name]
        if from_area.winners.size == 0:
            raise ValueError(f"Projecting from area with no assembly: {from_area}")
        max_idx = int(np.max(from_area.winners))
        if max_idx >= connectome.shape[0]:
            raise IndexError(
                f"Winner index {max_idx} exceeds connectome source size {connectome.shape[0]} "
                f"for connection {from_area_name} -> {target_area_name}."
            )


def accumulate_prev_inputs_explicit(
    target_n, from_stimuli, from_areas, connectomes_by_stimulus, 
    connectomes, area_by_name, target_area_name
):
    """Accumulate previous winner inputs for explicit areas."""
    prev_winner_inputs = np.zeros(target_n, dtype=np.float32)
    
    for stim in from_stimuli:
        stim_inputs = connectomes_by_stimulus[stim][target_area_name]
        prev_winner_inputs += stim_inputs
    
    for from_area_name in from_areas:
        connectome = connectomes[from_area_name][target_area_name]
        for w in area_by_name[from_area_name].winners:
            prev_winner_inputs += connectome[w]
    
    return prev_winner_inputs


def apply_area_to_area_plasticity(
    new_winners, from_areas, connectomes, area_by_name, 
    target_area_name, disable_plasticity
):
    """Apply area-to-area plasticity for explicit areas."""
    for from_area_name in from_areas:
        from_area_winners = area_by_name[from_area_name].winners
        the_connectome = connectomes[from_area_name][target_area_name]
        
        area_to_area_beta = (
            0 if disable_plasticity
            else area_by_name[target_area_name].beta_by_area[from_area_name]
        )
        
        for i in new_winners:
            for j in from_area_winners:
                the_connectome[j, i] *= 1.0 + area_to_area_beta


