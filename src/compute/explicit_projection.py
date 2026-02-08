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

try:
    from ..core.backend import get_xp
except ImportError:
    from core.backend import get_xp


class ExplicitProjectionEngine:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def validate_source_winners(self, connectome, source_winners) -> None:
        """
        Ensure that the maximum winner index does not exceed the source rows of the connectome.
        """
        xp = get_xp()
        if source_winners.size == 0:
            raise ValueError("Projecting from area with no assembly")
        max_idx = int(xp.max(source_winners))
        if max_idx >= connectome.shape[0]:
            raise IndexError(
                f"Winner index {max_idx} exceeds connectome source size {connectome.shape[0]}"
            )

    def accumulate_prev_inputs_explicit(
        self,
        target_n: int,
        stimuli_vectors: List,
        area_connectomes_and_winners: List[tuple],
    ):
        """
        Sum inputs from stimuli and area winners into a vector sized target_n.
        """
        xp = get_xp()
        prev_winner_inputs = xp.zeros(target_n, dtype=xp.float32)
        for vec in stimuli_vectors:
            prev_winner_inputs += vec.astype(xp.float32, copy=False)
        for connectome, winners in area_connectomes_and_winners:
            valid = winners[winners < connectome.shape[0]]
            if len(valid) > 0:
                prev_winner_inputs += connectome[valid].sum(axis=0)
        return prev_winner_inputs

    def apply_area_to_area_plasticity(
        self,
        connectome,
        pre_winner_rows,
        post_winner_cols,
        beta: float,
        disable: bool = False,
    ) -> None:
        """
        In-place plasticity scaling for explicit area connectome.
        """
        xp = get_xp()
        if disable or beta == 0.0:
            return
        factor = 1.0 + beta
        pre_winner_rows = xp.asarray(pre_winner_rows)
        post_winner_cols = xp.asarray(post_winner_cols)
        if len(pre_winner_rows) > 0 and len(post_winner_cols) > 0:
            ix = xp.ix_(pre_winner_rows, post_winner_cols)
            connectome[ix] *= factor


# Standalone functions for direct use in brain.py
def validate_source_winners(from_areas, connectomes, area_by_name, target_area_name):
    """Validate source winners against connectome bounds."""
    xp = get_xp()
    for from_area_name in from_areas:
        connectome = connectomes[from_area_name][target_area_name]
        from_area = area_by_name[from_area_name]
        if from_area.winners.size == 0:
            raise ValueError(f"Projecting from area with no assembly: {from_area}")
        max_idx = int(xp.max(from_area.winners))
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
    xp = get_xp()
    prev_winner_inputs = xp.zeros(target_n, dtype=xp.float32)

    for stim in from_stimuli:
        stim_inputs = connectomes_by_stimulus[stim][target_area_name]
        prev_winner_inputs += stim_inputs

    for from_area_name in from_areas:
        connectome = connectomes[from_area_name][target_area_name]
        winners = area_by_name[from_area_name].winners
        valid = winners[winners < connectome.shape[0]]
        if len(valid) > 0:
            prev_winner_inputs += connectome[valid].sum(axis=0)

    return prev_winner_inputs


def apply_area_to_area_plasticity(
    new_winners, from_areas, connectomes, area_by_name,
    target_area_name, disable_plasticity
):
    """Apply area-to-area plasticity for explicit areas."""
    xp = get_xp()
    for from_area_name in from_areas:
        from_area_winners = area_by_name[from_area_name].winners
        the_connectome = connectomes[from_area_name][target_area_name]

        area_to_area_beta = (
            0 if disable_plasticity
            else area_by_name[target_area_name].beta_by_area[from_area_name]
        )

        if area_to_area_beta == 0:
            continue
        factor = 1.0 + area_to_area_beta
        from_arr = xp.asarray(from_area_winners)
        to_arr = xp.asarray(new_winners)
        if len(from_arr) > 0 and len(to_arr) > 0:
            ix = xp.ix_(from_arr, to_arr)
            the_connectome[ix] *= factor
