# plasticity.py

"""
Synaptic plasticity mechanisms for neural assembly simulations.

This module will contain various plasticity rules and mechanisms
for updating synaptic weights in neural assemblies.
"""

import numpy as np
from typing import List, Tuple
from .utils import validate_finite, validate_finite_scalar, normalize_index_list

class PlasticityEngine:
    """
    Engine for synaptic plasticity mechanisms.
    """
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    
    def hebbian_update(self, weights: np.ndarray, pre_neurons: np.ndarray, 
                      post_neurons: np.ndarray, beta: float) -> None:
        """Apply Hebbian plasticity rule."""
        for pre in pre_neurons:
            weights[pre, post_neurons] *= (1 + beta)
    
    def anti_hebbian_update(self, weights: np.ndarray, pre_neurons: np.ndarray, 
                           post_neurons: np.ndarray, beta: float) -> None:
        """Apply anti-Hebbian plasticity rule."""
        for pre in pre_neurons:
            weights[pre, post_neurons] *= (1 - beta)
    
    def spike_timing_dependent_plasticity(self, weights: np.ndarray, 
                                        pre_times: np.ndarray, post_times: np.ndarray,
                                        delta_t: float, beta: float) -> None:
        """Apply spike-timing dependent plasticity (STDP)."""
        # Simplified STDP implementation
        for i, pre_time in enumerate(pre_times):
            for j, post_time in enumerate(post_times):
                dt = post_time - pre_time
                if dt > 0:  # Pre before post - LTP
                    weights[i, j] *= (1 + beta * np.exp(-dt / delta_t))
                elif dt < 0:  # Post before pre - LTD
                    weights[i, j] *= (1 - beta * np.exp(dt / delta_t))
    
    def homeostatic_scaling(self, weights: np.ndarray, target_activity: float, 
                           current_activity: float, eta: float = 0.01) -> None:
        """Apply homeostatic scaling to maintain target activity."""
        if current_activity > 0:
            scale_factor = 1 + eta * (target_activity - current_activity) / current_activity
            weights *= scale_factor

    def scale_stimulus_to_area(self, target_connectome: np.ndarray,
                               new_winner_indices: List[int],
                               beta: float,
                               disable: bool = False) -> np.ndarray:
        """
        Scale stimulus->area synapses for new winners by (1+beta).

        Mirrors brain.py logic where, for each i in new winners, vector[i] *= 1+beta.

        Args:
            target_connectome: 1D array of synapses into target area (length = w or _new_w)
            new_winner_indices: indices of new winners to strengthen
            beta: plasticity factor (>0 strengthens)
            disable: if True, returns unchanged copy

        Returns:
            np.ndarray: scaled copy of target_connectome
        """
        out = target_connectome.copy()
        if disable or beta == 0.0:
            return out
        if out.ndim != 1:
            # Handle sparse connectomes (empty arrays)
            if out.size == 0:
                return out
            raise ValueError("target_connectome must be 1D for stimulus->area scaling")
        validate_finite(out, "target_connectome")
        validate_finite_scalar(beta, "beta")
        factor = 1.0 + beta
        n = out.shape[0]
        for idx in normalize_index_list(new_winner_indices):
            if 0 <= idx < n:
                out[idx] *= factor
        return out

    def scale_area_to_area(self, connectome: np.ndarray,
                           pre_winner_rows: List[int],
                           post_winner_cols: List[int],
                           beta: float,
                           disable: bool = False) -> np.ndarray:
        """
        Scale area->area synapses for recent winners by (1+beta) on [rows in pre_winner_rows, cols in post_winner_cols].

        Mirrors brain.py logic: for i in post winners, for j in pre winners, connectome[j, i] *= 1+beta.

        Args:
            connectome: 2D array of synapses from source area (rows) to target area (cols)
            pre_winner_rows: indices of winners in source area
            post_winner_cols: indices of new winners in target area (columns)
            beta: plasticity factor
            disable: if True, returns unchanged copy

        Returns:
            np.ndarray: scaled copy of connectome
        """
        out = connectome.copy()
        if disable or beta == 0.0:
            return out
        if out.ndim != 2:
            # Handle sparse connectomes (empty arrays)
            if out.size == 0:
                return out
            raise ValueError("connectome must be 2D for area->area scaling")
        validate_finite(out, "connectome")
        validate_finite_scalar(beta, "beta")
        factor = 1.0 + beta
        rows, cols = out.shape
        norm_rows = normalize_index_list(pre_winner_rows)
        norm_cols = normalize_index_list(post_winner_cols)
        for j in norm_rows:
            if 0 <= j < rows:
                for i in norm_cols:
                    if 0 <= i < cols:
                        out[j, i] *= factor
        return out
