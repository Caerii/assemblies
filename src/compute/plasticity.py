# plasticity.py

"""
Synaptic plasticity mechanisms for neural assembly simulations.

This module will contain various plasticity rules and mechanisms
for updating synaptic weights in neural assemblies.
"""

import numpy as np
from typing import List
from .utils import validate_finite, validate_finite_scalar, normalize_index_list

try:
    from ..core.backend import get_xp, to_cpu
except ImportError:
    from core.backend import get_xp, to_cpu


class PlasticityEngine:
    """
    Engine for synaptic plasticity mechanisms.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng


    def hebbian_update(self, weights, pre_neurons, post_neurons, beta: float) -> None:
        """Apply Hebbian plasticity rule."""
        xp = get_xp()
        pre_neurons = xp.asarray(pre_neurons)
        post_neurons = xp.asarray(post_neurons)
        if len(pre_neurons) > 0 and len(post_neurons) > 0:
            ix = xp.ix_(pre_neurons, post_neurons)
            weights[ix] *= (1 + beta)

    def anti_hebbian_update(self, weights, pre_neurons, post_neurons, beta: float) -> None:
        """Apply anti-Hebbian plasticity rule."""
        xp = get_xp()
        pre_neurons = xp.asarray(pre_neurons)
        post_neurons = xp.asarray(post_neurons)
        if len(pre_neurons) > 0 and len(post_neurons) > 0:
            ix = xp.ix_(pre_neurons, post_neurons)
            weights[ix] *= (1 - beta)

    def spike_timing_dependent_plasticity(self, weights,
                                        pre_times, post_times,
                                        delta_t: float, beta: float) -> None:
        """Apply spike-timing dependent plasticity (STDP)."""
        xp = get_xp()
        pre_times = xp.asarray(pre_times)
        post_times = xp.asarray(post_times)
        # Vectorized STDP via outer difference
        dt = post_times[None, :] - pre_times[:, None]  # (n_pre, n_post)
        ltp_mask = dt > 0
        ltd_mask = dt < 0
        scale = xp.ones_like(dt)
        scale[ltp_mask] = 1 + beta * xp.exp(-dt[ltp_mask] / delta_t)
        scale[ltd_mask] = 1 - beta * xp.exp(dt[ltd_mask] / delta_t)
        weights *= scale

    def homeostatic_scaling(self, weights, target_activity: float,
                           current_activity: float, eta: float = 0.01) -> None:
        """Apply homeostatic scaling to maintain target activity."""
        if current_activity > 0:
            scale_factor = 1 + eta * (target_activity - current_activity) / current_activity
            weights *= scale_factor

    def scale_stimulus_to_area(self, target_connectome,
                               new_winner_indices: List[int],
                               beta: float,
                               disable: bool = False):
        """
        Scale stimulus->area synapses for new winners by (1+beta).

        Args:
            target_connectome: 1D array of synapses into target area
            new_winner_indices: indices of new winners to strengthen
            beta: plasticity factor (>0 strengthens)
            disable: if True, returns unchanged copy

        Returns:
            scaled copy of target_connectome
        """
        out = target_connectome.copy()
        if disable or beta == 0.0:
            return out
        if out.ndim != 1:
            if out.size == 0:
                return out
            raise ValueError("target_connectome must be 1D for stimulus->area scaling")
        validate_finite(np.asarray(to_cpu(out)), "target_connectome")
        validate_finite_scalar(beta, "beta")
        xp = get_xp()
        factor = 1.0 + beta
        n = out.shape[0]
        indices = xp.array([int(idx) for idx in normalize_index_list(new_winner_indices) if 0 <= idx < n])
        if len(indices) > 0:
            out[indices] *= factor
        return out

    def scale_area_to_area(self, connectome,
                           pre_winner_rows: List[int],
                           post_winner_cols: List[int],
                           beta: float,
                           disable: bool = False):
        """
        Scale area->area synapses for recent winners by (1+beta).

        Args:
            connectome: 2D array of synapses from source area to target area
            pre_winner_rows: indices of winners in source area
            post_winner_cols: indices of new winners in target area (columns)
            beta: plasticity factor
            disable: if True, returns unchanged copy

        Returns:
            scaled copy of connectome
        """
        out = connectome.copy()
        if disable or beta == 0.0:
            return out
        if out.ndim != 2:
            if out.size == 0:
                return out
            raise ValueError("connectome must be 2D for area->area scaling")
        validate_finite(np.asarray(to_cpu(out)), "connectome")
        validate_finite_scalar(beta, "beta")
        xp = get_xp()
        factor = 1.0 + beta
        rows, cols = out.shape
        valid_r = xp.array([int(j) for j in normalize_index_list(pre_winner_rows) if 0 <= j < rows])
        valid_c = xp.array([int(i) for i in normalize_index_list(post_winner_cols) if 0 <= i < cols])
        if len(valid_r) > 0 and len(valid_c) > 0:
            ix = xp.ix_(valid_r, valid_c)
            out[ix] *= factor
        return out
