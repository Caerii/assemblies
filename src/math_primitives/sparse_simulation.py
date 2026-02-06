# sparse_simulation.py

"""
Sparse simulation engine for neural assembly dynamics.

This module contains the complex sparse simulation algorithms extracted
from the root brain.py projection logic, including dynamic connectome expansion,
first-time winner tracking, and input distribution management.

Biological Context:
- Implements sparse simulation of neural assemblies
- Manages dynamic expansion of connectomes when new neurons fire
- Handles input distribution from multiple brain areas and stimuli
- Maintains synaptic connectivity patterns for learning

Assembly Calculus Context:
- Enables efficient simulation of large-scale neural networks
- Supports dynamic assembly formation and connectivity
- Models synaptic plasticity in sparse neural populations

Mathematical Foundation:
- Connectome expansion: Dynamic matrix resizing with padding
- Input distribution: Multinomial sampling across input sources
- Winner indexing: First-time vs repeat winner tracking
- Synaptic assignment: Bernoulli sampling for connectivity
"""

import numpy as np
import heapq
from typing import List, Dict, Tuple, Any

class SparseSimulationEngine:
    """
    Sparse simulation engine for neural assembly dynamics.

    This class implements the complex algorithms for sparse neural simulation
    extracted from the root brain.py, including dynamic connectome expansion,
    winner tracking, and input distribution management.
    """

    def __init__(self, rng: np.random.Generator):
        """
        Initialize the sparse simulation engine.

        Args:
            rng (np.random.Generator): Random number generator for reproducibility
        """
        self.rng = rng

    def calculate_input_distribution(self, input_sizes: List[int],
                                  first_winner_inputs: List[float]) -> List[np.ndarray]:
        """
        Calculate how inputs should be distributed across multiple sources.

        This implements the complex input distribution logic from root brain.py
        lines 757-773, which determines how synaptic connections from each
        input source should be assigned to new winners.

        Args:
            input_sizes (List[int]): Size of each input source
            first_winner_inputs (List[float]): Input strength for each new winner

        Returns:
            List[np.ndarray]: Input distribution for each new winner

        Biological Context:
            Each new winner receives synaptic input from multiple brain areas
            and stimuli. This method determines the connection strength from
            each input source based on the winner's total input strength.

        Mathematical Context:
            Uses multinomial-like distribution across input sources.
            For each new winner with input strength s, sample s connections
            from the total input space, then count how many come from each source.
        """
        if not input_sizes or not first_winner_inputs:
            return []

        num_inputs = len(input_sizes)
        total_k = sum(input_sizes)
        num_first_winners = len(first_winner_inputs)

        inputs_by_first_winner_index = [None] * num_first_winners

        for i in range(num_first_winners):
            input_strength = int(first_winner_inputs[i])

            # Sample input indices from the total input space
            input_indices = self.rng.choice(
                range(total_k), input_strength, replace=False
            )

            # Count how many inputs come from each source
            num_connections_by_input_index = np.zeros(num_inputs)
            total_so_far = 0

            for j in range(num_inputs):
                # Count inputs that fall within this source's range
                num_connections_by_input_index[j] = sum(
                    total_so_far + input_sizes[j] > idx >= total_so_far
                    for idx in input_indices
                )
                total_so_far += input_sizes[j]

            inputs_by_first_winner_index[i] = num_connections_by_input_index

        return inputs_by_first_winner_index

    def expand_connectome_dynamic(self, current_connectome: np.ndarray,
                                num_new_winners: int,
                                axis: int = 1) -> np.ndarray:
        """
        Dynamically expand connectome to accommodate new winners.

        This implements the dynamic connectome expansion logic from root brain.py
        lines 782-784, 806-808, etc., which handles the expansion of synaptic
        weight matrices when new neurons become active.

        Args:
            current_connectome (np.ndarray): Current synaptic weight matrix
            num_new_winners (int): Number of new winners to add
            axis (int): Axis to expand (0=rows, 1=columns)

        Returns:
            np.ndarray: Expanded connectome matrix

        Biological Context:
            When new neurons fire for the first time, the synaptic weight
            matrices need to be expanded to include connections to/from
            these new neurons. This maintains the full connectivity structure.

        Mathematical Context:
            Matrix expansion with zero-padding to maintain connectivity
            structure while accommodating new neural elements.
        """
        if num_new_winners <= 0:
            return current_connectome

        if axis == 0:  # Expand rows
            pad_width = ((0, num_new_winners), (0, 0))
        elif axis == 1:  # Expand columns
            pad_width = ((0, 0), (0, num_new_winners))
        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")

        expanded_connectome = np.pad(current_connectome, pad_width)
        return expanded_connectome

    def assign_synaptic_connections(self, connectome: np.ndarray,
                                  input_sources: List[np.ndarray],
                                  new_winner_indices: List[int],
                                  connection_probability: float = 0.05) -> np.ndarray:
        """
        Assign synaptic connections for new winners based on input distribution.

        This implements the synaptic connection assignment logic from root brain.py
        lines 788-790, 827-833, etc., which determines the initial synaptic
        weights for new winners based on their input distribution.

        Args:
            connectome (np.ndarray): Synaptic weight matrix to update
            input_sources (List[np.ndarray]): Input distributions for each winner
            new_winner_indices (List[int]): Indices of new winners
            connection_probability (float): Probability of random connections

        Returns:
            np.ndarray: Updated connectome matrix

        Biological Context:
            New winners establish synaptic connections based on their input
            patterns. Strong inputs become strong synapses, while weak or
            absent inputs may still form random connections.

        Mathematical Context:
            Direct assignment of connection strengths based on input distribution,
            with Bernoulli sampling for background connectivity.
        """
        updated_connectome = connectome.copy()
        num_inputs_processed = 0

        # Process each winner's input distribution
        for winner_idx, input_distribution in zip(new_winner_indices, input_sources):
            # Assign connections for this winner from all input sources
            for source_idx, strength in enumerate(input_distribution):
                # Direct connection strength assignment
                updated_connectome[winner_idx, source_idx] = strength

            num_inputs_processed += 1

        return updated_connectome

    def initialize_new_winner_synapses(self, connectome: np.ndarray,
                                     target_w: int,
                                     num_new_winners: int,
                                     input_sizes: List[int],
                                     p: float = 0.05) -> np.ndarray:
        """
        Initialize synaptic weights for new winners from external stimuli.

        This implements the logic from root brain.py lines 809-811, which
        initializes synaptic connections from stimuli that weren't active
        in the current projection round.

        Args:
            connectome (np.ndarray): Stimulus-to-area connectome
            target_w: Current number of ever-fired neurons
            num_new_winners (int): Number of new winners
            input_sizes (List[int]): Sizes of input stimuli
            connection_probability (float): Connection probability

        Returns:
            np.ndarray: Updated connectome with new winner synapses

        Biological Context:
            Even stimuli that weren't active in the current round can form
            connections to new winners through random synaptic formation.

        Mathematical Context:
            Bernoulli sampling to establish baseline connectivity from
            inactive stimuli to new neural elements.
        """
        if num_new_winners <= 0:
            return connectome

        updated_connectome = connectome.copy()

        # Initialize connections for new winners from external stimuli
        for i in range(num_new_winners):
            winner_idx = target_w + i
            for j, stim_size in enumerate(input_sizes):
                # Random connections from each stimulus
                connections = self.rng.binomial(
                    stim_size, p, size=1
                )[0]
                updated_connectome[winner_idx, j] = connections

        return updated_connectome

    def apply_plasticity_scaling(self, connectome: np.ndarray,
                               winner_indices: List[int],
                               plasticity_beta: float) -> np.ndarray:
        """
        Apply Hebbian plasticity scaling to synaptic weights.

        This implements the plasticity scaling logic from root brain.py
        lines 794-795, 837-839, which strengthens synapses to recently
        active neurons.

        Args:
            connectome (np.ndarray): Synaptic weight matrix
            winner_indices (List[int]): Indices of active winners
            plasticity_beta (float): Plasticity scaling factor

        Returns:
            np.ndarray: Connectome with applied plasticity scaling

        Biological Context:
            Hebbian plasticity strengthens synapses to neurons that
            participated in recent successful firing patterns.

        Mathematical Context:
            Multiplicative scaling of synaptic weights based on
            neural activity participation.
        """
        updated_connectome = connectome.copy()

        for i in winner_indices:
            for j in range(connectome.shape[0]):
                # Apply plasticity scaling
                updated_connectome[j, i] *= (1.0 + plasticity_beta)

        return updated_connectome

    def process_first_time_winners(self, all_potential_winners: List[float],
                                 target_area_w: int,
                                 target_area_k: int) -> Tuple[List[int], List[float], int]:
        """
        Process first-time winners and update their indices.

        This implements the first-time winner processing logic from root brain.py
        lines 736-747, which handles the indexing and tracking of neurons
        that fire for the first time.

        Args:
            all_potential_winners (List[float]): Input strengths for all potential winners
            target_area_w (int): Current number of ever-fired neurons
            target_area_k (int): Number of winners to select

        Returns:
            Tuple[List[int], List[float], int]: (new_winner_indices, first_winner_inputs, num_first_winners)

        Biological Context:
            First-time winners are assigned new indices in the neural population
            and their input strengths are tracked for connectome expansion.

        Mathematical Context:
            Index remapping to maintain consistent neural population structure
            while accommodating dynamic expansion.
        """
        # Select top k winners using heap algorithm
        new_winner_indices = heapq.nlargest(
            target_area_k,
            range(len(all_potential_winners)),
            key=lambda i: all_potential_winners[i]
        )

        first_winner_inputs = []
        num_first_winners_processed = 0

        # Process each winner
        for i in range(len(new_winner_indices)):
            if new_winner_indices[i] >= target_area_w:
                # First-time winner
                first_winner_inputs.append(all_potential_winners[new_winner_indices[i]])
                # Remap index to new position
                new_winner_indices[i] = target_area_w + num_first_winners_processed
                num_first_winners_processed += 1

        return new_winner_indices, first_winner_inputs, num_first_winners_processed

    def sample_new_winner_inputs(
        self,
        input_sizes: List[int],
        n: int,
        w: int,
        k: int,
        p: float,
    ) -> np.ndarray:
        """
        Sample potential input strengths for k new winner candidates using
        the truncated-normal approximation from the original brain.py.

        Each new candidate neuron receives a random number of inputs from the
        total input pool (stimuli + source areas). The distribution is
        Binomial(total_k, p) truncated to the top-(k/effective_n) quantile,
        approximated via a truncated normal.

        Args:
            input_sizes: Size of each input source (stimulus sizes + source area k values).
            n: Total neuron count of the target area.
            w: Number of neurons that have ever fired in the target area.
            k: Assembly size (number of winners to select).
            p: Connection probability.

        Returns:
            1D array of length k with sampled input strengths for new candidates.
        """
        import math
        from scipy.stats import binom, truncnorm

        total_k = sum(input_sizes)
        effective_n = n - w

        if effective_n <= k:
            raise RuntimeError(
                f"Remaining size of area too small to sample k new winners "
                f"(effective_n={effective_n}, k={k})."
            )

        quantile = (effective_n - k) / effective_n
        alpha = binom.ppf(quantile, total_k, p)

        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))
        a = (alpha - mu) / std

        samples = (
            mu + truncnorm.rvs(a, np.inf, scale=std, size=k, random_state=self.rng)
        ).round(0)
        np.clip(samples, 0, total_k, out=samples)
        return samples

    def compute_input_splits(
        self,
        input_sizes: List[int],
        first_winner_inputs: List[int],
    ) -> List[List[int]]:
        """
        Distribute each new winner's total input across source areas/stimuli
        proportional to their sizes.

        Args:
            input_sizes: Size of each input source.
            first_winner_inputs: Total input for each first-time winner.

        Returns:
            List of per-winner split vectors (one int per input source).
        """
        total_k = sum(input_sizes)
        if total_k == 0:
            return [[] for _ in first_winner_inputs]

        proportions = np.array(input_sizes, dtype=np.float64) / float(total_k)
        splits = []
        for total_in in first_winner_inputs:
            remaining = int(total_in)
            base = np.floor(proportions * remaining).astype(int)
            remainder = remaining - int(base.sum())
            if remainder > 0:
                frac = (proportions * remaining) - base
                order = np.argsort(-frac)
                for j in range(remainder):
                    base[order[j % len(base)]] += 1
            splits.append(base.tolist())
        return splits

    def get_simulation_method_info(self) -> Dict[str, Any]:
        """
        Get information about the sparse simulation methods.

        Returns:
            Dict[str, Any]: Information about the simulation methods
        """
        return {
            'module': 'sparse_simulation',
            'algorithms': [
                'dynamic_connectome_expansion',
                'input_distribution_calculation',
                'first_time_winner_processing',
                'synaptic_connection_assignment',
                'plasticity_scaling',
                'connectome_initialization'
            ],
            'biological_context': 'sparse_neural_assembly_dynamics',
            'mathematical_foundation': 'dynamic_matrix_operations_and_multinomial_sampling',
            'complexity': 'O(k * log k) for winner selection, O(total_k) for input distribution'
        }
