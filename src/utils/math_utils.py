# math_utils.py
"""
Mathematical Utilities for Neural Assembly Computation

This module provides mathematical primitives essential for neural assembly
simulation, including normalization, winner selection, and statistical
computations used in the Assembly Calculus framework.

Biological Context:
- Feature normalization ensures consistent input ranges for neural competition
- Winner selection implements sparse coding principles found in biological brains
- Statistical functions support scalable simulation of large neural populations

Assembly Calculus Context:
- Normalization enables fair competition between neural inputs
- Winner selection implements the core mechanism of assembly formation
- Statistical approximations enable efficient simulation of large networks

Mathematical Foundation:
- Min-max normalization: (x - min) / (max - min)
- Winner-take-all: Select top-k elements from input vector
- Binomial statistics: Support for probabilistic neural modeling
"""

import numpy as np
import heapq

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to [0, 1] range using min-max scaling.
    
    This is crucial for neural assembly computation as it ensures consistent
    input ranges for winner selection algorithms. Normalization prevents
    any single input from dominating the competition due to scale differences.

    Args:
        features (np.ndarray): Feature vector to normalize.
                              Can be neural inputs, synaptic weights, or activations.

    Returns:
        np.ndarray: Normalized feature vector with values in [0, 1].
                   Minimum value becomes 0, maximum becomes 1.
                   
    Biological Context:
        Normalization models the homeostatic mechanisms in biological neurons
        that maintain consistent firing rates and prevent runaway excitation.
        
    Mathematical Context:
        Formula: (x - min(x)) / (max(x) - min(x))
        - Preserves relative relationships between features
        - Maps all values to [0, 1] interval
        - Handles edge case where all features are identical
        
    Example:
        >>> features = np.array([1, 5, 3, 9, 2])
        >>> normalized = normalize_features(features)
        >>> print(normalized)  # [0.0, 0.5, 0.25, 1.0, 0.125]
    """
    return (features - np.min(features)) / (np.ptp(features) + 1e-6)

def select_top_k_indices(features: np.ndarray, k: int) -> np.ndarray:
    """
    Select the indices of the top-k values using argsort.
    
    This implements the core winner-take-all mechanism of neural assembly formation.
    Only the k neurons with highest inputs will fire, implementing sparse coding
    principles found in biological neural networks.

    Args:
        features (np.ndarray): Input feature vector (neural inputs).
        k (int): Number of winners to select (assembly size).

    Returns:
        np.ndarray: Indices of the top-k values, sorted in descending order.
        
    Biological Context:
        Implements winner-take-all competition in biological neural networks,
        where only the most strongly activated neurons fire. This creates
        sparse, efficient representations and prevents runaway excitation.
        
    Assembly Calculus Context:
        This is the fundamental mechanism for assembly formation. The k winners
        form a neural assembly representing a specific concept or computation.
        
    Mathematical Context:
        Uses argsort(-features) to find indices of largest values.
        Time complexity: O(n log n) due to sorting.
        
    Example:
        >>> inputs = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        >>> winners = select_top_k_indices(inputs, k=3)
        >>> print(winners)  # [1, 3, 2] (indices of top 3 values)
    """
    return np.argsort(-features)[:k]

def heapq_select_top_k(features: np.ndarray, k: int) -> np.ndarray:
    """
    Selects the indices of the top-k values using a heap for efficiency.

    Args:
        features (np.ndarray): Feature vector.
        k (int): Number of top indices to select.

    Returns:
        np.ndarray: Indices of the top-k values.
    """
    if k >= len(features):
        return np.arange(len(features))
    return np.array(heapq.nlargest(k, range(len(features)), features.take))

def binomial_ppf(quantile: float, n: int, p: float) -> float:
    """
    Calculates the inverse cumulative distribution function (percent point function)
    for a binomial distribution.

    Args:
        quantile (float): The quantile to compute the ppf at.
        n (int): Number of trials.
        p (float): Probability of success on each trial.

    Returns:
        float: The ppf value.
    """
    from scipy.stats import binom
    return binom.ppf(quantile, n, p)

# Additional utility functions can be added here
