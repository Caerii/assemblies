# utils.py

import numpy as np
import heapq

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalizes features between 0 and 1.

    Args:
        features (np.ndarray): Feature vector.

    Returns:
        np.ndarray: Normalized feature vector.
    """
    return (features - np.min(features)) / (np.ptp(features) + 1e-6)

def select_top_k_indices(features: np.ndarray, k: int) -> np.ndarray:
    """
    Selects the indices of the top-k values in a feature vector.

    Args:
        features (np.ndarray): Feature vector.
        k (int): Number of top indices to select.

    Returns:
        np.ndarray: Indices of the top-k values.
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
