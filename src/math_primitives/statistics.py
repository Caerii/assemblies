# statistics.py

"""
Statistical sampling algorithms for neural assembly simulations.

This module contains the complex statistical sampling methods extracted
from the root brain.py projection logic, including binomial PPF calculations,
truncated normal sampling, and quantile-based thresholding.

Biological Context:
- Implements statistical approximations for large-scale neural simulations
- Enables efficient sparse simulation of neural assemblies
- Models probabilistic synaptic connectivity and activation patterns

Mathematical Foundation:
- Binomial distribution: Models synaptic connection probabilities
- Truncated normal: Approximates binomial for large n
- Quantile functions: Determine activation thresholds
- Normal approximation: Improves computational efficiency

Assembly Calculus Context:
- Statistical sampling enables scalable assembly formation
- Quantile thresholds implement winner-take-all competition
- Probabilistic methods preserve assembly overlap properties
"""

import numpy as np
from scipy.stats import binom, truncnorm
import math
from typing import List, Tuple

class StatisticalEngine:
    """
    Advanced statistical sampling engine for neural assembly simulations.
    
    This class implements the complex statistical methods used in the root
    brain.py for efficient sparse simulation of neural assemblies. It handles
    binomial distributions, truncated normal approximations, and quantile-based
    thresholding for winner selection.
    
    Key Features:
    - Binomial PPF calculations for quantile thresholds
    - Truncated normal sampling for new winner generation
    - Normal approximation for computational efficiency
    - Quantile-based winner selection thresholds
    
    Performance:
    - Enables sparse simulation of large neural networks
    - Reduces computational complexity from O(n²) to O(k log n)
    - Maintains statistical accuracy of assembly formation
    """
    
    def __init__(self, rng: np.random.Generator):
        """
        Initialize statistical engine.
        
        Args:
            rng (np.random.Generator): Random number generator for reproducibility
        """
        self.rng = rng
        self._use_normal_ppf = False  # Future feature for advanced approximations
    
    def calculate_quantile_threshold(self, effective_n: int, k: int) -> float:
        """
        Calculate quantile threshold for winner selection.
        
        This implements the threshold calculation used in sparse simulation
        to determine which neurons are likely to be winners based on their
        input strength relative to the population.
        
        Args:
            effective_n (int): Effective population size (n - w)
            k (int): Number of winners to select
            
        Returns:
            float: Quantile threshold (0, 1) for winner selection
            
        Biological Context:
            This threshold implements the competitive dynamics of neural
            assemblies where only the most strongly activated neurons fire.
            The quantile ensures that exactly k neurons are selected from
            the available population.
            
        Mathematical Context:
            Quantile = (effective_n - k) / effective_n
            This gives the threshold below which (effective_n - k) neurons
            fall, leaving exactly k neurons above the threshold.
        """
        if effective_n <= 0:
            raise ValueError("Effective population size must be positive")
        if k <= 0:
            raise ValueError("Number of winners must be positive")
        if k >= effective_n:
            raise ValueError("Number of winners cannot exceed effective population size")
            
        return (effective_n - k) / effective_n
    
    def calculate_binomial_ppf(self, quantile: float, n: int, p: float) -> float:
        """
        Calculate binomial percent point function (PPF) for threshold.
        
        This computes the threshold value such that a binomial(n, p) random
        variable exceeds this value with probability (1 - quantile).
        
        Args:
            quantile (float): Quantile threshold (0, 1)
            n (int): Number of trials
            p (float): Success probability
            
        Returns:
            float: Threshold value for winner selection
            
        Biological Context:
            The PPF determines the minimum input strength required for a
            neuron to be selected as a winner. This implements the competitive
            threshold mechanism in neural assembly formation.
            
        Mathematical Context:
            PPF(quantile) = min{x : P(X ≤ x) ≥ quantile} where X ~ Binomial(n, p)
            This gives the threshold below which (quantile * 100)% of neurons fall.
        """
        if not 0 <= quantile <= 1:
            raise ValueError("Quantile must be between 0 and 1")
        if n <= 0:
            raise ValueError("Number of trials must be positive")
        if not 0 <= p <= 1:
            raise ValueError("Success probability must be between 0 and 1")
            
        return binom.ppf(quantile, n, p)
    
    def sample_truncated_normal_winners(self, alpha: float, total_k: int, 
                                      p: float, k: int) -> np.ndarray:
        """
        Sample potential new winners using truncated normal approximation.
        
        This implements the sophisticated statistical sampling method used
        in the root brain.py for generating potential new winners in sparse
        simulation mode. It uses a truncated normal approximation to the
        binomial distribution for computational efficiency.
        
        Args:
            alpha (float): Threshold from binomial PPF
            total_k (int): Total input size from all sources
            p (float): Connection probability
            k (int): Number of potential winners to sample
            
        Returns:
            np.ndarray: Array of input strengths for potential new winners
            
        Biological Context:
            This method generates realistic input strengths for new neurons
            that might become winners. The truncated normal approximation
            maintains the statistical properties of the true binomial
            distribution while being computationally efficient.
            
        Mathematical Context:
            Uses normal approximation: N(μ, σ²) where:
            - μ = total_k * p (mean)
            - σ = sqrt(total_k * p * (1-p)) (standard deviation)
            - Truncated at α to ∞ to avoid negative values
            - Rounded to integers to match discrete nature of inputs
        """
        if total_k <= 0:
            raise ValueError("Total input size must be positive")
        if not 0 <= p <= 1:
            raise ValueError("Connection probability must be between 0 and 1")
        if k <= 0:
            raise ValueError("Number of potential winners must be positive")
            
        # Calculate normal approximation parameters
        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))
        
        # Calculate truncation point
        a = (alpha - mu) / std
        
        # Sample from truncated normal distribution
        # Using np.inf as upper bound to avoid sampling above total_k
        samples = mu + truncnorm.rvs(a, np.inf, scale=std, size=k)
        
        # Round to integers and clamp to valid range
        rounded_samples = samples.round(0).astype(int)
        
        # Ensure no samples exceed total_k (theoretical maximum)
        rounded_samples = np.minimum(rounded_samples, total_k)
        
        return rounded_samples
    
    def sample_binomial_winners(self, n: int, p: float, size: int = 1) -> np.ndarray:
        """
        Sample winners directly from binomial distribution.
        
        This is the direct binomial sampling method used for explicit
        areas and simple cases where the normal approximation is not needed.
        
        Args:
            n (int): Number of trials
            p (float): Success probability
            size (int): Number of samples to generate
            
        Returns:
            np.ndarray: Array of binomial samples
        """
        if n <= 0:
            raise ValueError("Number of trials must be positive")
        if not 0 <= p <= 1:
            raise ValueError("Success probability must be between 0 and 1")
        if size <= 0:
            raise ValueError("Sample size must be positive")
            
        return self.rng.binomial(n, p, size=size)
    
    def calculate_input_statistics(self, input_sizes: List[int], p: float) -> Tuple[float, float]:
        """
        Calculate mean and variance for normal approximation.
        
        This computes the parameters needed for the normal approximation
        to the sum of multiple binomial distributions.
        
        Args:
            input_sizes (List[int]): List of input sizes from different sources
            p (float): Connection probability
            
        Returns:
            Tuple[float, float]: (mean, variance) of the combined distribution
            
        Mathematical Context:
            If X_i ~ Binomial(n_i, p) are independent, then:
            - E[ΣX_i] = Σ(n_i * p) = p * Σn_i
            - Var[ΣX_i] = Σ(n_i * p * (1-p)) = p * (1-p) * Σn_i
        """
        if not input_sizes:
            return 0.0, 0.0
            
        total_inputs = sum(input_sizes)
        mean = total_inputs * p
        variance = total_inputs * p * (1.0 - p)
        
        return mean, variance
    
    def calculate_input_statistics_root_style(self, input_sizes: List[int], p: float) -> Tuple[float, float]:
        """
        Calculate mean and variance using root brain.py style.

        This matches the exact calculation from the root brain.py implementation
        which uses sum of squared variances rather than sum of variances.

        Args:
            input_sizes (List[int]): List of input sizes from different sources
            p (float): Connection probability

        Returns:
            Tuple[float, float]: (mean, variance) using root brain.py method
        """
        if not input_sizes:
            return 0.0, 0.0

        normal_approx_mean = 0.0
        normal_approx_var = 0.0

        for local_k in input_sizes:
            normal_approx_mean += local_k * p
            normal_approx_var += ((local_k * p * (1 - p)) ** 2)

        return normal_approx_mean, normal_approx_var

    def calculate_input_statistics_standard(self, input_sizes: List[int], p: float) -> Tuple[float, float]:
        """
        Calculate mean and variance for normal approximation using standard binomial method.

        This computes the parameters needed for the normal approximation
        to the sum of multiple binomial distributions, using the standard
        binomial variance calculation that matches the active code in root brain.py.

        Args:
            input_sizes (List[int]): List of input sizes from different sources
            p (float): Connection probability

        Returns:
            Tuple[float, float]: (mean, variance) using standard binomial method
        """
        if not input_sizes:
            return 0.0, 0.0

        total_k = sum(input_sizes)

        # Standard binomial calculation (matches root brain.py lines 697-698)
        mean = total_k * p
        variance = total_k * p * (1.0 - p)

        return mean, variance
    
    def validate_sampling_parameters(self, effective_n: int, k: int, 
                                   total_k: int, p: float) -> None:
        """
        Validate parameters for statistical sampling.
        
        This ensures all parameters are within valid ranges for the
        statistical sampling algorithms.
        
        Args:
            effective_n (int): Effective population size
            k (int): Number of winners to select
            total_k (int): Total input size
            p (float): Connection probability
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if effective_n <= k:
            raise RuntimeError(
                f'Remaining size of area ({effective_n}) too small to sample '
                f'k new winners ({k}).'
            )
        if total_k <= 0:
            raise ValueError("Total input size must be positive")
        if not 0 <= p <= 1:
            raise ValueError("Connection probability must be between 0 and 1")
        if k <= 0:
            raise ValueError("Number of winners must be positive")
    
    def get_sampling_method_info(self) -> dict:
        """
        Get information about the sampling methods used.
        
        Returns:
            dict: Information about sampling methods and parameters
        """
        return {
            'normal_ppf_enabled': self._use_normal_ppf,
            'methods_available': [
                'binomial_direct',
                'truncated_normal_approximation',
                'quantile_thresholding'
            ],
            'approximation_accuracy': 'High (maintains statistical properties)',
            'computational_complexity': 'O(k) for sampling, O(log n) for PPF'
        }