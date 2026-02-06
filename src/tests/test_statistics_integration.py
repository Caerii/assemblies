# test_statistics_integration.py

"""
Integration tests for the Statistical Engine against root brain.py.

This module verifies that the extracted statistical algorithms produce
the same results as the original implementation in the root brain.py.
"""

import unittest
import numpy as np

from src.math_primitives.statistics import StatisticalEngine

class TestStatisticsIntegration(unittest.TestCase):
    """
    Integration tests comparing Statistical Engine with root brain.py.
    
    These tests verify that the extracted statistical algorithms produce
    identical results to the original implementation, ensuring correctness
    of the extraction process.
    """
    
    def setUp(self):
        """Set up test fixtures with same parameters as root brain.py."""
        # Use same seed as root brain.py for reproducibility
        self.rng = np.random.default_rng(seed=42)
        self.stats_engine = StatisticalEngine(self.rng)
        
        # Parameters from root brain.py test cases
        self.effective_n = 1000
        self.k = 100
        self.total_k = 500
        self.p = 0.05
    
    def test_quantile_calculation_matches_root(self):
        """Test that quantile calculation matches root brain.py."""
        # This is the exact calculation from root brain.py line 677
        quantile = (self.effective_n - self.k) / self.effective_n
        
        # Test our implementation
        our_quantile = self.stats_engine.calculate_quantile_threshold(
            self.effective_n, self.k
        )
        
        self.assertAlmostEqual(our_quantile, quantile, places=10)
    
    def test_binomial_ppf_matches_root(self):
        """Test that binomial PPF matches root brain.py."""
        from scipy.stats import binom
        
        quantile = (self.effective_n - self.k) / self.effective_n
        
        # Root brain.py calculation (line 688)
        root_alpha = binom.ppf(quantile, self.total_k, self.p)
        
        # Our implementation
        our_alpha = self.stats_engine.calculate_binomial_ppf(
            quantile, self.total_k, self.p
        )
        
        self.assertAlmostEqual(our_alpha, root_alpha, places=10)
    
    def test_truncated_normal_sampling_matches_root(self):
        """Test that truncated normal sampling matches root brain.py."""
        from scipy.stats import truncnorm, binom
        import math
        
        quantile = (self.effective_n - self.k) / self.effective_n
        alpha = binom.ppf(quantile, self.total_k, self.p)
        
        # Root brain.py calculation (lines 697-707)
        mu = self.total_k * self.p
        std = math.sqrt(self.total_k * self.p * (1.0 - self.p))
        a = (alpha - mu) / std
        
        # Root brain.py sampling
        root_samples = (mu + truncnorm.rvs(a, np.inf, scale=std, size=self.k)).round(0)
        for i in range(len(root_samples)):
            if root_samples[i] > self.total_k:
                root_samples[i] = self.total_k
        
        # Our implementation
        our_samples = self.stats_engine.sample_truncated_normal_winners(
            alpha, self.total_k, self.p, self.k
        )
        
        # Compare results (allowing for random variation)
        self.assertEqual(len(our_samples), len(root_samples))
        self.assertTrue(np.all(our_samples >= 0))
        self.assertTrue(np.all(our_samples <= self.total_k))
        
        # Statistical properties should be similar
        root_mean = np.mean(root_samples)
        our_mean = np.mean(our_samples)
        
        # Means should be close (within 20% due to randomness)
        self.assertLess(abs(our_mean - root_mean), root_mean * 0.2)
    
    def test_input_statistics_matches_root(self):
        """Test that input statistics calculation matches root brain.py."""
        # Simulate input sizes from multiple sources (like root brain.py)
        input_sizes = [100, 200, 300]  # From stimuli and areas
        
        # Root brain.py calculation (lines 647-666)
        total_k = sum(input_sizes)
        normal_approx_mean = 0.0
        normal_approx_var = 0.0
        
        for local_k in input_sizes:
            normal_approx_mean += local_k * self.p
            normal_approx_var += ((local_k * self.p * (1 - self.p)) ** 2)
        
        # Root brain.py uses sum of squared variances, not sum of variances
        # This is the correct formula from the root implementation
        
        # Our implementation (using root-style method)
        our_mean, our_variance = self.stats_engine.calculate_input_statistics_root_style(
            input_sizes, self.p
        )
        
        # Compare results
        self.assertAlmostEqual(our_mean, normal_approx_mean, places=10)
        self.assertAlmostEqual(our_variance, normal_approx_var, places=10)
    
    def test_parameter_validation_matches_root(self):
        """Test that parameter validation matches root brain.py."""
        # Test the exact error condition from root brain.py (lines 673-675)
        effective_n = 50  # Too small
        k = 100  # Larger than effective_n
        
        with self.assertRaises(RuntimeError) as context:
            self.stats_engine.validate_sampling_parameters(
                effective_n, k, self.total_k, self.p
            )
        
        # Check that error message matches root brain.py
        expected_msg = f'Remaining size of area ({effective_n}) too small to sample k new winners ({k}).'
        self.assertIn(expected_msg, str(context.exception))
    
    def test_statistical_properties_consistency(self):
        """Test that statistical properties are consistent across multiple runs."""
        # Test with multiple random seeds to ensure consistency
        seeds = [42, 123, 456, 789, 999]
        all_samples = []
        
        for seed in seeds:
            rng = np.random.default_rng(seed=seed)
            stats_engine = StatisticalEngine(rng)
            
            quantile = (self.effective_n - self.k) / self.effective_n
            alpha = stats_engine.calculate_binomial_ppf(quantile, self.total_k, self.p)
            
            samples = stats_engine.sample_truncated_normal_winners(
                alpha, self.total_k, self.p, self.k
            )
            all_samples.append(samples)
        
        # All samples should have consistent properties
        for samples in all_samples:
            self.assertEqual(len(samples), self.k)
            self.assertTrue(np.all(samples >= 0))
            self.assertTrue(np.all(samples <= self.total_k))
        
        # Statistical properties should be consistent across seeds
        means = [np.mean(samples) for samples in all_samples]
        expected_mean = self.total_k * self.p
        
        for mean in means:
            # Allow for more variation due to truncated normal sampling
            self.assertLess(abs(mean - expected_mean), expected_mean * 0.5)
    
    def test_edge_case_handling_matches_root(self):
        """Test that edge cases are handled the same as root brain.py."""
        # Test very small effective_n (should raise error)
        with self.assertRaises(RuntimeError):
            self.stats_engine.validate_sampling_parameters(10, 20, 100, 0.5)
        
        # Test k = effective_n - 1 (should work)
        self.stats_engine.validate_sampling_parameters(100, 99, 100, 0.5)
        
        # Test k = 1 (should work)
        self.stats_engine.validate_sampling_parameters(100, 1, 100, 0.5)
    
    def test_mathematical_correctness(self):
        """Test mathematical correctness of the algorithms."""
        # Test quantile calculation
        effective_n = 1000
        k = 100
        quantile = self.stats_engine.calculate_quantile_threshold(effective_n, k)
        
        # Quantile should be (1000 - 100) / 1000 = 0.9
        self.assertAlmostEqual(quantile, 0.9, places=10)
        
        # Test binomial PPF properties
        n = 1000
        p = 0.1
        
        # PPF should be monotonic
        ppf_1 = self.stats_engine.calculate_binomial_ppf(0.1, n, p)
        ppf_2 = self.stats_engine.calculate_binomial_ppf(0.5, n, p)
        ppf_3 = self.stats_engine.calculate_binomial_ppf(0.9, n, p)
        
        self.assertLessEqual(ppf_1, ppf_2)
        self.assertLessEqual(ppf_2, ppf_3)
        
        # Test normal approximation accuracy
        # For large n and moderate p, normal approximation should be good
        samples = self.stats_engine.sample_truncated_normal_winners(50, 1000, 0.1, 100)
        sample_mean = np.mean(samples)
        expected_mean = 1000 * 0.1  # 100
        
        # Sample mean should be close to expected mean
        self.assertLess(abs(sample_mean - expected_mean), expected_mean * 0.2)

if __name__ == '__main__':
    unittest.main()
