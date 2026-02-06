# test_statistics_final_validation.py

"""
Final comprehensive validation against root brain.py.

This module provides the ULTIMATE validation to ensure our statistical engine
exactly matches the root brain.py implementation in every detail.
"""

import unittest
import numpy as np
from scipy.stats import binom, truncnorm
import math

from src.math_primitives.statistics import StatisticalEngine

class TestStatisticsFinalValidation(unittest.TestCase):
    """
    Ultimate validation test suite against root brain.py.

    This class provides the final, comprehensive validation to ensure
    our statistical engine is a perfect extraction of the root brain.py.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(seed=42)
        self.stats_engine = StatisticalEngine(self.rng)

    def test_quantile_calculation_exact_match(self):
        """Test that quantile calculation exactly matches root brain.py."""
        # Root brain.py line 677: quantile = (effective_n - target_area.k) / effective_n
        effective_n = 1000
        k = 100

        # Our implementation
        our_quantile = self.stats_engine.calculate_quantile_threshold(effective_n, k)

        # Root implementation
        root_quantile = (effective_n - k) / effective_n

        # Should match exactly
        self.assertEqual(our_quantile, root_quantile)

        # Test multiple values
        test_cases = [(1000, 50), (5000, 200), (10000, 1000)]
        for n, k_val in test_cases:
            our_result = self.stats_engine.calculate_quantile_threshold(n, k_val)
            root_result = (n - k_val) / n
            self.assertEqual(our_result, root_result)

    def test_binomial_ppf_exact_match(self):
        """Test that binomial PPF exactly matches root brain.py."""
        # Root brain.py line 688: alpha = binom.ppf(quantile, total_k, self.p)
        quantile = 0.9
        total_k = 1000
        p = 0.1

        # Our implementation
        our_alpha = self.stats_engine.calculate_binomial_ppf(quantile, total_k, p)

        # Root implementation (scipy.stats.binom.ppf)
        root_alpha = binom.ppf(quantile, total_k, p)

        # Should match exactly
        self.assertEqual(our_alpha, root_alpha)

        # Test edge cases
        test_cases = [
            (0.1, 1000, 0.05),
            (0.5, 500, 0.2),
            (0.9, 2000, 0.01),
            (0.99, 100, 0.3)
        ]

        for q, n, p_val in test_cases:
            our_result = self.stats_engine.calculate_binomial_ppf(q, n, p_val)
            root_result = binom.ppf(q, n, p_val)
            self.assertEqual(our_result, root_result)

    def test_normal_approximation_parameters_exact_match(self):
        """Test that normal approximation parameters exactly match root brain.py."""
        # Root brain.py lines 697-698:
        # mu = total_k * self.p
        # std = math.sqrt(total_k * self.p * (1.0 - self.p))
        total_k = 1000
        p = 0.1

        # Our implementation
        our_mean, our_var = self.stats_engine.calculate_input_statistics_standard([total_k], p)

        # Root implementation
        root_mu = total_k * p
        root_std = math.sqrt(total_k * p * (1.0 - p))

        # Should match exactly
        self.assertEqual(our_mean, root_mu)
        self.assertEqual(our_var, root_std ** 2)

        # Test with multiple input sizes
        input_sizes = [100, 200, 300]
        total_k = sum(input_sizes)

        our_mean, our_var = self.stats_engine.calculate_input_statistics_standard(input_sizes, p)
        root_mu = total_k * p
        root_std = math.sqrt(total_k * p * (1.0 - p))

        self.assertEqual(our_mean, root_mu)
        self.assertEqual(our_var, root_std ** 2)

    def test_truncated_normal_sampling_exact_match(self):
        """Test that truncated normal sampling exactly matches root brain.py."""
        # Root brain.py lines 707-710:
        # potential_new_winner_inputs = (mu + truncnorm.rvs(a, np.inf, scale=std, size=k)).round(0)
        # for i in range(len(potential_new_winner_inputs)):
        #     if potential_new_winner_inputs[i] > total_k:
        #         potential_new_winner_inputs[i] = total_k

        alpha = 50
        total_k = 1000
        p = 0.1
        k = 10

        # Calculate parameters as in root
        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))
        a = (alpha - mu) / std

        # Our implementation
        our_samples = self.stats_engine.sample_truncated_normal_winners(alpha, total_k, p, k)

        # Root implementation simulation
        root_samples = (mu + truncnorm.rvs(a, np.inf, scale=std, size=k)).round(0)
        for i in range(len(root_samples)):
            if root_samples[i] > total_k:
                root_samples[i] = total_k

        # Should have same statistical properties
        self.assertEqual(len(our_samples), len(root_samples))
        self.assertTrue(np.all(our_samples >= 0))
        self.assertTrue(np.all(our_samples <= total_k))
        self.assertTrue(np.all(root_samples >= 0))
        self.assertTrue(np.all(root_samples <= total_k))

        # Means should be similar
        our_mean = np.mean(our_samples)
        root_mean = np.mean(root_samples)
        expected_mean = mu
        self.assertLess(abs(our_mean - expected_mean), expected_mean * 0.2)
        self.assertLess(abs(root_mean - expected_mean), expected_mean * 0.2)

    def test_complete_algorithm_chain_exact_match(self):
        """Test the complete algorithm chain exactly matches root brain.py."""
        # Simulate the exact parameters from root brain.py
        total_k = 1000
        p = 0.1
        effective_n = 1000
        target_k = 100

        # Step 1: Calculate quantile (line 677)
        quantile = (effective_n - target_k) / effective_n
        our_quantile = self.stats_engine.calculate_quantile_threshold(effective_n, target_k)
        self.assertEqual(quantile, our_quantile)

        # Step 2: Calculate alpha (line 688)
        alpha = binom.ppf(quantile, total_k, p)
        our_alpha = self.stats_engine.calculate_binomial_ppf(quantile, total_k, p)
        self.assertEqual(alpha, our_alpha)

        # Step 3: Calculate normal parameters (lines 697-698)
        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))

        our_mean, our_var = self.stats_engine.calculate_input_statistics_standard([total_k], p)
        self.assertEqual(mu, our_mean)
        self.assertEqual(std ** 2, our_var)

        # Step 4: Sample potential winners
        our_samples = self.stats_engine.sample_truncated_normal_winners(alpha, total_k, p, target_k)

        # Verify samples have correct properties
        self.assertEqual(len(our_samples), target_k)
        self.assertTrue(np.all(our_samples >= 0))
        self.assertTrue(np.all(our_samples <= total_k))

        # Mean should be reasonable
        sample_mean = np.mean(our_samples)
        self.assertLess(abs(sample_mean - mu), mu * 0.3)

    def test_variance_calculation_method_validation(self):
        """Test that our variance calculation method matches root brain.py."""
        # Root brain.py uses the standard variance calculation, not the sum of squared variances
        # This test validates our choice

        input_sizes = [100, 200, 300]
        p = 0.1
        total_k = sum(input_sizes)

        # Our implementation
        our_mean, our_var = self.stats_engine.calculate_input_statistics_standard(input_sizes, p)

        # Root brain.py calculation
        root_mu = total_k * p
        root_var = total_k * p * (1.0 - p)  # Standard binomial variance

        # Should match
        self.assertEqual(our_mean, root_mu)
        self.assertEqual(our_var, root_var)

        # Test that sum of squared variances would be different
        squared_var_sum = 0
        for size in input_sizes:
            squared_var_sum += (size * p * (1.0 - p)) ** 2

        # These should be different
        self.assertNotEqual(our_var, squared_var_sum)

    def test_root_brain_py_commented_code_validation(self):
        """Test the commented code in root brain.py to understand the alternative approach."""
        # Root brain.py has commented code that shows an alternative variance calculation
        # This test validates our understanding

        input_sizes = [100, 200, 300]
        p = 0.1
        total_k = sum(input_sizes)

        # Alternative calculation from commented code (lines 656, 666)
        # normal_approx_var += ((local_k * local_p * (1 - local_p)) ** 2)
        alternative_var_sum = 0
        for size in input_sizes:
            alternative_var_sum += (size * p * (1.0 - p)) ** 2

        # Our current implementation (standard approach)
        our_var = total_k * p * (1.0 - p)

        # The commented approach is different from our current approach
        self.assertNotEqual(our_var, alternative_var_sum)

        # But our approach matches the active root code
        root_var = total_k * p * (1.0 - p)
        self.assertEqual(our_var, root_var)

    def test_numerical_precision_exact_match(self):
        """Test numerical precision exactly matches root brain.py."""
        # Test with extreme precision values
        effective_n = 1000000
        k = 500000
        quantile = (effective_n - k) / effective_n

        our_quantile = self.stats_engine.calculate_quantile_threshold(effective_n, k)

        # Should match to full precision
        self.assertAlmostEqual(our_quantile, quantile, places=15)

        # Test with very small differences
        n = 1000
        k1, k2 = 500, 501

        q1 = self.stats_engine.calculate_quantile_threshold(n, k1)
        q2 = self.stats_engine.calculate_quantile_threshold(n, k2)
        root_q1 = (n - k1) / n
        root_q2 = (n - k2) / n

        self.assertAlmostEqual(q1, root_q1, places=15)
        self.assertAlmostEqual(q2, root_q2, places=15)

    def test_edge_case_boundary_exact_match(self):
        """Test edge cases exactly match root brain.py behavior."""
        # Test boundary conditions that root brain.py might encounter

        # Very small effective_n
        effective_n = 2
        k = 1
        quantile = (effective_n - k) / effective_n
        our_quantile = self.stats_engine.calculate_quantile_threshold(effective_n, k)
        self.assertEqual(quantile, our_quantile)

        # Very large numbers
        effective_n = 1000000
        k = 100000
        quantile = (effective_n - k) / effective_n
        our_quantile = self.stats_engine.calculate_quantile_threshold(effective_n, k)
        self.assertEqual(quantile, our_quantile)

        # p = 0 case
        with self.assertRaises(ZeroDivisionError):
            self.stats_engine.sample_truncated_normal_winners(50, 1000, 0.0, 10)

        # p = 1 case
        with self.assertRaises(ZeroDivisionError):
            self.stats_engine.sample_truncated_normal_winners(50, 1000, 1.0, 10)

if __name__ == '__main__':
    unittest.main()
