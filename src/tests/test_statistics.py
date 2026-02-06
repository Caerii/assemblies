# test_statistics.py

"""
Comprehensive tests for the Statistical Engine.

This module tests the complex statistical sampling algorithms extracted
from the root brain.py, ensuring they produce correct and consistent results.
"""

import unittest
import numpy as np

from src.math_primitives.statistics import StatisticalEngine

class TestStatisticalEngine(unittest.TestCase):
    """
    Test suite for the Statistical Engine.
    
    Tests the complex statistical sampling algorithms including:
    - Quantile threshold calculations
    - Binomial PPF computations
    - Truncated normal sampling
    - Parameter validation
    - Statistical accuracy
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(seed=42)
        self.stats_engine = StatisticalEngine(self.rng)
    
    def test_quantile_threshold_calculation(self):
        """Test quantile threshold calculation."""
        # Test basic case
        effective_n = 1000
        k = 100
        quantile = self.stats_engine.calculate_quantile_threshold(effective_n, k)
        expected = (1000 - 100) / 1000  # 0.9
        self.assertAlmostEqual(quantile, expected, places=6)
        
        # Test edge case: k = 1
        quantile = self.stats_engine.calculate_quantile_threshold(100, 1)
        expected = (100 - 1) / 100  # 0.99
        self.assertAlmostEqual(quantile, expected, places=6)
        
        # Test edge case: k = effective_n - 1
        quantile = self.stats_engine.calculate_quantile_threshold(100, 99)
        expected = (100 - 99) / 100  # 0.01
        self.assertAlmostEqual(quantile, expected, places=6)
    
    def test_quantile_threshold_validation(self):
        """Test quantile threshold parameter validation."""
        # Test invalid effective_n
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(0, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(-10, 10)
        
        # Test invalid k
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(100, 0)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(100, -10)
        
        # Test k >= effective_n
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(100, 100)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(100, 150)
    
    def test_binomial_ppf_calculation(self):
        """Test binomial PPF calculation."""
        # Test basic case
        quantile = 0.9
        n = 1000
        p = 0.05
        ppf = self.stats_engine.calculate_binomial_ppf(quantile, n, p)
        
        # PPF should be a non-negative number
        self.assertGreaterEqual(ppf, 0)
        self.assertIsInstance(ppf, (int, float))
        
        # Test edge cases
        ppf_0 = self.stats_engine.calculate_binomial_ppf(0.0, n, p)
        ppf_1 = self.stats_engine.calculate_binomial_ppf(1.0, n, p)
        
        # PPF at quantile 0.0 can be negative (represents impossible threshold)
        # PPF at quantile 1.0 should be the maximum possible value
        self.assertGreaterEqual(ppf_1, ppf_0)
        
        # Test that PPF increases with quantile
        ppf_05 = self.stats_engine.calculate_binomial_ppf(0.5, n, p)
        self.assertGreaterEqual(ppf_05, ppf_0)
        self.assertGreaterEqual(ppf_1, ppf_05)
    
    def test_binomial_ppf_validation(self):
        """Test binomial PPF parameter validation."""
        # Test invalid quantile
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(-0.1, 100, 0.5)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(1.1, 100, 0.5)
        
        # Test invalid n
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(0.5, 0, 0.5)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(0.5, -10, 0.5)
        
        # Test invalid p
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(0.5, 100, -0.1)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(0.5, 100, 1.1)
    
    def test_truncated_normal_sampling(self):
        """Test truncated normal sampling for new winners."""
        # Test basic case
        alpha = 50.0
        total_k = 1000
        p = 0.05
        k = 100
        
        samples = self.stats_engine.sample_truncated_normal_winners(
            alpha, total_k, p, k
        )
        
        # Check output properties
        self.assertEqual(len(samples), k)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= total_k))
        self.assertTrue(np.all(samples == samples.astype(int)))
        
        # Test statistical properties
        expected_mean = total_k * p
        sample_mean = np.mean(samples)
        
        # Sample mean should be reasonably close to expected mean
        # (allowing for statistical variation)
        self.assertLess(abs(sample_mean - expected_mean), expected_mean * 0.5)
    
    def test_truncated_normal_sampling_validation(self):
        """Test truncated normal sampling parameter validation."""
        # Test invalid total_k
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 0, 0.5, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, -10, 0.5, 10)
        
        # Test invalid p
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 100, -0.1, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 100, 1.1, 10)
        
        # Test invalid k
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 100, 0.5, 0)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 100, 0.5, -10)
    
    def test_binomial_sampling(self):
        """Test direct binomial sampling."""
        # Test basic case
        n = 1000
        p = 0.05
        size = 100
        
        samples = self.stats_engine.sample_binomial_winners(n, p, size)
        
        # Check output properties
        self.assertEqual(len(samples), size)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= n))
        self.assertTrue(np.all(samples == samples.astype(int)))
        
        # Test statistical properties
        expected_mean = n * p
        sample_mean = np.mean(samples)
        
        # Sample mean should be close to expected mean
        self.assertLess(abs(sample_mean - expected_mean), expected_mean * 0.3)
    
    def test_binomial_sampling_validation(self):
        """Test binomial sampling parameter validation."""
        # Test invalid n
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(0, 0.5, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(-10, 0.5, 10)
        
        # Test invalid p
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(100, -0.1, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(100, 1.1, 10)
        
        # Test invalid size
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(100, 0.5, 0)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(100, 0.5, -10)
    
    def test_input_statistics_calculation(self):
        """Test input statistics calculation."""
        # Test basic case
        input_sizes = [100, 200, 300]
        p = 0.05
        
        mean, variance = self.stats_engine.calculate_input_statistics(input_sizes, p)
        
        expected_mean = sum(input_sizes) * p  # 600 * 0.05 = 30
        expected_variance = sum(input_sizes) * p * (1 - p)  # 600 * 0.05 * 0.95 = 28.5
        
        self.assertAlmostEqual(mean, expected_mean, places=6)
        self.assertAlmostEqual(variance, expected_variance, places=6)
        
        # Test empty input sizes
        mean, variance = self.stats_engine.calculate_input_statistics([], p)
        self.assertEqual(mean, 0.0)
        self.assertEqual(variance, 0.0)
    
    def test_parameter_validation(self):
        """Test comprehensive parameter validation."""
        # Test valid parameters
        self.stats_engine.validate_sampling_parameters(1000, 100, 500, 0.05)
        
        # Test invalid effective_n
        with self.assertRaises(RuntimeError):
            self.stats_engine.validate_sampling_parameters(50, 100, 500, 0.05)
        
        # Test invalid total_k
        with self.assertRaises(ValueError):
            self.stats_engine.validate_sampling_parameters(1000, 100, 0, 0.05)
        
        # Test invalid p
        with self.assertRaises(ValueError):
            self.stats_engine.validate_sampling_parameters(1000, 100, 500, -0.1)
        
        with self.assertRaises(ValueError):
            self.stats_engine.validate_sampling_parameters(1000, 100, 500, 1.1)
        
        # Test invalid k
        with self.assertRaises(ValueError):
            self.stats_engine.validate_sampling_parameters(1000, 0, 500, 0.05)
    
    def test_sampling_method_info(self):
        """Test sampling method information."""
        info = self.stats_engine.get_sampling_method_info()
        
        # Check required keys
        required_keys = ['normal_ppf_enabled', 'methods_available', 
                        'approximation_accuracy', 'computational_complexity']
        for key in required_keys:
            self.assertIn(key, info)
        
        # Check methods available
        expected_methods = [
            'binomial_direct',
            'truncated_normal_approximation',
            'quantile_thresholding'
        ]
        for method in expected_methods:
            self.assertIn(method, info['methods_available'])
    
    def test_statistical_consistency(self):
        """Test statistical consistency across multiple samples."""
        # Test that multiple samples have consistent statistical properties
        alpha = 50.0
        total_k = 1000
        p = 0.05
        k = 100
        n_samples = 10
        
        all_samples = []
        for _ in range(n_samples):
            samples = self.stats_engine.sample_truncated_normal_winners(
                alpha, total_k, p, k
            )
            all_samples.append(samples)
        
        # Concatenate all samples
        combined_samples = np.concatenate(all_samples)
        
        # Check that the combined distribution has expected properties
        expected_mean = total_k * p
        sample_mean = np.mean(combined_samples)
        
        # With more samples, the mean should be closer to expected
        self.assertLess(abs(sample_mean - expected_mean), expected_mean * 0.2)
        
        # Check that all samples are within valid range
        self.assertTrue(np.all(combined_samples >= 0))
        self.assertTrue(np.all(combined_samples <= total_k))
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test very small k
        samples = self.stats_engine.sample_truncated_normal_winners(10, 100, 0.1, 1)
        self.assertEqual(len(samples), 1)
        self.assertTrue(0 <= samples[0] <= 100)
        
        # Test very small p
        samples = self.stats_engine.sample_truncated_normal_winners(5, 1000, 0.001, 10)
        self.assertEqual(len(samples), 10)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1000))
        
        # Test very large p
        samples = self.stats_engine.sample_truncated_normal_winners(50, 100, 0.9, 10)
        self.assertEqual(len(samples), 10)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 100))

if __name__ == '__main__':
    unittest.main()
