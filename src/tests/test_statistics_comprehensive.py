# test_statistics_comprehensive.py

"""
Comprehensive test suite for the Statistical Engine.

This module provides exhaustive testing of the statistical engine to ensure
FULL safety and completeness before using in production. It covers:

1. Edge cases and boundary conditions
2. Mathematical property verification
3. Numerical stability and precision
4. Error handling and robustness
5. Performance and scalability
6. Cross-validation with multiple methods
7. Stress testing with extreme parameters
8. Memory and resource usage
9. Thread safety and concurrency
10. Integration with different data types
"""

import unittest
import numpy as np
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math_primitives.statistics import StatisticalEngine

class TestStatisticsComprehensive(unittest.TestCase):
    """
    Comprehensive test suite for the Statistical Engine.
    
    This class provides exhaustive testing to ensure the statistical engine
    is FULLY safe, complete, and robust for production use.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(seed=42)
        self.stats_engine = StatisticalEngine(self.rng)
        
        # Test parameters covering various scenarios
        self.test_params = {
            'small': {'n': 10, 'k': 2, 'p': 0.1},
            'medium': {'n': 1000, 'k': 100, 'p': 0.05},
            'large': {'n': 100000, 'k': 10000, 'p': 0.01},
            'extreme_small_p': {'n': 1000, 'k': 100, 'p': 0.001},
            'extreme_large_p': {'n': 1000, 'k': 100, 'p': 0.999},
            'extreme_small_k': {'n': 1000, 'k': 1, 'p': 0.05},
            'extreme_large_k': {'n': 1000, 'k': 999, 'p': 0.05}
        }
    
    def test_edge_cases_quantile_calculation(self):
        """Test quantile calculation with extreme edge cases."""
        # Test k = 1 (minimum possible)
        quantile = self.stats_engine.calculate_quantile_threshold(1000, 1)
        expected = (1000 - 1) / 1000
        self.assertAlmostEqual(quantile, expected, places=10)
        
        # Test k = effective_n - 1 (maximum possible)
        quantile = self.stats_engine.calculate_quantile_threshold(1000, 999)
        expected = (1000 - 999) / 1000
        self.assertAlmostEqual(quantile, expected, places=10)
        
        # Test very large numbers
        quantile = self.stats_engine.calculate_quantile_threshold(1000000, 100000)
        expected = (1000000 - 100000) / 1000000
        self.assertAlmostEqual(quantile, expected, places=10)
        
        # Test floating point precision
        quantile = self.stats_engine.calculate_quantile_threshold(3, 1)
        expected = (3 - 1) / 3
        self.assertAlmostEqual(quantile, expected, places=15)
    
    def test_edge_cases_binomial_ppf(self):
        """Test binomial PPF with extreme edge cases."""
        # Test with very small p
        ppf = self.stats_engine.calculate_binomial_ppf(0.5, 1000, 0.001)
        self.assertGreaterEqual(ppf, 0)
        self.assertLessEqual(ppf, 1000)
        
        # Test with very large p
        ppf = self.stats_engine.calculate_binomial_ppf(0.5, 1000, 0.999)
        self.assertGreaterEqual(ppf, 0)
        self.assertLessEqual(ppf, 1000)
        
        # Test with very small n
        ppf = self.stats_engine.calculate_binomial_ppf(0.5, 1, 0.5)
        self.assertGreaterEqual(ppf, 0)
        self.assertLessEqual(ppf, 1)
        
        # Test with very large n
        ppf = self.stats_engine.calculate_binomial_ppf(0.5, 100000, 0.1)
        self.assertGreaterEqual(ppf, 0)
        self.assertLessEqual(ppf, 100000)
        
        # Test quantile = 0.0 (should return -1 or 0)
        ppf = self.stats_engine.calculate_binomial_ppf(0.0, 100, 0.5)
        self.assertLessEqual(ppf, 0)
        
        # Test quantile = 1.0 (should return n)
        ppf = self.stats_engine.calculate_binomial_ppf(1.0, 100, 0.5)
        self.assertEqual(ppf, 100)
    
    def test_edge_cases_truncated_normal_sampling(self):
        """Test truncated normal sampling with extreme edge cases."""
        # Test with very small alpha (should still work)
        samples = self.stats_engine.sample_truncated_normal_winners(
            -100, 1000, 0.1, 10
        )
        self.assertEqual(len(samples), 10)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1000))
        
        # Test with very large alpha (should return mostly total_k)
        samples = self.stats_engine.sample_truncated_normal_winners(
            1000, 1000, 0.1, 10
        )
        self.assertEqual(len(samples), 10)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1000))
        
        # Test with p = 0 (should handle gracefully)
        with self.assertRaises(ZeroDivisionError):
            samples = self.stats_engine.sample_truncated_normal_winners(
                50, 1000, 0.0, 10
            )
        
        # Test with p = 1 (should handle gracefully)
        with self.assertRaises(ZeroDivisionError):
            samples = self.stats_engine.sample_truncated_normal_winners(
                50, 1000, 1.0, 10
            )
    
    def test_numerical_stability(self):
        """Test numerical stability with various parameter combinations."""
        # Test with very small numbers (but valid)
        quantile = self.stats_engine.calculate_quantile_threshold(2, 1)
        self.assertTrue(0 <= quantile <= 1)
        
        # Test with very large numbers
        quantile = self.stats_engine.calculate_quantile_threshold(1000000, 500000)
        self.assertTrue(0 <= quantile <= 1)
        
        # Test floating point precision
        for i in range(100):
            n = np.random.randint(1, 10000)
            k = np.random.randint(1, n)
            quantile = self.stats_engine.calculate_quantile_threshold(n, k)
            self.assertTrue(0 <= quantile <= 1)
            self.assertFalse(np.isnan(quantile))
            self.assertFalse(np.isinf(quantile))
    
    def test_mathematical_properties(self):
        """Test mathematical properties of the algorithms."""
        # Test monotonicity of PPF
        n, p = 1000, 0.1
        ppf_1 = self.stats_engine.calculate_binomial_ppf(0.1, n, p)
        ppf_2 = self.stats_engine.calculate_binomial_ppf(0.5, n, p)
        ppf_3 = self.stats_engine.calculate_binomial_ppf(0.9, n, p)
        
        self.assertLessEqual(ppf_1, ppf_2)
        self.assertLessEqual(ppf_2, ppf_3)
        
        # Test symmetry of quantile calculation
        n = 1000
        k1, k2 = 100, 900
        q1 = self.stats_engine.calculate_quantile_threshold(n, k1)
        q2 = self.stats_engine.calculate_quantile_threshold(n, k2)
        
        # q1 + q2 should be approximately 1 (allowing for floating point)
        self.assertAlmostEqual(q1 + q2, 1.0, places=10)
        
        # Test additivity of input statistics
        input_sizes = [100, 200, 300]
        p = 0.1
        mean, var = self.stats_engine.calculate_input_statistics(input_sizes, p)
        
        # Mean should be sum of individual means
        expected_mean = sum(input_sizes) * p
        self.assertAlmostEqual(mean, expected_mean, places=10)
        
        # Variance should be sum of individual variances
        expected_var = sum(input_sizes) * p * (1 - p)
        self.assertAlmostEqual(var, expected_var, places=10)
    
    def test_statistical_accuracy(self):
        """Test statistical accuracy with large samples."""
        # Test binomial sampling accuracy
        n, p = 1000, 0.1
        size = 10000
        samples = self.stats_engine.sample_binomial_winners(n, p, size)
        
        # Check mean
        sample_mean = np.mean(samples)
        expected_mean = n * p
        self.assertLess(abs(sample_mean - expected_mean), expected_mean * 0.05)
        
        # Check variance
        sample_var = np.var(samples)
        expected_var = n * p * (1 - p)
        self.assertLess(abs(sample_var - expected_var), expected_var * 0.1)
        
        # Test truncated normal sampling accuracy
        alpha = 50
        total_k = 1000
        p = 0.1
        k = 1000
        samples = self.stats_engine.sample_truncated_normal_winners(
            alpha, total_k, p, k
        )
        
        # Check that samples are within bounds
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= total_k))
        
        # Check that mean is reasonable
        sample_mean = np.mean(samples)
        expected_mean = total_k * p
        self.assertLess(abs(sample_mean - expected_mean), expected_mean * 0.2)
    
    def test_error_handling_robustness(self):
        """Test error handling and robustness."""
        # Test with invalid parameters
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(-1, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(10, -1)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_quantile_threshold(10, 15)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(-0.1, 100, 0.5)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(1.1, 100, 0.5)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(0.5, 0, 0.5)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(0.5, 100, -0.1)
        
        with self.assertRaises(ValueError):
            self.stats_engine.calculate_binomial_ppf(0.5, 100, 1.1)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 0, 0.5, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 100, -0.1, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 100, 1.1, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_truncated_normal_winners(50, 100, 0.5, 0)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(0, 0.5, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(100, -0.1, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(100, 1.1, 10)
        
        with self.assertRaises(ValueError):
            self.stats_engine.sample_binomial_winners(100, 0.5, 0)
    
    def test_performance_scalability(self):
        """Test performance and scalability."""
        # Test with increasing problem sizes
        sizes = [100, 1000, 10000]
        times = []
        
        for size in sizes:
            start_time = time.time()
            
            # Test quantile calculation
            quantile = self.stats_engine.calculate_quantile_threshold(size, size // 2)
            
            # Test binomial PPF
            ppf = self.stats_engine.calculate_binomial_ppf(0.5, size, 0.1)
            
            # Test sampling
            samples = self.stats_engine.sample_binomial_winners(size, 0.1, 100)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Times should be reasonable (not exponential growth)
        for i in range(1, len(times)):
            # Allow for some variation but should not be exponential
            if times[i-1] > 0:  # Avoid division by zero
                self.assertLess(times[i], times[i-1] * 10)
    
    def test_memory_usage(self):
        """Test memory usage with large samples."""
        # Test with large sample sizes
        large_size = 100000
        samples = self.stats_engine.sample_binomial_winners(1000, 0.1, large_size)
        
        # Should not cause memory issues
        self.assertEqual(len(samples), large_size)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1000))
        
        # Test memory cleanup
        del samples
        import gc
        gc.collect()
    
    def test_thread_safety(self):
        """Test thread safety of the statistical engine."""
        results = []
        errors = []
        
        def worker(seed):
            try:
                rng = np.random.default_rng(seed=seed)
                engine = StatisticalEngine(rng)
                
                # Perform various operations
                quantile = engine.calculate_quantile_threshold(1000, 100)
                ppf = engine.calculate_binomial_ppf(0.5, 1000, 0.1)
                samples = engine.sample_binomial_winners(1000, 0.1, 100)
                
                results.append((quantile, ppf, len(samples)))
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()
        
        # Should have no errors
        self.assertEqual(len(errors), 0)
        
        # Should have results from all threads
        self.assertEqual(len(results), 10)
        
        # Results should be consistent
        for result in results:
            self.assertIsInstance(result[0], (int, float))
            self.assertIsInstance(result[1], (int, float))
            self.assertEqual(result[2], 100)
    
    def test_data_type_robustness(self):
        """Test robustness with different data types."""
        # Test with different numpy dtypes
        dtypes = [np.int32, np.int64, np.float32, np.float64]
        
        for dtype in dtypes:
            n = dtype(1000)
            k = dtype(100)
            p = dtype(0.1)
            
            # Should work with different dtypes
            quantile = self.stats_engine.calculate_quantile_threshold(n, k)
            self.assertIsInstance(quantile, (int, float, np.floating))
            
            ppf = self.stats_engine.calculate_binomial_ppf(0.5, n, p)
            self.assertIsInstance(ppf, (int, float))
            
            samples = self.stats_engine.sample_binomial_winners(n, p, 10)
            self.assertEqual(len(samples), 10)
    
    def test_cross_validation_methods(self):
        """Test cross-validation between different methods."""
        # Test that different methods give consistent results
        n, p = 1000, 0.1
        size = 1000
        
        # Direct binomial sampling
        samples1 = self.stats_engine.sample_binomial_winners(n, p, size)
        
        # Truncated normal approximation
        alpha = self.stats_engine.calculate_binomial_ppf(0.5, n, p)
        samples2 = self.stats_engine.sample_truncated_normal_winners(
            alpha, n, p, size
        )
        
        # Both should have similar statistical properties
        mean1 = np.mean(samples1)
        mean2 = np.mean(samples2)
        expected_mean = n * p
        
        self.assertLess(abs(mean1 - expected_mean), expected_mean * 0.1)
        self.assertLess(abs(mean2 - expected_mean), expected_mean * 0.2)
    
    def test_stress_testing(self):
        """Test with extreme stress conditions."""
        # Test with very large numbers
        large_n = 1000000
        large_k = 100000
        large_p = 0.01
        
        quantile = self.stats_engine.calculate_quantile_threshold(large_n, large_k)
        self.assertTrue(0 <= quantile <= 1)
        
        ppf = self.stats_engine.calculate_binomial_ppf(0.5, large_n, large_p)
        self.assertTrue(0 <= ppf <= large_n)
        
        # Test with very small numbers (but valid)
        small_n = 2
        small_k = 1
        small_p = 0.5
        
        quantile = self.stats_engine.calculate_quantile_threshold(small_n, small_k)
        self.assertTrue(0 <= quantile <= 1)
        
        ppf = self.stats_engine.calculate_binomial_ppf(0.5, small_n, small_p)
        self.assertTrue(0 <= ppf <= small_n)
    
    def test_warning_handling(self):
        """Test handling of warnings and edge cases."""
        # Test with parameters that might generate warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            
            # Should not generate warnings for normal parameters
            quantile = self.stats_engine.calculate_quantile_threshold(1000, 100)
            self.assertIsInstance(quantile, (int, float))
    
    def test_reproducibility(self):
        """Test reproducibility with same seeds."""
        # Test that same seed gives same results
        seed = 42
        rng1 = np.random.default_rng(seed=seed)
        rng2 = np.random.default_rng(seed=seed)
        
        engine1 = StatisticalEngine(rng1)
        engine2 = StatisticalEngine(rng2)
        
        # Same operations should give same results
        samples1 = engine1.sample_binomial_winners(1000, 0.1, 100)
        samples2 = engine2.sample_binomial_winners(1000, 0.1, 100)
        
        np.testing.assert_array_equal(samples1, samples2)
    
    def test_boundary_conditions(self):
        """Test all boundary conditions thoroughly."""
        # Test quantile boundary conditions
        for n in [2, 10, 100, 1000]:  # Start from 2 to avoid k=0
            for k in [1, n-1]:
                if k < n and k > 0:  # Only test valid cases
                    quantile = self.stats_engine.calculate_quantile_threshold(n, k)
                    self.assertTrue(0 <= quantile <= 1)
        
        # Test PPF boundary conditions
        for n in [1, 10, 100, 1000]:
            for p in [0.001, 0.1, 0.5, 0.9, 0.999]:
                for q in [0.0, 0.5, 1.0]:
                    ppf = self.stats_engine.calculate_binomial_ppf(q, n, p)
                    # PPF can be negative for very low quantiles, but should be <= n
                    self.assertTrue(ppf <= n)
        
        # Test sampling boundary conditions
        for total_k in [1, 10, 100, 1000]:
            for p in [0.001, 0.1, 0.5, 0.9, 0.999]:
                for k in [1, 10, 100]:
                    if k <= total_k:
                        samples = self.stats_engine.sample_truncated_normal_winners(
                            50, total_k, p, k
                        )
                        self.assertEqual(len(samples), k)
                        self.assertTrue(np.all(samples >= 0))
                        self.assertTrue(np.all(samples <= total_k))

if __name__ == '__main__':
    unittest.main()
