# test_statistics_performance.py

"""
Performance and stress tests for the Statistical Engine.

This module provides comprehensive performance testing to ensure the statistical
engine can handle production workloads efficiently and reliably.
"""

import unittest
import numpy as np
import time
import gc
from concurrent.futures import ThreadPoolExecutor

# Try to import psutil, but don't fail if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.math_primitives.statistics import StatisticalEngine

class TestStatisticsPerformance(unittest.TestCase):
    """
    Performance and stress tests for the Statistical Engine.
    
    This class provides comprehensive performance testing to ensure
    the statistical engine can handle production workloads.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(seed=42)
        self.stats_engine = StatisticalEngine(self.rng)
        
        # Performance test parameters
        self.performance_params = {
            'small': {'n': 100, 'k': 10, 'p': 0.1, 'size': 1000},
            'medium': {'n': 10000, 'k': 1000, 'p': 0.05, 'size': 10000},
            'large': {'n': 100000, 'k': 10000, 'p': 0.01, 'size': 100000},
            'extreme': {'n': 1000000, 'k': 100000, 'p': 0.001, 'size': 1000000}
        }
    
    def test_quantile_calculation_performance(self):
        """Test performance of quantile calculation."""
        # Test with increasing problem sizes
        sizes = [100, 1000, 10000, 100000]
        times = []
        
        for n in sizes:
            k = n // 2
            
            start_time = time.time()
            for _ in range(1000):  # Repeat for accurate timing
                quantile = self.stats_engine.calculate_quantile_threshold(n, k)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Should be O(1) - constant time
        for i in range(1, len(times)):
            # Times should be roughly constant (within 2x variation)
            if times[0] > 0:  # Avoid division by zero
                self.assertLess(times[i], times[0] * 2)
    
    def test_binomial_ppf_performance(self):
        """Test performance of binomial PPF calculation."""
        # Test with increasing problem sizes
        sizes = [100, 1000, 10000, 100000]
        times = []
        
        for n in sizes:
            p = 0.1
            
            start_time = time.time()
            for _ in range(100):  # Repeat for accurate timing
                ppf = self.stats_engine.calculate_binomial_ppf(0.5, n, p)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Should be reasonable performance
        for i in range(len(times)):
            self.assertLess(times[i], 1.0)  # Should complete within 1 second
    
    def test_sampling_performance(self):
        """Test performance of sampling operations."""
        # Test binomial sampling performance
        n, p = 10000, 0.1
        sizes = [1000, 10000, 100000, 1000000]
        
        for size in sizes:
            start_time = time.time()
            samples = self.stats_engine.sample_binomial_winners(n, p, size)
            end_time = time.time()
            
            elapsed = end_time - start_time
            self.assertLess(elapsed, 5.0)  # Should complete within 5 seconds
            self.assertEqual(len(samples), size)
    
    def test_truncated_normal_performance(self):
        """Test performance of truncated normal sampling."""
        alpha = 50
        total_k = 10000
        p = 0.1
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            start_time = time.time()
            samples = self.stats_engine.sample_truncated_normal_winners(
                alpha, total_k, p, size
            )
            end_time = time.time()
            
            elapsed = end_time - start_time
            self.assertLess(elapsed, 5.0)  # Should complete within 5 seconds
            self.assertEqual(len(samples), size)
    
    def test_memory_usage(self):
        """Test memory usage with large samples."""
        # Test memory usage with large samples
        n = 100000
        p = 0.1
        size = 1000000
        
        # Generate large sample
        samples = self.stats_engine.sample_binomial_winners(n, p, size)
        
        # Basic memory test - should not crash
        self.assertEqual(len(samples), size)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= n))
        
        # If psutil is available, test actual memory usage
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate large sample
            samples = self.stats_engine.sample_binomial_winners(n, p, size)
            
            # Get memory usage after sampling
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Memory increase should be reasonable (less than 1GB)
            self.assertLess(memory_increase, 1000)
        
        # Clean up
        del samples
        gc.collect()
    
    def test_concurrent_performance(self):
        """Test performance under concurrent access."""
        def worker(seed):
            rng = np.random.default_rng(seed=seed)
            engine = StatisticalEngine(rng)
            
            # Perform various operations
            quantile = engine.calculate_quantile_threshold(1000, 100)
            ppf = engine.calculate_binomial_ppf(0.5, 1000, 0.1)
            samples = engine.sample_binomial_winners(1000, 0.1, 1000)
            
            return len(samples)
        
        # Test with multiple threads
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(100)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(elapsed, 10.0)
        
        # All results should be correct
        self.assertEqual(len(results), 100)
        self.assertTrue(all(r == 1000 for r in results))
    
    def test_stress_testing(self):
        """Test under stress conditions."""
        # Test with maximum reasonable parameters
        n = 1000000
        k = 100000
        p = 0.01
        
        # Should not crash or hang
        quantile = self.stats_engine.calculate_quantile_threshold(n, k)
        self.assertTrue(0 <= quantile <= 1)
        
        ppf = self.stats_engine.calculate_binomial_ppf(0.5, n, p)
        self.assertTrue(0 <= ppf <= n)
        
        # Test with many small operations
        for _ in range(1000):
            samples = self.stats_engine.sample_binomial_winners(100, 0.1, 10)
            self.assertEqual(len(samples), 10)
    
    def test_scalability(self):
        """Test scalability with increasing problem sizes."""
        # Test quantile calculation scalability
        sizes = [100, 1000, 10000, 100000]
        times = []
        
        for n in sizes:
            k = n // 2
            
            start_time = time.time()
            quantile = self.stats_engine.calculate_quantile_threshold(n, k)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Should scale well (not exponential)
        for i in range(1, len(times)):
            if times[i-1] > 0:  # Avoid division by zero
                self.assertLess(times[i], times[i-1] * 10)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Test that memory usage is reasonable
        n = 100000
        p = 0.1
        size = 1000000
        
        # Generate large sample
        samples = self.stats_engine.sample_binomial_winners(n, p, size)
        
        # Basic memory test - should not crash
        self.assertEqual(len(samples), size)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= n))
        
        # If psutil is available, test actual memory usage
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Generate large sample
            samples = self.stats_engine.sample_binomial_winners(n, p, size)
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = peak_memory - initial_memory
            
            # Memory usage should be reasonable
            expected_memory = size * 8 / 1024 / 1024  # 8 bytes per int64
            self.assertLess(memory_usage, expected_memory * 2)  # Allow 2x overhead
        
        # Clean up
        del samples
        gc.collect()
    
    def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        # Test with valid parameters
        start_time = time.time()
        for _ in range(1000):
            try:
                quantile = self.stats_engine.calculate_quantile_threshold(1000, 100)
            except Exception:
                pass
        valid_time = time.time() - start_time
        
        # Test with invalid parameters (should fail fast)
        start_time = time.time()
        for _ in range(1000):
            try:
                quantile = self.stats_engine.calculate_quantile_threshold(-1, 100)
            except Exception:
                pass
        invalid_time = time.time() - start_time
        
        # Error handling should not be significantly slower
        if valid_time > 0:  # Avoid division by zero
            self.assertLess(invalid_time, valid_time * 2)
    
    def test_batch_processing_performance(self):
        """Test performance of batch processing operations."""
        # Test processing many small operations vs one large operation
        n, p = 1000, 0.1
        
        # Many small operations
        start_time = time.time()
        for _ in range(100):
            samples = self.stats_engine.sample_binomial_winners(n, p, 100)
        small_ops_time = time.time() - start_time
        
        # One large operation
        start_time = time.time()
        samples = self.stats_engine.sample_binomial_winners(n, p, 10000)
        large_op_time = time.time() - start_time
        
        # Large operation should be reasonably efficient (allow some variance)
        self.assertLess(large_op_time, small_ops_time * 1.5)  # Allow 50% overhead
    
    def test_parameter_sensitivity(self):
        """Test performance sensitivity to different parameters."""
        # Test with different p values
        n = 10000
        p_values = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
        times = []
        
        for p in p_values:
            start_time = time.time()
            samples = self.stats_engine.sample_binomial_winners(n, p, 10000)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Performance should not vary dramatically with p
        max_time = max(times)
        min_time = min(times)
        if min_time > 0:  # Avoid division by zero
            self.assertLess(max_time, min_time * 5)  # Should not vary by more than 5x
    
    def test_reproducibility_performance(self):
        """Test that reproducibility doesn't significantly impact performance."""
        # Test with same seed (reproducible)
        start_time = time.time()
        for _ in range(100):
            rng = np.random.default_rng(seed=42)
            engine = StatisticalEngine(rng)
            samples = engine.sample_binomial_winners(1000, 0.1, 100)
        reproducible_time = time.time() - start_time
        
        # Test with different seeds (non-reproducible)
        start_time = time.time()
        for i in range(100):
            rng = np.random.default_rng(seed=i)
            engine = StatisticalEngine(rng)
            samples = engine.sample_binomial_winners(1000, 0.1, 100)
        non_reproducible_time = time.time() - start_time
        
        # Performance should be similar (allow some variance)
        self.assertLess(abs(reproducible_time - non_reproducible_time),
                       max(reproducible_time, non_reproducible_time) * 0.8)  # Allow 80% difference

if __name__ == '__main__':
    unittest.main()
