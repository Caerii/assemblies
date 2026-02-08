# test_statistics.py

"""
Comprehensive tests for the Statistical Engine.

This module tests the complex statistical sampling algorithms extracted
from the root brain.py, ensuring they produce correct and consistent results.

Test classes:
- TestStatisticalEngine: Core functionality and validation (16 tests)
- TestStatisticsEdgeCases: Thread safety, data types, boundary sweeps, stress testing
- TestStatisticsRootParity: Exact match against scipy binom.ppf and root brain.py chains
- TestStatisticsMathematical: Mathematical properties, convergence, distribution, KS test
"""

import math
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats
from scipy.stats import binom

from src.compute.statistics import StatisticalEngine


# ---------------------------------------------------------------------------
# 1. TestStatisticalEngine -- original 16 tests, unchanged
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. TestStatisticsEdgeCases -- unique tests from comprehensive
# ---------------------------------------------------------------------------

class TestStatisticsEdgeCases(unittest.TestCase):
    """
    Edge-case and robustness tests sourced from the comprehensive suite.

    Covers thread safety, data-type robustness, exhaustive boundary sweeps,
    numerical stability fuzz, cross-method validation, stress testing,
    warning handling, and seed reproducibility.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(seed=42)
        self.stats_engine = StatisticalEngine(self.rng)

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
        """Test robustness with different numpy data types."""
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

    def test_boundary_conditions(self):
        """Test all boundary conditions thoroughly."""
        # Quantile boundary sweep
        for n in [2, 10, 100, 1000]:
            for k in [1, n - 1]:
                if 0 < k < n:
                    quantile = self.stats_engine.calculate_quantile_threshold(n, k)
                    self.assertTrue(0 <= quantile <= 1)

        # PPF boundary sweep
        for n in [1, 10, 100, 1000]:
            for p in [0.001, 0.1, 0.5, 0.9, 0.999]:
                for q in [0.0, 0.5, 1.0]:
                    ppf = self.stats_engine.calculate_binomial_ppf(q, n, p)
                    self.assertTrue(ppf <= n)

        # Sampling boundary sweep
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

    def test_numerical_stability(self):
        """Test numerical stability with random parameter combinations."""
        rng = np.random.default_rng(seed=99)
        for _ in range(100):
            n = rng.integers(1, 10000)
            k = rng.integers(1, n)
            quantile = self.stats_engine.calculate_quantile_threshold(int(n), int(k))
            self.assertTrue(0 <= quantile <= 1)
            self.assertFalse(np.isnan(quantile))
            self.assertFalse(np.isinf(quantile))

    def test_cross_validation_methods(self):
        """Test cross-validation between sampling methods."""
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
        expected_mean = n * p
        self.assertLess(abs(np.mean(samples1) - expected_mean), expected_mean * 0.1)
        self.assertLess(abs(np.mean(samples2) - expected_mean), expected_mean * 0.2)

    def test_stress_testing(self):
        """Test with extreme stress conditions."""
        # Very large numbers
        large_n = 1000000
        large_k = 100000

        quantile = self.stats_engine.calculate_quantile_threshold(large_n, large_k)
        self.assertTrue(0 <= quantile <= 1)

        ppf = self.stats_engine.calculate_binomial_ppf(0.5, large_n, 0.01)
        self.assertTrue(0 <= ppf <= large_n)

        # Very small (but valid) numbers
        quantile = self.stats_engine.calculate_quantile_threshold(2, 1)
        self.assertTrue(0 <= quantile <= 1)

        ppf = self.stats_engine.calculate_binomial_ppf(0.5, 2, 0.5)
        self.assertTrue(0 <= ppf <= 2)

    def test_warning_handling(self):
        """Test that normal parameters produce no warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quantile = self.stats_engine.calculate_quantile_threshold(1000, 100)
            self.assertIsInstance(quantile, (int, float))

    def test_reproducibility(self):
        """Test reproducibility with identical seeds."""
        seed = 42
        rng1 = np.random.default_rng(seed=seed)
        rng2 = np.random.default_rng(seed=seed)

        engine1 = StatisticalEngine(rng1)
        engine2 = StatisticalEngine(rng2)

        samples1 = engine1.sample_binomial_winners(1000, 0.1, 100)
        samples2 = engine2.sample_binomial_winners(1000, 0.1, 100)

        np.testing.assert_array_equal(samples1, samples2)


# ---------------------------------------------------------------------------
# 3. TestStatisticsRootParity -- exact-match tests from integration and
#    final_validation
# ---------------------------------------------------------------------------

class TestStatisticsRootParity(unittest.TestCase):
    """
    Root-parity tests ensuring exact match against scipy binom.ppf and
    the original root brain.py algorithm chain.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(seed=42)
        self.stats_engine = StatisticalEngine(self.rng)

    def test_binomial_ppf_exact_match_scipy(self):
        """Test that binomial PPF exactly matches scipy.stats.binom.ppf."""
        test_cases = [
            (0.9, 1000, 0.1),
            (0.1, 1000, 0.05),
            (0.5, 500, 0.2),
            (0.9, 2000, 0.01),
            (0.99, 100, 0.3),
        ]

        for q, n, p in test_cases:
            our_result = self.stats_engine.calculate_binomial_ppf(q, n, p)
            root_result = binom.ppf(q, n, p)
            self.assertEqual(
                our_result, root_result,
                f"Mismatch for q={q}, n={n}, p={p}: {our_result} != {root_result}"
            )

    def test_root_style_variance_calculation(self):
        """Test root-style variance (sum of squared individual variances)."""
        input_sizes = [100, 200, 300]
        p = 0.1

        our_mean, our_var = self.stats_engine.calculate_input_statistics_root_style(
            input_sizes, p
        )

        expected_mean = sum(input_sizes) * p
        expected_var = sum((s * p * (1 - p)) ** 2 for s in input_sizes)

        self.assertAlmostEqual(our_mean, expected_mean, places=10)
        self.assertAlmostEqual(our_var, expected_var, places=10)

    def test_complete_algorithm_chain(self):
        """Test the complete algorithm chain exactly matches root brain.py."""
        total_k = 1000
        p = 0.1
        effective_n = 1000
        target_k = 100

        # Step 1: quantile
        quantile = (effective_n - target_k) / effective_n
        our_quantile = self.stats_engine.calculate_quantile_threshold(effective_n, target_k)
        self.assertEqual(quantile, our_quantile)

        # Step 2: alpha via binom.ppf
        alpha = binom.ppf(quantile, total_k, p)
        our_alpha = self.stats_engine.calculate_binomial_ppf(quantile, total_k, p)
        self.assertEqual(alpha, our_alpha)

        # Step 3: normal parameters
        mu = total_k * p
        std_sq = total_k * p * (1.0 - p)
        our_mean, our_var = self.stats_engine.calculate_input_statistics_standard(
            [total_k], p
        )
        self.assertEqual(mu, our_mean)
        self.assertEqual(std_sq, our_var)

        # Step 4: sample
        our_samples = self.stats_engine.sample_truncated_normal_winners(
            alpha, total_k, p, target_k
        )
        self.assertEqual(len(our_samples), target_k)
        self.assertTrue(np.all(our_samples >= 0))
        self.assertTrue(np.all(our_samples <= total_k))
        self.assertLess(abs(np.mean(our_samples) - mu), mu * 0.3)

    def test_numerical_precision(self):
        """Test numerical precision exactly matches root brain.py."""
        # Large-number precision
        effective_n = 1000000
        k = 500000
        quantile = (effective_n - k) / effective_n
        our_quantile = self.stats_engine.calculate_quantile_threshold(effective_n, k)
        self.assertAlmostEqual(our_quantile, quantile, places=15)

        # Small-difference precision
        n = 1000
        k1, k2 = 500, 501
        q1 = self.stats_engine.calculate_quantile_threshold(n, k1)
        q2 = self.stats_engine.calculate_quantile_threshold(n, k2)
        root_q1 = (n - k1) / n
        root_q2 = (n - k2) / n
        self.assertAlmostEqual(q1, root_q1, places=15)
        self.assertAlmostEqual(q2, root_q2, places=15)

    def test_variance_method_standard_vs_root(self):
        """Test that standard variance differs from root-style squared variance."""
        input_sizes = [100, 200, 300]
        p = 0.1
        total_k = sum(input_sizes)

        our_mean, our_var = self.stats_engine.calculate_input_statistics_standard(
            input_sizes, p
        )
        root_mu = total_k * p
        root_var = total_k * p * (1.0 - p)
        self.assertEqual(our_mean, root_mu)
        self.assertEqual(our_var, root_var)

        # Squared-variance approach is different
        squared_var = sum((s * p * (1.0 - p)) ** 2 for s in input_sizes)
        self.assertNotEqual(our_var, squared_var)

    def test_parameter_validation_error_message(self):
        """Test that parameter validation error message matches root brain.py."""
        effective_n = 50
        k = 100

        with self.assertRaises(RuntimeError) as ctx:
            self.stats_engine.validate_sampling_parameters(
                effective_n, k, 500, 0.05
            )

        expected_msg = (
            f'Remaining size of area ({effective_n}) too small to '
            f'sample k new winners ({k}).'
        )
        self.assertIn(expected_msg, str(ctx.exception))

    def test_edge_case_p_zero_and_one(self):
        """Test that p=0 and p=1 raise ZeroDivisionError in truncated normal."""
        with self.assertRaises(ZeroDivisionError):
            self.stats_engine.sample_truncated_normal_winners(50, 1000, 0.0, 10)

        with self.assertRaises(ZeroDivisionError):
            self.stats_engine.sample_truncated_normal_winners(50, 1000, 1.0, 10)


# ---------------------------------------------------------------------------
# 4. TestStatisticsMathematical -- rigorous mathematical property tests
# ---------------------------------------------------------------------------

class TestStatisticsMathematical(unittest.TestCase):
    """
    Mathematical property tests for the Statistical Engine.

    Covers quantile properties over exhaustive ranges, PPF monotonicity and
    symmetry, convergence, distribution properties, and a KS test comparing
    binomial samples against a normal approximation.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(seed=42)
        self.stats_engine = StatisticalEngine(self.rng)

    def test_quantile_mathematical_properties(self):
        """Test quantile properties over exhaustive (n, k) range."""
        # Quantile always in [0, 1]
        for n in range(1, 100):
            for k in range(1, n):
                quantile = self.stats_engine.calculate_quantile_threshold(n, k)
                self.assertGreaterEqual(quantile, 0.0)
                self.assertLessEqual(quantile, 1.0)

        # Quantile decreases as k increases
        n = 1000
        quantiles = []
        for k in range(1, n, 10):
            quantiles.append(
                (k, self.stats_engine.calculate_quantile_threshold(n, k))
            )
        quantiles.sort(key=lambda x: x[0])
        for i in range(1, len(quantiles)):
            self.assertGreaterEqual(quantiles[i - 1][1], quantiles[i][1])

        # Complementary k values sum to 1
        n = 1000
        q1 = self.stats_engine.calculate_quantile_threshold(n, 100)
        q2 = self.stats_engine.calculate_quantile_threshold(n, 900)
        self.assertAlmostEqual(q1 + q2, 1.0, places=10)

    def test_ppf_monotonicity_and_symmetry(self):
        """Test PPF monotonicity and p=0.5 symmetry."""
        n, p = 1000, 0.1
        quantiles = np.linspace(0.0, 1.0, 11)
        ppfs = [self.stats_engine.calculate_binomial_ppf(q, n, p) for q in quantiles]

        for i in range(1, len(ppfs)):
            self.assertGreaterEqual(ppfs[i], ppfs[i - 1])

        # Boundary values
        self.assertLessEqual(ppfs[0], 0)
        self.assertEqual(ppfs[-1], n)

        # p = 0.5 symmetry: PPF(0.25) + PPF(0.75) ~ n
        n_sym = 100
        ppf_025 = self.stats_engine.calculate_binomial_ppf(0.25, n_sym, 0.5)
        ppf_075 = self.stats_engine.calculate_binomial_ppf(0.75, n_sym, 0.5)
        self.assertAlmostEqual(ppf_025 + ppf_075, n_sym, places=1)

    def test_convergence_properties(self):
        """Test that sample means converge as sample size grows."""
        n = 1000
        p = 0.1
        expected_mean = n * p

        for size in [100, 1000, 10000]:
            samples = self.stats_engine.sample_binomial_winners(n, p, size)
            error = abs(np.mean(samples) - expected_mean)
            self.assertLess(error, expected_mean * 0.5)

    def test_distribution_properties(self):
        """Test that binomial samples have correct distributional shape."""
        n = 100
        p = 0.3
        size = 10000

        samples = self.stats_engine.sample_binomial_winners(n, p, size)

        # Integer-valued in [0, n]
        self.assertTrue(np.all(samples == samples.astype(int)))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= n))

        # Mean
        expected_mean = n * p
        self.assertLess(abs(np.mean(samples) - expected_mean), expected_mean * 0.1)

        # Variance
        expected_var = n * p * (1 - p)
        self.assertLess(abs(np.var(samples) - expected_var), expected_var * 0.2)

    def test_normal_approximation_ks_test(self):
        """Test normal approximation quality with a KS test."""
        n = 1000
        p = 0.1
        expected_mean = n * p
        expected_std = math.sqrt(n * p * (1 - p))

        n_samples = 10000
        samples = self.stats_engine.sample_binomial_winners(n, p, n_samples)

        # Mean and variance checks
        self.assertLess(
            abs(np.mean(samples) - expected_mean), expected_mean * 0.05
        )
        self.assertLess(
            abs(np.var(samples, ddof=1) - n * p * (1 - p)),
            n * p * (1 - p) * 0.1,
        )

        # Two-sample KS test against normal samples
        theoretical = np.random.default_rng(seed=0).normal(
            expected_mean, expected_std, n_samples
        )
        _, p_value = stats.ks_2samp(samples, theoretical)
        self.assertGreater(p_value, 1e-10)

    def test_edge_case_binomial_extremes(self):
        """Test binomial sampling at p=0, p=1, and n=1."""
        # p = 0 -> all zeros
        samples = self.stats_engine.sample_binomial_winners(100, 0.0, 1000)
        self.assertTrue(np.all(samples == 0))

        # p = 1 -> all n
        samples = self.stats_engine.sample_binomial_winners(100, 1.0, 1000)
        self.assertTrue(np.all(samples == 100))

        # n = 1 -> Bernoulli
        samples = self.stats_engine.sample_binomial_winners(1, 0.5, 1000)
        self.assertTrue(np.all((samples == 0) | (samples == 1)))

    def test_quantile_numerical_precision(self):
        """Test numerical precision of quantile differences."""
        n = 1000
        k1, k2 = 500, 501

        q1 = self.stats_engine.calculate_quantile_threshold(n, k1)
        q2 = self.stats_engine.calculate_quantile_threshold(n, k2)

        # Difference should be exactly 1/n
        self.assertAlmostEqual(abs(q1 - q2), 1.0 / n, places=10)

        # Large n, exact half
        quantile = self.stats_engine.calculate_quantile_threshold(1000000, 500000)
        self.assertAlmostEqual(quantile, 0.5, places=10)


if __name__ == '__main__':
    unittest.main()
