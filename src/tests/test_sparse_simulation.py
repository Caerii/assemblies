# test_sparse_simulation.py

"""
Comprehensive tests for the Sparse Simulation Engine.

This module tests the complex sparse simulation algorithms extracted
from the root brain.py, ensuring they produce correct and consistent results.
"""

import unittest
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.compute.sparse_simulation import SparseSimulationEngine

class TestSparseSimulationEngine(unittest.TestCase):
    """
    Test suite for the Sparse Simulation Engine.

    Tests the complex sparse simulation algorithms including:
    - Dynamic connectome expansion
    - Input distribution calculation
    - First-time winner processing
    - Synaptic connection assignment
    - Parameter validation
    - Statistical accuracy
    """

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(seed=42)
        self.sparse_engine = SparseSimulationEngine(self.rng)

    def test_input_distribution_calculation(self):
        """Test input distribution calculation."""
        # Test basic case
        input_sizes = [100, 200, 300]
        first_winner_inputs = [50.0, 30.0, 70.0]

        distributions = self.sparse_engine.calculate_input_distribution(
            input_sizes, first_winner_inputs
        )

        # Should have one distribution per winner
        self.assertEqual(len(distributions), len(first_winner_inputs))

        # Each distribution should have same number of sources
        for dist in distributions:
            self.assertEqual(len(dist), len(input_sizes))

        # Total connections should match input strength
        for i, (dist, expected) in enumerate(zip(distributions, first_winner_inputs)):
            total_connections = int(np.sum(dist))
            self.assertEqual(total_connections, int(expected))

    def test_dynamic_connectome_expansion(self):
        """Test dynamic connectome expansion."""
        # Test row expansion
        current_connectome = np.array([[1, 2], [3, 4]])
        expanded_rows = self.sparse_engine.expand_connectome_dynamic(
            current_connectome, 2, axis=0
        )

        expected_rows = np.array([
            [1, 2],
            [3, 4],
            [0, 0],
            [0, 0]
        ])
        np.testing.assert_array_equal(expanded_rows, expected_rows)

        # Test column expansion
        expanded_cols = self.sparse_engine.expand_connectome_dynamic(
            current_connectome, 2, axis=1
        )

        expected_cols = np.array([
            [1, 2, 0, 0],
            [3, 4, 0, 0]
        ])
        np.testing.assert_array_equal(expanded_cols, expected_cols)

    def test_synaptic_connection_assignment(self):
        """Test synaptic connection assignment."""
        # Create test connectome
        connectome = np.zeros((5, 3))
        input_sources = [np.array([2, 1, 3]), np.array([1, 2, 0])]
        new_winner_indices = [3, 4]

        updated_connectome = self.sparse_engine.assign_synaptic_connections(
            connectome, input_sources, new_winner_indices
        )

        # Check that connections were assigned
        self.assertEqual(updated_connectome[3, 0], 2)  # First source to first winner
        self.assertEqual(updated_connectome[3, 1], 1)  # Second source to first winner
        self.assertEqual(updated_connectome[4, 0], 1)  # First source to second winner
        self.assertEqual(updated_connectome[4, 1], 2)  # Second source to second winner

    def test_first_time_winner_processing(self):
        """Test first-time winner processing."""
        all_potential_winners = [10.0, 5.0, 8.0, 12.0, 3.0]
        target_area_w = 2
        target_area_k = 3

        new_winners, first_inputs, num_first = self.sparse_engine.process_first_time_winners(
            all_potential_winners, target_area_w, target_area_k
        )

        # Should select top 3 winners
        self.assertEqual(len(new_winners), 3)

        # Should identify first-time winners (indices >= target_area_w)
        # Order may vary due to heap algorithm, but should contain the right values
        expected_first_inputs_set = {8.0, 12.0}
        self.assertEqual(set(first_inputs), expected_first_inputs_set)
        self.assertEqual(num_first, 2)

    def test_plasticity_scaling(self):
        """Test Hebbian plasticity scaling."""
        connectome = np.array([
            [1.0, 2.0, 0.5],
            [0.8, 1.5, 0.3],
            [0.6, 1.0, 0.7]
        ])
        winner_indices = [0, 2]
        plasticity_beta = 0.1

        scaled_connectome = self.sparse_engine.apply_plasticity_scaling(
            connectome, winner_indices, plasticity_beta
        )

        # Winners should have scaled connections
        self.assertAlmostEqual(scaled_connectome[0, 0], 1.1)  # 1.0 * 1.1
        self.assertAlmostEqual(scaled_connectome[1, 0], 0.88)  # 0.8 * 1.1
        self.assertAlmostEqual(scaled_connectome[2, 0], 0.66)  # 0.6 * 1.1
        self.assertAlmostEqual(scaled_connectome[0, 2], 0.55)  # 0.5 * 1.1
        self.assertAlmostEqual(scaled_connectome[1, 2], 0.33)  # 0.3 * 1.1
        self.assertAlmostEqual(scaled_connectome[2, 2], 0.77)  # 0.7 * 1.1

        # Non-winner columns should remain unchanged
        self.assertAlmostEqual(scaled_connectome[0, 1], 2.0)
        self.assertAlmostEqual(scaled_connectome[1, 1], 1.5)
        self.assertAlmostEqual(scaled_connectome[2, 1], 1.0)

    def test_new_winner_synapse_initialization(self):
        """Test initialization of synapses for new winners."""
        connectome = np.zeros((5, 3))
        target_w = 3
        num_new_winners = 2
        input_sizes = [100, 200]
        p = 0.05

        # Set seed for reproducible test
        test_rng = np.random.default_rng(seed=123)
        test_engine = SparseSimulationEngine(test_rng)

        updated_connectome = test_engine.initialize_new_winner_synapses(
            connectome, target_w, num_new_winners, input_sizes, p
        )

        # Should have connections for new winners (rows 3, 4)
        self.assertEqual(updated_connectome.shape, (5, 3))

        # New winners should have some connections
        total_new_connections = np.sum(updated_connectome[target_w:, :])
        self.assertGreater(total_new_connections, 0)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with empty inputs
        distributions = self.sparse_engine.calculate_input_distribution([], [])
        self.assertEqual(distributions, [])

        # Test connectome expansion with invalid parameters
        connectome = np.array([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            self.sparse_engine.expand_connectome_dynamic(connectome, 2, axis=2)

    def test_statistical_correctness(self):
        """Test statistical correctness of algorithms."""
        # Test input distribution with known values
        input_sizes = [10, 20, 30]
        first_winner_inputs = [15.0]

        distributions = self.sparse_engine.calculate_input_distribution(
            input_sizes, first_winner_inputs
        )

        # Total connections should equal input strength
        total_connections = int(np.sum(distributions[0]))
        self.assertEqual(total_connections, 15)

        # All connections should be non-negative
        self.assertTrue(np.all(distributions[0] >= 0))
        # Should be integers (may be float type but integer values)
        self.assertTrue(np.all(distributions[0] == np.round(distributions[0])))

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with zero input sizes
        input_sizes = [0, 100, 0]
        first_winner_inputs = [10.0]

        distributions = self.sparse_engine.calculate_input_distribution(
            input_sizes, first_winner_inputs
        )

        # Should handle zero sizes gracefully
        self.assertEqual(len(distributions), 1)
        self.assertEqual(len(distributions[0]), 3)

        # Test connectome expansion with zero new winners
        connectome = np.array([[1, 2], [3, 4]])
        expanded = self.sparse_engine.expand_connectome_dynamic(connectome, 0)
        np.testing.assert_array_equal(expanded, connectome)

    def test_reproducibility(self):
        """Test reproducibility with same seeds."""
        # Test that same seed gives same results
        seed = 42
        rng1 = np.random.default_rng(seed=seed)
        rng2 = np.random.default_rng(seed=seed)

        engine1 = SparseSimulationEngine(rng1)
        engine2 = SparseSimulationEngine(rng2)

        input_sizes = [100, 200]
        first_winner_inputs = [50.0, 30.0]

        # Same operations should give same results
        dist1 = engine1.calculate_input_distribution(input_sizes, first_winner_inputs)
        dist2 = engine2.calculate_input_distribution(input_sizes, first_winner_inputs)

        for d1, d2 in zip(dist1, dist2):
            np.testing.assert_array_equal(d1, d2)

    def test_algorithm_complexity(self):
        """Test algorithm complexity and scaling."""
        # Test with increasing problem sizes
        sizes = [10, 100, 1000]

        for size in sizes:
            input_sizes = [size, size * 2]
            first_winner_inputs = [size * 0.5]

            # Should complete without issues
            distributions = self.sparse_engine.calculate_input_distribution(
                input_sizes, first_winner_inputs
            )

            self.assertEqual(len(distributions), 1)
            self.assertEqual(len(distributions[0]), 2)


class TestSparseSimulationExtremes(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=123)
        self.engine = SparseSimulationEngine(self.rng)

    def test_process_first_winners_extremes(self):
        winners = [1.0, 2.0, 3.0]
        # k = 0
        new, first, num_first = self.engine.process_first_time_winners(winners, target_area_w=2, target_area_k=0)
        self.assertEqual(new, [])
        self.assertEqual(first, [])
        self.assertEqual(num_first, 0)
        # k > len
        new, first, num_first = self.engine.process_first_time_winners(winners, target_area_w=2, target_area_k=10)
        self.assertEqual(len(new), 3)
        self.assertEqual(set(first), {3.0})
        self.assertEqual(num_first, 1)
        # all indices < w (no first-time)
        new, first, num_first = self.engine.process_first_time_winners(winners, target_area_w=5, target_area_k=2)
        self.assertEqual(num_first, 0)
        self.assertEqual(first, [])

    def test_distribution_with_zero_sources(self):
        input_sizes = [0, 0, 50]
        first_inputs = [10.0]
        dists = self.engine.calculate_input_distribution(input_sizes, first_inputs)
        self.assertEqual(len(dists), 1)
        self.assertEqual(len(dists[0]), 3)
        self.assertEqual(int(np.sum(dists[0])), 10)
        self.assertTrue(np.all(dists[0][:2] == 0))

    def test_distribution_empty_inputs(self):
        self.assertEqual(self.engine.calculate_input_distribution([], []), [])
        self.assertEqual(self.engine.calculate_input_distribution([10, 20], []), [])

    def test_expand_connectome_extremes(self):
        base = np.array([[1, 2], [3, 4]], dtype=float)
        # zero expansion
        np.testing.assert_array_equal(self.engine.expand_connectome_dynamic(base, 0, axis=1), base)
        np.testing.assert_array_equal(self.engine.expand_connectome_dynamic(base, 0, axis=0), base)
        # column expansion
        ex_cols = self.engine.expand_connectome_dynamic(base, 3, axis=1)
        self.assertEqual(ex_cols.shape, (2, 5))
        self.assertTrue(np.all(ex_cols[:, :2] == base))
        self.assertTrue(np.all(ex_cols[:, 2:] == 0))
        # row expansion
        ex_rows = self.engine.expand_connectome_dynamic(base, 2, axis=0)
        self.assertEqual(ex_rows.shape, (4, 2))
        self.assertTrue(np.all(ex_rows[:2, :] == base))
        self.assertTrue(np.all(ex_rows[2:, :] == 0))
        # invalid axis
        with self.assertRaises(ValueError):
            self.engine.expand_connectome_dynamic(base, 1, axis=3)

    def test_assign_synapses_mismatched_lengths(self):
        conn = np.zeros((5, 3))
        input_sources = [np.array([2, 0, 1])]  # only one winner provided
        new_winners = [4, 3]  # two winners specified
        updated = self.engine.assign_synaptic_connections(conn, input_sources, new_winners)
        # Winner 4 gets assigned; winner 3 remains zero
        self.assertEqual(updated[4, 0], 2)
        self.assertEqual(updated[4, 2], 1)
        self.assertTrue(np.all(updated[3, :] == 0))

    def test_initialize_synapses_p_extremes(self):
        conn = np.zeros((6, 2), dtype=int)
        # p = 0
        out0 = self.engine.initialize_new_winner_synapses(conn, target_w=4, num_new_winners=2, input_sizes=[100, 200], p=0.0)
        self.assertTrue(np.all(out0[4:, :] == 0))
        # p = 1
        out1 = self.engine.initialize_new_winner_synapses(conn, target_w=4, num_new_winners=2, input_sizes=[5, 3], p=1.0)
        # new rows equal to input sizes exactly
        self.assertTrue(np.all(out1[4:, 0] == 5))
        self.assertTrue(np.all(out1[4:, 1] == 3))

    def test_plasticity_scaling_extremes(self):
        conn = np.array([[1.0, 2.0], [3.0, 4.0]])
        # beta = 0 (no change)
        out0 = self.engine.apply_plasticity_scaling(conn, winner_indices=[0], plasticity_beta=0.0)
        np.testing.assert_array_almost_equal(out0, conn)
        # very large beta
        outL = self.engine.apply_plasticity_scaling(conn, winner_indices=[1], plasticity_beta=10.0)
        self.assertAlmostEqual(outL[0,1], 2.0 * 11.0)
        self.assertAlmostEqual(outL[1,1], 4.0 * 11.0)

    def test_concurrency(self):
        # Ensure no race conditions in read-only ops
        def worker(seed):
            rng = np.random.default_rng(seed)
            eng = SparseSimulationEngine(rng)
            _ = eng.calculate_input_distribution([10, 20, 30], [5.0, 10.0])
            base = np.zeros((3,3))
            _ = eng.expand_connectome_dynamic(base, 2, axis=1)
            return True
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(worker, range(16)))
        self.assertTrue(all(results))


if __name__ == '__main__':
    unittest.main()
