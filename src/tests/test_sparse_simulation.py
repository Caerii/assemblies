# test_sparse_simulation.py

"""
Comprehensive tests for the Sparse Simulation Engine.

This module tests the complex sparse simulation algorithms extracted
from the root brain.py, ensuring they produce correct and consistent results.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math_primitives.sparse_simulation import SparseSimulationEngine

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

if __name__ == '__main__':
    unittest.main()
