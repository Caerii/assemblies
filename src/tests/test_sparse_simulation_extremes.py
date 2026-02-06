# test_sparse_simulation_extremes.py

"""
Extreme and boundary tests for the Sparse Simulation Engine.

Covers pathological and edge cases to ensure robustness:
- Zero/negative sizes, empty inputs, very large sizes
- Degenerate first-winner scenarios (k=0, k>len, all indices < w)
- Distribution when some sources are zero-length
- Expansion with zero/negative new winners and both axes
- Assignment with mismatched lengths and empty sources
- Initialization with p=0 and p=1, huge inputs
- Plasticity scaling with beta=0, very large beta
- Concurrency safety
"""

import unittest
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.math_primitives.sparse_simulation import SparseSimulationEngine


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
