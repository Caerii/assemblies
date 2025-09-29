# test_plasticity_extremes.py

import unittest
import numpy as np
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math_primitives.plasticity import PlasticityEngine
from math_primitives.sparse_simulation import SparseSimulationEngine


class TestPlasticityExtremes(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=123)
        self.pe = PlasticityEngine(self.rng)
        self.se = SparseSimulationEngine(self.rng)

    def test_stimulus_to_area_types_and_shapes(self):
        vec_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = self.pe.scale_stimulus_to_area(vec_f32, [0, 2], beta=0.25)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_almost_equal(out, np.array([1.25, 2.0, 3.75], dtype=np.float32))

        vec = np.array([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            self.pe.scale_stimulus_to_area(vec, [0], beta=0.1)

    def test_area_to_area_types_and_shapes(self):
        mat = np.ones((3, 4), dtype=np.float64)
        out = self.pe.scale_area_to_area(mat, [0, 5], [1, -1], beta=0.5)
        self.assertEqual(out.dtype, np.float64)
        expected = np.ones((3, 4))
        expected[0, 1] *= 1.5
        np.testing.assert_array_almost_equal(out, expected)

        mat1d = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            self.pe.scale_area_to_area(mat1d, [0], [0], beta=0.1)

    def test_randomized_consistency(self):
        mat = self.rng.random((10, 8))
        base = mat.copy()
        pre_rows = [0, 3, 7]
        post_cols = [2, 5]
        beta = 0.2
        out = self.pe.scale_area_to_area(mat, pre_rows, post_cols, beta)
        factor = 1.0 + beta
        for r in pre_rows:
            for c in post_cols:
                if 0 <= r < base.shape[0] and 0 <= c < base.shape[1]:
                    self.assertAlmostEqual(out[r, c], base[r, c] * factor)
        unaffected = [(1, 1), (9, 7)]
        for r, c in unaffected:
            self.assertAlmostEqual(out[r, c], base[r, c])

    def test_concurrency_safety(self):
        def worker(seed):
            rng = np.random.default_rng(seed)
            pe = PlasticityEngine(rng)
            se = SparseSimulationEngine(rng)
            mat = rng.random((20, 10))
            vec = rng.random(30)
            _ = pe.scale_area_to_area(mat, [0, 1, 2], [3, 4], beta=0.1)
            _ = pe.scale_stimulus_to_area(vec, [5, 10, 15], beta=0.2)
            _ = se.expand_connectome_dynamic(mat, 5, axis=1)
            return True
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(worker, range(16)))
        self.assertTrue(all(results))

    def test_integration_with_sparse_sim(self):
        connectome = np.zeros((5, 3))
        inputs_by_winner = [np.array([2, 1, 0]), np.array([1, 0, 2])]
        new_winner_rows = [3, 4]
        assigned = self.se.assign_synaptic_connections(connectome, inputs_by_winner, new_winner_rows)
        pre_rows = [0, 1, 2, 3, 4]
        post_cols = [1, 2]  # valid column indices in (5,3)
        beta = 0.3
        scaled = self.pe.scale_area_to_area(assigned, pre_rows, post_cols, beta)
        factor = 1.0 + beta
        self.assertAlmostEqual(scaled[0, 1], assigned[0, 1] * factor)
        self.assertAlmostEqual(scaled[4, 2], assigned[4, 2] * factor)

    def test_disable_flags(self):
        vec = np.array([1.0, 2.0, 3.0])
        out = self.pe.scale_stimulus_to_area(vec, [1], beta=0.9, disable=True)
        np.testing.assert_array_almost_equal(out, vec)
        mat = np.eye(4)
        out = self.pe.scale_area_to_area(mat, [0, 1], [2, 3], beta=0.9, disable=True)
        np.testing.assert_array_almost_equal(out, mat)

    def test_invalid_values_and_indices(self):
        vec = np.array([1.0, np.nan, 3.0])
        with self.assertRaises(ValueError):
            self.pe.scale_stimulus_to_area(vec, [1], beta=0.1)
        vec2 = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.pe.scale_stimulus_to_area(vec2, [1.5], beta=0.1)
        mat = np.ones((2, 2))
        with self.assertRaises(ValueError):
            self.pe.scale_area_to_area(mat, [0], [0], beta=np.inf)
        mat2 = np.array([[1.0, np.inf], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            self.pe.scale_area_to_area(mat2, [0], [1], beta=0.1)


if __name__ == '__main__':
    unittest.main()
