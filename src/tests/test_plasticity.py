# test_plasticity.py

import unittest
import numpy as np

from ..math_primitives.plasticity import PlasticityEngine


class TestPlasticityEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PlasticityEngine(np.random.default_rng(seed=42))

    def test_stimulus_to_area_scaling_basic(self):
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        new_winners = [1, 3]
        out = self.engine.scale_stimulus_to_area(vec, new_winners, beta=0.1, disable=False)
        np.testing.assert_array_almost_equal(out, np.array([1.0, 2.2, 3.0, 4.4]))

    def test_stimulus_to_area_scaling_disabled(self):
        vec = np.array([1.0, 2.0, 3.0])
        out = self.engine.scale_stimulus_to_area(vec, [0, 2], beta=0.5, disable=True)
        np.testing.assert_array_almost_equal(out, vec)

    def test_area_to_area_scaling_basic(self):
        mat = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
        from_rows = [0, 2]
        to_cols = [1]
        out = self.engine.scale_area_to_area(mat, from_rows, to_cols, beta=0.2, disable=False)
        expected = mat.copy()
        expected[0, 1] *= 1.2
        expected[2, 1] *= 1.2
        np.testing.assert_array_almost_equal(out, expected)

    def test_area_to_area_scaling_disabled(self):
        mat = np.ones((2, 2))
        out = self.engine.scale_area_to_area(mat, [0, 1], [0, 1], beta=5.0, disable=True)
        np.testing.assert_array_almost_equal(out, mat)

    def test_bounds_safety(self):
        vec = np.array([1.0, 2.0])
        out = self.engine.scale_stimulus_to_area(vec, [0, 5], beta=1.0)
        np.testing.assert_array_almost_equal(out, np.array([2.0, 2.0]))

        mat = np.ones((2, 2))
        out = self.engine.scale_area_to_area(mat, [0, 3], [1, -1], beta=0.5)
        expected = np.ones((2, 2))
        expected[0, 1] *= 1.5
        np.testing.assert_array_almost_equal(out, expected)

    def test_extreme_beta(self):
        mat = np.array([[2.0]])
        out = self.engine.scale_area_to_area(mat, [0], [0], beta=0.0)
        np.testing.assert_array_almost_equal(out, mat)
        out = self.engine.scale_area_to_area(mat, [0], [0], beta=10.0)
        self.assertAlmostEqual(out[0, 0], 2.0 * 11.0)


if __name__ == '__main__':
    unittest.main()
