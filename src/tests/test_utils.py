# test_utils.py

import unittest
import numpy as np
import utils

class TestUtils(unittest.TestCase):

    def test_normalize_features(self):
        features = np.array([1, 2, 3, 4, 5])
        normalized = utils.normalize_features(features)
        self.assertTrue(np.all(normalized >= 0) and np.all(normalized <= 1))
        self.assertEqual(normalized[0], 0)
        self.assertEqual(normalized[-1], 1)

    def test_select_top_k_indices(self):
        features = np.array([10, 20, 30, 40, 50])
        top_indices = utils.select_top_k_indices(features, k=3)
        self.assertTrue(np.array_equal(top_indices, [4, 3, 2]))

    def test_heapq_select_top_k(self):
        features = np.array([10, 20, 30, 40, 50])
        top_indices = utils.heapq_select_top_k(features, k=3)
        self.assertTrue(np.array_equal(top_indices, [4, 3, 2]))

    def test_binomial_ppf(self):
        quantile = 0.5
        n = 10
        p = 0.5
        ppf_value = utils.binomial_ppf(quantile, n, p)
        self.assertEqual(ppf_value, 5)

if __name__ == '__main__':
    unittest.main()
