# test_utils.py

import unittest
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.math_utils import normalize_features, select_top_k_indices, heapq_select_top_k, binomial_ppf

class TestUtils(unittest.TestCase):

    def test_normalize_features(self):
        features = np.array([1, 2, 3, 4, 5])
        normalized = normalize_features(features)
        self.assertTrue(np.all(normalized >= 0) and np.all(normalized <= 1))
        self.assertEqual(normalized[0], 0)
        self.assertAlmostEqual(normalized[-1], 1, places=6)

    def test_select_top_k_indices(self):
        features = np.array([10, 20, 30, 40, 50])
        top_indices = select_top_k_indices(features, k=3)
        self.assertTrue(np.array_equal(top_indices, [4, 3, 2]))

    def test_heapq_select_top_k(self):
        features = np.array([10, 20, 30, 40, 50])
        top_indices = heapq_select_top_k(features, k=3)
        self.assertTrue(np.array_equal(top_indices, [4, 3, 2]))

    def test_binomial_ppf(self):
        quantile = 0.5
        n = 10
        p = 0.5
        ppf_value = binomial_ppf(quantile, n, p)
        self.assertEqual(ppf_value, 5)

if __name__ == '__main__':
    unittest.main()
