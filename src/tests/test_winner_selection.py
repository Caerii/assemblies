# test_winner_selection.py

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from math_primitives.winner_selection import WinnerSelector
from math_primitives.sparse_simulation import SparseSimulationEngine


class TestWinnerSelection(unittest.TestCase):
    def setUp(self):
        self.seed = 2024
        self.rng = np.random.default_rng(self.seed)
        self.selector = WinnerSelector(np.random.default_rng(self.seed))
        self.sparse = SparseSimulationEngine(np.random.default_rng(self.seed))

    def test_top_k_argsort_basic(self):
        x = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        k = 3
        idx = self.selector.select_top_k_indices(x, k)
        # Manually compute expected via argsort
        expected = np.argsort(-x)[:k]
        np.testing.assert_array_equal(idx, expected)

    def test_heapq_top_k_equivalence_of_values(self):
        # For ties, indices may differ; validate chosen values match top-k multiset
        x = self.rng.random(100)
        x[5] = x[6]  # introduce a tie
        k = 10
        idx_heap = self.selector.heapq_select_top_k(x, k)
        idx_sort = np.argsort(-x)[:k]
        # Compare sorted values for both selections
        values_heap = np.sort(x[idx_heap])[::-1]
        values_sort = np.sort(x[idx_sort])[::-1]
        np.testing.assert_allclose(values_heap, values_sort, rtol=0, atol=0)

    def test_k_zero_and_k_ge_n(self):
        x = self.rng.random(10)
        idx0 = self.selector.select_top_k_indices(x, 0)
        self.assertEqual(len(idx0), 0)
        idx_all = self.selector.heapq_select_top_k(x, 100)
        np.testing.assert_array_equal(np.sort(idx_all), np.arange(len(x)))

    def test_first_time_winner_parity_with_sparse_engine(self):
        # Build combined vector: prev winners (w) + potentials (k)
        w = 7
        k = 4
        prev_inputs = self.rng.integers(0, 50, size=w)
        potentials = self.rng.integers(0, 50, size=k)
        all_inputs = np.concatenate([prev_inputs, potentials])
        # Reference: sparse engine process_first_time_winners
        new_idx_ref, first_inputs_ref, num_first_ref = self.sparse.process_first_time_winners(
            all_inputs.tolist(), target_area_w=w, target_area_k=k
        )
        # WinnerSelector using heapq over the same array
        heap_idx = self.selector.heapq_select_top_k(all_inputs, k)
        # Emulate remap/first-time extraction here
        first_inputs = []
        new_idx = list(heap_idx)
        num_first = 0
        for i in range(len(new_idx)):
            if new_idx[i] >= w:
                first_inputs.append(all_inputs[new_idx[i]])
                new_idx[i] = w + num_first
                num_first += 1
        # Compare outputs
        self.assertEqual(num_first, num_first_ref)
        np.testing.assert_array_equal(np.array(first_inputs), np.array(first_inputs_ref))
        np.testing.assert_array_equal(np.array(new_idx), np.array(new_idx_ref))

    def test_large_scale_consistency(self):
        # Ensure methods work for large vectors and agree on value multiset
        x = self.rng.random(20000)
        k = 100
        idx_heap = self.selector.heapq_select_top_k(x, k)
        idx_sort = np.argsort(-x)[:k]
        values_heap = np.sort(x[idx_heap])[::-1]
        values_sort = np.sort(x[idx_sort])[::-1]
        np.testing.assert_allclose(values_heap, values_sort, rtol=0, atol=0)

    def test_combined_selection_with_mask_and_tie_policy(self):
        w = 5
        k = 3
        prev = np.array([10, 9, 8, 7, 6], dtype=float)
        pot = np.array([10, 10, 5], dtype=float)  # ties on value 10
        all_inputs = np.concatenate([prev, pot])
        # inhibit index 1 and 5
        mask = np.zeros_like(all_inputs, dtype=bool)
        mask[1] = True
        mask[5] = True
        new_idx, first_inputs, num_first, original_indices = self.selector.select_combined_winners(
            all_inputs, target_area_w=w, target_area_k=k, method="heapq", mask_inhibit=mask, tie_policy="value_then_index"
        )
        # Verify that inhibited indices are not in winners (check pre-remap indices)
        self.assertNotIn(1, original_indices)
        self.assertNotIn(5, original_indices)
        # Verify count and first inputs length match
        self.assertEqual(len(new_idx), k)
        self.assertEqual(num_first, len(first_inputs))

    def test_mask_all_inhibited_returns_empty(self):
        # If all candidates are inhibited, selection should return empty lists
        w = 3
        k = 5
        all_inputs = np.array([5., 4., 3., 2., 1.])
        mask = np.ones_like(all_inputs, dtype=bool)
        new_idx, first_inputs, num_first, original_indices = self.selector.select_combined_winners(
            all_inputs, target_area_w=w, target_area_k=k, method="heapq", mask_inhibit=mask, tie_policy="value_then_index"
        )
        self.assertEqual(new_idx, [])
        self.assertEqual(original_indices, [])
        self.assertEqual(first_inputs, [])
        self.assertEqual(num_first, 0)

    def test_tie_breaking_value_then_index_heapq(self):
        # With equal values, smaller index should come first under value_then_index
        x = np.array([10., 10., 10., 9., 8.])
        w = 0
        k = 3
        new_idx, first_inputs, num_first, original_indices = self.selector.select_combined_winners(
            x, target_area_w=w, target_area_k=k, method="heapq", tie_policy="value_then_index"
        )
        # Expect indices [0,1,2] due to tie-breaking by index
        self.assertEqual(original_indices, [0, 1, 2])

    def test_tie_breaking_value_then_index_argsort(self):
        # Same as heapq case, but via argsort path
        x = np.array([10., 10., 10., 9., 8.])
        w = 0
        k = 3
        new_idx, first_inputs, num_first, original_indices = self.selector.select_combined_winners(
            x, target_area_w=w, target_area_k=k, method="argsort", tie_policy="value_then_index"
        )
        self.assertEqual(original_indices, [0, 1, 2])

    def test_mask_nonboolean_coercion(self):
        # Non-boolean masks should be coerced safely
        x = np.array([5., 4., 3., 2., 1.])
        w = 0
        k = 2
        mask = np.array([0, 1, 0, 0, 0], dtype=int)  # inhibit index 1
        new_idx, first_inputs, num_first, original_indices = self.selector.select_combined_winners(
            x, target_area_w=w, target_area_k=k, method="heapq", mask_inhibit=mask
        )
        self.assertNotIn(1, original_indices)
        self.assertEqual(len(original_indices), 2)

    def test_nan_inputs_raise(self):
        # NaN/Inf in inputs should raise
        x = np.array([1., np.nan, 2.])
        with self.assertRaises(ValueError):
            self.selector.select_combined_winners(x, target_area_w=0, target_area_k=1)


if __name__ == '__main__':
    unittest.main()
