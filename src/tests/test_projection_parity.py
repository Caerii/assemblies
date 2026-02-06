# test_projection_parity.py

"""
Parity test: root brain.py projection pieces vs extracted engines.

This test reconstructs a minimal projection step using the original
root-style formulas (scipy.stats) and compares against the extracted
StatisticalEngine, SparseSimulationEngine, and PlasticityEngine.
"""

import unittest
import numpy as np
import heapq
from scipy.stats import binom, truncnorm

from src.math_primitives.statistics import StatisticalEngine
from src.math_primitives.sparse_simulation import SparseSimulationEngine
from src.math_primitives.plasticity import PlasticityEngine


class TestProjectionParity(unittest.TestCase):
    def setUp(self):
        self.seed = 777
        self.rng = np.random.default_rng(self.seed)
        self.stats = StatisticalEngine(np.random.default_rng(self.seed))
        self.sparse = SparseSimulationEngine(np.random.default_rng(self.seed))
        self.plastic = PlasticityEngine(np.random.default_rng(self.seed))

    def test_projection_parity_single_round(self):
        # Parameters (mirroring root fields)
        p = 0.1
        k = 5
        effective_n = 1000
        w = 20  # number of ever fired
        # Inputs come from two stimuli and one area
        input_sizes = [120, 180, 60]  # total_k = 360
        total_k = sum(input_sizes)

        # Previous winners' inputs (deterministic for parity)
        prev_winner_count = w
        prev_winner_inputs = self.rng.integers(low=0, high=total_k, size=prev_winner_count)

        # ---- Root-style path ----
        root_quantile = (effective_n - k) / effective_n
        root_alpha = binom.ppf(root_quantile, total_k, p)
        mu = total_k * p
        std = np.sqrt(total_k * p * (1.0 - p))
        a = (root_alpha - mu) / std
        root_potential = (mu + truncnorm.rvs(a, np.inf, scale=std, size=k, random_state=self.seed)).round(0)
        root_potential = np.clip(root_potential, 0, total_k)
        root_all = np.concatenate([prev_winner_inputs, root_potential])
        root_new_winners = heapq.nlargest(k, range(len(root_all)), key=root_all.__getitem__)
        root_first_inputs = []
        root_new_winners_remap = list(root_new_winners)
        num_first = 0
        for i in range(k):
            if root_new_winners_remap[i] >= w:
                root_first_inputs.append(root_all[root_new_winners_remap[i]])
                root_new_winners_remap[i] = w + num_first
                num_first += 1
        # Input distribution for each first-time winner
        root_inputs_by_first = []
        for i in range(num_first):
            s = int(root_first_inputs[i])
            indices = self.rng.choice(range(total_k), s, replace=False)
            alloc = np.zeros(len(input_sizes))
            total_so_far = 0
            for j, size in enumerate(input_sizes):
                alloc[j] = np.sum((indices >= total_so_far) & (indices < total_so_far + size))
                total_so_far += size
            root_inputs_by_first.append(alloc)
        root_inputs_by_first = [np.array(x) for x in root_inputs_by_first]

        # Plasticity examples: vector and matrix scaling
        vec = np.zeros(w + num_first)
        mat = np.zeros((50, w + num_first))
        root_vec_scaled = vec.copy()
        for i in root_new_winners_remap:
            root_vec_scaled[i] *= (1.0 + 0.3)
        root_mat_scaled = mat.copy()
        pre_rows = [0, 1, 2]
        for j in pre_rows:
            for i in root_new_winners_remap:
                root_mat_scaled[j, i] *= (1.0 + 0.3)

        # ---- Extracted pipeline ----
        ext_quantile = self.stats.calculate_quantile_threshold(effective_n, k)
        self.assertAlmostEqual(ext_quantile, root_quantile)
        ext_alpha = self.stats.calculate_binomial_ppf(ext_quantile, total_k, p)
        self.assertEqual(ext_alpha, root_alpha)
        ext_potential = self.stats.sample_truncated_normal_winners(ext_alpha, total_k, p, k)
        ext_all = np.concatenate([prev_winner_inputs, ext_potential])
        # winner selection
        ext_new_winners, ext_first_inputs, ext_num_first = self.sparse.process_first_time_winners(
            ext_all.tolist(), w, k
        )
        self.assertEqual(ext_num_first, num_first)
        np.testing.assert_array_equal(np.array(ext_first_inputs), np.array(root_first_inputs))
        # input distribution
        ext_inputs_by_first = self.sparse.calculate_input_distribution(input_sizes, ext_first_inputs)
        self.assertEqual(len(ext_inputs_by_first), len(root_inputs_by_first))
        for e, r in zip(ext_inputs_by_first, root_inputs_by_first):
            np.testing.assert_array_equal(e, r)
        # plasticity scaling
        ext_vec_scaled = self.plastic.scale_stimulus_to_area(vec, ext_new_winners, beta=0.3)
        ext_mat_scaled = self.plastic.scale_area_to_area(mat, pre_rows, ext_new_winners, beta=0.3)
        np.testing.assert_array_equal(ext_vec_scaled, root_vec_scaled)
        np.testing.assert_array_equal(ext_mat_scaled, root_mat_scaled)

    def test_projection_parity_multi_rounds(self):
        # Parameters
        p = 0.1
        k = 7
        effective_n = 2000
        w = 15
        input_sizes = [150, 250, 100]  # total_k = 500
        total_k = sum(input_sizes)

        # Track vectors/matrices for root and extracted
        root_vec = np.zeros(w)
        ext_vec = np.zeros(w)
        root_mat = np.zeros((60, w))
        ext_mat = np.zeros((60, w))
        pre_rows = [0, 1, 2, 3]

        rounds = 5
        for _ in range(rounds):
            # Prev winner inputs (deterministic per round)
            prev_winner_inputs = self.rng.integers(low=0, high=total_k, size=w)

            # ---- Root path ----
            root_quantile = (effective_n - k) / effective_n
            root_alpha = binom.ppf(root_quantile, total_k, p)
            mu = total_k * p
            std = np.sqrt(total_k * p * (1.0 - p))
            a = (root_alpha - mu) / std
            rvs = truncnorm.rvs(a, np.inf, scale=std, size=k, random_state=self.rng)
            root_potential = (mu + rvs).round(0)
            root_potential = np.clip(root_potential, 0, total_k)
            root_all = np.concatenate([prev_winner_inputs, root_potential])
            root_new_winners = heapq.nlargest(k, range(len(root_all)), key=root_all.__getitem__)
            root_first_inputs = []
            root_new_winners_remap = list(root_new_winners)
            num_first = 0
            for i in range(k):
                if root_new_winners_remap[i] >= w:
                    root_first_inputs.append(root_all[root_new_winners_remap[i]])
                    root_new_winners_remap[i] = w + num_first
                    num_first += 1
            # Input distribution
            root_inputs_by_first = []
            for i in range(num_first):
                s = int(root_first_inputs[i])
                indices = self.rng.choice(range(total_k), s, replace=False)
                alloc = np.zeros(len(input_sizes))
                total_so_far = 0
                for j, size in enumerate(input_sizes):
                    alloc[j] = np.sum((indices >= total_so_far) & (indices < total_so_far + size))
                    total_so_far += size
                root_inputs_by_first.append(alloc)
            root_inputs_by_first = [np.array(x) for x in root_inputs_by_first]

            # Ensure vec/mat sized to current w + num_first
            if num_first > 0:
                root_vec = np.pad(root_vec, (0, num_first))
                root_mat = np.pad(root_mat, ((0, 0), (0, num_first)))
                ext_vec = np.pad(ext_vec, (0, num_first))
                ext_mat = np.pad(ext_mat, ((0, 0), (0, num_first)))

            # Apply root plasticity
            for i in root_new_winners_remap:
                root_vec[i] *= (1.0 + 0.2)
            for j in pre_rows:
                for i in root_new_winners_remap:
                    root_mat[j, i] *= (1.0 + 0.2)

            # ---- Extracted path ----
            ext_quantile = self.stats.calculate_quantile_threshold(effective_n, k)
            self.assertAlmostEqual(ext_quantile, root_quantile)
            ext_alpha = self.stats.calculate_binomial_ppf(ext_quantile, total_k, p)
            self.assertEqual(ext_alpha, root_alpha)
            ext_potential = self.stats.sample_truncated_normal_winners(ext_alpha, total_k, p, k)
            ext_all = np.concatenate([prev_winner_inputs, ext_potential])
            ext_new_winners, ext_first_inputs, ext_num_first = self.sparse.process_first_time_winners(
                ext_all.tolist(), w, k
            )
            self.assertEqual(ext_num_first, num_first)
            np.testing.assert_array_equal(np.array(ext_first_inputs), np.array(root_first_inputs))
            ext_inputs_by_first = self.sparse.calculate_input_distribution(input_sizes, ext_first_inputs)
            for e, r in zip(ext_inputs_by_first, root_inputs_by_first):
                np.testing.assert_array_equal(e, r)
            # Apply extracted plasticity
            ext_vec = self.plastic.scale_stimulus_to_area(ext_vec, ext_new_winners, beta=0.2)
            ext_mat = self.plastic.scale_area_to_area(ext_mat, pre_rows, ext_new_winners, beta=0.2)

            # Compare after round
            np.testing.assert_array_equal(ext_vec, root_vec)
            np.testing.assert_array_equal(ext_mat, root_mat)

            # Update ever-fired count
            w += num_first
            effective_n = max(effective_n - num_first, 1)



if __name__ == '__main__':
    unittest.main()


