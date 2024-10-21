#! /usr/bin/python

import brain_util as bu
import simulations
import unittest
import time

class TestBrainFunction(unittest.TestCase):
    def test_projection(self):
        start_time = time.time()
        print("Testing Projection...")
        w = simulations.project_sim(1000000, 1000, 0.001, 0.05, 25)
        elapsed_time = time.time() - start_time
        self.assertEqual(w[-2], w[-1], "Projection test failed.")
        print(f"Projection completed in {elapsed_time:.2f} seconds.")

    def test_pattern_completion(self):
        start_time = time.time()
        print("Testing Pattern Completion...")
        (_, winners) = simulations.pattern_com(
            100000, 317, 0.05, 0.05, 25, 0.5, 5)
        elapsed_time = time.time() - start_time
        self.assertGreaterEqual(bu.overlap(winners[24], winners[29]), 300, "Pattern completion test failed.")
        print(f"Pattern Completion completed in {elapsed_time:.2f} seconds.")

    def test_association(self):
        start_time = time.time()
        print("Testing Association...")
        (_, winners) = simulations.association_sim(100000, 317, 0.05, 0.1, 10)
        elapsed_time = time.time() - start_time
        self.assertLessEqual(bu.overlap(winners[9], winners[19]), 2, "Association test failed at early overlap.")
        self.assertGreaterEqual(bu.overlap(winners[9], winners[29]), 100, "Association test failed at later overlap 1.")
        self.assertGreaterEqual(bu.overlap(winners[19], winners[29]), 100, "Association test failed at later overlap 2.")
        self.assertGreaterEqual(bu.overlap(winners[9], winners[39]), 20, "Association test failed at latest overlap.")
        print(f"Association test completed in {elapsed_time:.2f} seconds.")

    def test_merge(self):
        start_time = time.time()
        print("Testing Merge...")
        (w_a, w_b, w_c) = simulations.merge_sim(100000, 317, 0.01, 0.05, 50)
        elapsed_time = time.time() - start_time
        self.assertLessEqual(w_a[-1], 3200, "Merge test failed for area A.")
        self.assertLessEqual(w_b[-1], 3200, "Merge test failed for area B.")
        self.assertLessEqual(w_c[-1], 6400, "Merge test failed for area C.")
        print(f"Merge test completed in {elapsed_time:.2f} seconds.")

if __name__ == '__main__':
    unittest.main()
