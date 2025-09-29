# test_connectome.py

import unittest
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.connectome import Connectome

class TestConnectome(unittest.TestCase):

    def setUp(self):
        self.connectome = Connectome(source_size=100, target_size=200, p=0.05)

    def test_initialization(self):
        self.assertEqual(self.connectome.source_size, 100)
        self.assertEqual(self.connectome.target_size, 200)
        self.assertEqual(self.connectome.weights.shape, (100, 200))

    def test_compute_inputs(self):
        pre_neurons = np.array([0, 1, 2])
        inputs = self.connectome.compute_inputs(pre_neurons)
        self.assertEqual(len(inputs), 200)
        self.assertTrue(np.all(inputs >= 0))

    def test_update_weights(self):
        pre_neurons = np.array([0, 1, 2])
        post_neurons = np.array([10, 20, 30])
        weights_before = self.connectome.weights.copy()
        self.connectome.update_weights(pre_neurons, post_neurons, beta=0.1)
        for pre in pre_neurons:
            self.assertTrue(np.all(self.connectome.weights[pre, post_neurons] == weights_before[pre, post_neurons] * 1.1))

    def test_expand(self):
        self.connectome.expand(new_source_size=10, new_target_size=20)
        self.assertEqual(self.connectome.source_size, 110)
        self.assertEqual(self.connectome.target_size, 220)
        self.assertEqual(self.connectome.weights.shape, (110, 220))

if __name__ == '__main__':
    unittest.main()
