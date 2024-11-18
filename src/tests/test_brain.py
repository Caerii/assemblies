# test_brain.py

import unittest
import numpy as np
from brain import Brain
from area import Area
from stimulus import Stimulus

class TestBrain(unittest.TestCase):

    def setUp(self):
        self.brain = Brain(p=0.01, seed=42)
        self.brain.add_area("Area1", n=1000, k=100, beta=0.05, explicit=True)
        self.brain.add_area("Area2", n=800, k=80, beta=0.05, explicit=False)
        self.brain.add_stimulus("Stim1", size=500)

    def test_add_area(self):
        self.assertIn("Area1", self.brain.areas)
        self.assertIn("Area2", self.brain.areas)
        self.assertEqual(self.brain.areas["Area1"].n, 1000)
        self.assertEqual(self.brain.areas["Area2"].n, 800)

    def test_add_stimulus(self):
        self.assertIn("Stim1", self.brain.stimuli)
        self.assertEqual(self.brain.stimuli["Stim1"].size, 500)

    def test_project(self):
        external_inputs = {"Area1": np.arange(100)}
        projections = {"Area1": ["Area2"]}
        self.brain.project(external_inputs, projections)
        self.assertTrue(len(self.brain.areas["Area2"].winners) > 0)

    def test_connectomes_initialization(self):
        self.assertIn("Stim1", self.brain.connectomes_by_stimulus)
        self.assertIn("Area1", self.brain.connectomes)
        self.assertIn("Area2", self.brain.connectomes)
        self.assertIn("Area1", self.brain.connectomes["Area1"])
        self.assertIn("Area2", self.brain.connectomes["Area1"])
        self.assertIn("Area1", self.brain.connectomes["Area2"])
        self.assertIn("Area2", self.brain.connectomes["Area2"])

if __name__ == '__main__':
    unittest.main()
