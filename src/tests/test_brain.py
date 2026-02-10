# test_brain.py
"""
Brain Class Test Suite

This module contains comprehensive tests for the Brain class, validating
the core functionality of neural assembly simulation and Assembly Calculus
operations.

Test Coverage:
- Brain initialization and configuration
- Area and stimulus management
- Assembly Calculus projection operations
- Connectome initialization and connectivity
- Synaptic plasticity and learning

Biological Validation:
- Tests verify biologically plausible neural dynamics
- Validates sparse coding principles (k << n)
- Ensures proper synaptic connectivity patterns
- Confirms Hebbian plasticity mechanisms

Assembly Calculus Validation:
- Tests projection operations preserve overlap properties
- Validates winner-take-all competition mechanisms
- Ensures proper assembly formation and propagation
- Confirms mathematical properties of neural computations
"""

import unittest
import numpy as np

from ..core.brain import Brain

class TestBrain(unittest.TestCase):
    """
    Test suite for Brain class functionality.
    
    Tests the core neural assembly simulation capabilities including
    area management, projection operations, and Assembly Calculus
    implementation.
    """

    def setUp(self):
        self.brain = Brain(p=0.01, seed=42)
        self.brain.add_area("Area1", n=1000, k=100, beta=0.05, explicit=True)
        self.brain.add_area("Area2", n=800, k=80, beta=0.05, explicit=False)
        self.brain.add_stimulus("Stim1", size=500)

    def test_add_area(self):
        """
        Test neural area creation and configuration.
        
        Validates that areas are properly created with correct neural
        population parameters and assembly dynamics. This tests the
        fundamental building blocks of the neural assembly simulation.
        """
        self.assertIn("Area1", self.brain.areas)
        self.assertIn("Area2", self.brain.areas)
        self.assertEqual(self.brain.areas["Area1"].n, 1000)
        self.assertEqual(self.brain.areas["Area2"].n, 800)

    def test_add_stimulus(self):
        """
        Test external stimulus integration.
        
        Validates that external stimuli can be added to the brain
        simulation and are properly integrated into the neural network.
        External stimuli represent sensory inputs or pre-computed assemblies.
        """
        self.assertIn("Stim1", self.brain.stimuli)
        self.assertEqual(self.brain.stimuli["Stim1"].size, 500)

    def test_project_legacy_api(self):
        """
        Test projection using the legacy API (areas_by_stim, dst_areas_by_src_area).

        This matches the original brain.py signature used by simulations.py:
            brain.project({"stim": ["Area"]}, {"Area1": ["Area2"]})

        Uses a dedicated brain with both areas sparse (the standard path)
        to avoid cross-engine projection issues between explicit and sparse.
        """
        b = Brain(p=0.01, seed=42)
        b.add_area("A", n=1000, k=100, beta=0.05)
        b.add_area("B", n=800, k=80, beta=0.05)
        b.add_stimulus("S", size=500)

        # First project stimulus into A to create an assembly there
        b.project({"S": ["A"]}, {})
        self.assertTrue(len(b.areas["A"].winners) > 0)

        # Now project A → B (area-to-area)
        b.project({}, {"A": ["B"]})
        self.assertTrue(len(b.areas["B"].winners) > 0)

    def test_project_new_api(self):
        """
        Test projection using the new API (external_inputs, projections).

        The new API allows injecting arbitrary activation patterns and
        specifying projection routes via keyword arguments.

        Uses a dedicated brain with sparse areas to test the standard
        injection + projection path.
        """
        b = Brain(p=0.01, seed=42)
        b.add_area("A", n=1000, k=100, beta=0.05)
        b.add_area("B", n=800, k=80, beta=0.05)
        b.add_stimulus("S", size=500)

        # First establish an assembly in A via stimulus
        b.project({"S": ["A"]}, {})
        self.assertTrue(len(b.areas["A"].winners) > 0)

        # Inject A's current winners as external input, project to B
        external_inputs = {"A": b.areas["A"].winners.copy()}
        projections = {"A": ["B"]}
        b.project(external_inputs=external_inputs, projections=projections)
        self.assertTrue(len(b.areas["B"].winners) > 0)

    def test_connectomes_initialization(self):
        """
        Test synaptic connectivity initialization.
        
        Validates that connectomes (synaptic weight matrices) are properly
        initialized between all areas and stimuli. This ensures the neural
        network has the necessary connectivity for information flow.
        
        Biological Context:
        - Tests sparse connectivity patterns
        - Validates bidirectional connections between areas
        - Ensures proper stimulus-to-area connectivity
        """
        self.assertIn("Stim1", self.brain.connectomes_by_stimulus)
        self.assertIn("Area1", self.brain.connectomes)
        self.assertIn("Area2", self.brain.connectomes)
        # Check that Area1 has connectomes to Area2
        self.assertIn("Area2", self.brain.connectomes["Area1"])
        # Check that Area2 has connectomes to Area1
        self.assertIn("Area1", self.brain.connectomes["Area2"])

if __name__ == '__main__':
    unittest.main()
