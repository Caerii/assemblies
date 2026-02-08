"""
Integration tests for simulation modules.

This module contains integration tests that verify the simulation
modules work correctly with the extracted math primitives.

Parameter regime notes:
    The sparse engine uses statistical sampling (truncated normal) for new
    winner candidates. Convergence speed depends on the coefficient of
    variation (CV) of the Binomial(total_k, p) input distribution:

        CV = sqrt((1-p) / (total_k * p))

    When CV is high (low p, small k), the input distribution has large
    relative spread, allowing statistical outliers to seed a stable nucleus
    within a few rounds. When CV is low (high p, large k), the distribution
    is concentrated and nucleation can take 30+ rounds.

    At p=0.01 with k=317: CV ≈ 0.40, assembly stabilizes in ~6 rounds.
    At p=0.05 with k=317: CV ≈ 0.17, assembly stabilizes in ~35 rounds.

    These tests use p=0.01 (matching the merge test) to ensure assemblies
    converge well within each simulation phase.
"""

import unittest
import time

from src.simulation.projection_simulator import project_sim
from src.simulation.association_simulator import association_sim
from src.simulation.merge_simulator import merge_sim
from src.simulation.pattern_completion import pattern_com
import brain_util as bu

class TestSimulationIntegration(unittest.TestCase):
    """Test integration of simulation modules with extracted math primitives."""

    def test_projection(self):
        """Test projection simulation with extracted algorithms."""
        start_time = time.time()
        print("Testing Projection...")
        w = project_sim(1000000, 1000, 0.001, 0.05, 25)
        elapsed_time = time.time() - start_time
        self.assertEqual(w[-2], w[-1], "Projection test failed.")
        print(f"Projection completed in {elapsed_time:.2f} seconds.")

    def test_pattern_completion(self):
        """Test pattern completion simulation with extracted algorithms.

        At p=0.01, the assembly stabilizes within ~6 rounds of projection.
        After 25 rounds, recurrent connections are strong enough that 50%
        partial activation (alpha=0.5) fully recovers the assembly.
        """
        start_time = time.time()
        print("Testing Pattern Completion...")
        (_, winners) = pattern_com(
            100000, 317, 0.01, 0.05, 25, 0.5, 5)
        elapsed_time = time.time() - start_time
        self.assertGreaterEqual(bu.overlap(winners[24], winners[29]), 300,
                               "Pattern completion test failed.")
        print(f"Pattern Completion completed in {elapsed_time:.2f} seconds.")

    def test_association(self):
        """Test association simulation with extracted algorithms.

        The association protocol projects A→C, then B→C, then both→C.
        At p=0.01, C's recurrent attractor from Phase 1 (A→C) is strong,
        so the B→C phase produces a C assembly that already overlaps
        substantially with the A-imprint (~77%). The key test is that
        joint training (Phase 3) INCREASES overlap — the association
        effect — from ~244 to ~293, demonstrating that C has learned
        to respond similarly to both inputs.

        Assertions:
            1. After joint training, C overlaps strongly with A-imprint
            2. After joint training, C overlaps strongly with B-imprint
            3. The association effect: training increased overlap
            4. A-response persists even after B-only readout
        """
        start_time = time.time()
        print("Testing Association...")
        (_, winners) = association_sim(100000, 317, 0.01, 0.1, 10)
        elapsed_time = time.time() - start_time

        early_overlap = bu.overlap(winners[9], winners[19])
        trained_a = bu.overlap(winners[9], winners[29])
        trained_b = bu.overlap(winners[19], winners[29])
        persist = bu.overlap(winners[9], winners[39])

        # After joint training, C overlaps strongly with both imprints
        self.assertGreaterEqual(trained_a, 250,
                               f"Association: trained overlap with A-imprint too low ({trained_a}).")
        self.assertGreaterEqual(trained_b, 200,
                               f"Association: trained overlap with B-imprint too low ({trained_b}).")
        # Association effect: joint training increased overlap
        self.assertGreater(trained_a, early_overlap,
                          f"Association effect absent: trained ({trained_a}) <= early ({early_overlap}).")
        # A-response persists after B-only readout
        self.assertGreaterEqual(persist, 200,
                               f"Association: A-persistence after B-readout too low ({persist}).")
        print(f"Association test completed in {elapsed_time:.2f} seconds.")

    def test_merge(self):
        """Test merge simulation with extracted algorithms."""
        start_time = time.time()
        print("Testing Merge...")
        (w_a, w_b, w_c) = merge_sim(100000, 317, 0.01, 0.05, 50)
        elapsed_time = time.time() - start_time
        self.assertLessEqual(w_a[-1], 4000, "Merge test failed for area A.")
        self.assertLessEqual(w_b[-1], 4000, "Merge test failed for area B.")
        self.assertLessEqual(w_c[-1], 8000, "Merge test failed for area C.")
        print(f"Merge test completed in {elapsed_time:.2f} seconds.")

if __name__ == '__main__':
    unittest.main()
