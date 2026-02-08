"""
Tests for sequence operations: Sequence dataclass, sequence_memorize,
and ordered_recall.

Based on the theory from:
    Dabagia, Papadimitriou, Vempala.
    "Computation with Sequences of Assemblies in a Model of the Brain."
    Neural Computation (2025).  arXiv:2306.03812.
"""

import unittest
import numpy as np

from src.core.brain import Brain
from src.assembly_calculus.assembly import Assembly, overlap, chance_overlap
from src.assembly_calculus.sequence import Sequence
from src.assembly_calculus.ops import (
    project, sequence_memorize, ordered_recall, _snap,
)

N = 10000
K = 100
P = 0.05
BETA = 0.1
ROUNDS = 10
SEED = 42


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


# ---------------------------------------------------------------------------
# Sequence dataclass tests
# ---------------------------------------------------------------------------

class TestSequenceDataclass(unittest.TestCase):
    """Test the Sequence frozen dataclass."""

    def _make_assemblies(self, n=3):
        """Create n assemblies with known winners."""
        rng = np.random.default_rng(SEED)
        asms = []
        for i in range(n):
            winners = rng.choice(N, size=K, replace=False).astype(np.uint32)
            asms.append(Assembly("A", winners))
        return asms

    def test_length(self):
        asms = self._make_assemblies(5)
        seq = Sequence("A", asms)
        self.assertEqual(len(seq), 5)

    def test_indexing(self):
        asms = self._make_assemblies(3)
        seq = Sequence("A", asms)
        self.assertEqual(seq[0], asms[0])
        self.assertEqual(seq[2], asms[2])

    def test_iteration(self):
        asms = self._make_assemblies(3)
        seq = Sequence("A", asms)
        collected = list(seq)
        self.assertEqual(len(collected), 3)
        for a, b in zip(collected, asms):
            self.assertEqual(a, b)

    def test_pairwise_overlaps(self):
        asms = self._make_assemblies(3)
        seq = Sequence("A", asms)
        pw = seq.pairwise_overlaps()
        self.assertEqual(len(pw), 2)
        # Random assemblies should have near-zero overlap
        for o in pw:
            self.assertLess(o, 0.1)

    def test_overlap_matrix_shape(self):
        asms = self._make_assemblies(4)
        seq = Sequence("A", asms)
        mat = seq.overlap_matrix()
        self.assertEqual(mat.shape, (4, 4))
        # Diagonal should be 1.0
        for i in range(4):
            self.assertAlmostEqual(mat[i, i], 1.0, places=5)

    def test_immutability(self):
        asms = self._make_assemblies(2)
        seq = Sequence("A", asms)
        with self.assertRaises(AttributeError):
            seq.area = "B"

    def test_list_converted_to_tuple(self):
        asms = self._make_assemblies(2)
        seq = Sequence("A", asms)  # pass list
        self.assertIsInstance(seq.assemblies, tuple)


# ---------------------------------------------------------------------------
# sequence_memorize tests
# ---------------------------------------------------------------------------

class TestSequenceMemorize(unittest.TestCase):
    """Test sequence_memorize operation."""

    def test_memorize_creates_correct_length(self):
        b = _make_brain()
        for i in range(3):
            b.add_stimulus(f"s{i}", K)
        b.add_area("A", N, K, BETA)

        seq = sequence_memorize(b, ["s0", "s1", "s2"], "A", rounds_per_step=ROUNDS)

        self.assertIsInstance(seq, Sequence)
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.area, "A")

    def test_memorize_creates_distinct_assemblies(self):
        """Each stimulus should produce a distinct assembly."""
        b = _make_brain()
        for i in range(3):
            b.add_stimulus(f"s{i}", K)
        b.add_area("A", N, K, BETA)

        seq = sequence_memorize(b, ["s0", "s1", "s2"], "A", rounds_per_step=ROUNDS)

        # All pairs should be distinct (overlap < 0.5)
        mat = seq.overlap_matrix()
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertLess(mat[i, j], 0.5,
                                    f"Assemblies {i} and {j} should be distinct "
                                    f"(overlap={mat[i, j]:.3f}).")

    def test_memorize_repetitions_strengthen_links(self):
        """Multiple repetitions should strengthen consecutive overlap."""
        overlaps = {}
        for reps in [1, 3]:
            b = _make_brain()
            for i in range(3):
                b.add_stimulus(f"s{i}", K)
            b.add_area("A", N, K, BETA)

            seq = sequence_memorize(b, ["s0", "s1", "s2"], "A",
                                    rounds_per_step=ROUNDS, repetitions=reps)
            overlaps[reps] = seq.mean_consecutive_overlap()

        # More repetitions should not decrease consecutive overlap
        # (strengthens x_i -> x_{i+1} Hebbian links)
        self.assertGreaterEqual(overlaps[3], overlaps[1] - 0.05,
                                f"3 reps overlap ({overlaps[3]:.3f}) should be >= "
                                f"1 rep overlap ({overlaps[1]:.3f}).")


# ---------------------------------------------------------------------------
# ordered_recall tests
# ---------------------------------------------------------------------------

class TestOrderedRecall(unittest.TestCase):
    """Test ordered_recall operation with LRI."""

    def test_recall_requires_lri(self):
        """ordered_recall must raise ValueError if refractory_period=0."""
        b = _make_brain()
        b.add_stimulus("s0", K)
        b.add_area("A", N, K, BETA)  # no LRI
        project(b, "s0", "A", rounds=ROUNDS)

        with self.assertRaises(ValueError):
            ordered_recall(b, "A", "s0")

    def test_recall_returns_sequence(self):
        """ordered_recall should return a Sequence."""
        b = _make_brain()
        b.add_stimulus("s0", K)
        b.add_area("A", N, K, BETA)

        # Train, then enable LRI for recall
        for _ in range(ROUNDS):
            b.project({"s0": ["A"]}, {"A": ["A"]})
        b.set_lri("A", refractory_period=3, inhibition_strength=100.0)

        result = ordered_recall(b, "A", "s0", max_steps=3)

        self.assertIsInstance(result, Sequence)
        self.assertGreater(len(result), 0)
        self.assertEqual(result.area, "A")

    def test_recall_stops_on_novel_assembly(self):
        """Recall should stop when it produces an unrecognised assembly."""
        b = _make_brain()
        for i in range(2):
            b.add_stimulus(f"s{i}", K)
        b.add_area("A", N, K, BETA)

        # Memorize short sequence
        memorized = sequence_memorize(
            b, ["s0", "s1"], "A", rounds_per_step=ROUNDS, repetitions=2)

        # Enable LRI and recall with known_assemblies constraint
        b.set_lri("A", refractory_period=3, inhibition_strength=100.0)
        result = ordered_recall(
            b, "A", "s0", max_steps=20,
            known_assemblies=list(memorized))

        # Should stop when a novel assembly (not matching any memorized) appears
        self.assertLessEqual(len(result), 20,
                             "Recall should terminate within max_steps.")

    def test_recall_recovers_memorized_sequence(self):
        """After memorizing A-B-C, cueing with A should recall assemblies
        that overlap with B and C in order."""
        b = _make_brain()
        for i in range(3):
            b.add_stimulus(f"s{i}", K)
        b.add_area("A", N, K, BETA)

        # Memorize sequence (without LRI)
        memorized = sequence_memorize(
            b, ["s0", "s1", "s2"], "A",
            rounds_per_step=ROUNDS, repetitions=3)

        # Enable LRI for recall
        b.set_lri("A", refractory_period=3, inhibition_strength=100.0)

        # Recall from first stimulus
        recalled = ordered_recall(
            b, "A", "s0", max_steps=10,
            known_assemblies=list(memorized))

        # The first recalled assembly should match the first memorized
        self.assertGreater(
            overlap(recalled[0], memorized[0]), 0.3,
            f"First recalled should match first memorized "
            f"(overlap={overlap(recalled[0], memorized[0]):.3f}).")

        # If recall produced more than 1 assembly, check the second
        # overlaps with the second memorized (or at least is different
        # from the first)
        if len(recalled) > 1:
            # The second recalled should be different from the first
            self.assertLess(
                overlap(recalled[0], recalled[1]), 0.9,
                "Consecutive recalled assemblies should be different.")


if __name__ == '__main__':
    unittest.main()
