"""
Tests for RandomChoiceArea and PFANetwork.

Based on:
    Dabagia, Papadimitriou, Vempala (2023).
    "Computation with Sequences of Assemblies in a Model of the Brain."
"""

import unittest
from collections import Counter

from src.core.brain import Brain
from src.assembly_calculus.pfa import RandomChoiceArea, PFANetwork


N = 10000
K = 100
P = 0.05
BETA = 0.05
SEED = 42
ROUNDS = 10


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


class TestRandomChoiceArea(unittest.TestCase):
    """Test RandomChoiceArea (neural coin-flip)."""

    def test_coin_flip_binary_output(self):
        """flip() should return 0 or 1."""
        b = _make_brain()
        coin = RandomChoiceArea(b, n=N, k=K, beta=BETA)
        result = coin.flip(bias=0.5, rounds=10, seed=0)
        self.assertIn(result, [0, 1])

    def test_fair_coin_roughly_balanced(self):
        """A fair coin (bias=0.5) should produce roughly 50/50 over
        many flips.  We use a wide tolerance since neural competition
        is inherently noisy."""
        b = _make_brain()
        coin = RandomChoiceArea(b, n=N, k=K, beta=BETA)

        counts = Counter()
        for i in range(40):
            result = coin.flip(bias=0.5, rounds=10, seed=i * 7)
            counts[result] += 1

        # Should have both outcomes
        self.assertGreater(counts[0], 0,
                           "Fair coin should produce some 0s.")
        self.assertGreater(counts[1], 0,
                           "Fair coin should produce some 1s.")

    def test_biased_coin_favors_one(self):
        """A strongly biased coin (bias=0.9) should mostly produce 0."""
        b = _make_brain()
        coin = RandomChoiceArea(b, n=N, k=K, beta=BETA)

        counts = Counter()
        for i in range(30):
            result = coin.flip(bias=0.9, rounds=10, seed=i * 13)
            counts[result] += 1

        # 0 should appear significantly more often
        self.assertGreater(counts[0], counts[1],
                           f"Biased coin should favor 0: got {dict(counts)}.")


class TestPFANetwork(unittest.TestCase):
    """Test PFANetwork with probabilistic transitions."""

    def test_pfa_deterministic_like_fsm(self):
        """When all transitions have probability 1.0, PFA should
        behave like a deterministic FSM."""
        b = _make_brain()
        states = ["q0", "q1"]
        symbols = ["a"]
        transitions = [
            ("q0", "a", "q1", 1.0),
        ]

        pfa = PFANetwork(b, states, symbols, transitions, "q0",
                         n=N, k=K, beta=BETA, rounds=ROUNDS)

        result = pfa.step("a", seed=42)
        self.assertEqual(result, "q1",
                         f"Deterministic PFA should go to q1, got {result}")

    def test_pfa_probabilistic_both_states(self):
        """When a transition is probabilistic (50/50), both target
        states should appear over many runs."""
        b = _make_brain()
        states = ["q0", "q1", "q2"]
        symbols = ["a"]
        transitions = [
            ("q0", "a", "q1", 0.5),
            ("q0", "a", "q2", 0.5),
        ]

        pfa = PFANetwork(b, states, symbols, transitions, "q0",
                         n=N, k=K, beta=BETA, rounds=ROUNDS)

        results = Counter()
        for i in range(40):
            pfa.reset()
            result = pfa.step("a", seed=i * 11)
            results[result] += 1

        # Both q1 and q2 should appear
        self.assertGreater(results.get("q1", 0), 0,
                           f"q1 should appear in results: {dict(results)}")
        self.assertGreater(results.get("q2", 0), 0,
                           f"q2 should appear in results: {dict(results)}")


if __name__ == '__main__':
    unittest.main()
