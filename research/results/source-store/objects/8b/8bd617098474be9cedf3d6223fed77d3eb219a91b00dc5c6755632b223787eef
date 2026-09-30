"""
PFANetwork routing tests. Coin mechanism/null controls live in
 test_coin_construction.py. Legacy fairness and seed-bias frequency assertions
 were removed: they described an invalid instrument and do not establish
 calibration for the explicit attractor construction.

Based on:
    Dabagia, Papadimitriou, Vempala (2023).
    "Computation with Sequences of Assemblies in a Model of the Brain."
"""

import unittest

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.pfa import PFANetwork, SeedMixtureChoice


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

    def test_pfa_explicit_seed_mixture_returns_a_declared_target(self):
        """This tests routing only; seed weights do not certify outcome frequencies."""
        b = _make_brain()
        states = ["q0", "q1", "q2"]
        symbols = ["a"]
        transitions = [
            ("q0", "a", "q1", 0.5),
            ("q0", "a", "q2", 0.5),
        ]

        pfa = PFANetwork(b, states, symbols, transitions, "q0",
                         n=N, k=K, beta=BETA, rounds=ROUNDS,
                         choice=SeedMixtureChoice(2000, 200, 3., rounds_train=10))

        for seed in range(3):
            pfa.reset()
            self.assertIn(pfa.step("a", seed=seed), {"q1", "q2"})


if __name__ == '__main__':
    unittest.main()
