"""
Tests for FSMNetwork: finite state machine via neural assemblies.

Based on:
    Dabagia, Papadimitriou, Vempala (2023).
    "Computation with Sequences of Assemblies in a Model of the Brain."
"""

import unittest

from src.core.brain import Brain
from src.assembly_calculus.fsm import FSMNetwork
from src.assembly_calculus.assembly import overlap


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


class TestFSMNetwork(unittest.TestCase):
    """Test FSMNetwork with simple finite automata."""

    def test_state_readout_matches(self):
        """Each trained state should be readable from the state lexicon."""
        b = _make_brain()
        states = ["q0", "q1"]
        symbols = ["a"]
        transitions = [("q0", "a", "q1")]

        fsm = FSMNetwork(b, states, symbols, transitions, "q0",
                         n=N, k=K, beta=BETA, rounds=ROUNDS)

        # Each state in the lexicon should be distinct
        ov = overlap(fsm.state_lexicon["q0"], fsm.state_lexicon["q1"])
        self.assertLess(ov, 0.3,
                        f"States should be distinct (overlap={ov:.3f}).")

    def test_two_state_toggle(self):
        """Simple A↔B toggle: q0 -a-> q1, q1 -a-> q0."""
        b = _make_brain()
        states = ["q0", "q1"]
        symbols = ["a"]
        transitions = [
            ("q0", "a", "q1"),
            ("q1", "a", "q0"),
        ]

        fsm = FSMNetwork(b, states, symbols, transitions, "q0",
                         n=N, k=K, beta=BETA, rounds=ROUNDS)

        self.assertEqual(fsm.current_state, "q0")

        new_state = fsm.step("a")
        self.assertEqual(new_state, "q1",
                         f"Expected q1 after step, got {new_state}")

    def test_reset_returns_to_initial(self):
        """reset() should restore the FSM to its initial state."""
        b = _make_brain()
        states = ["q0", "q1"]
        symbols = ["a"]
        transitions = [("q0", "a", "q1")]

        fsm = FSMNetwork(b, states, symbols, transitions, "q0",
                         n=N, k=K, beta=BETA, rounds=ROUNDS)

        fsm.step("a")
        self.assertEqual(fsm.current_state, "q1")

        fsm.reset()
        self.assertEqual(fsm.current_state, "q0")

    def test_run_produces_trajectory(self):
        """run() should return state trajectory for a symbol sequence."""
        b = _make_brain()
        states = ["q0", "q1"]
        symbols = ["a", "b"]
        transitions = [
            ("q0", "a", "q1"),
            ("q1", "b", "q0"),
        ]

        fsm = FSMNetwork(b, states, symbols, transitions, "q0",
                         n=N, k=K, beta=BETA, rounds=ROUNDS)

        trajectory = fsm.run(["a", "b", "a"])
        self.assertEqual(len(trajectory), 3)
        # q0 -a-> q1, q1 -b-> q0, q0 -a-> q1
        self.assertEqual(trajectory[0], "q1")
        self.assertEqual(trajectory[1], "q0")
        self.assertEqual(trajectory[2], "q1")


if __name__ == '__main__':
    unittest.main()
