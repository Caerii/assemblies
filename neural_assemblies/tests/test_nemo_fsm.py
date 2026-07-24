"""Tests for NEMO FSM / Markov PFA with refracted arc."""

from __future__ import annotations

import unittest

from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.nemo_fsm import (
    AlternatingMarkovNetwork,
    NemoArcFSM,
    NemoMarkovPFA,
)


class TestNemoArcFSM(unittest.TestCase):
    def test_deterministic_transition(self):
        b = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
        fsm = NemoArcFSM(
            b, states=["q0", "q1"], symbols=["a"],
            transitions=[("q0", "a", "q1"), ("q1", "a", "q0")],
            n=2000, k=40, beta=0.1, rounds=6,
        )
        fsm.train_transition("a", "q0", "q1")
        nxt = fsm.step_symbol("a", "q0")
        self.assertEqual(nxt, "q1")


class TestNemoMarkovPFA(unittest.TestCase):
    def test_alternating_markov_samples_both_states(self):
        b = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
        transitions = [
            ("q0", "flip", "q0", 0.5),
            ("q0", "flip", "q1", 0.5),
            ("q1", "flip", "q0", 0.5),
            ("q1", "flip", "q1", 0.5),
        ]
        net = AlternatingMarkovNetwork(
            b, ["q0", "q1"], transitions, "q0", n=2000, k=40, beta=0.1,
        )
        seen = set()
        for i in range(30):
            seen.add(net.sample_step(seed=100 + i * 11))
        self.assertIn("q0", seen)
        self.assertIn("q1", seen)

    def test_refracted_arc_reset(self):
        b = Brain(p=0.05, save_winners=True, seed=7, engine="numpy_sparse")
        pfa = NemoMarkovPFA(
            b, ["q0", "q1"],
            [("q0", "flip", "q1", 1.0), ("q1", "flip", "q0", 1.0)],
            "q0", n=2000, k=40,
        )
        pfa.reset()
        self.assertEqual(pfa.current_state, "q0")


class TestMarkovChainModel(unittest.TestCase):
    def test_samples_both_states(self):
        from neural_assemblies.programs.markov_coin import MarkovChainModel

        b = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
        traces = [("q0", "flip", "q0"), ("q0", "flip", "q1"), ("q1", "flip", "q0"), ("q1", "flip", "q1")]
        model = MarkovChainModel(b, traces, "q0", n=2000, k=40)
        seen = set(model.run(25, seed_base=100))
        self.assertIn("q0", seen)
        self.assertIn("q1", seen)
