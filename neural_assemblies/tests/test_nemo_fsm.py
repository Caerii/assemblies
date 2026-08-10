"""Tests for NEMO FSM / Markov PFA with refracted arc."""

from __future__ import annotations

import unittest

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.nemo_fsm import (
    AlternatingMarkovNetwork,
    NemoArcFSM,
    NemoMarkovPFA,
)


class TestNemoArcFSM(unittest.TestCase):
    """The FSM must DECIDE, and an untrained one must not.

    The previous test asserted that `step_symbol("a", "q0") == "q1"` -- but
    that method returned a table lookup, so the assertion held on an untrained
    network. A behavioural test whose degenerate arm also passes is not a test;
    every case here therefore comes with its untrained control.
    """

    def _build(self, seed=42):
        b = Brain(p=0.05, save_winners=True, seed=seed, engine="numpy_sparse")
        return b, NemoArcFSM(
            b, states=["q0", "q1"], symbols=["a"],
            transitions=[("q0", "a", "q1"), ("q1", "a", "q0")],
            n=2000, k=40, beta=0.1, rounds=6,
        )

    def test_trained_transition_is_read_from_the_assembly(self):
        _, fsm = self._build()
        fsm.train_from_list([("a", "q0", "q1"), ("a", "q1", "q0")],
                            presentations=10)
        self.assertEqual(fsm.run(["a"], start_state="q0"), ["q1"])
        self.assertEqual(fsm.run(["a"], start_state="q1"), ["q0"])

    def test_untrained_fsm_does_not_decide(self):
        """The control the old test lacked: with no training, the alternating
        machine cannot produce the alternating trajectory."""
        _, fsm = self._build()
        trajectory = fsm.run(["a", "a", "a", "a"], start_state="q0")
        self.assertNotEqual(trajectory, ["q1", "q0", "q1", "q0"],
                            "an untrained FSM reproduced the transition table")

    def test_running_does_not_learn_or_charge_bias(self):
        """`run` is a readout: no plasticity, and no refraction charged.

        The reference tests with ``update=False``, which stops learning and
        bias accumulation together. If running charged bias, one step of a
        sequence would change the next.
        """
        b, fsm = self._build()
        fsm.train_from_list([("a", "q0", "q1"), ("a", "q1", "q0")],
                            presentations=5)
        bias = b._engine._areas[fsm.arc_area]._cumulative_bias
        before = float(np.sum(np.asarray(bias)))
        first = fsm.run(["a", "a", "a"], start_state="q0")
        bias = b._engine._areas[fsm.arc_area]._cumulative_bias
        self.assertEqual(float(np.sum(np.asarray(bias))), before,
                         "running the FSM charged refraction bias")
        self.assertEqual(first, fsm.run(["a", "a", "a"], start_state="q0"),
                         "a readout changed the machine it was reading")

    def test_training_does_not_wipe_the_arc_bias(self):
        """Refraction must accumulate ACROSS transitions.

        `train_transition` used to call `clear_refracted_bias` on entry, so the
        bias never survived one transition and refraction was inert -- the
        mechanism that keeps the arc from collapsing onto its more-exposed
        conjunct was configured and doing nothing.
        """
        b, fsm = self._build()
        fsm.train_from_list([("a", "q0", "q1")], presentations=1)
        after_one = float(np.sum(np.asarray(
            b._engine._areas[fsm.arc_area]._cumulative_bias)))
        fsm.train_from_list([("a", "q1", "q0")], presentations=1)
        after_two = float(np.sum(np.asarray(
            b._engine._areas[fsm.arc_area]._cumulative_bias)))
        self.assertGreater(after_one, 0.0, "no bias charged during training")
        self.assertGreater(after_two, after_one,
                           "bias did not survive across transitions")


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
