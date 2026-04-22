import unittest

from neural_assemblies.assembly_calculus import FSMNetwork, PFANetwork, Transition, TransitionMap
from neural_assemblies.core.brain import Brain


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


class TestTransitionMap(unittest.TestCase):
    def test_normalizes_tuples_and_objects(self):
        transition_map = TransitionMap([
            ("q0", "a", "q1"),
            Transition("q1", "a", "q0"),
        ])

        self.assertEqual(
            transition_map.as_tuples(),
            [("q0", "a", "q1"), ("q1", "a", "q0")],
        )

    def test_probability_validation_requires_complete_mass(self):
        with self.assertRaises(ValueError):
            TransitionMap([
                ("q0", "a", "q1", 0.6),
                ("q0", "a", "q2", 0.3),
            ]).validate_probability_mass()

    def test_deterministic_table_rejects_probabilistic_entries(self):
        with self.assertRaises(ValueError):
            TransitionMap([
                ("q0", "a", "q1", 0.5),
                ("q0", "a", "q2", 0.5),
            ]).deterministic_table()


class TestTransitionIntegration(unittest.TestCase):
    def test_fsm_accepts_transition_objects(self):
        brain = _make_brain()
        fsm = FSMNetwork(
            brain,
            states=["q0", "q1"],
            symbols=["a"],
            transitions=[Transition("q0", "a", "q1")],
            initial_state="q0",
            n=N,
            k=K,
            beta=BETA,
            rounds=ROUNDS,
        )

        self.assertEqual(fsm.step("a"), "q1")
        self.assertEqual(fsm.transition_table[("q0", "a")], "q1")

    def test_pfa_accepts_transition_objects(self):
        brain = _make_brain()
        pfa = PFANetwork(
            brain,
            states=["q0", "q1", "q2"],
            symbols=["a"],
            transitions=[
                Transition("q0", "a", "q1", 0.5),
                Transition("q0", "a", "q2", 0.5),
            ],
            initial_state="q0",
            n=N,
            k=K,
            beta=BETA,
            rounds=ROUNDS,
        )

        self.assertIn(pfa.step("a", seed=1), {"q1", "q2"})


if __name__ == "__main__":
    unittest.main()
