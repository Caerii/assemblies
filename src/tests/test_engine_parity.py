"""
Engine integration tests: verify that Brain + engine produces correct results,
and that the engine standalone matches Brain + engine exactly.
"""

import numpy as np
import unittest

from src.core.brain import Brain
from src.core.engine import create_engine
from src.core.backend import to_cpu


SEED = 42
N, K, P, BETA = 5000, 50, 0.05, 0.05


def _sorted_winners(brain_or_engine, area_name, is_engine=False):
    """Extract sorted compact winners as numpy uint32."""
    if is_engine:
        return np.sort(brain_or_engine.get_winners(area_name))
    return np.sort(np.array(to_cpu(brain_or_engine.areas[area_name].winners), dtype=np.uint32))


class TestEngineParity(unittest.TestCase):
    """Verify Brain+engine integration and engine standalone consistency."""

    def test_projection_parity(self):
        """Two Brain instances with same engine type and seed produce identical results."""
        rounds = 15

        b1 = Brain(p=P, save_size=True, seed=SEED, w_max=20.0, engine="numpy_sparse")
        b1.add_area("A", n=N, k=K, beta=BETA)
        b1.add_stimulus("s", size=K)
        for _ in range(rounds):
            b1.project({"s": ["A"]}, {})

        b2 = Brain(p=P, save_size=True, seed=SEED, w_max=20.0, engine="numpy_sparse")
        b2.add_area("A", n=N, k=K, beta=BETA)
        b2.add_stimulus("s", size=K)
        for _ in range(rounds):
            b2.project({"s": ["A"]}, {})

        self.assertEqual(b1.areas["A"].saved_w, b2.areas["A"].saved_w)
        np.testing.assert_array_equal(
            _sorted_winners(b1, "A"),
            _sorted_winners(b2, "A"),
        )

    def test_engine_direct_projection_parity(self):
        """Engine standalone matches Brain+engine exactly."""
        rounds = 15

        b1 = Brain(p=P, save_size=True, seed=SEED, w_max=20.0, engine="numpy_sparse")
        b1.add_area("A", n=N, k=K, beta=BETA)
        b1.add_stimulus("s", size=K)
        for _ in range(rounds):
            b1.project({"s": ["A"]}, {})

        engine = create_engine("numpy_sparse", p=P, seed=SEED, w_max=20.0)
        engine.add_area("A", n=N, k=K, beta=BETA)
        engine.add_stimulus("s", size=K)
        engine_w = []
        for _ in range(rounds):
            r = engine.project_into("A", from_stimuli=["s"], from_areas=[])
            engine_w.append(r.num_ever_fired)

        self.assertEqual(b1.areas["A"].saved_w, engine_w)
        np.testing.assert_array_equal(
            _sorted_winners(b1, "A"),
            _sorted_winners(engine, "A", is_engine=True),
        )

    def test_multi_area_parity(self):
        """Stimulus -> A, then A -> B — two identical Brains match."""
        b1 = Brain(p=P, save_size=True, seed=SEED, w_max=20.0, engine="numpy_sparse")
        b1.add_area("A", n=N, k=K, beta=BETA)
        b1.add_area("B", n=N, k=K, beta=BETA)
        b1.add_stimulus("s", size=K)
        for _ in range(10):
            b1.project({"s": ["A"]}, {})
        for _ in range(10):
            b1.project({}, {"A": ["B"]})

        b2 = Brain(p=P, save_size=True, seed=SEED, w_max=20.0, engine="numpy_sparse")
        b2.add_area("A", n=N, k=K, beta=BETA)
        b2.add_area("B", n=N, k=K, beta=BETA)
        b2.add_stimulus("s", size=K)
        for _ in range(10):
            b2.project({"s": ["A"]}, {})
        for _ in range(10):
            b2.project({}, {"A": ["B"]})

        for area_name in ["A", "B"]:
            self.assertEqual(
                b1.areas[area_name].saved_w,
                b2.areas[area_name].saved_w,
                f"w mismatch for area {area_name}",
            )
            np.testing.assert_array_equal(
                _sorted_winners(b1, area_name),
                _sorted_winners(b2, area_name),
            )

    def test_association_parity(self):
        """Simultaneous stim -> A + A -> A (self-recurrent)."""
        b1 = Brain(p=P, save_size=True, seed=SEED, w_max=20.0, engine="numpy_sparse")
        b1.add_area("A", n=N, k=K, beta=BETA)
        b1.add_stimulus("s", size=K)
        for _ in range(10):
            b1.project({"s": ["A"]}, {"A": ["A"]})

        b2 = Brain(p=P, save_size=True, seed=SEED, w_max=20.0, engine="numpy_sparse")
        b2.add_area("A", n=N, k=K, beta=BETA)
        b2.add_stimulus("s", size=K)
        for _ in range(10):
            b2.project({"s": ["A"]}, {"A": ["A"]})

        self.assertEqual(b1.areas["A"].saved_w, b2.areas["A"].saved_w)
        np.testing.assert_array_equal(
            _sorted_winners(b1, "A"),
            _sorted_winners(b2, "A"),
        )

    def test_engine_name_property(self):
        """Brain.engine_name returns correct value."""
        b1 = Brain(p=P)
        # Default is auto-detected (numpy_sparse or cuda_implicit)
        self.assertIn(b1.engine_name, ["numpy_sparse", "numpy_explicit", "cuda_implicit"])

        b2 = Brain(p=P, engine="numpy_sparse")
        self.assertEqual(b2.engine_name, "numpy_sparse")

    def test_assembly_convergence(self):
        """Verify assemblies converge (saved_w stabilizes) after repeated projection."""
        rounds = 20
        b = Brain(p=P, save_size=True, seed=SEED, w_max=20.0, engine="numpy_sparse")
        b.add_area("A", n=N, k=K, beta=BETA)
        b.add_stimulus("s", size=K)
        for _ in range(rounds):
            b.project({"s": ["A"]}, {})

        # After enough rounds, saved_w should stabilize (not grow unboundedly)
        w_values = b.areas["A"].saved_w
        self.assertTrue(len(w_values) == rounds)
        # The last few values should be close to each other
        self.assertLess(w_values[-1] - w_values[-2], K,
                        "saved_w should stabilize, not keep growing by k each round")


if __name__ == "__main__":
    unittest.main()
