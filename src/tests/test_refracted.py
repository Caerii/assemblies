"""
Tests for refracted mode: cumulative bias inhibition.

Refracted mode is distinct from LRI. LRI uses a sliding window of recent
winners with a decaying penalty. Refracted mode accumulates a permanent bias:
each time a neuron fires, its bias grows, making it progressively harder to
fire again.
"""

import unittest
import numpy as np

from src.core.brain import Brain
from src.assembly_calculus.ops import _snap
from src.assembly_calculus.assembly import overlap


N = 10000
K = 100
P = 0.05
BETA = 0.1
SEED = 42


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


class TestRefractedMode(unittest.TestCase):
    """Test refracted mode (cumulative bias inhibition)."""

    def test_no_refracted_by_default(self):
        """Areas without refracted mode should behave normally:
        repeated projection converges to a stable assembly."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)

        for _ in range(10):
            b.project({"s": ["A"]}, {"A": ["A"]})
        snap1 = _snap(b, "A")
        for _ in range(5):
            b.project({"s": ["A"]}, {"A": ["A"]})
        snap2 = _snap(b, "A")

        # Should converge (high overlap)
        self.assertGreater(overlap(snap1, snap2), 0.9,
                           "Without refracted mode, assembly should be stable.")

    def test_cumulative_bias_shifts_winners(self):
        """With refracted mode, repeated projection should shift the
        assembly because previously-fired neurons accumulate penalty."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA,
                    refracted=True, refracted_strength=5.0)

        # Initial projection to establish assembly
        for _ in range(5):
            b.project({"s": ["A"]}, {"A": ["A"]})
        snap_early = _snap(b, "A")

        # Many more projections — cumulative bias should push winners away
        for _ in range(20):
            b.project({"s": ["A"]}, {"A": ["A"]})
        snap_late = _snap(b, "A")

        # Assembly should have shifted (lower overlap than without refracted)
        ov = overlap(snap_early, snap_late)
        self.assertLess(ov, 0.8,
                        f"Refracted mode should shift assembly "
                        f"(overlap={ov:.3f}, expected < 0.8).")

    def test_clear_bias_resets(self):
        """After clearing refracted bias and disabling refracted mode,
        the assembly should reconverge to a stable attractor."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA,
                    refracted=True, refracted_strength=5.0)

        # Build up bias — assembly drifts
        for _ in range(15):
            b.project({"s": ["A"]}, {"A": ["A"]})

        # Clear bias AND disable refracted mode
        b.clear_refracted_bias("A")
        b.set_refracted("A", False)

        # Re-project — should converge to stable assembly
        for _ in range(10):
            b.project({"s": ["A"]}, {"A": ["A"]})
        snap_reset = _snap(b, "A")

        for _ in range(5):
            b.project({"s": ["A"]}, {"A": ["A"]})
        snap_stable = _snap(b, "A")
        self.assertGreater(overlap(snap_reset, snap_stable), 0.7,
                           "After clearing bias and disabling refracted, "
                           "assembly should restabilize.")

    def test_refracted_and_lri_independent(self):
        """Both refracted mode and LRI can be active simultaneously
        without interfering with each other."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA,
                    refractory_period=3, inhibition_strength=50.0,
                    refracted=True, refracted_strength=3.0)

        # Should not raise — both modes coexist
        for _ in range(10):
            b.project({"s": ["A"]}, {"A": ["A"]})

        snap = _snap(b, "A")
        self.assertEqual(len(snap.winners), K)

    def test_set_refracted_at_runtime(self):
        """Refracted mode can be enabled after area creation."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)

        # Build stable assembly without refracted
        for _ in range(10):
            b.project({"s": ["A"]}, {"A": ["A"]})
        snap_before = _snap(b, "A")

        # Enable refracted at runtime
        b.set_refracted("A", True, strength=5.0)

        # Continue projecting — should start shifting
        for _ in range(20):
            b.project({"s": ["A"]}, {"A": ["A"]})
        snap_after = _snap(b, "A")

        ov = overlap(snap_before, snap_after)
        self.assertLess(ov, 0.9,
                        f"Runtime-enabled refracted should shift assembly "
                        f"(overlap={ov:.3f}).")


if __name__ == '__main__':
    unittest.main()
