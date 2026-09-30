"""
Tests for refracted mode: cumulative bias inhibition.

Refracted mode is distinct from LRI. LRI uses a sliding window of recent
winners with a decaying penalty. Refracted mode accumulates a permanent bias:
each time a neuron fires, its bias grows, making it progressively harder to
fire again.
"""

import contextlib
import os
import unittest
import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.assembly_calculus.assembly import overlap


N = 10000
K = 100
P = 0.05
BETA = 0.1
SEED = 42


def _make_brain(**kwargs):
    defaults = dict(p=P, save_winners=True, seed=SEED, engine="numpy_sparse")
    defaults.update(kwargs)
    return Brain(**defaults)


@contextlib.contextmanager
def _constant_refraction():
    """Run a block under the pre-correction constant-increment rule."""
    old = os.environ.get("ASSEMBLIES_CONSTANT_REFRACTION")
    os.environ["ASSEMBLIES_CONSTANT_REFRACTION"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("ASSEMBLIES_CONSTANT_REFRACTION", None)
        else:
            os.environ["ASSEMBLIES_CONSTANT_REFRACTION"] = old


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


class TestRefractionRule(unittest.TestCase):
    """The accumulation RULE, asserted directly.

    The tests above assert thresholds ("the assembly shifted"), which a wrong
    rule also passes -- a constant increment shifts assemblies too, it just
    stops doing so once Hebbian growth outruns it. These assert the law from
    `core/_homeostasis.py`, whose evidence is in
    `research/experiments/seq_arc_refraction_reference.py`.
    """

    def test_increment_is_proportional_to_raw_drive(self):
        """Charge is (net + bias) * strength -- the RAW drive, pre-subtraction."""
        from neural_assemblies.core._homeostasis import refraction_increment

        net = np.array([100.0, 10.0])
        bias = np.array([44.0, 0.0])
        got = refraction_increment(net, bias, 0.1)
        np.testing.assert_allclose(got, np.array([14.4, 1.0]))

    def test_constant_rule_available_for_ab(self):
        """The pre-correction rule stays reachable behind an env flag.

        No module reload: the flag is read inside `refraction_increment` on
        every call, so engines that bound the function at import time honour it
        too. That is what makes the A/B usable from a running experiment.
        """
        from neural_assemblies.core._homeostasis import refraction_increment

        with _constant_refraction():
            np.testing.assert_allclose(
                refraction_increment(np.array([100.0]), np.array([44.0]), 0.1),
                np.array([0.1]))

    def _bias_sum(self, brain, area):
        bias = brain._engine._areas[area]._cumulative_bias
        return float(np.sum(np.asarray(bias)))

    def _one_projection_charge(self):
        """Bias charged by a single projection, after a short warm-up."""
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA, refracted=True, refracted_strength=0.1)
        for _ in range(5):
            b.project({"s": ["A"]}, {"A": ["A"]})
        before = self._bias_sum(b, "A")
        b.project({"s": ["A"]}, {"A": ["A"]})
        return self._bias_sum(b, "A") - before

    def test_engine_charges_via_the_shared_rule(self):
        """The engine must call `_homeostasis`, not carry a private copy.

        Under the constant rule the charge is EXACTLY ``strength * k``, a value
        the drive-proportional rule cannot produce. Flipping the flag and
        seeing the engine's charge change is what proves the shared owner is
        the one in force -- three engines carried three copies of this rule and
        all three were wrong together ([[pricing-law-implemented-twice]]).
        """
        with _constant_refraction():
            constant_charge = self._one_projection_charge()
        proportional_charge = self._one_projection_charge()

        self.assertAlmostEqual(constant_charge, 0.1 * K, places=4)
        self.assertGreater(proportional_charge, 0.0, "no bias was charged")
        self.assertNotAlmostEqual(
            proportional_charge, 0.1 * K, places=4,
            msg="engine is still charging a constant increment")

    def test_charge_is_gated_on_plasticity(self):
        """A no-learn pass must not charge bias.

        The reference charges inside `RefractedArea.update`, so `update=False`
        stops learning and charging together. Ours did not, which meant a
        read-out sequence altered the trajectory it was observing, one step
        changing the next.
        """
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA, refracted=True, refracted_strength=0.1)
        for _ in range(5):
            b.project({"s": ["A"]}, {"A": ["A"]})

        before = self._bias_sum(b, "A")
        b.disable_plasticity = True
        try:
            for _ in range(5):
                b.project({"s": ["A"]}, {"A": ["A"]})
        finally:
            b.disable_plasticity = False
        self.assertEqual(self._bias_sum(b, "A"), before,
                         "bias was charged during a no-learn projection")

    def test_engine_without_refraction_refuses(self):
        """Configuring refraction where it cannot run must raise, not no-op.

        Ablated from the reference FSM, refraction takes the mod-3 task from
        3/3 seeds to 0/3, so an engine that silently ignores the request does
        not return a slightly different number -- it runs a different
        experiment. See [[silent-no-op-dead-fibers]].
        """
        b = Brain(p=P, save_winners=True, seed=SEED, engine="numpy_exact")
        with self.assertRaises(NotImplementedError):
            b.add_area("A", N, K, BETA, refracted=True, refracted_strength=0.1)

    def test_requesting_the_default_is_not_a_request(self):
        """refracted=False stays a no-op on an engine that lacks the mechanism."""
        b = Brain(p=P, save_winners=True, seed=SEED, engine="numpy_exact")
        b.add_area("A", N, K, BETA, refracted=False)   # must not raise
        self.assertIn("A", b.areas)

    def test_masked_readout_requires_an_actual_boolean(self):
        b = _make_brain()
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA, refracted=True, refracted_strength=0.1)
        with self.assertRaises(TypeError):
            b.set_masked_readout("A", 1)


if __name__ == '__main__':
    unittest.main()
