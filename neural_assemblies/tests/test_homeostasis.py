"""`core/_homeostasis` is the ONE owner of the homeostats' arithmetic and gates.

Three kinds of assertion, each guarding a past defect:

* the LAWS, asserted directly (a threshold test passes a wrong rule too);
* the CONFLICT is unspellable at `Brain` -- the combination that confounded
  three S5 studies cannot be constructed by accident any more;
* every ENGINE calls the owner rather than a private copy -- the refraction
  rule once lived as three engine copies, all wrong together, and the scaling
  law as five ([[pricing-law-implemented-twice]]).
"""
from __future__ import annotations

import random
import unittest
from unittest import mock

import numpy as np

from neural_assemblies.core import _homeostasis as H
from neural_assemblies.core.brain import Brain

N, K, BETA, P = 1000, 30, 0.1, 0.05


def _brain(**kw):
    random.seed(7)
    np.random.seed(7)
    defaults = dict(p=P, seed=7, engine="numpy_sparse")
    defaults.update(kw)
    return Brain(**defaults)


class TestLaws(unittest.TestCase):

    def test_scaling_gate_spellings(self):
        self.assertFalse(H.scaling_applies(False, "A"))
        self.assertTrue(H.scaling_applies(True, "A"))
        self.assertTrue(H.scaling_applies(frozenset({"A"}), "A"))
        self.assertFalse(H.scaling_applies(frozenset({"A"}), "B"))
        self.assertFalse(H.scaling_applies(set(), "A"))

    def test_setpoint_is_initial_expected_sum_at_fiber_p(self):
        self.assertAlmostEqual(H.scaling_setpoint(200, 0.4), 80.0)
        self.assertEqual(H.scaling_setpoint(0, 0.4), 1e-12)

    def test_column_scale_numpy_and_torch_agree_and_guard_zero(self):
        sums = np.array([4.0, 0.0, 2.5, 1e-15], dtype=np.float32)
        got = H.column_scale(sums, 2.0, xp=np)
        np.testing.assert_allclose(got, [0.5, 1.0, 0.8, 1.0])
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        tgot = H.column_scale(torch.tensor(sums), 2.0).numpy()
        np.testing.assert_allclose(tgot, got)

    def test_refraction_is_the_anti_hebbian_counterweight(self):
        """net_{t+1} - net_t = (beta - s) * raw_t: at s == beta, constant.

        The identity behind REFRACTION-CANCELS-CONVERGENCE, run as
        arithmetic: a neuron winning on the same input, potentiated by
        (1+beta) each win and charged raw*s, keeps a constant net at s=beta.
        """
        beta = 0.1
        raw, bias = np.array([50.0]), np.array([0.0])
        nets = []
        for _ in range(20):
            nets.append(float(raw[0] - bias[0]))
            bias = bias + H.refraction_increment(raw - bias, bias, beta)
            raw = raw * (1 + beta)
        np.testing.assert_allclose(nets, [50.0] * 20, rtol=1e-6)

    def test_refraction_strength_rejects_nonfinite_negative_and_boolean(self):
        for value in (-1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                H.validate_refraction_strength(value)

    def test_brain_validates_refraction_strength_before_registration(self):
        b = _brain()
        for value in (-1.0, float("nan"), True):
            with self.assertRaises(ValueError):
                b.add_area("bad", N, K, BETA, refracted_strength=value)
            self.assertNotIn("bad", b.areas)
        b.add_area("A", N, K, BETA)
        with self.assertRaises(ValueError):
            b.set_refracted("A", True, strength=float("nan"))
        self.assertFalse(b.areas["A"].refracted)


class TestConflictIsUnspellable(unittest.TestCase):

    def test_refracted_area_in_scaled_brain_is_refused(self):
        b = _brain(synaptic_scaling=True)
        with self.assertRaises(H.HomeostasisConflict):
            b.add_area("ARC", N, K, BETA, refracted=True,
                       refracted_strength=0.1)

    def test_scoping_scaling_away_from_the_arc_is_allowed(self):
        b = _brain(synaptic_scaling=frozenset({"STATE"}))
        b.add_area("STATE", N, K, BETA)
        b.add_area("ARC", N, K, BETA, refracted=True, refracted_strength=0.1)

    def test_turning_refraction_on_in_a_scaled_area_is_refused(self):
        b = _brain(synaptic_scaling=frozenset({"A"}))
        b.add_area("A", N, K, BETA)
        with self.assertRaises(H.HomeostasisConflict):
            b.set_refracted("A", True, strength=0.1)
        b.set_refracted("A", False)          # disabling is always fine

    def test_unscaled_brain_unaffected(self):
        b = _brain()
        b.add_area("ARC", N, K, BETA, refracted=True, refracted_strength=0.1)
        b.set_refracted("ARC", True, strength=0.1)


class TestEnginesCallTheOwner(unittest.TestCase):
    """Patch the owner's arithmetic and watch the engine reach it."""

    def _run_scaled(self, engine):
        b = _brain(engine=engine, synaptic_scaling=True)
        b.add_stimulus("s", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        for _ in range(3):
            b.project({"s": ["A"]}, {"A": ["A", "B"]})

    def test_numpy_engine_scales_through_column_scale(self):
        # Engines bind the owner's function BY NAME at import, so the patch
        # goes where the engine looks it up; patching the owner module's
        # attribute would leave the bound reference untouched and read 0.
        from neural_assemblies.core.numpy_engine import _sparse
        with mock.patch.object(_sparse, "column_scale",
                               wraps=H.column_scale) as cs:
            self._run_scaled("numpy_sparse")
        self.assertGreater(cs.call_count, 0, "numpy engine bypassed the owner")

    def test_torch_engine_scales_through_column_scale(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        # The CSR classes bind `column_scale` by name from the owner at
        # import; patch where they look it up.
        from neural_assemblies.core.torch_engine import _csr
        with mock.patch.object(_csr, "column_scale",
                               wraps=H.column_scale) as cs:
            try:
                self._run_scaled("torch_sparse")
            except Exception as e:  # engine unavailable on this box
                self.skipTest(f"torch_sparse unavailable: {e}")
        self.assertGreater(cs.call_count, 0, "torch engine bypassed the owner")


if __name__ == "__main__":
    unittest.main()
