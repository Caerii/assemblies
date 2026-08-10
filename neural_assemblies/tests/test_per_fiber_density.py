"""Per-fiber connection density on the sparse engine.

The regime condition kp >= 3 ln n is per-AREA, so an organ needing a dense
local regime inside a sparse brain needs density set per fiber. `k` is not a
substitute: raising it spends capacity and forces `n` up with it.

Two properties are load-bearing and both are asserted here: a fiber gets the
density it asked for, and a brain that never asks is UNCHANGED.
"""

from __future__ import annotations

import unittest

import numpy as np

from neural_assemblies.core.brain import Brain


def _brain(p=0.05, seed=1):
    return Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse",
                 norm_init=False)


def _wire(b, beta=0.0):
    for name in ("SRC_A", "SRC_B", "TGT"):
        b.add_area(name, 3000, 60, beta)
    b.add_stimulus("sa", 60)
    b.add_stimulus("sb", 60)
    return b


class TestPerFiberDensity(unittest.TestCase):
    def test_each_fiber_draws_its_own_density(self):
        """Two fibers into ONE target, in one brain, at different densities.

        Measured on the raw structural block rather than on materialized
        neurons: the ones that materialize are exactly those that won k-WTA,
        which favours neurons with more afferents, so a realized density read
        over winners is biased upward. That bias applies equally to the
        homogeneous case and is not what this test is about.
        """
        b = _wire(_brain())
        b.add_connectivity("SRC_A", "TGT", 0.40)
        eng = b._engine
        dense = np.asarray(eng._init_area_block("SRC_A", "TGT", 0, 400, 0, 400))
        sparse = np.asarray(eng._init_area_block("SRC_B", "TGT", 0, 400, 0, 400))
        self.assertAlmostEqual(float((dense != 0).mean()), 0.40, delta=0.02)
        self.assertAlmostEqual(float((sparse != 0).mean()), 0.05, delta=0.02)

    def test_requesting_the_global_p_is_not_a_request(self):
        """It stays a no-op, and leaves the engine on its homogeneous paths."""
        b = _wire(_brain(p=0.05))
        b.add_connectivity("SRC_A", "TGT", 0.05)
        self.assertFalse(b._engine.heterogeneous(),
                         "asking for the density already in force must not "
                         "switch the engine onto heterogeneous code paths")

    def test_no_op_call_leaves_results_bit_identical(self):
        """The safety property the whole change rests on."""
        def run(with_call):
            b = _wire(_brain(p=0.05), beta=0.1)
            if with_call:
                b.add_connectivity("SRC_A", "TGT", 0.05)   # the global value
            for _ in range(3):
                b.project({"sa": ["SRC_A"], "sb": ["SRC_B"]}, {})
            for _ in range(3):
                b.project({}, {"SRC_A": ["TGT"], "SRC_B": ["TGT"]})
            return np.asarray(b.areas["TGT"].winners).copy()

        np.testing.assert_array_equal(run(False), run(True))

    def test_a_denser_fiber_changes_the_outcome(self):
        """Manipulation check: if nothing moves, the setting is inert."""
        def run(dense):
            b = _wire(_brain(p=0.05), beta=0.1)
            if dense:
                b.add_connectivity("SRC_A", "TGT", 0.40)
            for _ in range(3):
                b.project({"sa": ["SRC_A"], "sb": ["SRC_B"]}, {})
            for _ in range(3):
                b.project({}, {"SRC_A": ["TGT"], "SRC_B": ["TGT"]})
            return set(int(i) for i in np.asarray(b.areas["TGT"].winners))

        self.assertNotEqual(run(False), run(True),
                            "per-fiber density had no effect on the winners")

    def test_setting_density_after_traffic_raises(self):
        """Connectivity is STRUCTURAL: it decides which synapses exist.

        Changing it once a fiber has carried drive would leave potentiation on
        synapses that no longer exist and silently rewrite formed assemblies.
        """
        b = _wire(_brain(), beta=0.1)
        for _ in range(3):
            b.project({"sa": ["SRC_A"]}, {})
        b.project({}, {"SRC_A": ["TGT"]})
        with self.assertRaises(RuntimeError):
            b.add_connectivity("SRC_A", "TGT", 0.40)

    def test_unknown_endpoints_raise(self):
        b = _wire(_brain())
        with self.assertRaises(KeyError):
            b.add_connectivity("NOPE", "TGT", 0.4)
        with self.assertRaises(KeyError):
            b.add_connectivity("SRC_A", "NOPE", 0.4)

    def test_stimulus_fibers_take_a_density_too(self):
        """A conjunction is routinely driven by a stimulus and an area, so the
        stimulus side has to be settable or only half the drive is tunable."""
        b = _wire(_brain())
        b.add_connectivity("sa", "TGT", 0.40)
        self.assertEqual(b._engine._p_for("sa", "TGT"), 0.40)
        self.assertEqual(b._engine._p_for("sb", "TGT"), 0.05)

    def test_regime_audit_reads_the_override(self):
        """The audit must price the fiber at ITS density, not the global one."""
        from neural_assemblies.diagnostics import regime_audit

        b = _wire(_brain())
        b.add_connectivity("SRC_A", "TGT", 0.40)
        found = {r.area: r for r in regime_audit(b, {"TGT": ["SRC_A", "SRC_B"]})}
        tgt = found["TGT"]
        self.assertAlmostEqual(tgt.afferents["SRC_A"], 60 * 0.40, places=6)
        self.assertAlmostEqual(tgt.afferents["SRC_B"], 60 * 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
