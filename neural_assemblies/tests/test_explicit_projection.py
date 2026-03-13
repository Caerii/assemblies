# test_explicit_projection.py

import unittest
import numpy as np

from brain import Brain


class TestExplicitProjection(unittest.TestCase):
    def setUp(self):
        self.brain = Brain(p=0.1, save_size=True, save_winners=True, seed=123)

    def test_explicit_target_marks_ever_fired_and_no_first_winners(self):
        # Create explicit target area and a stimulus
        self.brain.add_area("T", n=20, k=5, beta=0.05, explicit=True)
        self.brain.add_stimulus("S", size=50)

        # Project from stimulus S into explicit area T
        areas_by_stim = {"S": ["T"]}
        dst_areas_by_src_area = {}
        self.brain.project(areas_by_stim, dst_areas_by_src_area)

        area = self.brain.area_by_name["T"]
        # In explicit mode, no first-time winners are tracked in the non-explicit pipeline
        self.assertEqual(area.num_first_winners, 0)
        # Winners should be set and ever_fired flags true for them
        self.assertEqual(len(area.winners), area.k)
        self.assertTrue(np.all(area.ever_fired[area.winners]))
        # num_ever_fired matches count of true flags
        self.assertEqual(area.num_ever_fired, int(np.sum(area.ever_fired)))

    def test_index_bounds_error_from_source_area(self):
        # Explicit target and explicit source with too-large winner index
        self.brain.add_area("SRC", n=10, k=3, beta=0.05, explicit=True)
        self.brain.add_area("T", n=12, k=3, beta=0.05, explicit=True)

        # Manually set a winner index that exceeds SRC->T connectome source rows (n=10)
        src_area = self.brain.area_by_name["SRC"]
        src_area.winners = np.array([0, 5, 999], dtype=np.uint32)

        # Connectome SRC->T exists from add_area path
        areas_by_stim = {}
        dst_areas_by_src_area = {"SRC": ["T"]}

        with self.assertRaises(IndexError):
            self.brain.project(areas_by_stim, dst_areas_by_src_area)

    def test_explicit_plasticity_scaling_applied(self):
        # Explicit source and target; verify plasticity scaling modifies connectome entries for winners
        b = self.brain
        b.add_area("SRC", n=10, k=3, beta=0.2, explicit=True)
        b.add_area("T", n=12, k=3, beta=0.2, explicit=True)
        # Set a known winner set in SRC
        src = b.area_by_name["SRC"]
        src.winners = np.array([0, 1, 2], dtype=np.uint32)
        # Ensure source area reports a valid assembly size
        src.w = src.winners.size
        # Precondition: set some nonzero weights on rows [0..2] so scaling is observable
        conn = b.connectomes["SRC"]["T"]
        conn[:3, :] = 1.0
        conn_before = conn.copy()
        b.project({}, {"SRC": ["T"]})
        conn_after = b.connectomes["SRC"]["T"]
        # For winner columns in T (post-winners), rows [0,1,2] should have been scaled by (1+beta)
        beta = b.area_by_name["T"].beta_by_area["SRC"]
        # Note: winners are set on T during project; use its _new_winners
        t = b.area_by_name["T"]
        cols = t._new_winners
        # Winner columns should be scaled on rows 0..2 exactly by (1+beta)
        for c in cols:
            before = conn_before[:3, c]
            after = conn_after[:3, c]
            np.testing.assert_allclose(after, before * (1.0 + beta), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()


