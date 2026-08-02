# test_area.py

import unittest
import numpy as np

from ..core.area import Area

class TestArea(unittest.TestCase):

    def setUp(self):
        self.area = Area(name="TestArea", n=1000, k=100, beta=0.05, explicit=True)

    def test_initialization(self):
        self.assertEqual(self.area.name, "TestArea")
        self.assertEqual(self.area.n, 1000)
        self.assertEqual(self.area.k, 100)
        self.assertEqual(self.area.beta, 0.05)
        self.assertTrue(self.area.explicit)
        self.assertEqual(len(self.area.winners), 0)

    def test_winners_setting(self):
        winners = np.arange(100)
        self.area.winners = winners
        self.assertEqual(len(self.area.winners), 100)
        self.assertEqual(self.area.w, 100)
        self.assertTrue(np.array_equal(self.area.winners, winners))

    def test_fix_and_unfix_assembly(self):
        self.area.winners = np.arange(100)
        self.area.fix_assembly()
        self.assertTrue(self.area.fixed_assembly)
        self.area.unfix_assembly()
        self.assertFalse(self.area.fixed_assembly)

    def test_update_beta_by_stimulus_refuses_rather_than_no_ops(self):
        # This used to write `beta_by_stimulus` and return happily, while the
        # engine went on reading `beta_by_source`. See test_per_fiber_beta.py.
        with self.assertRaises(NotImplementedError) as ctx:
            self.area.update_beta_by_stimulus("Stim1", 0.1)
        self.assertIn("update_plasticities", str(ctx.exception))

    def test_update_beta_by_area_refuses_rather_than_no_ops(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self.area.update_beta_by_area("Area1", 0.2)
        # The message must name the route that works, not just the failure.
        self.assertIn("brain.update_plasticity('Area1', 'TestArea', 0.2)",
                      str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
