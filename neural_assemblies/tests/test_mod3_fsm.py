"""Unit tests for mod-3 FSM demo."""

from __future__ import annotations

import unittest

from neural_assemblies.programs.mod3_fsm import mod3_transition_table, run_mod3_fsm_demo


class TestMod3Fsm(unittest.TestCase):
    def test_transition_table_size(self):
        table = mod3_transition_table()
        self.assertEqual(len(table), 33)

    def test_demo_accepts_divisible_by_three(self):
        result = run_mod3_fsm_demo(presentations=10, n=1500, k=30, rounds=5)
        self.assertTrue(result.positive_accepted)
        self.assertTrue(result.negative_rejected)


if __name__ == "__main__":
    unittest.main()
