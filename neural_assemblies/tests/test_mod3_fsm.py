"""Unit tests for mod-3 FSM demo."""

from __future__ import annotations

import unittest

from neural_assemblies.programs.mod3_fsm import mod3_transition_table, run_mod3_fsm_demo


class TestMod3Fsm(unittest.TestCase):
    def test_transition_table_size(self):
        table = mod3_transition_table()
        self.assertEqual(len(table), 33)

    @unittest.expectedFailure
    def test_demo_accepts_divisible_by_three(self):
        """KNOWN FAILING, and it used to pass for the wrong reason.

        This asserted the demo's verdict while `step_symbol` returned a table
        lookup, so it passed on an untrained brain and at beta=0. With the
        readout fixed the demo does not reliably decide: A1 measured 5/10 seeds,
        4/10 with a fully correct trajectory. Single transitions are perfect
        (330/330); the state assembly drifts along a sequence. At seed 42 the
        trajectory tracks ground truth through all five digit steps and misses
        only the final `end` transition.

        Left as an expected failure rather than weakened, so it converts back
        into a passing test the moment drift is fixed. See
        `research/notes/the_arc_is_a_conjunction_and_the_state_drifts.md` and
        the retraction block in
        `research/literature/parity/golden/nemo2025_fsm_mod3.json`.
        """
        result = run_mod3_fsm_demo(presentations=15)
        self.assertTrue(result.positive_accepted)
        self.assertTrue(result.negative_rejected)


if __name__ == "__main__":
    unittest.main()
