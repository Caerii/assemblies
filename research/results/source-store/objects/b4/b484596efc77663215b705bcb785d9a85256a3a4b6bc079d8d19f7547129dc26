"""The refraction-convergence instrument's relocation events (Amendment 1 of
PREREG_refraction_convergence.md): a curve that is stable between wholesale
relocations must read as a few events, a churning curve as one event that
never ends, and a stable curve as none."""
from __future__ import annotations

import unittest

from research.experiments.refraction_convergence import (
    RELOCATED, STABLE, relocation_events,
)


class RelocationEvents(unittest.TestCase):
    def test_stable_curve_has_no_event(self):
        self.assertEqual(relocation_events([1.0] * 239), ([], []))

    def test_relocations_are_maximal_runs_with_their_start_rounds(self):
        # consecutive[i] compares rounds i+1 and i+2; a drop at index 43
        # means the winners of round 45 differ from those of round 44.
        curve = [1.0] * 239
        curve[43] = 0.0
        curve[44] = 0.3          # the same event, two rounds long
        curve[100] = 0.9         # a partial re-ranking below the stable bar
        starts, lengths = relocation_events(curve)
        self.assertEqual(starts, [45, 102])
        self.assertEqual(lengths, [2, 1])

    def test_churn_is_one_event_that_never_ends(self):
        starts, lengths = relocation_events([0.01] * 239)
        self.assertEqual((starts, lengths), ([2], [239]))

    def test_threshold_is_the_registered_constant(self):
        curve = [1.0] * 10
        curve[5] = STABLE          # at the bar is stable
        self.assertEqual(relocation_events(curve), ([], []))
        curve[5] = STABLE - 1e-9   # below it is an event
        self.assertEqual(relocation_events(curve), ([7], [1]))
        self.assertLess(RELOCATED, STABLE)

    def test_a_broken_threshold_would_miss_a_relocation(self):
        # true negative: with the event threshold at 0, a wholesale
        # relocation whose overlap is exactly 0 would not register
        curve = [1.0] * 10
        curve[5] = 0.0
        self.assertEqual(relocation_events(curve, stable=0.0), ([], []))
        self.assertEqual(relocation_events(curve), ([7], [1]))


if __name__ == "__main__":
    unittest.main()
