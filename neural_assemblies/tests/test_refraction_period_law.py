"""The period-law sweep's design (PREREG_refraction_period_law.md).

The study asks whether the relocation period follows
`ln(w_max)/ln(1+beta) + (1-1/w_max)/beta` away from the single point it was
measured at. For that to be a test rather than a description, three things
have to hold before any data exists: the cells must sweep each parameter
separately through the measured point, the run length must be a declared
function of the PREDICTION rather than of the measurement, and the formula
must actually make the two qualitative claims the bars rest on.
"""
from __future__ import annotations

import math
import unittest

from research.experiments.refraction_convergence import clip_period
from research.experiments.refraction_period_law import (
    CELLS, MIN_ROUNDS, PERIODS_WANTED, cell_name, rounds_for,
)

MEASURED_POINT = (20.0, 0.10)


class TheCellsAreADesign(unittest.TestCase):
    def test_the_measured_point_is_in_the_sweep(self):
        self.assertIn(MEASURED_POINT, CELLS)

    def test_each_parameter_is_swept_with_the_other_held(self):
        w_fixed, b_fixed = MEASURED_POINT
        betas = sorted(b for w, b in CELLS if w == w_fixed)
        w_maxes = sorted(w for w, b in CELLS if b == b_fixed)
        self.assertGreaterEqual(len(betas), 3, "need a beta sweep")
        self.assertGreaterEqual(len(w_maxes), 3, "need a w_max sweep")
        # both sweeps must straddle the measured point, or a monotonicity bar
        # would be read off one side only
        self.assertLess(min(betas), b_fixed)
        self.assertGreater(max(betas), b_fixed)
        self.assertLess(min(w_maxes), w_fixed)
        self.assertGreater(max(w_maxes), w_fixed)

    def test_w_max_moves_further_than_beta_so_the_ordering_bar_can_fail(self):
        # PL-4 claims beta is the stronger lever DESPITE w_max moving further.
        # If w_max moved less, the bar would be trivially satisfiable.
        w_fixed, b_fixed = MEASURED_POINT
        betas = [b for w, b in CELLS if w == w_fixed]
        w_maxes = [w for w, b in CELLS if b == b_fixed]
        self.assertGreater(max(w_maxes) / min(w_maxes), max(betas) / min(betas))

    def test_cell_names_are_distinct(self):
        self.assertEqual(len({cell_name(w, b) for w, b in CELLS}), len(CELLS))


class TheRunLengthIsDeclared(unittest.TestCase):
    def test_rounds_come_from_the_prediction_not_the_data(self):
        for w, b in CELLS:
            expected = max(MIN_ROUNDS, math.ceil(PERIODS_WANTED * clip_period(w, b)))
            self.assertEqual(rounds_for(w, b), expected, (w, b))

    def test_every_cell_can_show_at_least_five_relocations(self):
        for w, b in CELLS:
            self.assertGreaterEqual(rounds_for(w, b) / clip_period(w, b), 5.0, (w, b))

    def test_a_slower_cell_is_given_more_rounds(self):
        self.assertGreater(rounds_for(20.0, 0.05), rounds_for(20.0, 0.20))


class TheFormulaMakesTheClaims(unittest.TestCase):
    def test_the_period_falls_with_beta_and_rises_with_w_max(self):
        for a, b in zip((0.20, 0.10), (0.10, 0.05)):
            self.assertLess(clip_period(20.0, a), clip_period(20.0, b))
        for a, b in zip((5.0, 20.0), (20.0, 100.0)):
            self.assertLess(clip_period(a, 0.10), clip_period(b, 0.10))

    def test_beta_is_inverse_and_w_max_is_logarithmic(self):
        # a fourfold change in beta must move the period more than a
        # twentyfold change in w_max; this is what PL-4 asserts of the data
        beta_ratio = clip_period(20.0, 0.05) / clip_period(20.0, 0.20)
        wmax_ratio = clip_period(100.0, 0.10) / clip_period(5.0, 0.10)
        self.assertGreater(beta_ratio, 3.0)
        self.assertLess(wmax_ratio, 3.0)
        self.assertGreater(beta_ratio, wmax_ratio)

    def test_the_small_beta_approximation_is_the_one_quoted(self):
        # (ln w_max + 1) / beta, used to explain the levers
        for w, b in ((20.0, 0.05), (20.0, 0.02), (100.0, 0.02)):
            approx = (math.log(w) + 1.0) / b
            self.assertLess(abs(approx - clip_period(w, b)) / clip_period(w, b), 0.02)

    def test_a_flat_formula_would_fail_every_monotonicity_bar(self):
        # true negative: if the period did not depend on its parameters, the
        # sweep could not distinguish the law from a constant, and these are
        # the comparisons that would collapse
        def flat(w_max=20.0, beta=0.10):
            del w_max, beta
            return 40.93
        self.assertEqual(flat(20.0, 0.05), flat(20.0, 0.20))
        self.assertNotEqual(clip_period(20.0, 0.05), clip_period(20.0, 0.20))


if __name__ == "__main__":
    unittest.main()
