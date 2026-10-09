"""The interval estimates behind P8's crossings table: a crossing is read where
the paper says it is, a jump between two ladder points collapses the bootstrap
but not the bracket, and a graded curve gets a real interval."""
import math

import numpy as np

from research.experiments import memory_budget_figures as mf


def test_the_crossing_is_log_interpolated_and_bracketed():
    pts = [(0.10, 1.0), (0.12, 0.6), (0.14, 0.0)]
    c, r0, r1 = mf.crossing(pts, 0.5, bracket=True)
    t = (0.6 - 0.5) / 0.6
    assert math.isclose(c, math.exp(math.log(0.12) + t * math.log(0.14 / 0.12)))
    assert (r0, r1) == (0.12, 0.14)
    assert mf.crossing([(0.1, 1.0), (0.2, 0.9)], 0.5) is None


def test_an_all_or_none_jump_collapses_the_bootstrap_but_not_the_bracket():
    curve = [(0.10, [1.0] * 20), (0.11, [1.0] * 20), (0.12, [0.0] * 20)]
    point, lo, hi, share, b0, b1 = mf.bootstrap(curve, 0.5, np.random.default_rng(0))
    assert lo == hi == point and share == 1.0
    assert (b0, b1) == (0.11, 0.12)


def test_a_graded_curve_gets_a_real_interval():
    rng = np.random.default_rng(1)
    curve = [(0.10 * 1.05 ** j, list((rng.random(20) < 1 - j / 10).astype(float))) for j in range(11)]
    point, lo, hi, share, _, _ = mf.bootstrap(curve, 0.5, np.random.default_rng(2))
    assert lo < point < hi and share > 0.9
