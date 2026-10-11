"""The laws the refraction-memory programme measured and every later study is sized by.

    theta(n, k, p)        the assembly calculus's projection-convergence threshold,
                          sqrt((1 - p) ln n / (p k)): the learning-rate scale (Amendment 17
                          registered the completion optimum as a fraction of it; the sequence
                          studies write at beta = theta)
    above_floor(n, k, p)  the regime the n/k laws need: k p >= 3 ln n
    unit(n, k, p)         the load unit: the sequence length L at load rho = 1,
                          n^2 p / (k ln n) (Amendment 37's load law, rho = L / unit)

Owned here since 2026-10-10. Before, theta and above_floor lived in memory_threshold_law
(Amendment 17) and unit in memory_load_law (Amendment 37); both re-export them.
"""
from __future__ import annotations

import math


def theta(n, k, p):
    """The convergence-threshold form sqrt((1 - p) ln n / (p k))."""
    return math.sqrt((1 - p) * math.log(n) / (p * k))


def above_floor(n, k, p):
    return k * p >= 3 * math.log(n)


def unit(n, k, p):
    """The L at rho = 1."""
    return n * n * p / (k * math.log(n))
