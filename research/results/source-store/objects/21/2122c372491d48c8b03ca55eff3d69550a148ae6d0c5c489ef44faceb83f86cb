"""The lexicon regime study's bars (PREREG_word_capacity.md Amendment 6), on
synthetic sweeps: a best plasticity set by n_LEX p (strong below 50, weak
from 200) must pass; one set by n alone must fail the equal-n p agreement
(E2), and one set by p alone must fail the boundary (E1)."""
import math

from research.experiments import word_capacity_regime_run as rg


def _observations(star):
    out = {}
    for p in rg.PROBABILITIES:
        sweep = {}
        for beta in rg.BETAS:
            cells = {}
            for name in rg.CELLS:
                n = rg.n_of(name)
                v = 1000.0 * (1 - 0.1 * (math.log(beta) - math.log(star(n, p))) ** 2)
                if p == 0.05 and beta == 0.1 and name in rg.A5:
                    v = rg.A5[name]
                cells[name] = {"ceiling": {"mean": max(v, 1.0)}, "censored_seeds": 0}
            sweep[f"{beta:g}"] = {"beta": beta, "cells": cells}
        out[p] = {"p": p, "sweep": sweep}
    return out


def _by_np(n, p):
    # strong write (at or above the grid's top) at n p <= 50, weak from 200,
    # one smooth function of n p
    return min(1.6, 0.05 * (100.0 / (n * p)) ** 3)


def test_a_regime_set_by_n_p_passes():
    bars = rg.evaluate(_observations(_by_np))["bars"]
    assert all(bars.values()), bars


def test_a_regime_set_by_n_alone_fails_e2():
    bars = rg.evaluate(_observations(lambda n, p: 0.05 * (2000.0 / n) ** 3))["bars"]
    assert not bars["E2"]


def test_a_regime_set_by_p_alone_fails_e1():
    bars = rg.evaluate(_observations(lambda n, p: 0.004 / p))["bars"]
    assert not bars["E1"]
