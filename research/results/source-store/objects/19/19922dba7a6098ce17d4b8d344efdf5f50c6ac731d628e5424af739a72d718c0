"""The recall-law study's bars (PREREG_refraction_memory.md Amendment 14), on
synthetic sweeps: a refracted memory that follows the synapse-count law must
pass W2 and one that follows (n/k)^2 must fail it; equal (n/k, k p) cells
must agree for W3; and the multiplier is judged on distinct completion."""
import math

import pytest

from research.experiments import memory_recall_law as rl

pytestmark = pytest.mark.requires_torch


def _window(value):
    return {"never": value <= 0, "upper_censored": False, "upper": value,
            "lower": None, "last_above": value}


def _sweep(betas, peak_index, value, rank1):
    return {f"{b:g}": {"beta": b, "c_first_item": 4,
                       "windows": {"complete_distinct": _window(value if i == peak_index
                                                                 else 0.6 * value),
                                   "complete": _window(value), "rank1": _window(rank1)}}
            for i, b in enumerate(betas)}


def _observations(law, control=lambda n, k, p: 5.0):
    cells = {}
    for (n, k, p), block in rl.CELLS.items():
        ref = law(n, k, p)
        cells[f"{n}/{k}/{p:g}"] = {
            "n": n, "k": k, "p": p, "block": block,
            "refracted": _sweep(rl.refracted_betas(k, p), 2, ref, 2 * ref),
            "control": _sweep(list(rl.CONTROL_BETAS), 1, control(n, k, p), 20.0),
        }
    return {"cells": cells}


def _synapse(n, k, p):
    return 0.06 * rl.synapse_scale(n, k, p)


def test_the_plan_centres_the_refracted_sweep_on_the_law():
    for (n, k, p) in rl.CELLS:
        betas = rl.refracted_betas(k, p)
        assert len(betas) == 5 and abs(betas[2] / rl.beta_pred(k, p) - 1) < 1e-3
    assert rl.beta_pred(60, 0.5) == rl.beta_pred(120, 0.25) == rl.beta_pred(240, 0.125)


def test_a_synapse_count_memory_passes_and_an_n_over_k_memory_fails_w2():
    a13 = {nkp: _synapse(*nkp) for nkp in rl.A13}
    bars = rl.evaluate(_observations(_synapse), a13=a13)["bars"]
    assert bars["WV"] and bars["W1"] and bars["W2"] and bars["W3"] and bars["W4"]
    square = rl.evaluate(_observations(lambda n, k, p: 0.4 * (n / k) ** 2), a13=a13)["bars"]
    assert not square["W2"]


def test_w3_fails_when_p_matters_beyond_k_p():
    def p_dependent(n, k, p):
        return _synapse(n, k, p) * (1.0 if p == 0.5 else 0.5)
    bars = rl.evaluate(_observations(p_dependent))["bars"]
    assert not bars["W3"]


def test_the_multiplier_is_judged_on_distinct_completion():
    bars = rl.evaluate(_observations(_synapse, control=lambda n, k, p: _synapse(n, k, p) / 3))["bars"]
    assert not bars["W1"]
