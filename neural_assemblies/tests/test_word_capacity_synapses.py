"""The vocabulary synapse study's bars (PREREG_word_capacity.md Amendments 4 and 5),
on synthetic ceilings: a synapse-limited lexicon (V* proportional to n p)
with a fan-in-scaled best learning rate must pass, and a neuron-limited one
(V* independent of p) must fail L1; a FEAT-bound cell is excluded, not
counted."""
import math

from research.experiments import word_capacity_synapses_run as ws

N = {"A": 1000, "B": 2000, "C": 4000}


def _observations(law, star):
    out = {}
    for p in ws.PROBABILITIES:
        sweep = {}
        for beta in ws.BETAS:
            cells = {}
            for cell in ws.CELLS:
                peak = law(cell, p)
                v = peak * (1 - 0.3 * (math.log(beta) - math.log(star(p))) ** 2)
                cells[cell] = {"ceiling": {"mean": max(v, 1.0)}, "censored_seeds": 0}
            sweep[f"{beta:g}"] = {"beta": beta, "cells": cells}
        out[p] = {"p": p, "sweep": sweep}
    return out


def _scaled_star(p):
    return 0.1 * math.sqrt(0.05 / p)


def test_a_synapse_limited_lexicon_passes():
    def law(cell, p):
        v = 1.5 * N[cell] * p
        return v
    obs = _observations(law, _scaled_star)
    for cell, value in ws.PART2.items():          # the instrument replays Part 2
        obs[0.05]["sweep"]["0.1"]["cells"][cell]["ceiling"]["mean"] = value
    bars = ws.evaluate(obs)["bars"]
    assert bars["L0"] and bars["L1"] and bars["L2"] and bars["L3"] and bars["L4"]


def test_a_neuron_limited_lexicon_fails_l1():
    bars = ws.evaluate(_observations(lambda cell, p: 0.075 * N[cell], _scaled_star))["bars"]
    assert not bars["L1"]


def test_a_fixed_learning_rate_fails_l4():
    bars = ws.evaluate(_observations(lambda cell, p: 1.5 * N[cell] * p, lambda p: 0.1))["bars"]
    assert bars["L3"] and not bars["L4"]


def test_optimum_finds_a_vertex_and_refuses_an_edge():
    betas = list(ws.BETAS)
    assert abs(ws.optimum(betas, [1 - (math.log(b) - math.log(0.08)) ** 2 for b in betas])
               / 0.08 - 1) < 0.05
    assert ws.optimum(betas, [1, 2, 3, 4, 5]) is None


def _optimum_observations(law, star):
    out = {}
    for p in ws.PROBABILITIES:
        sweep = {}
        for beta in ws.OPT_BETAS:
            cells = {}
            for cell in ws.CELLS:
                peak = law(cell, p)
                v = peak * (1 - 0.15 * (math.log(beta) - math.log(star(cell, p))) ** 2)
                cells[cell] = {"ceiling": {"mean": max(v, 1.0)}, "censored_seeds": 0}
            sweep[f"{beta:g}"] = {"beta": beta, "cells": cells}
        out[p] = {"p": p, "sweep": sweep}
    return out


def test_amendment_5_passes_a_synapse_limited_load_dependent_lexicon():
    def star(cell, p):
        return {"A": 0.4, "B": 0.1, "C": 0.04}[cell] * math.sqrt(0.05 / p)
    obs = _optimum_observations(lambda cell, p: 1.5 * N[cell] * p, star)
    bars = ws.evaluate_optimum(obs)["bars"]
    assert bars["M1"] and bars["M2"] and bars["M3"], bars


def test_amendment_5_fails_a_size_blind_optimum():
    obs = _optimum_observations(lambda cell, p: 1.5 * N[cell] * p,
                                lambda cell, p: 0.1 * math.sqrt(0.05 / p))
    assert not ws.evaluate_optimum(obs)["bars"]["M2"]
