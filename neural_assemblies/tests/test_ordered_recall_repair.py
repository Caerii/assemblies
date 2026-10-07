"""The ordered-recall repair study's bars (PREREG_ordered_recall_reproduction.md
Amendment 2), on synthetic rows: a chain that advances on the bridge passes
OR-1 to OR-3 and OR-6 to OR-8 and fails OR-4; a chain that stays on its cue
fails OR-1."""
from research.experiments import ordered_recall_repair as rr

SEEDS = list(range(20))


def _rows(*, one=7, no_inhibition=5, sampled=0, xfail=0, weak=0, cue=0.95, in_order=True):
    row = {"one_round": {"steps": one, "in_order": in_order, "cue": cue},
           "one_round_no_inhibition": {"steps": no_inhibition, "in_order": True, "cue": cue},
           "one_round_sampled": {"steps": sampled, "in_order": True, "cue": cue},
           "xfail_construction": {"steps": xfail, "in_order": True, "cue": 0.3},
           "one_round_xfail_beta": {"steps": weak, "in_order": True, "cue": cue}}
    return {"rows": {str(s): row for s in SEEDS}}


def test_a_bridge_driven_chain_passes_all_but_the_inhibition_null():
    bars = rr.evaluate(_rows())["bars"]
    assert all(bars[b] for b in ("OR-1", "OR-2", "OR-3", "OR-6", "OR-7", "OR-8")), bars
    assert not bars["OR-4"] and not bars["OR-5"], bars


def test_a_chain_that_needs_inhibition_passes_or4():
    assert rr.evaluate(_rows(no_inhibition=0))["bars"]["OR-4"]


def test_a_chain_that_stays_on_its_cue_fails_or1():
    assert not rr.evaluate(_rows(one=0))["bars"]["OR-1"]


def test_steps_count_only_consecutive_matches_after_the_cue():
    from neural_assemblies.assembly_calculus.assembly import Assembly
    import numpy as np
    stored = [Assembly("A", np.arange(10 * i, 10 * i + 10, dtype=np.uint32)) for i in range(4)]
    recalled = [stored[0], stored[1], Assembly("A", np.arange(500, 510, dtype=np.uint32)), stored[3]]
    out = rr.score(stored, recalled)
    assert out["steps"] == 1 and out["in_order"] and out["cue"] == 1.0
