import copy

from research.compare_migration import compare


def test_a1_comparison_catches_wrong_exactness_despite_correct_label():
    row = {"seed": 1, "p": .3, "length": 2000, "accuracy": 1.,
           "exact_fraction": 1., "first_error": None, "prefix_correct": {"2000": True}}
    baseline = {"rows": [row]}
    candidate = {"run": {"seeds": [1], "parameters": {"p_values": [.3]}},
                 "observations": {"rows": [copy.deepcopy(row)]}}
    assert compare(candidate, baseline, "a1")["numerical_match"]
    candidate["observations"]["rows"][0]["exact_fraction"] = .5
    assert not compare(candidate, baseline, "a1")["numerical_match"]
    candidate["observations"]["rows"] = []
    assert not compare(candidate, baseline, "a1")["numerical_match"]


def test_capacity_comparison_uses_explicit_seed_order_and_k():
    baseline = {"B/100": {"8": {"rank1": [.1, .2, .3]}},
                "B/100/ceiling": {"k": 10}}
    candidate = {"run": {"seeds": [13, 7, 19], "parameters": {
        "arms": ["B"], "nk": [[100, 10]], "configuration": {"checkpoints": [8]}}},
        "observations": {"cells": [{"arm": "B", "n": 100, "k": 10,
                                    "checkpoints": {"8": {"rank1": [.2, .1, .3]}}}]}}
    assert compare(candidate, baseline, "capacity", [7, 13, 19])["numerical_match"]
    candidate["observations"]["cells"][0]["k"] = 20
    assert not compare(candidate, baseline, "capacity", [7, 13, 19])["numerical_match"]
