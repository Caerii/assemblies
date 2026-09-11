import copy
import pytest

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
    candidate["observations"]["cells"][0]["k"] = 10.0
    assert not compare(candidate, baseline, "capacity", [7, 13, 19])["numerical_match"]
    candidate["observations"]["cells"][0]["k"] = 20
    assert not compare(candidate, baseline, "capacity", [7, 13, 19])["numerical_match"]


@pytest.mark.parametrize("field,value", [("seed", True), ("seed", 1.0),
    ("length", 2000.001), ("length", 2000.0), ("first_error", 1000001)])
def test_a1_integer_identity_is_exact_not_a_tolerance(field, value):
    row = {"seed": 1, "p": .3, "length": 2000, "first_error": 1000}
    if field == "first_error":
        row.update(length=2000000, first_error=1000000)
    candidate = {"run": {"seeds": [1], "parameters": {"p_values": [.3]}},
                 "observations": {"rows": [{**row, field: value}]}}
    assert not compare(candidate, {"rows": [row]}, "a1")["numerical_match"]


def test_duplicate_historical_rows_cannot_hide_conflicting_evidence():
    row = {"seed": 1, "p": .3, "length": 2000, "accuracy": 1.}
    candidate = {"run": {"seeds": [1], "parameters": {"p_values": [.3]}},
                 "observations": {"rows": [row]}}
    baseline = {"rows": [{**row, "accuracy": .1}, row]}
    with pytest.raises(ValueError, match="duplicate evidence"):
        compare(candidate, baseline, "a1")


def test_duplicate_json_object_keys_reject_before_comparison(tmp_path):
    from research.compare_migration import _load_json
    path = tmp_path / "reference.json"
    path.write_text('{"rows": [], "rows": [{"seed": 1}]}')
    with pytest.raises(ValueError, match="duplicate evidence"):
        _load_json(path)


def test_continuous_measurements_keep_tolerance_but_nested_bools_do_not():
    from research.compare_migration import _equal
    assert _equal(1.000001, 1.0)
    assert not _equal([1], [True])
    assert not _equal([1.0], [1])


def test_historical_float_identity_is_not_reinterpreted_as_an_integer():
    row = {"seed": 1.0, "p": .3, "length": 2000}
    candidate = {"run": {"seeds": [1], "parameters": {"p_values": [.3]}},
                 "observations": {"rows": [row]}}
    with pytest.raises(ValueError, match="historical A1 seed"):
        compare(candidate, {"rows": [row]}, "a1")


def test_capacity_reference_seed_keys_are_integer_identities():
    candidate = {"run": {}, "observations": {}}
    with pytest.raises(ValueError, match="reference seed order"):
        compare(candidate, {}, "capacity", [True, 2, 3])


def test_paired_capacity_comparison_checks_both_named_conditions():
    def baseline(values, *, checkpoints=(8,)):
        return {"B/100": {str(m): {"rank1": values[m]} for m in checkpoints},
                "B/100/ceiling": {"k": 10}}

    control = baseline({8: [.1, .2, .3]})
    treatment = baseline({8: [.7, .8, .9], 16: [.6, .7, .8]}, checkpoints=(8, 16))
    conditions = {
        "control": {"checkpoints": [8, 16]},
        "refracted": {"checkpoints": [8, 16]},
    }
    candidate = {
        "run": {"seeds": [13, 7, 19], "parameters": {
            "arms": ["B"], "nk": [[100, 10]], "conditions": conditions}},
        "observations": {"conditions": {
            "control": {"cells": {"B/100/10": {
                "arm": "B", "n": 100, "k": 10,
                "checkpoints": {"8": {"rank1": [.2, .1, .3]},
                                "16": {"rank1": [.2, .1, .3]}}}}},
            "refracted": {"cells": {"B/100/10": {
                "arm": "B", "n": 100, "k": 10,
                "checkpoints": {"8": {"rank1": [.8, .7, .9]},
                                "16": {"rank1": [.7, .6, .8]}}}}},
        }},
    }
    result = compare(candidate, control, "capacity-paired", [7, 13, 19],
                     treatment_baseline=treatment)
    assert result["numerical_match"] and result["comparisons"] == 9
    candidate["observations"]["conditions"]["refracted"]["cells"][
        "B/100/10"]["checkpoints"]["16"]["rank1"][0] = 0.0
    assert not compare(candidate, control, "capacity-paired", [7, 13, 19],
                       treatment_baseline=treatment)["numerical_match"]


def test_paired_capacity_comparison_requires_both_references():
    candidate = {"run": {"seeds": [1, 2, 3]}, "observations": {}}
    with pytest.raises(ValueError, match="treatment reference"):
        compare(candidate, {}, "capacity-paired", [1, 2, 3])
