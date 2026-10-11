"""research.replay: a committed run reruns from its recorded inputs, and only
identical observations pass -- whole or by cells."""
import json

import pytest

from research import replay as R


def _record(tmp_path, observations, cells=None):
    run = {"script": "unused.py", "parameters": {"cells": cells or []}, "seeds": [1, 2, 3],
           "mode": "study"}
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"run": run, "observations": observations}))
    return path


def _cells():
    return [{"n": 100, "k": 10, "p": 0.5}, {"n": 200, "k": 10, "p": 0.5}]


def _measure(record):
    return {"cells": {R.cell_key(c): {"value": c["n"] * 0.1} for c in record["parameters"]["cells"]},
            "verdict": "UNJUDGED"}


def test_an_exact_rerun_is_identical(tmp_path):
    path = _record(tmp_path, _measure({"parameters": {"cells": _cells()}}), _cells())
    report = R.replay(path, root=tmp_path, measure=_measure)
    assert report["identical"] and report["differs"] == []


def test_a_changed_value_is_named(tmp_path):
    recorded = _measure({"parameters": {"cells": _cells()}})
    recorded["cells"]["200/10/0.5"]["value"] = 20.000001
    report = R.replay(_record(tmp_path, recorded, _cells()), root=tmp_path, measure=_measure)
    assert not report["identical"] and report["differs"] == ["cells"]


def test_a_cell_subset_reruns_and_compares_only_those_cells(tmp_path):
    recorded = _measure({"parameters": {"cells": _cells()}})
    recorded["cells"]["200/10/0.5"]["value"] = -1.0            # not rerun, not compared
    seen = []

    def measure(record):
        seen.append([R.cell_key(c) for c in record["parameters"]["cells"]])
        return _measure(record)
    report = R.replay(_record(tmp_path, recorded, _cells()), cells=["100/10/0.5"],
                      root=tmp_path, measure=measure)
    assert seen == [["100/10/0.5"]] and report["identical"]


def test_an_unknown_cell_is_refused(tmp_path):
    path = _record(tmp_path, {}, _cells())
    with pytest.raises(ValueError):
        R.replay(path, cells=["999/1/0.5"], root=tmp_path, measure=_measure)


def test_the_measure_is_the_function_the_runner_called(tmp_path):
    """the runner calls measure(record); a study whose measure(spec, seeds, device) is a helper
    passed experiment(record) to the runner, and that is what a replay must call"""
    helper = tmp_path / "helper_study.py"
    helper.write_text(
        "def measure(spec, seeds, device):\n    return 'helper'\n\n\n"
        "def experiment(record):\n    return 'experiment'\n", encoding="utf-8")
    assert R.measure_of(helper)(None) == "experiment"
    direct = tmp_path / "direct_study.py"
    direct.write_text(
        "def measure(record):\n    return 'measure'\n\n\n"
        "def experiment(record):\n    return 'experiment'\n", encoding="utf-8")
    assert R.measure_of(direct)(None) == "measure"
