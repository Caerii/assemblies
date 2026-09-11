"""Word-capacity runs must bind model, protocol, seeds, and immutable output."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from neural_assemblies import ExecutionKind, ExecutionSemantics, describe_hashed_aligner
from research.experiments import word_capacity as capacity
from research.experiments import word_capacity_run as study


ROOT = Path(__file__).resolve().parents[2]


def execution():
    semantics = describe_hashed_aligner(
        p=capacity.U.P, beta=capacity.U.BETA,
        rounds_word=capacity.ROUNDS_HASHED,
    )
    return ExecutionSemantics(
        ExecutionKind.ALIGNMENT, {"default": semantics},
    ).to_dict()


def record(**changes):
    value = {
        "protocol": study.PROTOCOL,
        "protocol_version": study.VERSION,
        "engine": "scheduled_aligner",
        "execution_semantics": execution(),
        "parameters": study.parameters(True),
        "seeds": [9, 2, 7],
        "mode": "smoke",
    }
    value.update(changes)
    return value


def test_measure_consumes_recorded_profile_and_keeps_per_seed_curves(monkeypatch):
    calls = []

    def run_cell(name, seeds, sizes, **kwargs):
        calls.append((name, seeds, sizes, kwargs))
        return {8: [1., .875, .75], 16: [.5, .625, .75]}

    monkeypatch.setattr(capacity, "run_cell", run_cell)
    observed = study.measure(record())
    assert observed["verdict"] == "VOID"
    assert observed["curves"] == {
        "A": {"8": [1., .875, .75], "16": [.5, .625, .75]},
    }
    _, seeds, sizes, kwargs = calls[0]
    assert seeds == [9, 2, 7] and sizes == [8, 16]
    assert kwargs["engine"] == "scheduled"
    assert kwargs["aligner_semantics"] == execution()["profiles"]["default"]
    assert observed["report"]["cells"]["A"]["ceiling"]["seed_ids"] == [9, 2, 7]
    assert all(bar["status"] == "VOID" for bar in observed["report"]["bars"].values())


@pytest.mark.parametrize("damage", [
    ("connection_probability", .2),
    ("rounds_per_pair", 3),
    ("cell_definitions", {}),
    ("vocabulary_sizes", [16, 8]),
    ("cells", ["A", "A"]),
    ("feature_area", [20, 21]),
])
def test_protocol_drift_fails_before_alignment(monkeypatch, damage):
    parameters = deepcopy(study.parameters(True))
    parameters[damage[0]] = damage[1]
    monkeypatch.setattr(
        capacity, "run_cell", lambda *args, **kwargs: pytest.fail("drift reached alignment"),
    )
    with pytest.raises(ValueError):
        study.measure(record(parameters=parameters))


def test_hashed_engine_refuses_scheduled_only_feature_area_before_alignment(monkeypatch):
    parameters = study.parameters(True)
    parameters["feature_area"] = [4000, 100]
    monkeypatch.setattr(
        capacity, "run_cell", lambda *args, **kwargs: pytest.fail("bad area reached alignment"),
    )
    with pytest.raises(ValueError, match="fixed feature area"):
        study.measure(record(engine="hashed_aligner", parameters=parameters))


def test_cli_supplies_complete_schema8_runner_inputs(monkeypatch):
    calls = []
    monkeypatch.setattr(study, "run_experiment", lambda **kwargs: calls.append(kwargs))
    study.main(["--tag", "fixture", "--smoke", "--seeds", "9", "2", "7"])
    sent = calls[0]
    assert sent["protocol"] == study.PROTOCOL and sent["protocol_version"] == study.VERSION
    assert sent["engine"] == "scheduled_aligner" and sent["seeds"] == [9, 2, 7]
    assert sent["parameters"] == study.parameters(True)
    assert sent["aligner_semantics"] == describe_hashed_aligner(
        p=capacity.U.P, beta=capacity.U.BETA,
        rounds_word=capacity.ROUNDS_HASHED,
    )


def test_legacy_entry_requires_a_tag_before_measurement(monkeypatch):
    monkeypatch.setattr(
        study, "run_experiment", lambda **kwargs: pytest.fail("missing tag reached runner"),
    )
    with pytest.raises(SystemExit) as exc:
        capacity.main(["--smoke"])
    assert exc.value.code == 2


def test_schema8_cell_a_replay_exactly_preserves_registered_curve():
    historical = json.loads((
        ROOT / "research/results/aligner/word_capacity_results_scheduled_feat4000x100.json"
    ).read_text())
    replay = json.loads((
        ROOT / "research/results/runs/aligner.word-capacity/"
        "word-capacity-cell-a-schema8-replay-20260911/results.json"
    ).read_text())
    run = replay["run"]
    assert run["schema_version"] == 8
    assert run["engine"] == "scheduled_aligner"
    assert run["execution_semantics"]["kind"] == "alignment"
    assert run["seeds"] == historical["seeds"]
    assert replay["observations"]["curves"]["A"] == historical["cells"]["A"]
    cell = replay["observations"]["report"]["cells"]["A"]
    assert cell["ceiling"]["seed_ids"] == historical["seeds"]
    assert cell["censored_seeds"] == 1
    assert cell["ceiling"]["mean"] == pytest.approx(73.82614696009215)
