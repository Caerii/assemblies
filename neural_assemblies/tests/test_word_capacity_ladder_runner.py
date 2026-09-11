"""The FEAT ladder must use one immutable, evidence-producing boundary."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from neural_assemblies import ExecutionKind, ExecutionSemantics, describe_hashed_aligner
from research.experiments import word_capacity as capacity
from research.experiments import word_capacity_ladder_run as study
from research.experiments.word_capacity_protocol import REGISTERED_PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


def execution():
    profile = describe_hashed_aligner(
        p=REGISTERED_PROTOCOL.connection_probability,
        beta=REGISTERED_PROTOCOL.plasticity,
        rounds_word=REGISTERED_PROTOCOL.rounds_per_pair,
    )
    return ExecutionSemantics(
        ExecutionKind.ALIGNMENT, {"default": profile},
    ).to_dict()


def record(**changes):
    value = {
        "protocol": study.PROTOCOL,
        "protocol_version": study.VERSION,
        "engine": "scheduled_aligner",
        "execution_semantics": execution(),
        "parameters": study.selected_protocol(smoke=True).to_parameters(),
        "seeds": [9, 2, 7],
        "mode": "smoke",
    }
    value.update(changes)
    return value


def test_measure_consumes_one_protocol_per_selected_rung(monkeypatch):
    calls = []

    def run_cell(name, seeds, sizes, **kwargs):
        calls.append((name, seeds, sizes, kwargs))
        return {8: [1., .875, .75], 16: [.5, .625, .75]}

    monkeypatch.setattr(capacity, "run_cell", run_cell)
    result = study.measure(record())
    assert result["verdict"] == "VOID"
    assert set(result["curves"]) == {"A:1000x50"}
    assert result["rungs"]["A:1000x50"]["ceiling"]["seed_ids"] == [9, 2, 7]
    name, seeds, sizes, kwargs = calls[0]
    assert (name, seeds, sizes) == ("A", [9, 2, 7], (8, 16))
    assert kwargs["protocol"].cells == ("A",)
    assert kwargs["protocol"].feature_area == (1000, 50)
    assert kwargs["aligner_semantics"] == execution()["profiles"]["default"]


@pytest.mark.parametrize("field,value,error", [
    ("cells", ["B"], "only registered cells A and C"),
    ("category_count", 5, "does not implement changed constants"),
    ("feature_ladder", [[2000, 50], [1000, 50]], "registered order"),
])
def test_ladder_protocol_drift_fails_before_alignment(
    monkeypatch, field, value, error,
):
    parameters = deepcopy(study.selected_protocol(smoke=True).to_parameters())
    parameters[field] = value
    if field == "feature_ladder":
        parameters["feature_area"] = value[0]
    monkeypatch.setattr(
        capacity, "run_cell",
        lambda *args, **kwargs: pytest.fail("invalid protocol reached alignment"),
    )
    with pytest.raises(ValueError, match=error):
        study.measure(record(parameters=parameters))


def test_cli_writes_the_complete_ladder_protocol(monkeypatch):
    calls = []
    monkeypatch.setattr(study, "run_experiment", lambda **kwargs: calls.append(kwargs))
    study.main([
        "--tag", "fixture", "--smoke", "--seeds", "9", "2", "7",
        "--feature-areas", "1000:50",
    ])
    sent = calls[0]
    assert sent["protocol"] == study.PROTOCOL
    assert sent["protocol_version"] == "3.3"
    assert sent["engine"] == "scheduled_aligner"
    assert sent["parameters"] == study.selected_protocol(smoke=True).to_parameters()


def test_legacy_ladder_entry_refuses_unrecorded_execution():
    with pytest.raises(RuntimeError, match="word_capacity_ladder_run"):
        capacity.ladder(["A"], [1, 2, 3], [8, 16])


def test_protocol33_ladder_replay_preserves_registered_rung():
    historical = json.loads((
        ROOT / "research/results/aligner/word_capacity_ladder.json"
    ).read_text())
    replay = json.loads((
        ROOT / "research/results/runs/aligner.word-capacity-ladder/"
        "protocol33-a-1000x50-replay-20260911/results.json"
    ).read_text())
    assert replay["run"]["schema_version"] == 8
    assert replay["run"]["protocol_version"] == "3.3"
    assert replay["run"]["seeds"] == historical["seeds"]
    assert replay["observations"]["curves"]["A:1000x50"] == (
        historical["curves"]["A:1000x50"]
    )
