"""Word-capacity runs must bind model, protocol, seeds, and immutable output."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from neural_assemblies import ExecutionKind, ExecutionSemantics, describe_hashed_aligner
from research.experiments import word_capacity as capacity
from research.experiments import word_capacity_run as study
from research.experiments.word_capacity_protocol import (
    REGISTERED_PROTOCOL, SMOKE_PROTOCOL, WordCapacityProtocol,
)


ROOT = Path(__file__).resolve().parents[2]


def execution():
    semantics = describe_hashed_aligner(
        p=REGISTERED_PROTOCOL.connection_probability,
        beta=REGISTERED_PROTOCOL.plasticity,
        rounds_word=REGISTERED_PROTOCOL.rounds_per_pair,
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
    assert seeds == [9, 2, 7] and sizes == (8, 16)
    assert kwargs["engine"] == "scheduled"
    assert kwargs["protocol"] == SMOKE_PROTOCOL
    assert kwargs["aligner_semantics"] == execution()["profiles"]["default"]
    assert observed["report"]["cells"]["A"]["ceiling"]["seed_ids"] == [9, 2, 7]
    assert all(bar["status"] == "VOID" for bar in observed["report"]["bars"].values())


@pytest.mark.parametrize("damage", [
    ("connection_probability", .2),
    ("plasticity", float("nan")),
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


def test_registered_per_brain_protocol_refuses_hashed_backend_before_alignment(monkeypatch):
    monkeypatch.setattr(
        capacity, "run_cell", lambda *args, **kwargs: pytest.fail("bad area reached alignment"),
    )
    with pytest.raises(ValueError, match="requires scheduled_aligner"):
        study.measure(record(engine="hashed_aligner"))


def test_public_execution_rejects_unknown_engine_and_unselected_cell():
    with pytest.raises(ValueError, match="unknown word-capacity engine"):
        capacity.run_cell("A", [1, 2, 3], [8, 16], engine="typo",
                          protocol=SMOKE_PROTOCOL)
    with pytest.raises(ValueError, match="not selected"):
        capacity.run_cell("B", [1, 2, 3], [8, 16], engine="scheduled",
                          protocol=SMOKE_PROTOCOL)
    with pytest.raises(ValueError, match="vocabulary sizes"):
        capacity.run_cell("A", [1, 2, 3], [8, 32], engine="scheduled",
                          protocol=SMOKE_PROTOCOL)


def test_corpus_seed_scope_is_a_backend_contract():
    shared = replace(SMOKE_PROTOCOL, corpus_seed_scope="shared-batch")
    with pytest.raises(ValueError, match="per-brain"):
        capacity.run_cell("A", [1, 2, 3], [8, 16], engine="scheduled",
                          protocol=shared)
    with pytest.raises(ValueError, match="shared-batch"):
        capacity.run_cell("A", [1, 2, 3], [8, 16], engine="hashed",
                          protocol=SMOKE_PROTOCOL)


def test_cli_supplies_complete_schema8_runner_inputs(monkeypatch):
    calls = []
    monkeypatch.setattr(study, "run_experiment", lambda **kwargs: calls.append(kwargs))
    study.main(["--tag", "fixture", "--smoke", "--seeds", "9", "2", "7"])
    sent = calls[0]
    assert sent["protocol"] == study.PROTOCOL and sent["protocol_version"] == study.VERSION
    assert sent["engine"] == "scheduled_aligner" and sent["seeds"] == [9, 2, 7]
    assert sent["parameters"] == study.parameters(True)
    assert sent["aligner_semantics"] == describe_hashed_aligner(
        p=REGISTERED_PROTOCOL.connection_probability,
        beta=REGISTERED_PROTOCOL.plasticity,
        rounds_word=REGISTERED_PROTOCOL.rounds_per_pair,
    )


def test_legacy_entry_requires_a_tag_before_measurement(monkeypatch):
    monkeypatch.setattr(
        study, "run_experiment", lambda **kwargs: pytest.fail("missing tag reached runner"),
    )
    with pytest.raises(SystemExit) as exc:
        capacity.main(["--smoke"])
    assert exc.value.code == 2


def test_protocol33_cell_a_replay_exactly_preserves_registered_curve():
    historical = json.loads((
        ROOT / "research/results/aligner/word_capacity_results_scheduled_feat4000x100.json"
    ).read_text())
    replay = json.loads((
        ROOT / "research/results/runs/aligner.word-capacity/"
        "word-capacity-protocol33-cell-a-replay-20260911/results.json"
    ).read_text())
    run = replay["run"]
    assert run["schema_version"] == 8
    assert run["protocol_version"] == "3.3"
    assert run["engine"] == "scheduled_aligner"
    assert run["execution_semantics"]["kind"] == "alignment"
    assert run["seeds"] == historical["seeds"]
    assert run["parameters"] == REGISTERED_PROTOCOL.select(cells=("A",)).to_parameters()
    assert replay["observations"]["curves"]["A"] == historical["cells"]["A"]
    cell = replay["observations"]["report"]["cells"]["A"]
    assert cell["ceiling"]["seed_ids"] == historical["seeds"]
    assert cell["censored_seeds"] == 1
    assert cell["ceiling"]["mean"] == pytest.approx(73.82614696009215)


def test_protocol_is_immutable_and_round_trips_complete_parameters():
    assert WordCapacityProtocol.from_parameters(
        REGISTERED_PROTOCOL.to_parameters()
    ) == REGISTERED_PROTOCOL
    with pytest.raises(FrozenInstanceError):
        REGISTERED_PROTOCOL.threshold = .8


@pytest.mark.parametrize("curve,seeds,error", [
    ({16: [1., 1., 1.]}, [1, 1, 2], "unique"),
    ({16: [1., 1.]}, [1, 2, 3], "one value per seed"),
    ({32: [1., 1., 1.]}, [1, 2, 3], "protocol-grid prefix"),
    ({16: [1., float("nan"), 1.]}, [1, 2, 3], "finite probabilities"),
])
def test_capacity_curve_shape_fails_before_statistics(curve, seeds, error):
    with pytest.raises(ValueError, match=error):
        capacity.ceilings(curve, seeds, protocol=REGISTERED_PROTOCOL)


def test_corpus_generation_consumes_protocol_instead_of_module_globals():
    two_categories = replace(REGISTERED_PROTOCOL, category_count=2)
    _, targets, _, features = capacity.corpus(16, 42, protocol=two_categories)
    assert {bundle[0] for bundle in targets.values()} == {"CAT_0", "CAT_1"}
    assert "CAT_2" not in features


def test_numpy_alignment_receives_protocol_probability_and_plasticity(monkeypatch):
    calls = []

    class Reconstruction:
        @staticmethod
        def overlap(_assembly):
            return 0

    class Aligner:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        @staticmethod
        def train(_experience, _rng):
            pass

        @staticmethod
        def bundle_assembly(bundle):
            return bundle

        @staticmethod
        def reconstruct(_word):
            return Reconstruction()

    protocol = replace(
        SMOKE_PROTOCOL, connection_probability=.2, plasticity=.3,
        minimum_exposures=1,
    )
    monkeypatch.setattr(capacity.U, "Aligner", Aligner)
    capacity.type_accuracy(42, 8, 1000, 50, 50, protocol=protocol)
    assert calls[0]["p"] == .2
    assert calls[0]["beta"] == .3
    assert (calls[0]["feat_n"], calls[0]["feat_k"]) == protocol.feature_area
