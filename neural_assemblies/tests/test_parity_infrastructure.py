"""Parity registry and in-process golden verification."""

from __future__ import annotations

import pytest

from neural_assemblies.parity.executors import EXECUTORS
from neural_assemblies.parity.registry import (
    claim_index,
    get_protocol,
    list_protocols,
    load_registry,
    resolve_golden_path,
)
from neural_assemblies.parity.runner import verify_protocol


def test_registry_loads_and_resolves_goldens():
    data = load_registry()
    assert data.get("version") == 1
    protocols = data.get("protocols", [])
    assert len(protocols) >= 9
    for row in protocols:
        proto = get_protocol(row["protocol_id"])
        path = resolve_golden_path(proto)
        assert path.endswith(row["golden"].split("/")[-1])


def test_claim_index_covers_primary_claims():
    idx = claim_index()
    assert "COIN24-E01" in idx
    assert "SEQ25-M05" in idx
    assert "SEQ25-E10" in idx
    assert "COIN24-E04" in idx
    assert "COIN24-E02" in idx
    assert "COLT22-E04" in idx
    assert idx["COIN24-E01"].protocol_id == "coin2024_demo"


def test_list_protocols_matches_registry():
    assert len(list_protocols()) == len(load_registry()["protocols"])


@pytest.mark.parametrize("protocol_id", [
    "coin2024_demo", "coin2024_compete", "coin2024_softmax", "coin2024_markov_arc",
    "direct2026_pearl",
])
def test_retracted_golden_stops_before_executor(monkeypatch, protocol_id):
    from neural_assemblies.parity.executors import RetractedProtocol

    def forbidden():
        raise AssertionError("retracted protocol reached its executor")

    monkeypatch.setitem(EXECUTORS, protocol_id, forbidden)
    with pytest.raises(RetractedProtocol, match="RETRACTED"):
        verify_protocol(protocol_id)


def test_cli_reports_retraction_as_a_distinct_status(capsys):
    import json
    from neural_assemblies.parity.cli import main

    assert main(["verify", "coin2024_demo"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "retracted"


@pytest.mark.parametrize(
    "protocol_id",
    sorted(EXECUTORS.keys()),
)
def test_inprocess_verify(protocol_id: str):
    # A RETRACTED golden is skipped, not failed. Nothing is wrong with the
    # code; the RECORD is withdrawn, so there is nothing to verify against and
    # a discrepancy would be meaningless. Skipping keeps the protocol visible
    # in the run -- deleting it is how a retraction gets quietly forgotten.
    from neural_assemblies.parity.executors import RetractedProtocol
    from neural_assemblies.programs.colt_mnist_data import DatasetUnavailable

    try:
        result = verify_protocol(protocol_id)
    except (RetractedProtocol, DatasetUnavailable) as exc:
        pytest.skip(str(exc))
    assert result.passed, f"{protocol_id}: {result.diffs or result.message}"


@pytest.mark.slow
def test_nemo2025_curriculum_registered():
    proto = get_protocol("nemo2025_curriculum")
    assert proto.slow
    assert proto.repro_command
