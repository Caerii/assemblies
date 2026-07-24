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


@pytest.mark.parametrize(
    "protocol_id",
    [
        pytest.param(
            pid,
            marks=pytest.mark.xfail(
                reason="DIRECT directional binding is vacuous: forward/reverse/do metrics are provably insensitive to the learned CAUSE->BIND connectome (wipe-test + cue-swap + feedforward probes); pre-norm_init values measured degree-hub substrate overlap, not binding. Not re-baselined -- that would pin the artifact.",
                strict=False,
            ),
        )
        if pid == "direct2026_pearl"
        else pid
        for pid in sorted(EXECUTORS.keys())
    ],
)
def test_inprocess_verify(protocol_id: str):
    result = verify_protocol(protocol_id)
    assert result.passed, f"{protocol_id}: {result.diffs or result.message}"


@pytest.mark.slow
def test_nemo2025_curriculum_registered():
    proto = get_protocol("nemo2025_curriculum")
    assert proto.slow
    assert proto.repro_command
