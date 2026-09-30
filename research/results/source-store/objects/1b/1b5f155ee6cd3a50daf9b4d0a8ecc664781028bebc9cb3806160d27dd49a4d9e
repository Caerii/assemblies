"""Parity manifests are immutable evidence artifacts."""

import json

import pytest

from neural_assemblies.parity.protocol import ProtocolResult
from neural_assemblies.parity.runner import write_manifest


def test_write_manifest_refuses_to_overwrite(tmp_path):
    result = ProtocolResult(
        protocol_id="contract-test",
        claim_id="contract-test",
        passed=True,
        backend="numpy",
    )
    path = tmp_path / "manifest.json"
    write_manifest(result, path)
    assert json.loads(path.read_text(encoding="utf-8"))["protocol_id"] == "contract-test"
    with pytest.raises(FileExistsError):
        write_manifest(result, path)
