"""Corruption detection and exact replay contract for R1n evidence."""
import base64
import hashlib
import json
import lzma
from pathlib import Path
import pytest

from research.multimodal.unknown_drift import Protocol, Scenario, run
from research.multimodal.verify_unknown_drift import read_evidence, verify_replay


def _archive(tmp_path):
    p = Protocol(n=48, k=4, rounds=8, seeds=(0, 1),
                 scenarios=(Scenario('stable'), Scenario('shifted', (4,), (2,))))
    result = run(p)
    raw = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    archive = tmp_path / 'evidence.txt'
    archive.write_text(base64.b64encode(lzma.compress(raw)).decode(), encoding='ascii')
    return archive, hashlib.sha256(raw).hexdigest(), result


def test_archived_evidence_replay(tmp_path):
    path, sha, result = _archive(tmp_path)
    loaded, got = read_evidence(path, expected_sha256=sha)
    assert loaded == json.loads(json.dumps(result)) and got == sha
    assert verify_replay(loaded) is True


def test_bad_checksum_rejected(tmp_path):
    path, sha, result = _archive(tmp_path)
    with pytest.raises(ValueError, match='checksum mismatch'):
        read_evidence(path, expected_sha256='0'*64)


def test_corrupted_base64_rejected(tmp_path):
    path, sha, result = _archive(tmp_path)
    path.write_text('????')
    with pytest.raises(ValueError, match='invalid'):
        read_evidence(path, expected_sha256=sha)


def test_tampered_report_replay_rejected(tmp_path):
    path, sha, result = _archive(tmp_path)
    result['trials'][0]['rows'][0]['after_exact']['global_history'] = 42.
    with pytest.raises(AssertionError, match='disagrees'):
        verify_replay(result)
