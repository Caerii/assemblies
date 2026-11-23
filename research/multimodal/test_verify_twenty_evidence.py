from pathlib import Path
import pytest
from research.multimodal.verify_twenty_evidence import verify_archive


def test_archived_campaign_authenticity():
    result=verify_archive()
    assert result['status']=='verified'
    assert result['stage_count']==20
    assert result['learning_updates']==8000
    assert result['recomputed'] is False


def test_corrupt_archive_fails_closed(tmp_path):
    path=tmp_path/'bad.base64'
    path.write_text('not base64')
    with pytest.raises(Exception):verify_archive(path)
    from research.multimodal.verify_twenty_evidence import ARCHIVE
    data=ARCHIVE.read_text()
    path.write_text(data[:50] + 'A' + data[51:])
    with pytest.raises(ValueError,match='mismatch'):
        verify_archive(path)


def test_full_replay_matches_archived_protocol():
    # Slow, but verifies exact source-to-evidence agreement over all seeds.
    result=verify_archive(recompute=True)
    assert result['recomputed'] is True
