"""Evidence immutable hash and deterministic replay gates."""
import base64
import lzma
from pathlib import Path
import pytest
from research.multimodal.verify_noncommutative import verify, EVIDENCE


def test_canonical_archive_verified():
    x = verify()
    assert x['cells'] == 360
    assert not x['replayed']


def test_corrupt_or_missing_archive_rejected(tmp_path):
    with pytest.raises(AssertionError, match='decode'):
        verify(tmp_path / 'nonexistent')
    p = tmp_path / 'fake.base64'
    p.write_text('%%%invalid%%%', encoding='ascii')
    with pytest.raises(AssertionError, match='decode'):
        verify(p)
    p.write_text(base64.b64encode(lzma.compress(b'{"changed":true}')).decode(), encoding='ascii')
    with pytest.raises(AssertionError, match='hash mismatch'):
        verify(p)


def test_full_replay():
    x = verify(replay=True)
    assert x['replayed']
    assert x['cells'] == 360
