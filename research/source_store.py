"""Content-addressed storage for the source a run executed.

Source-linked specification: research/README.md#recoverable-source

Every shared-runner run records the exact source it ran (``git_commit``,
``source_sha256``, the script, the registration and its input artifacts), and
until 2026-09-30 kept those bytes as a per-run ``source.zip``. Consecutive runs
share almost all of their source, and a zip is an opaque binary that git
cannot deduplicate: 71 archives took 313 MB of the tree while their distinct
content was 35.5 MB.

A run now keeps a small ``source.manifest.json`` naming its members in archive
order, each by the SHA-256 of its bytes, and the bytes live once in
``research/results/source-store/objects/<ab>/<sha256>``. The manifest also
carries the digest the run record binds (``archive_sha256``): the zip is a
deterministic function of the member names and bytes (the runner writes every
member with a fixed timestamp and DEFLATE), and every migrated archive was
rebuilt byte-for-byte from the store before its zip was removed.

The objects are stored raw and marked ``-text`` in ``.gitattributes``: a byte
changed by a line-ending conversion would change its digest.
"""
from __future__ import annotations

import hashlib
import io
import json
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'research' / 'results' / 'source-store' / 'objects'
MANIFEST = 'source.manifest.json'
ARCHIVE = 'source.zip'
FORMAT = 'assemblies-source-manifest/1'


def object_path(digest: str, store: Path = STORE) -> Path:
    return store / digest[:2] / digest


def put(data: bytes, store: Path = STORE) -> str:
    """Store ``data`` once under its SHA-256; an existing object must match."""
    digest = hashlib.sha256(data).hexdigest()
    path = object_path(digest, store)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'source-store object {digest} is corrupt')
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)
    return digest


def get(digest: str, store: Path = STORE) -> bytes:
    """The stored bytes for ``digest``, verified."""
    data = object_path(digest, store).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'source-store object {digest} does not match its name')
    return data


def build_archive(members) -> bytes:
    """The runner's archive for ``members`` [(name, bytes)], byte-for-byte."""
    buffer = io.BytesIO()
    with ZipFile(buffer, 'x') as archive:
        for name, data in members:
            entry = ZipInfo(name)  # fixed timestamp: preserve bytes, not checkout metadata
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, data)
    return buffer.getvalue()


def write_manifest(directory: Path, members, archive_sha256: str, *,
                   store: Path = STORE, rebuilt: bool) -> Path:
    """Store the member bytes and write the run's manifest (refuses to overwrite)."""
    names = [name for name, _ in members]
    if len(names) != len(set(names)):
        raise ValueError('duplicate source archive members')
    document = {
        'format': FORMAT,
        'archive_sha256': archive_sha256,
        'rebuild': {'method': 'zipfile ZipInfo(name), ZIP_DEFLATED, default level, archive order',
                    'zlib': zlib.ZLIB_RUNTIME_VERSION,
                    'verified_byte_identical': bool(rebuilt)},
        'members': [[name, put(data, store)] for name, data in members],
    }
    path = directory / MANIFEST
    with open(path, 'x', encoding='utf-8', newline='\n') as handle:
        json.dump(document, handle, indent=1)
        handle.write('\n')
    return path


def read_manifest(directory: Path) -> dict:
    document = json.loads((directory / MANIFEST).read_text(encoding='utf-8'))
    if (not isinstance(document, dict) or document.get('format') != FORMAT
            or set(document) != {'format', 'archive_sha256', 'rebuild', 'members'}
            or not isinstance(document['members'], list)):
        raise ValueError('source manifest is malformed')
    for entry in document['members']:
        if (not isinstance(entry, list) or len(entry) != 2
                or not all(isinstance(x, str) for x in entry)):
            raise ValueError('source manifest member entries must be [name, sha256]')
    return document


def rebuild_archive(directory: Path, store: Path = STORE) -> bytes:
    """Reassemble the run's archive from the store (for a deep check)."""
    document = read_manifest(directory)
    return build_archive([(name, get(digest, store)) for name, digest in document['members']])


def migrate_archive(directory: Path, store: Path = STORE) -> str:
    """Replace ``source.zip`` with a manifest, only if the store rebuilds it exactly.

    Returns 'migrated', 'kept' (rebuild differs: the zip stays and nothing is
    written), or 'absent'.
    """
    path = directory / ARCHIVE
    if not path.exists():
        return 'absent'
    raw = path.read_bytes()
    recorded = hashlib.sha256(raw).hexdigest()
    with ZipFile(io.BytesIO(raw)) as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    if hashlib.sha256(build_archive(members)).hexdigest() != recorded:
        return 'kept'
    write_manifest(directory, members, recorded, store=store, rebuilt=True)
    if hashlib.sha256(rebuild_archive(directory, store)).hexdigest() != recorded:
        (directory / MANIFEST).unlink()
        return 'kept'
    path.unlink()
    return 'migrated'
