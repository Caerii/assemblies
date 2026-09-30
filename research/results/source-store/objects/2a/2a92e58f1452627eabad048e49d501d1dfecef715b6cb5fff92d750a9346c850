"""Check recoverable source without extracting or importing archived code.

Source-linked specification: research/README.md#recoverable-source

A run's source is either a sibling ``source.zip`` (runs before 2026-09-30,
unless migrated) or a ``source.manifest.json`` whose members live in the
content-addressed store (research/source_store.py). Both are checked the same
way at the member level: the archived source must reproduce ``source_sha256``,
and the archived script, registration and inputs must match their recorded
digests. The run record binds the archive by ``source_archive.sha256``; a
manifest must carry that same digest, and ``deep=True`` additionally rebuilds
the archive from the store and requires it byte-for-byte.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from research import source_store


def store_for(directory: Path) -> Path:
    """The store beside the runs tree: <results>/source-store/objects for <results>/runs/<protocol>/<tag>."""
    return directory.parent.parent.parent / 'source-store' / 'objects'


def _members(directory: Path, recorded: str):
    """(names in archive order, reader, errors) for whichever form the run keeps."""
    zip_path = directory / source_store.ARCHIVE
    if zip_path.exists():
        raw = zip_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != recorded:
            return None, None, ['source archive digest mismatch']
        archive = ZipFile(io.BytesIO(raw))
        return archive.namelist(), archive.read, []
    if (directory / source_store.MANIFEST).exists():
        document = source_store.read_manifest(directory)
        if document['archive_sha256'] != recorded:
            return None, None, ['source manifest does not carry the recorded archive digest']
        names = [name for name, _ in document['members']]
        digests = dict(document['members'])
        store = store_for(directory)
        return names, (lambda name: source_store.get(digests[name], store)), []
    return None, None, ['source archive missing: the run keeps neither source.zip nor source.manifest.json']


def validate_source_archive(directory: Path, record: dict, *, deep: bool = False) -> list[str]:
    """Bind exact archived bytes to the source and protocol digests in a run."""
    reference = record.get('source_archive')
    if (not isinstance(reference, dict) or set(reference) != {'file', 'sha256'}
            or reference['file'] != 'source.zip'):
        return ['source_archive must identify the sibling source.zip and its SHA-256']
    try:
        names, read, errors = _members(directory, reference['sha256'])
        if errors or names is None or read is None:
            return errors or ['unreadable source archive']
        if len(names) != len(set(names)):
            return ['duplicate source archive members']
        for name in names:
            parts = PurePosixPath(name).parts
            if (not parts or '\\' in name or ':' in name
                    or any(part in {'.', '..'} for part in name.split('/'))
                    or PurePosixPath(name).is_absolute()
                    or name not in {'script', 'registration'} and not name.startswith(('source/', 'inputs/'))):
                return ['invalid source archive member']
        inventory = record.get('source_inventory')
        if not isinstance(inventory, str) or not inventory:
            return ['source archive needs an inventory identifier']
        digest = hashlib.sha256(inventory.encode() + b'\0')
        for name in sorted(n for n in names if n.startswith('source/')):
            digest.update(name[len('source/'):].encode() + b'\0')
            digest.update(hashlib.sha256(read(name)).digest())
        errors = []
        if digest.hexdigest() != record['source_sha256']:
            errors.append('archived source does not reproduce source_sha256')
        for field in ('script', 'registration'):
            if hashlib.sha256(read(field)).hexdigest() != record[field + '_sha256']:
                errors.append(f'archived {field} digest mismatch')
        archived_inputs = {name[len('inputs/'):] for name in names if name.startswith('inputs/')}
        if record.get('schema_version', 0) >= 4 or archived_inputs:
            inputs = record.get('input_artifacts')
            if not isinstance(inputs, dict):
                errors.append('archived inputs need an input_artifacts mapping')
            elif archived_inputs != set(inputs):
                errors.append('archived input inventory differs from input_artifacts')
            else:
                for name, expected in inputs.items():
                    if hashlib.sha256(read('inputs/' + name)).hexdigest() != expected:
                        errors.append(f'archived input digest mismatch: {name}')
        if deep and not (directory / source_store.ARCHIVE).exists():
            rebuilt = source_store.rebuild_archive(directory, store_for(directory))
            if hashlib.sha256(rebuilt).hexdigest() != reference['sha256']:
                errors.append('the store does not rebuild the recorded source archive')
        return errors
    except (OSError, ValueError, KeyError, BadZipFile, RuntimeError) as exc:
        return [f'unreadable source archive: {exc}']
