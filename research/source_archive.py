"""Check recoverable source without extracting or importing archived code.

Source-linked specification: research/README.md#recoverable-source
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


def validate_source_archive(directory: Path, record: dict) -> list[str]:
    """Bind exact archived bytes to the source and protocol digests in a run."""
    reference = record.get('source_archive')
    if (not isinstance(reference, dict) or set(reference) != {'file', 'sha256'}
            or reference['file'] != 'source.zip'):
        return ['source_archive must identify the sibling source.zip and its SHA-256']
    try:
        path = directory / 'source.zip'
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
            return ['source archive digest mismatch']
        with ZipFile(path) as archive:
            names = archive.namelist()
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
                digest.update(hashlib.sha256(archive.read(name)).digest())
            errors = []
            if digest.hexdigest() != record['source_sha256']:
                errors.append('archived source does not reproduce source_sha256')
            for field in ('script', 'registration'):
                if hashlib.sha256(archive.read(field)).hexdigest() != record[field + '_sha256']:
                    errors.append(f'archived {field} digest mismatch')
            archived_inputs = {name[len('inputs/'):] for name in names if name.startswith('inputs/')}
            if record.get('schema_version') in (4, 5) or archived_inputs:
                inputs = record.get('input_artifacts')
                if not isinstance(inputs, dict):
                    errors.append('archived inputs need an input_artifacts mapping')
                elif archived_inputs != set(inputs):
                    errors.append('archived input inventory differs from input_artifacts')
                else:
                    for name, expected in inputs.items():
                        if hashlib.sha256(archive.read('inputs/' + name)).hexdigest() != expected:
                            errors.append(f'archived input digest mismatch: {name}')
            return errors
    except (OSError, ValueError, KeyError, BadZipFile, RuntimeError) as exc:
        return [f'unreadable source archive: {exc}']
