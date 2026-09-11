"""Shared experiment entry: validate, reserve, execute, retain attributable output.

This owns execution records and storage. research.harness owns paired studies;
neural_assemblies.diagnostics owns statistics. A completed run is not a PASS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from neural_assemblies.core.environment import environment_record
from research.json_documents import write_new_document as _write_new
from research.source_archive import validate_source_archive

ROOT = Path(__file__).resolve().parents[1]
# Source-linked specification: research/README.md#source-identity
SOURCE_INVENTORY = 'source-inputs-v3'
_SOURCE_SUFFIXES = frozenset({
    '.py', '.rs', '.cu', '.cuh', '.c', '.cc', '.cpp', '.h', '.hpp',
    '.lean', '.dfy', '.ts', '.tsx', '.js', '.mjs', '.toml', '.lock',
    '.yaml', '.yml', '.cmake', '.ps1', '.bat', '.cmd', '.sh',
})
_SOURCE_NAMES = frozenset({'lean-toolchain', 'CMakeLists.txt', 'Makefile'})


def _is_source_input(path: Path) -> bool:
    return (path.suffix in _SOURCE_SUFFIXES or path.name in _SOURCE_NAMES
            or (path.suffix == '.json' and path.parts[:1] == ('formal',))
            or (path.suffix == '.json' and path.parts[:2] == ('neural_assemblies', 'ir')))


_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')


def experiment_parser(description: str, *, engines: tuple[str, ...],
                      default_seeds: tuple[int, ...]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(default_seeds),
                        help='explicit independent seed identities')
    parser.add_argument('--tag', required=True, help='unique run name; existing runs are never overwritten')
    parser.add_argument('--engine', choices=engines, default=engines[0])
    parser.add_argument('--smoke', action='store_true', help='API check; observations are VOID as scientific evidence')
    return parser


def _repo_file(path: str | Path) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError(f'run input must name an existing repository file: {path}')
    return resolved


def _source_paths() -> list[str]:
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    return sorted({name for name in paths if name and _is_source_input(Path(name))})


def _archive_bytes(archive: ZipFile, name: str, data: bytes) -> None:
    entry = ZipInfo(name)  # Fixed timestamp; preserve bytes, not checkout metadata.
    entry.compress_type = ZIP_DEFLATED
    archive.writestr(entry, data)


def _source_identity(archive: ZipFile | None = None) -> dict:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    digest = hashlib.sha256(SOURCE_INVENTORY.encode() + b'\0')
    for name in _source_paths():
        data = _repo_file(name).read_bytes()
        digest.update(name.encode() + b'\0')
        digest.update(hashlib.sha256(data).digest())
        if archive is not None:
            _archive_bytes(archive, 'source/' + name, data)
    return {'git_commit': commit, 'source_sha256': digest.hexdigest()}


def run_experiment(*, script: str | Path, protocol: str, protocol_version: str,
                   registration: str | Path, engine: str, seeds: list[int], tag: str,
                   parameters: Mapping, measure: Callable[[dict], Mapping],
                   smoke: bool = False, minimum_study_seeds: int = 3,
                   output_root: Path | None = None,
                   input_artifacts: tuple[str, ...] = ()) -> Path:
    """Execute one resolved protocol; return its immutable results file.

    `measure(record)` receives a JSON snapshot of the resolved inputs. It must
    return JSON observations, including its own explicit scientific verdict.
    Completion alone never means an adoption bar was satisfied. Reserve before
    invoking it, and preserve failure records rather than reusing the tag.
    """
    for name, value in [('tag', tag), ('protocol', protocol), ('protocol_version', protocol_version)]:
        if not isinstance(value, str) or not _NAME.fullmatch(value):
            raise ValueError(f'{name} must be a nonempty simple name (letters, digits, dot, dash, underscore)')
    if not isinstance(engine, str) or not _NAME.fullmatch(engine) or engine == 'auto':
        raise ValueError('record the resolved engine; auto is not provenance')
    if type(smoke) is not bool or type(minimum_study_seeds) is not int or minimum_study_seeds < 3:
        raise ValueError('smoke must be boolean and minimum_study_seeds an integer of at least three')
    seeds = list(seeds)
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique integer identities')
    minimum = 3 if smoke else max(3, minimum_study_seeds, 20 if engine.startswith('hashed') else 3)
    if len(seeds) < minimum:
        raise ValueError(f'{engine} {"smoke" if smoke else "study"} requires at least {minimum} unique seeds')
    script_path = _repo_file(script)
    registration_path = _repo_file(registration)
    inputs = {path: hashlib.sha256(_repo_file(path).read_bytes()).hexdigest()
              for path in input_artifacts}
    record = dict(schema_version=3, environment=environment_record(), source_inventory=SOURCE_INVENTORY, protocol=protocol, protocol_version=protocol_version,
                  script=script_path.relative_to(ROOT).as_posix(),
                  script_sha256=hashlib.sha256(script_path.read_bytes()).hexdigest(),
                  registration=registration_path.relative_to(ROOT).as_posix(),
                  registration_sha256=hashlib.sha256(registration_path.read_bytes()).hexdigest(),
                  engine=engine, seeds=seeds, tag=tag, parameters=dict(parameters), input_artifacts=inputs,
                  mode='smoke' if smoke else 'study', scientific_status='VOID' if smoke else 'UNJUDGED',
                  started_utc=datetime.now(timezone.utc).isoformat(), **_source_identity())
    # Freeze nested caller-owned mappings/lists into a distinct JSON value.
    record = json.loads(json.dumps(record, allow_nan=False))
    parent = (output_root or ROOT / 'research' / 'results' / 'runs') / protocol
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / tag
    directory.mkdir()  # atomic reservation; raises FileExistsError before measure
    # Source-linked specification: research/README.md#recoverable-source
    # Reserve first; no measurement may run without a complete source capture.
    archive_path = directory / 'source.zip'
    with ZipFile(archive_path, 'x') as archive:
        captured = _source_identity(archive)
        for field, path in [('script', script_path), ('registration', registration_path)]:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != record[field + '_sha256']:
                raise RuntimeError(f'{field} changed while capturing source')
            _archive_bytes(archive, field, data)
    if captured != {k: record[k] for k in ('git_commit', 'source_sha256')}:
        raise RuntimeError('source changed while capturing source')
    record['source_archive'] = {'file': 'source.zip',
                                'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    _write_new(directory / 'run.json', record)
    try:
        observations = measure(json.loads(json.dumps(record)))
        if not isinstance(observations, Mapping):
            raise ValueError('experiment must return an observations mapping')
        if environment_record() != record['environment']:
            raise RuntimeError('repository environment changed during the run; observations cannot be adopted')
        if _source_identity() != {k: record[k] for k in ('git_commit', 'source_sha256')}:
            raise RuntimeError('source changed during the run; observations cannot be adopted')
        if hashlib.sha256(registration_path.read_bytes()).hexdigest() != record['registration_sha256']:
            raise RuntimeError('registration changed during the run')
        if any(hashlib.sha256(_repo_file(path).read_bytes()).hexdigest() != digest
               for path, digest in inputs.items()):
            raise RuntimeError('input artifact changed during the run')
        archive_errors = validate_source_archive(directory, record)
        if archive_errors:
            raise RuntimeError('; '.join(archive_errors))
        path = directory / 'results.json'
        _write_new(path, {'run': record, 'status': 'complete', 'observations': dict(observations)})
        return path
    except BaseException as exc:
        _write_new(directory / 'failure.json', {'run': record, 'status': 'failed',
                                               'error_type': type(exc).__name__, 'error': str(exc)})
        raise


EXPERIMENTS = {'historical-phase': 'research.experiments.historical_phase',
               'historical-scaling': 'research.experiments.historical_scaling',
               'historical-projection': 'research.experiments.historical_projection',
               'historical-noise': 'research.experiments.historical_noise',
               'context-noise': 'research.experiments.context_noise',
               'a1-learning-null': 'research.experiments.seq_a1_learning_null',
               'a1-horizon': 'research.experiments.seq_a1_horizon_hashed',
               'capacity-scaling': 'research.experiments.seq_capacity_scaling'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run a migrated experiment with immutable provenance.')
    parser.add_argument('experiment', choices=sorted(EXPERIMENTS))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    module = importlib.import_module(EXPERIMENTS[args.experiment])
    module.main(args.arguments)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
