"""Shared experiment entry: validate, reserve, execute, retain attributable output.

This owns execution records and storage. research.harness owns paired studies;
neural_assemblies.diagnostics owns statistics. A completed run is not a PASS.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, cast
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from neural_assemblies.core.environment import environment_record
from neural_assemblies.core.semantics import (
    ALIGNER_ENGINE_NAMES, AlignerSemantics, AlignmentStore, BRAIN_ENGINE_NAMES, ExecutionKind,
    ExecutionSemantics, ModelSemantics, NormalizationMode, ORGAN_ENGINE_KINDS,
    OrganSemantics, PlasticityRule, describe_brain_model,
)
from research.json_documents import (
    encode_document, snapshot_document, write_new_document as _write_new,
)
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
_ATTACHMENT_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.json\.gz\Z')
BRAIN_ENGINES = BRAIN_ENGINE_NAMES
ORGAN_ENGINES = ORGAN_ENGINE_KINDS
ALIGNER_ENGINES = ALIGNER_ENGINE_NAMES


@dataclass(frozen=True)
class ExperimentOutput:
    """Small indexed observations plus lossless, digest-bound raw JSON sidecars.

    Specification: research/README.md#raw-evidence-attachments
    """

    observations: Mapping
    json_attachments: Mapping[str, Any] = field(default_factory=dict)


# The runner accepts either the compact observation mapping used by small
# studies or the lossless envelope used when a study retains raw JSON.  Keep
# this type next to the writer so adapters cannot accidentally promise a
# narrower return shape than the storage boundary supports.
MeasureOutput = Mapping[str, Any] | ExperimentOutput


def _encode_attachments(attachments: Mapping[str, Any]) -> dict[str, tuple[bytes, dict]]:
    if not isinstance(attachments, Mapping):
        raise ValueError('json_attachments must be a mapping')
    encoded = {}
    for name, value in attachments.items():
        if (not isinstance(name, str) or not _ATTACHMENT_NAME.fullmatch(name)
                or name in {'results.json.gz', 'run.json.gz', 'failure.json.gz'}):
            raise ValueError('attachment names must be safe sibling *.json.gz names')
        raw = encode_document(value).encode('utf-8')
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        encoded[name] = (compressed, {
            'media_type': 'application/json', 'content_encoding': 'gzip',
            'sha256': hashlib.sha256(compressed).hexdigest(),
            'decoded_sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(compressed), 'decoded_bytes': len(raw),
        })
    return encoded


def _write_new_bytes(path: Path, value: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(value)
        stream.flush()


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
                   parameters: Mapping, measure: Callable[[dict], MeasureOutput],
                   smoke: bool = False, minimum_study_seeds: int = 3,
                   output_root: Path | None = None,
                   input_artifacts: tuple[str, ...] = (),
                   expected_input_digests: Mapping[str, str] | None = None,
                   model_semantics: ModelSemantics | Mapping | None = None,
                   organ_semantics: OrganSemantics | Mapping | None = None,
                   aligner_semantics: AlignerSemantics | Mapping | None = None) -> Path:
    """Execute one resolved protocol; return its immutable results file.

    `measure(record)` receives a JSON snapshot of the resolved inputs. It must
    return JSON observations, including its own explicit scientific verdict.
    Completion alone never means an adoption bar was satisfied. Reserve before
    invoking it, and preserve failure records rather than reusing the tag.
    Optional expected_input_digests binds prior parsing to the complete captured
    input inventory, using canonical repository-relative names (research/README.md
    #historical-experiment-parameter-files). Mismatch fails before reservation.
    Brain engines require a complete ModelSemantics profile; hashed organs
    require one or more OrganSemantics profiles. The runner writes the strict
    discriminated ExecutionSemantics envelope and rejects disagreement before
    reserving a tag. Organ implementations consume their recorded profile again
    before allocating device state.
    """
    for name, value in [('tag', tag), ('protocol', protocol), ('protocol_version', protocol_version)]:
        if not isinstance(value, str) or not _NAME.fullmatch(value):
            raise ValueError(f'{name} must be a nonempty simple name (letters, digits, dot, dash, underscore)')
    if not isinstance(engine, str) or not _NAME.fullmatch(engine) or engine == 'auto':
        raise ValueError('record the resolved engine; auto is not provenance')
    if engine in BRAIN_ENGINES:
        if organ_semantics is not None or aligner_semantics is not None:
            raise ValueError('Brain engines cannot claim organ or aligner semantics')
        if model_semantics is None:
            raise ValueError(
                f'{engine} runs require a complete model_semantics document'
            )
        requested_model = ModelSemantics.normalize(model_semantics)
        actual_model = describe_brain_model(
            engine,
            p=.05,
            seed=0,
            w_max=requested_model.weight_ceiling,
            norm_init=(
                requested_model.normalization
                is NormalizationMode.INVERSE_INDEGREE
            ),
        )
        mismatch = requested_model.mismatch(actual_model)
        if mismatch:
            raise ValueError(
                f'{engine} does not implement requested model_semantics: {mismatch}'
            )
        execution_document = ExecutionSemantics(
            ExecutionKind.BRAIN, {'default': requested_model},
        ).to_dict()
        schema_version = 7
    elif engine in ORGAN_ENGINES:
        if (model_semantics is not None or aligner_semantics is not None
                or organ_semantics is None):
            raise ValueError(
                f'{engine} requires organ_semantics exclusively'
            )
        if (isinstance(organ_semantics, OrganSemantics)
                or (isinstance(organ_semantics, Mapping)
                    and 'organ' in organ_semantics)):
            profiles = {'default': OrganSemantics.normalize(organ_semantics)}
        elif isinstance(organ_semantics, Mapping):
            profiles = dict(organ_semantics)
        else:
            raise TypeError('organ_semantics must be an OrganSemantics or profile mapping')
        execution_document = ExecutionSemantics(
            ExecutionKind.ORGAN, profiles,
        ).to_dict()
        profiles_document = cast(
            Mapping[str, Mapping[str, Any]], execution_document['profiles']
        )
        wrong = {
            name: profile['organ']
            for name, profile in profiles_document.items()
            if profile['organ'] != ORGAN_ENGINES[engine].value
        }
        if wrong:
            raise ValueError(
                f'{engine} cannot implement organ_semantics profiles: {wrong}'
            )
        schema_version = 7
    elif engine in ALIGNER_ENGINES:
        if (model_semantics is not None or organ_semantics is not None
                or aligner_semantics is None):
            raise ValueError(f'{engine} requires aligner_semantics exclusively')
        requested_aligner = AlignerSemantics.normalize(aligner_semantics)
        if engine == "scheduled_aligner" and (
            requested_aligner.cross_store is not AlignmentStore.PRESENT_ONLY
            or requested_aligner.cross_weight_ceiling is not None
            or requested_aligner.cross_plasticity
            is not PlasticityRule.MULTIPLICATIVE_UNBOUNDED
            or requested_aligner.anchor_plasticity is not PlasticityRule.NONE
        ):
            raise ValueError(
                "scheduled_aligner does not implement requested aligner_semantics"
            )
        execution_document = ExecutionSemantics(
            ExecutionKind.ALIGNMENT, {"default": requested_aligner},
        ).to_dict()
        schema_version = 8
    else:
        raise ValueError(f'unknown execution engine: {engine}')
    if type(smoke) is not bool or type(minimum_study_seeds) is not int or minimum_study_seeds < 3:
        raise ValueError('smoke must be boolean and minimum_study_seeds an integer of at least three')
    seeds = list(seeds)
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique integer identities')
    hashed_study = engine.startswith('hashed') or engine in ALIGNER_ENGINES
    minimum = 3 if smoke else max(3, minimum_study_seeds, 20 if hashed_study else 3)
    if len(seeds) < minimum:
        raise ValueError(f'{engine} {"smoke" if smoke else "study"} requires at least {minimum} unique seeds')
    script_path = _repo_file(script)
    registration_path = _repo_file(registration)
    input_bytes = {}
    for path in input_artifacts:
        resolved = _repo_file(path)
        name = resolved.relative_to(ROOT).as_posix()
        if name in input_bytes:
            raise ValueError(f'duplicate input artifact: {name}')
        input_bytes[name] = resolved.read_bytes()
    inputs = {name: hashlib.sha256(data).hexdigest() for name, data in input_bytes.items()}
    if expected_input_digests is not None and inputs != dict(expected_input_digests):
        raise ValueError('input artifacts differ from the configuration snapshot')
    record = dict(schema_version=schema_version, environment=environment_record(), source_inventory=SOURCE_INVENTORY, protocol=protocol, protocol_version=protocol_version,
                  script=script_path.relative_to(ROOT).as_posix(),
                  script_sha256=hashlib.sha256(script_path.read_bytes()).hexdigest(),
                  registration=registration_path.relative_to(ROOT).as_posix(),
                  registration_sha256=hashlib.sha256(registration_path.read_bytes()).hexdigest(),
                  engine=engine, execution_semantics=execution_document, seeds=seeds,
                  tag=tag, parameters=dict(parameters), input_artifacts=inputs,
                  mode='smoke' if smoke else 'study', scientific_status='VOID' if smoke else 'UNJUDGED',
                  started_utc=datetime.now(timezone.utc).isoformat(), **_source_identity())
    # Freeze nested caller-owned mappings/lists into a distinct JSON value.
    record = snapshot_document(record)
    parent = (output_root or ROOT / 'research' / 'results' / 'runs') / protocol
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / tag
    directory.mkdir()  # atomic reservation; raises FileExistsError before measure
    # Source-linked specification: research/README.md#recoverable-source
    # Reserve first; no measurement may run without a complete source capture.
    archive_path = directory / 'source.zip'
    try:
        with ZipFile(archive_path, 'x') as archive:
            captured = _source_identity(archive)
            for name, data in input_bytes.items():
                _archive_bytes(archive, 'inputs/' + name, data)
            for field, path in [('script', script_path), ('registration', registration_path)]:
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != record[field + '_sha256']:
                    raise RuntimeError(f'{field} changed while capturing source')
                _archive_bytes(archive, field, data)
        if captured != {k: record[k] for k in ('git_commit', 'source_sha256')}:
            raise RuntimeError('source changed while capturing source')
        record['source_archive'] = {
            'file': 'source.zip',
            'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        }
        _write_new(directory / 'run.json', record)
    except BaseException as exc:
        # Reservation is intentionally early.  If capture fails, retain the
        # reservation and its reason instead of leaving an unexplainable
        # orphan that a later audit cannot distinguish from an interrupted run.
        _write_new(directory / 'failure.json', {
            'run': record,
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
        })
        raise
    try:
        measured = measure(snapshot_document(record))
        if isinstance(measured, ExperimentOutput):
            observations = measured.observations
            attachments = _encode_attachments(measured.json_attachments)
        else:
            observations, attachments = measured, {}
        if not isinstance(observations, Mapping):
            raise ValueError('experiment must return an observations mapping')
        # Validate summaries before writing any attachment or completed record.
        observations = snapshot_document(dict(observations))
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
        for name, (content, _metadata) in attachments.items():
            _write_new_bytes(directory / name, content)
        path = directory / 'results.json'
        _write_new(path, {'run': record, 'status': 'complete', 'observations': observations,
                          'attachments': {name: metadata for name, (_content, metadata)
                                          in attachments.items()}})
        return path
    except BaseException as exc:
        _write_new(directory / 'failure.json', {'run': record, 'status': 'failed',
                                               'error_type': type(exc).__name__, 'error': str(exc)})
        raise


EXPERIMENTS = {'historical-merge': 'research.experiments.historical_merge',
               'historical-association': 'research.experiments.historical_association',
               'historical-phase': 'research.experiments.historical_phase',
               'historical-scaling': 'research.experiments.historical_scaling',
               'historical-projection': 'research.experiments.historical_projection',
               'historical-noise': 'research.experiments.historical_noise',
               'context-noise': 'research.experiments.context_noise',
               'a1-learning-null': 'research.experiments.seq_a1_learning_null',
               'a1-fsm-parity': 'research.experiments.seq_a1_fsm_parity',
               'a1-exactness-sweep': 'research.experiments.seq_a1_exactness_sweep',
               'a1-local-regime': 'research.experiments.seq_a1_local_regime',
               'a2-refraction-load': 'research.experiments.seq_a2_refraction_load',
               'refraction-memory-numpy': 'research.experiments.refraction_memory_numpy',
               'a1-horizon': 'research.experiments.seq_a1_horizon_hashed',
               'temporal-positions': 'research.experiments.seq_temporal_positions',
               'temporal-memory-high-order': 'research.experiments.seq_tm_high_order',
               's5-soft-census': 'research.experiments.seq_s5_soft_census_hashed',
               'per-fiber-plasticity': 'research.experiments.per_fiber_plasticity',
               'capacity-scaling': 'research.experiments.seq_capacity_scaling'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run a migrated experiment with immutable provenance.')
    parser.add_argument('--list', action='store_true', dest='list_experiments',
                        help='list registered experiment commands without importing them')
    parser.add_argument('experiment', nargs='?', choices=sorted(EXPERIMENTS))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.list_experiments:
        for command, module in sorted(EXPERIMENTS.items()):
            print(f'{command}\t{module}')
        return 0
    if args.experiment is None:
        parser.error('an experiment command is required (use --list to inspect the registry)')
    module = importlib.import_module(EXPERIMENTS[args.experiment])
    module.main(args.arguments)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
