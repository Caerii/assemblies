"""Old unsafe invocations must stop before compute or evidence replacement."""
import json
from pathlib import Path

import pytest

from research import runner
from neural_assemblies import (
    describe_brain_model,
    describe_hashed_aligner,
    describe_hashed_arc_fsm,
)


FIXTURE_MODEL = describe_brain_model(
    "numpy_exact", norm_init=False,
).to_dict()
FIXTURE_ORGAN = describe_hashed_arc_fsm().to_dict()
FIXTURE_ALIGNER = describe_hashed_aligner().to_dict()


@pytest.fixture
def run(tmp_path, monkeypatch):
    # Exercise storage and protocol validation without scanning the whole repo
    # per test or invoking an actual experiment.
    monkeypatch.setattr(runner, '_source_paths', lambda: [Path(__file__).relative_to(runner.ROOT).as_posix()])
    def execute(**kwargs):
        values = dict(script=Path(__file__), protocol='audit.fixture', protocol_version='1',
                      registration='research/notes/sequence/DESIGN_sequence_port.md',
                      engine='numpy_exact', seeds=[1, 2, 3], tag='fixture',
                      model_semantics=FIXTURE_MODEL,
                      parameters={'n': 100, 'k': 10}, measure=lambda record: {'values': [.1, .2, .3]},
                      output_root=tmp_path)
        values.update(kwargs)
        if "model_semantics" not in kwargs:
            values["model_semantics"] = (
                describe_brain_model(values["engine"], norm_init=False).to_dict()
                if values["engine"] in runner.BRAIN_ENGINES
                else None
            )
        if (values["engine"] in runner.ORGAN_ENGINES
                and "organ_semantics" not in kwargs):
            values["organ_semantics"] = FIXTURE_ORGAN
        if (values["engine"] in runner.ALIGNER_ENGINES
                and "aligner_semantics" not in kwargs):
            values["aligner_semantics"] = FIXTURE_ALIGNER
        return runner.run_experiment(**values)
    return execute


def test_reused_tag_refuses_before_compute_and_preserves_bytes(run):
    path = run()
    original = path.read_bytes()
    calls = []
    with pytest.raises(FileExistsError):
        run(measure=lambda record: calls.append(record))
    assert calls == []
    assert path.read_bytes() == original


@pytest.mark.parametrize('kwargs', [dict(seeds=[1, 1, 2]), dict(seeds=[1, 2]),
                                   dict(tag=''), dict(tag='../escape'), dict(engine='auto'),
                                   dict(engine=None), dict(smoke='false'),
                                   dict(model_semantics=None),
                                   dict(engine='hashed_arc_fsm', organ_semantics=None),
                                   dict(engine='hashed_arc_fsm', seeds=list(range(19))),
                                   dict(engine='hashed_aligner', aligner_semantics=None),
                                   dict(engine='scheduled_aligner', seeds=list(range(19)))])
def test_invalid_run_stops_before_compute(run, kwargs):
    calls = []
    with pytest.raises(ValueError):
        run(measure=lambda record: calls.append(record), **kwargs)
    assert calls == []


def test_wrong_engine_profile_stops_before_reservation(run, tmp_path):
    with pytest.raises(ValueError, match="does not implement requested"):
        run(engine="numpy_sparse", model_semantics=FIXTURE_MODEL)
    assert not (tmp_path / "audit.fixture" / "fixture").exists()


def test_wrong_organ_kind_stops_before_reservation(run, tmp_path):
    from neural_assemblies import describe_assembly_memory

    with pytest.raises(ValueError, match="cannot implement organ_semantics"):
        run(
            engine="hashed_arc_fsm",
            organ_semantics=describe_assembly_memory(),
        )
    assert not (tmp_path / "audit.fixture" / "fixture").exists()


def test_scheduled_aligner_rejects_dense_profile_before_reservation(run, tmp_path):
    dense = describe_hashed_aligner(store="dense", scaling=False)
    with pytest.raises(ValueError, match="does not implement"):
        run(engine="scheduled_aligner", aligner_semantics=dense)
    assert not (tmp_path / "audit.fixture" / "fixture").exists()


def test_alignment_run_uses_schema8_and_canonical_profile(run):
    from research.evidence import validate_artifact

    path = run(engine="hashed_aligner", smoke=True)
    record = json.loads(path.read_text())["run"]
    assert record["schema_version"] == 8
    assert record["execution_semantics"] == {
        "kind": "alignment", "profiles": {"default": FIXTURE_ALIGNER},
    }
    assert validate_artifact(path) == []


def test_smoke_record_is_void_and_includes_resolved_inputs(run):
    path = run(smoke=True, engine='hashed_arc_fsm')
    result = json.loads(path.read_text())
    assert result['run']['scientific_status'] == 'VOID'
    assert result['run']['seeds'] == [1, 2, 3]
    assert result['run']['parameters'] == {'n': 100, 'k': 10}
    assert result['run']['execution_semantics'] == {
        'kind': 'organ', 'profiles': {'default': FIXTURE_ORGAN},
    }
    assert result['run']['registration_sha256']
    assert result['status'] == 'complete'


def test_large_raw_json_is_a_digest_bound_compressed_attachment(run):
    from research.evidence import load_json_attachment, validate_artifact
    raw = {'frames': [{'seed': seed, 'neurons': list(range(200))}
                      for seed in range(20)]}
    path = run(measure=lambda _record: runner.ExperimentOutput(
        {'verdict': 'UNADOPTED', 'raw': {'attachment': 'raw-frames.json.gz'}},
        {'raw-frames.json.gz': raw}))
    payload = json.loads(path.read_text())
    assert payload['run']['schema_version'] == 7
    assert '"frames": [' not in path.read_text()
    assert set(payload['attachments']) == {'raw-frames.json.gz'}
    metadata = payload['attachments']['raw-frames.json.gz']
    assert metadata['bytes'] < metadata['decoded_bytes']
    assert validate_artifact(path) == []
    assert load_json_attachment(path, 'raw-frames.json.gz') == raw


def test_brain_run_records_canonical_model_semantics(run):
    path = run()
    record = json.loads(path.read_text())['run']
    assert record['execution_semantics'] == {
        'kind': 'brain', 'profiles': {'default': FIXTURE_MODEL},
    }


@pytest.mark.parametrize('damage', ['missing', 'unknown', 'noncanonical'])
def test_validator_rejects_invalid_execution_semantics(run, damage):
    from research.evidence import validate_artifact

    path = run()
    payload = json.loads(path.read_text())
    semantics = payload['run']['execution_semantics']['profiles']['default']
    if damage == 'missing':
        semantics = None
    elif damage == 'unknown':
        semantics['connectome'] = 'probably-exact'
    else:
        semantics['weight_ceiling'] = 20
    payload['run']['execution_semantics']['profiles']['default'] = semantics
    path.write_text(json.dumps(payload), encoding='utf-8')
    (path.parent / 'run.json').write_text(
        json.dumps(payload['run']), encoding='utf-8'
    )
    assert any('execution_semantics' in error for error in validate_artifact(path))


def test_attachment_encoding_is_deterministic():
    first = runner._encode_attachments({'raw.json.gz': {'values': [3, 1, 2]}})
    second = runner._encode_attachments({'raw.json.gz': {'values': [3, 1, 2]}})
    assert first == second


@pytest.mark.parametrize('damage', ['missing', 'changed', 'extra', 'metadata', 'json'])
def test_attachment_damage_or_inventory_drift_is_rejected(run, damage):
    import gzip
    from research.evidence import validate_artifact
    path = run(measure=lambda _record: runner.ExperimentOutput(
        {'verdict': 'UNADOPTED'}, {'raw.json.gz': {'values': [1, 2, 3]}}))
    attachment = path.parent / 'raw.json.gz'
    payload = json.loads(path.read_text())
    if damage == 'missing':
        attachment.unlink()
    elif damage == 'changed':
        attachment.write_bytes(attachment.read_bytes() + b'x')
    elif damage == 'extra':
        (path.parent / 'unrecorded.json.gz').write_bytes(b'extra')
    elif damage == 'metadata':
        payload['attachments']['raw.json.gz']['decoded_sha256'] = '0' * 64
        path.write_text(json.dumps(payload))
    else:
        malformed = gzip.compress(b'{"a": 1, "a": 2}', mtime=0)
        attachment.write_bytes(malformed)
        import hashlib
        item = payload['attachments']['raw.json.gz']
        item.update(sha256=hashlib.sha256(malformed).hexdigest(), bytes=len(malformed),
                    decoded_sha256=hashlib.sha256(b'{"a": 1, "a": 2}').hexdigest(),
                    decoded_bytes=len(b'{"a": 1, "a": 2}'))
        path.write_text(json.dumps(payload))
    assert validate_artifact(path)


@pytest.mark.parametrize('name', ['../raw.json.gz', 'sub/raw.json.gz', 'raw.json',
                                  'results.json.gz'])
def test_unsafe_attachment_names_fail_before_sidecar_write(run, name, tmp_path):
    with pytest.raises(ValueError, match='attachment names'):
        run(measure=lambda _record: runner.ExperimentOutput({}, {name: [1, 2, 3]}))
    directory = tmp_path / 'audit.fixture/fixture'
    assert (directory / 'failure.json').exists()
    assert not list(directory.glob('*.json.gz'))


def test_nonfinite_attachment_fails_without_publishing_partial_sidecar(run, tmp_path):
    with pytest.raises(ValueError, match='JSON compliant|nonfinite'):
        run(measure=lambda _record: runner.ExperimentOutput(
            {}, {'raw.json.gz': {'value': float('nan')}}))
    directory = tmp_path / 'audit.fixture/fixture'
    assert (directory / 'failure.json').exists()
    assert not (directory / 'results.json').exists()
    assert not (directory / 'raw.json.gz').exists()


def test_failure_keeps_record_and_reserves_tag(run):
    def fail(record):
        raise RuntimeError('constructed failure')
    with pytest.raises(RuntimeError, match='constructed'):
        run(measure=fail)
    with pytest.raises(FileExistsError):
        run()


def test_missing_dataset_stops_golden_executor(monkeypatch):
    from neural_assemblies.parity import runner as parity
    from neural_assemblies.programs import colt_mnist_data as data
    calls = []
    monkeypatch.setattr(data, 'find_mnist_dir', lambda: None)
    monkeypatch.setitem(parity.EXECUTORS, 'colt2022_mnist_notebook', lambda: calls.append(1))
    with pytest.raises(data.DatasetUnavailable, match='synthetic fallback'):
        parity.verify_protocol('colt2022_mnist_notebook')
    assert calls == []


def test_evidence_validator_rejects_a_dangling_edge(run):
    from research.evidence import validate_artifact

    path = run()
    assert validate_artifact(path) == []
    result = json.loads(path.read_text())
    result['run']['registration'] = 'research/notes/does-not-exist.md'
    path.write_text(json.dumps(result), encoding='utf-8')
    (path.parent / 'run.json').write_text(json.dumps(result['run']), encoding='utf-8')
    assert any('dangling registration' in error for error in validate_artifact(path))


def test_original_script_entry_refuses_missing_tag_before_experiment(monkeypatch):
    from research.experiments import seq_a1_horizon_hashed as horizon
    calls = []
    monkeypatch.setattr(horizon, 'experiment', lambda record: calls.append(record))
    with pytest.raises(SystemExit) as exc:
        horizon.main([])
    assert exc.value.code == 2
    assert calls == []


def test_history_resolves_parent_relative_links(tmp_path, monkeypatch):
    from research import evidence

    (tmp_path / 'docs').mkdir()
    (tmp_path / 'research' / 'results').mkdir(parents=True)
    (tmp_path / 'research' / 'results' / 'fixture.json').write_text('{}')
    (tmp_path / 'docs' / 'PREREG_fixture.md').write_text(
        '[result](../research/results/fixture.json)', encoding='utf-8')
    monkeypatch.setattr(evidence.subprocess, 'check_output',
                        lambda *a, **kw: b'docs/PREREG_fixture.md\0research/results/fixture.json\0')
    audit = evidence.audit_history(tmp_path)
    assert audit['unresolved_references'] == []
    assert audit['candidate_orphan_results'] == []
    assert audit['preregistrations_without_resolved_result_links'] == []


@pytest.mark.parametrize('field,value', [('mode', []), ('parameters', []),
                                        ('git_commit', 'unknown'), ('engine', 'auto')])
def test_malformed_record_is_reported_without_crashing(run, field, value):
    from research.evidence import validate_artifact

    path = run()
    payload = json.loads(path.read_text())
    payload['run'][field] = value
    path.write_text(json.dumps(payload), encoding='utf-8')
    (path.parent / 'run.json').write_text(json.dumps(payload['run']), encoding='utf-8')
    assert validate_artifact(path)


@pytest.fixture
def source_repo(tmp_path, monkeypatch):
    """Real Git discovery, including untracked source; no actual studies."""
    import subprocess
    root = tmp_path / 'source'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root / 'study.py').write_text('# initial source')
    subprocess.run(['git', 'add', 'study.py'], cwd=root, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], cwd=root, check=True)
    monkeypatch.setattr(runner, 'ROOT', root)
    return root


@pytest.mark.parametrize('name', [
    'formal/AssemblyIR/Learning.lean', 'formal/lean-toolchain',
    'scripts/cuda-dev.cmd',
    'formal/lake-manifest.json', 'neural_assemblies/ir/v1/projection.schema.json',
    'cpp/kernel.cuh', 'cpp/kernel.hpp', 'cpp/kernel.c',
    '.github/workflows/research-contracts.yml',
])
def test_source_identity_covers_verification_and_build_inputs(source_repo, name):
    path = source_repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    before = runner._source_identity()
    path.write_text('initial')
    added = runner._source_identity()
    path.write_text('changed')
    changed = runner._source_identity()
    assert before['git_commit'] == added['git_commit'] == changed['git_commit']
    assert len({item['source_sha256'] for item in (before, added, changed)}) == 3


def test_result_output_does_not_change_source_identity(source_repo):
    before = runner._source_identity()
    path = source_repo / 'research/results/runs/test/tag/results.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"observations": {}}')
    assert runner._source_identity() == before


def test_specification_mutation_fails_run_and_preserves_reservation(source_repo):
    spec = source_repo / 'formal/AssemblyIR.lean'
    spec.parent.mkdir()
    spec.write_text('-- before')
    (source_repo / 'registration.md').write_text('fixture protocol')
    def measure(record):
        spec.write_text('-- changed during measurement')
        return {'values': [1, 2, 3]}
    with pytest.raises(RuntimeError, match='source changed'):
        runner.run_experiment(script='study.py', protocol='fixture', protocol_version='1',
                              registration='registration.md', engine='numpy_exact',
                              seeds=[1, 2, 3], tag='mutation', parameters={},
                              model_semantics=FIXTURE_MODEL, measure=measure)
    directory = source_repo / 'research/results/runs/fixture/mutation'
    assert (directory / 'run.json').exists()
    assert not (directory / 'results.json').exists()
    assert json.loads((directory / 'failure.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('name', ['ASSEMBLIES_STREAM_INIT', 'EMERGENT_DEV_CURRICULUM',
                                  'NEURAL_ASSEMBLIES_NO_RUST'])
def test_environment_change_prevents_completed_results(run, monkeypatch, tmp_path, name):
    monkeypatch.setenv(name, 'before')
    def measure(record):
        monkeypatch.setenv(name, 'after')
        return {'values': [1, 2, 3]}
    with pytest.raises(RuntimeError, match='environment changed'):
        run(measure=measure)
    directory = tmp_path / 'audit.fixture/fixture'
    assert not (directory / 'results.json').exists()
    failure = json.loads((directory / 'failure.json').read_text())
    assert failure['status'] == 'failed'
    assert failure['run'] == json.loads((directory / 'run.json').read_text())


def test_record_carries_shared_environment_fingerprint_without_raw_values(run, monkeypatch):
    from neural_assemblies.assembly_calculus.emergent.evaluation import sweep
    monkeypatch.setenv('NEURAL_ASSEMBLIES_NO_RUST', 'private-test-value')
    path = run()
    record = json.loads(path.read_text())['run']
    assert record['schema_version'] == 7
    fingerprint = record['environment']['variables_sha256']
    assert fingerprint['NEURAL_ASSEMBLIES_NO_RUST'] == dict(
        sweep._training_env_signature())['NEURAL_ASSEMBLIES_NO_RUST']
    assert 'private-test-value' not in path.read_text()


@pytest.mark.parametrize('environment', [None, [], {},
    {'policy': 'unknown', 'variables_sha256': {}},
    {'policy': 'repository-environment-v1', 'variables_sha256': {'NEURAL_ASSEMBLIES_NO_RUST': 'raw'}},
])
def test_validator_rejects_malformed_environment(run, environment):
    from research.evidence import validate_artifact
    path = run()
    payload = json.loads(path.read_text())
    payload['run']['environment'] = environment
    path.write_text(json.dumps(payload), encoding='utf-8')
    (path.parent / 'run.json').write_text(json.dumps(payload['run']), encoding='utf-8')
    assert any('environment' in error for error in validate_artifact(path))


def test_validator_keeps_historical_schema_one_readable(run):
    from research.evidence import validate_artifact
    path = run()
    payload = json.loads(path.read_text())
    payload['run']['schema_version'] = 1
    payload['run'].pop('environment', None)
    path.write_text(json.dumps(payload), encoding='utf-8')
    (path.parent / 'run.json').write_text(json.dumps(payload['run']), encoding='utf-8')
    assert validate_artifact(path) == []


def test_unrelated_environment_change_does_not_invalidate_run(run, monkeypatch):
    def measure(record):
        monkeypatch.setenv('UNRELATED_APPLICATION_SETTING', 'changed')
        return {'values': [1, 2, 3]}
    assert run(measure=measure).exists()



def test_run_retains_the_configuration_consumed_by_measurement(run):
    from neural_assemblies import HomeostasisConfig, Brain
    config = HomeostasisConfig(norm_init=True, synaptic_scaling={'B', 'A'})
    def measure(record):
        consumed = HomeostasisConfig.from_document(record['parameters']['homeostasis'])
        brain = Brain(p=.1, engine='numpy_sparse', **consumed.as_kwargs())
        return {'executed_homeostasis': HomeostasisConfig.from_engine(brain._engine).to_document()}
    path = run(parameters={'homeostasis': config.to_document()}, measure=measure)
    payload = json.loads(path.read_text())
    assert payload['observations']['executed_homeostasis'] == payload['run']['parameters']['homeostasis']
    assert payload['run']['parameters']['homeostasis'] == config.to_document()



def test_run_records_and_executes_competition_document(run):
    from neural_assemblies import Brain, ThresholdPolicy
    from neural_assemblies.diagnostics import read_assembly
    from neural_assemblies.ir.competition import policy_from_document, policy_to_document
    document = policy_to_document(ThresholdPolicy(k=2, threshold=5))
    def measure(record):
        winners = []
        for seed in record['seeds']:
            brain = Brain(p=.1, seed=seed, engine=record['engine'], norm_init=False)
            brain.add_area('A', 4, 2, winner_policy=policy_from_document(record['parameters']['competition']))
            brain.project({}, {}, external_drive={'A': [9, 4, 3, 1]})
            winners.append(read_assembly(brain, 'A').tolist())
        return {'winners': winners}
    path = run(engine='numpy_explicit', parameters={'competition': document}, measure=measure)
    payload = json.loads(path.read_text())
    assert payload['run']['parameters']['competition'] == document
    assert payload['observations']['winners'] == [[0], [0], [0]]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999", "1e-999"])
def test_nonrepresentable_observation_json_is_not_valid_evidence(run, token):
    from research.evidence import validate_artifact
    path = run()
    payload = json.loads(path.read_text())
    payload["observations"] = {"value": "TOKEN"}
    path.write_text(json.dumps(payload).replace('"TOKEN"', token))
    assert any("JSON" in error or "binary64" in error for error in validate_artifact(path))


def test_duplicate_document_members_cannot_replace_evidence(run):
    from research.evidence import validate_artifact
    path = run()
    text = path.read_text().rstrip()
    path.write_text(text[:-1] + ', "observations": {"verdict": "PASS"}}')
    assert any("duplicate evidence" in error for error in validate_artifact(path))


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_embedded_record_types_must_match_reserved_identity(run, replacement):
    from research.evidence import validate_artifact
    path = run(parameters={"count": 1})
    payload = json.loads(path.read_text())
    payload["run"]["parameters"]["count"] = replacement
    path.write_text(json.dumps(payload))
    assert any("differs" in error for error in validate_artifact(path))


def test_strict_document_roundtrip_keeps_subnormals_and_large_integers(tmp_path):
    from research.json_documents import encode_document, load_document
    value = {"tiny": 5e-324, "large": 2**80, "zero": -0.0, "flag": False}
    path = tmp_path/"document.json"
    path.write_text(encode_document(value))
    assert encode_document(load_document(path)) == encode_document(value)


def test_source_archive_preserves_checkout_bytes_and_untracked_code(source_repo):
    from zipfile import ZipFile
    from research.evidence import validate_artifact
    script = b'# mixed endings\r\n# exact bytes\n'
    registration = b'protocol\r\n'
    (source_repo / 'study.py').write_bytes(script)
    (source_repo / 'registration.md').write_bytes(registration)
    (source_repo / 'untracked.py').write_bytes(b'# uncommitted\r\n')
    path = runner.run_experiment(
        script='study.py', registration='registration.md', protocol='fixture',
        protocol_version='1', engine='numpy_exact', seeds=[1, 2, 3], tag='capture',
        parameters={}, model_semantics=FIXTURE_MODEL,
        measure=lambda record: {'value': 0})
    with ZipFile(path.parent / 'source.zip') as archive:
        assert archive.read('source/study.py') == script
        assert archive.read('script') == script
        assert archive.read('registration') == registration
        assert archive.read('source/untracked.py') == b'# uncommitted\r\n'
    # Later checkout changes cannot rewrite the captured evidence.
    (source_repo / 'study.py').write_bytes(b'# later version')
    assert validate_artifact(path, root=source_repo) == []


def test_active_graph_rejects_valid_result_not_linked_from_registration(source_repo):
    import subprocess
    from research.evidence import validate_active_evidence_graph
    registration = source_repo / 'registration.md'
    registration.write_text('registered fixture')
    runner.run_experiment(
        script='study.py', registration='registration.md', protocol='fixture',
        protocol_version='1', engine='numpy_exact', seeds=[1, 2, 3], tag='linked',
        parameters={}, model_semantics=FIXTURE_MODEL,
        measure=lambda _record: {'verdict': 'UNADOPTED'})
    subprocess.run(['git', 'add', 'registration.md', 'research/results/runs'],
                   cwd=source_repo, check=True)
    errors = validate_active_evidence_graph(source_repo)
    assert any('does not link this result' in error for error in errors)
    registration.write_text(
        '[result](research/results/runs/fixture/linked/results.json)')
    assert validate_active_evidence_graph(source_repo) == []


@pytest.mark.parametrize('damage', ['missing', 'bytes', 'source', 'registration', 'duplicate', 'escape'])
def test_source_archive_damage_is_rejected_even_with_updated_container_digest(run, damage):
    import hashlib
    import warnings
    from zipfile import ZipFile
    from research.evidence import validate_artifact
    path = run()
    archive_path = path.parent / 'source.zip'
    payload = json.loads(path.read_text())
    if damage == 'missing':
        archive_path.unlink()
    elif damage == 'bytes':
        archive_path.write_bytes(b'not a zip')
    else:
        with ZipFile(archive_path) as archive:
            entries = [(name, archive.read(name)) for name in archive.namelist()]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with ZipFile(archive_path, 'w') as archive:
                for name, data in entries:
                    changed = (damage == 'source' and name.startswith('source/')
                               or damage == 'registration' and name == 'registration')
                    archive.writestr(name, b'changed' if changed else data)
                if damage == 'duplicate':
                    archive.writestr(*entries[0])
                if damage == 'escape':
                    archive.writestr('source/../escape.py', b'bad')
        payload['run']['source_archive']['sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(payload))
        (path.parent / 'run.json').write_text(json.dumps(payload['run']))
    assert any('archive' in error or 'archived' in error for error in validate_artifact(path))


def test_archive_mutation_during_measurement_prevents_completion(run, tmp_path):
    def measure(record):
        (tmp_path / 'audit.fixture/fixture/source.zip').write_bytes(b'changed')
        return {'value': 1}
    with pytest.raises(RuntimeError, match='archive digest mismatch'):
        run(measure=measure)
    directory = tmp_path / 'audit.fixture/fixture'
    assert (directory / 'failure.json').exists()
    assert not (directory / 'results.json').exists()


@pytest.mark.parametrize('version', [1, 2])
def test_historical_records_do_not_require_source_capture(run, version):
    from research.evidence import validate_artifact
    path = run()
    payload = json.loads(path.read_text())
    payload['run']['schema_version'] = version
    payload['run'].pop('source_archive')
    (path.parent / 'source.zip').unlink()
    path.write_text(json.dumps(payload))
    (path.parent / 'run.json').write_text(json.dumps(payload['run']))
    assert validate_artifact(path) == []


@pytest.mark.parametrize('damage', [None, 'changed', 'missing', 'extra'])
def test_run_inputs_are_recoverable_and_bound_to_record(source_repo, damage):
    import hashlib
    from zipfile import ZipFile
    from research.evidence import validate_artifact
    data = b'{"size": 60}\r\n'
    (source_repo / 'parameters.json').write_bytes(data)
    (source_repo / 'registration.md').write_text('fixture protocol')
    path = runner.run_experiment(
        script='study.py', registration='registration.md', protocol='fixture',
        protocol_version='1', engine='numpy_exact', seeds=[1, 2, 3], tag='capture',
        parameters={'size': 60}, input_artifacts=('./parameters.json',),
        model_semantics=FIXTURE_MODEL,
        measure=lambda record: {'value': 0})
    payload = json.loads(path.read_text())
    assert payload['run']['input_artifacts'] == {'parameters.json': hashlib.sha256(data).hexdigest()}
    archive_path = path.parent / 'source.zip'
    with ZipFile(archive_path) as archive:
        assert archive.read('inputs/parameters.json') == data
        entries = {name: archive.read(name) for name in archive.namelist()}
    # Later checkout edits do not alter the original captured input.
    (source_repo / 'parameters.json').write_text('{"size": 80}')
    assert validate_artifact(path, root=source_repo) == []
    if damage is None:
        return
    if damage == 'changed':
        entries['inputs/parameters.json'] = b'changed'
    elif damage == 'missing':
        del entries['inputs/parameters.json']
    else:
        entries['inputs/unrecorded.json'] = b'{}'
    with ZipFile(archive_path, 'w') as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    payload['run']['source_archive']['sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    (path.parent / 'run.json').write_text(json.dumps(payload['run']))
    assert any('archived input' in error for error in validate_artifact(path, root=source_repo))


def test_duplicate_input_aliases_fail_before_measurement(source_repo):
    (source_repo / 'registration.md').write_text('fixture protocol')
    (source_repo / 'parameters.json').write_text('{}')
    with pytest.raises(ValueError, match='duplicate input artifact'):
        runner.run_experiment(
            script='study.py', registration='registration.md', protocol='fixture',
            protocol_version='1', engine='numpy_exact', seeds=[1, 2, 3], tag='duplicate',
            parameters={}, input_artifacts=('parameters.json', './parameters.json'),
            model_semantics=FIXTURE_MODEL,
            measure=lambda record: pytest.fail('duplicate inputs reached computation'))
    assert not (source_repo / 'research/results/runs').exists()


def test_historical_schema_three_inputs_are_not_claimed_recoverable(run):
    from zipfile import ZipFile
    from research.evidence import validate_artifact
    path = run()
    payload = json.loads(path.read_text())
    payload['run']['schema_version'] = 3
    name = 'research/notes/sequence/DESIGN_sequence_port.md'
    payload['run']['input_artifacts'] = {name: 'a' * 64}
    path.write_text(json.dumps(payload))
    (path.parent / 'run.json').write_text(json.dumps(payload['run']))
    with ZipFile(path.parent / 'source.zip') as archive:
        assert not any(name.startswith('inputs/') for name in archive.namelist())
    assert validate_artifact(path) == []


def test_input_mutation_keeps_original_bytes_and_records_failure(source_repo):
    from zipfile import ZipFile
    (source_repo / 'registration.md').write_text('fixture protocol')
    input_path = source_repo / 'parameters.json'
    input_path.write_bytes(b'{"size": 60}')
    def measure(record):
        input_path.write_bytes(b'{"size": 80}')
        return {'value': 1}
    with pytest.raises(RuntimeError, match='input artifact changed'):
        runner.run_experiment(
            script='study.py', registration='registration.md', protocol='fixture',
            protocol_version='1', engine='numpy_exact', seeds=[1, 2, 3], tag='mutation',
            parameters={}, input_artifacts=('parameters.json',),
            model_semantics=FIXTURE_MODEL, measure=measure)
    directory = source_repo / 'research/results/runs/fixture/mutation'
    assert (directory / 'failure.json').exists()
    assert not (directory / 'results.json').exists()
    with ZipFile(directory / 'source.zip') as archive:
        assert archive.read('inputs/parameters.json') == b'{"size": 60}'


@pytest.mark.parametrize('change', ['bytes', 'inventory'])
def test_configuration_snapshot_mismatch_fails_before_reservation(source_repo, change):
    import hashlib
    data = b'{"size": 60}'
    (source_repo / 'parameters.json').write_bytes(data)
    (source_repo / 'registration.md').write_text('fixture protocol')
    expected = {'parameters.json': hashlib.sha256(data).hexdigest()}
    if change == 'bytes':
        (source_repo / 'parameters.json').write_bytes(b'{"size": 80}')
    else:
        expected['unlisted.json'] = 'a' * 64
    with pytest.raises(ValueError, match='configuration snapshot'):
        runner.run_experiment(
            script='study.py', registration='registration.md', protocol='fixture',
            protocol_version='1', engine='numpy_exact', seeds=[1, 2, 3], tag='changed',
            parameters={'size': 60}, input_artifacts=('parameters.json',),
            expected_input_digests=expected,
            model_semantics=FIXTURE_MODEL,
            measure=lambda record: pytest.fail('mismatched snapshot reached measurement'))
    assert not (source_repo / 'research/results/runs').exists()


def test_parameter_cli_runs_with_archived_overrides(source_repo, monkeypatch):
    from zipfile import ZipFile
    from research.experiments import _historical
    from research.experiments.base import ExperimentResult
    from research.evidence import validate_artifact
    monkeypatch.setattr(_historical, 'ROOT', source_repo)
    (source_repo / 'registration.md').write_text('fixture protocol')
    data = b'{"size": 7}\r\n'
    (source_repo / 'parameters.json').write_bytes(data)
    class Producer:
        def __init__(self, **kwargs):
            pass
        def run(self, **kwargs):
            model = kwargs.pop('model_semantics')
            assert model == describe_brain_model(
                'numpy_explicit', norm_init=False,
            ).to_dict()
            assert kwargs == {'size': 7, 'rounds': 5, 'seed_ids': [9, 2, 7]}
            return ExperimentResult('fixture', success=True, parameters=kwargs)
    spec = _historical.HistoricalStudy(
        'fixture', '1', 'registration.md', Path('study.py'), Producer,
        lambda smoke: {'size': 3, 'rounds': 5}, 'fixture scope')
    spec.main(['--tag', 'override', '--smoke', '--seeds', '9', '2', '7',
               '--parameters', 'parameters.json'])
    path = source_repo / 'research/results/runs/fixture/override/results.json'
    assert validate_artifact(path, root=source_repo) == []
    payload = json.loads(path.read_text())
    assert payload['run']['parameters'] == {'size': 7, 'rounds': 5}
    assert payload['observations']['verdict'] == 'VOID'
    with ZipFile(path.parent / 'source.zip') as archive:
        assert archive.read('inputs/parameters.json') == data
