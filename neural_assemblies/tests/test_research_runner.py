"""Old unsafe invocations must stop before compute or evidence replacement."""
import json
from pathlib import Path

import pytest

from research import runner


@pytest.fixture
def run(tmp_path, monkeypatch):
    # Exercise storage and protocol validation without scanning the whole repo
    # per test or invoking an actual experiment.
    monkeypatch.setattr(runner, '_source_identity', lambda: {'git_commit': 'b' * 40, 'source_sha256': 'a' * 64})
    def execute(**kwargs):
        values = dict(script=Path(__file__), protocol='audit.fixture', protocol_version='1',
                      registration='research/notes/sequence/DESIGN_sequence_port.md',
                      engine='numpy_exact', seeds=[1, 2, 3], tag='fixture',
                      parameters={'n': 100, 'k': 10}, measure=lambda record: {'values': [.1, .2, .3]},
                      output_root=tmp_path)
        values.update(kwargs)
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
                                   dict(engine='hashed_arc_fsm', seeds=list(range(19)))])
def test_invalid_run_stops_before_compute(run, kwargs):
    calls = []
    with pytest.raises(ValueError):
        run(measure=lambda record: calls.append(record), **kwargs)
    assert calls == []


def test_smoke_record_is_void_and_includes_resolved_inputs(run):
    path = run(smoke=True, engine='hashed_arc_fsm')
    result = json.loads(path.read_text())
    assert result['run']['scientific_status'] == 'VOID'
    assert result['run']['seeds'] == [1, 2, 3]
    assert result['run']['parameters'] == {'n': 100, 'k': 10}
    assert result['run']['registration_sha256']
    assert result['status'] == 'complete'


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
                              seeds=[1, 2, 3], tag='mutation', parameters={}, measure=measure)
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
    assert record['schema_version'] == 2
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
