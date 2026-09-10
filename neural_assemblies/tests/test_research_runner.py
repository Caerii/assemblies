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
