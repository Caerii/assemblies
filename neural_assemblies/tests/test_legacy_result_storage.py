"""Old result APIs must preserve files and reject silently lossy evidence."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from research.experiments import base
from research.experiments.stability.test_noise_robustness import NoiseRobustnessExperiment
from research.json_documents import write_new_document
from research import runner


def test_old_and_new_runners_share_the_exclusive_writer():
    assert runner._write_new is write_new_document
    assert base.write_new_document is write_new_document


def test_legacy_save_roundtrip_and_overwrite_refusal(tmp_path):
    result = base.ExperimentResult('fixture', parameters={'n': 100, 'enabled': False},
                                   raw_data={'seeds': [1, 2, 3], 'values': [.1, .2, .3]})
    path = tmp_path / 'nested' / 'result.json'
    result.save(path)
    assert base.ExperimentResult.load(path) == result
    original = path.read_bytes()
    result.metrics['new'] = 1
    with pytest.raises(FileExistsError):
        result.save(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize('value', [float('nan'), float('inf'), object(), np.array([1., 2.])])
def test_unrepresentable_data_fails_before_creating_parent(tmp_path, value):
    path = tmp_path / 'absent' / 'result.json'
    with pytest.raises((TypeError, ValueError)):
        base.ExperimentResult('fixture', raw_data={'value': value}).save(path)
    assert not path.parent.exists()


@pytest.mark.parametrize('text', [
    '{"experiment_name":"first","experiment_name":"second"}',
    '{"experiment_name":"fixture","duration_seconds":NaN}',
    '{"experiment_name":"fixture","duration_seconds":1e400}',
])
def test_legacy_load_rejects_ambiguous_or_nonfinite_documents(tmp_path, text):
    path = tmp_path / 'result.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        base.ExperimentResult.load(path)


def test_same_second_save_result_cannot_replace_evidence(tmp_path, monkeypatch):
    fixed = base.datetime(2026, 1, 1)
    monkeypatch.setattr(base, 'datetime', SimpleNamespace(now=lambda: fixed))
    experiment = NoiseRobustnessExperiment(results_dir=tmp_path, verbose=False)
    first = base.ExperimentResult('fixture', metrics={'value': 1})
    path = experiment.save_result(first)
    with pytest.raises(FileExistsError):
        experiment.save_result(base.ExperimentResult('fixture', metrics={'value': 2}))
    assert json.loads(path.read_text())['metrics']['value'] == 1


@pytest.mark.parametrize('test', [base.ttest_vs_null([.1, .2, .3], .0), base.paired_ttest([.1, .4, .3], [.0, .1, .2])])
def test_finite_statistics_keep_boolean_types_on_disk(tmp_path, test):
    path = tmp_path / 'stats.json'
    base.ExperimentResult('fixture', metrics={'test': test}).save(path)
    assert type(base.ExperimentResult.load(path).metrics['test']['significant']) is bool


def test_noise_study_saves_explicit_degenerate_statistics(tmp_path, monkeypatch):
    from research.experiments.stability import test_noise_robustness as study
    monkeypatch.setattr(study, 'run_stimulus_recovery_trial', lambda *a, **kw: 1.)
    monkeypatch.setattr(study, 'run_autonomous_recovery_trial', lambda *a, **kw: 1.)
    monkeypatch.setattr(study, 'run_association_recovery_trial', lambda *a, **kw: {'b_recovery': 1., 'a_intact': 1.})
    result = study.NoiseRobustnessExperiment(results_dir=tmp_path, verbose=False).run(n=60, k=6, n_seeds=3)
    path = tmp_path / 'noise.json'
    result.save(path)
    saved = base.ExperimentResult.load(path)
    for arm in ('h1_stimulus_recovery', 'h2_autonomous_recovery', 'h3_association_recovery'):
        for cell in saved.metrics[arm]:
            test = cell['test_vs_chance']
            assert test['degenerate'] == 'zero_variance' and test['significant'] is False
            assert test['t'] is test['p'] is test['d'] is None
    assert saved.raw_data['cells'][0]['values'] == [1., 1., 1.]


@pytest.mark.parametrize('status', ['False', 'True', 0, 1, None, np.bool_(False)])
def test_execution_status_cannot_be_a_truthy_nonboolean(status):
    with pytest.raises(ValueError, match='boolean execution status'):
        base.ExperimentResult('fixture', success=status)


def test_invalid_loaded_execution_status_is_not_coerced(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text('{"experiment_name":"fixture","success":"False"}')
    with pytest.raises(ValueError, match='boolean execution status'):
        base.ExperimentResult.load(path)


def test_mutated_execution_status_fails_before_output_creation(tmp_path):
    result = base.ExperimentResult('fixture')
    with pytest.raises(ValueError, match='boolean execution status'):
        result.success = 'False'
    assert result.success is True
    # Serialization also validates state supplied through reflection/deserialization.
    result.__dict__['success'] = 'False'
    path = tmp_path / 'absent' / 'result.json'
    with pytest.raises(ValueError, match='boolean execution status'):
        result.save(path)
    assert not path.parent.exists()


def test_failed_execution_remains_a_boolean_failure_after_roundtrip(tmp_path):
    result = base.ExperimentResult('fixture', error_message='constructed failure')
    result.success = False
    path = tmp_path / 'result.json'
    result.save(path)
    assert base.ExperimentResult.load(path).success is False


@pytest.mark.parametrize('document', [{"experiment_name": "fixture"}, [], None])
def test_missing_execution_status_cannot_default_to_success(tmp_path, document):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='explicitly contain'):
        base.ExperimentResult.load(path)


@pytest.mark.parametrize('left,right,reason', [([1.,1.,1.],[0.,0.,0.],'zero_variance'), ([1.,1.,1.],[1.,1.,1.],'at_null')])
def test_paired_report_keeps_constant_effect_but_not_fake_test(left, right, reason):
    report = base.summarize_paired(left, right, seed_ids=[9,2,7])
    assert report['values'] == [a-b for a,b in zip(left,right)]
    assert report['summary']['mean'] == left[0]-right[0]
    assert report['test']['degenerate'] == reason
    assert report['test']['t'] is report['test']['p'] is report['test']['d'] is None
    assert report['test']['significant'] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('left,right,seeds', [
    ([1,2,3],[1,2],[1,2,3]), ([1,2],[1,2],[1,2]),
    ([1,2,3],[1,2,3],[1,1,2]), ([1,2,3],[1,float('nan'),3],[1,2,3]),
])
def test_invalid_pairing_is_not_truncated_or_dropped(left, right, seeds):
    with pytest.raises(ValueError):
        base.summarize_paired(left,right,seed_ids=seeds)


def test_nonconstant_paired_report_matches_independent_scipy_test():
    from scipy.stats import ttest_rel
    left, right = [.25,.75,1.], [.125,.25,.5]
    report = base.summarize_paired(left,right,seed_ids=[9,2,7])
    expected = ttest_rel(left,right)
    assert report['test']['t'] == pytest.approx(expected.statistic)
    assert report['test']['p'] == pytest.approx(expected.pvalue)
