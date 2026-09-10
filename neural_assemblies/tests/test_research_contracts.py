"""Construct the misleading result first, then require the API to reject it."""
import pytest

from neural_assemblies.diagnostics import ensemble, ensemble_from_values, paired_delta
from neural_assemblies.assembly_calculus.emergent.training.perf import resolve_engine
from research.experiments.base import summarize
from research.harness import Criteria, study


def test_explicit_engine_survives_environment_override(monkeypatch):
    monkeypatch.setenv('ASSEMBLIES_ENGINE', 'numpy_sparse')
    assert resolve_engine('numpy_exact') == 'numpy_exact'
    assert resolve_engine('auto') == 'numpy_sparse'


def test_duplicate_seeds_rejected_before_any_work():
    calls = []
    with pytest.raises(ValueError, match='unique'):
        ensemble(lambda s: calls.append(s), [1, 1, 1])
    with pytest.raises(ValueError, match='unique'):
        study(arms={'a': lambda s: calls.append(s), 'b': lambda s: calls.append(s)},
              seeds=[1, 1, 1])
    assert calls == []


@pytest.mark.parametrize('keys', [[1, 2], [1, 1, 2]])
def test_bad_cell_identity_is_not_an_ensemble(keys):
    with pytest.raises(ValueError, match='keys'):
        ensemble_from_values([.1, .2, .3], keys=keys)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_readings_are_not_silently_dropped(bad):
    with pytest.raises(ValueError, match='seeds'):
        ensemble_from_values([.1, .2, bad], keys=[11, 12, 13])


def test_pairing_uses_seed_identity_and_preserves_it():
    a = ensemble_from_values([.2, .4, .6], keys=[1, 2, 3])
    permuted = ensemble_from_values([.3, .1, .2], keys=[3, 1, 2])
    with pytest.raises(ValueError, match='same seed keys'):
        paired_delta(a, permuted)
    b = ensemble_from_values([.1, .2, .3], keys=[1, 2, 3])
    delta = paired_delta(a, b)
    assert delta.values == pytest.approx([.1, .2, .3])
    assert delta.keys == (1, 2, 3)


def test_absent_registered_metric_cannot_pass():
    with pytest.raises(ValueError, match='no reported metrics'):
        study(arms={'a': lambda s: {'other': .4}, 'b': lambda s: {'other': .6}},
              seeds=[1, 2, 3], criteria={'accuracy': Criteria(above=.5)})
    with pytest.raises(ValueError, match='no metrics'):
        study(arms={'a': lambda s: {}, 'b': lambda s: {}}, seeds=[1, 2, 3],
              criteria={'accuracy': Criteria(above=.5)})


def test_empty_bar_is_not_a_registered_verdict():
    with pytest.raises(ValueError, match='no evaluable condition'):
        study(arms={'a': lambda s: {'m': .4}, 'b': lambda s: {'m': .6}},
              seeds=[1, 2, 3], criteria={'m': Criteria()})


def test_exploratory_metric_does_not_hide_a_judged_failure():
    result = study(arms={'a': lambda s: {'m': .9, 'extra': 1.},
                        'b': lambda s: {'m': .1, 'extra': 1.}},
                   seeds=[1, 2, 3], criteria={'m': Criteria(above=.5)})
    assert result.verdict == 'FAIL'
    assert result.metrics['extra'].verdict == 'UNJUDGED'
    assert not result.passed


@pytest.mark.parametrize('count', [3, 12, 20, 50])
def test_legacy_summary_uses_canonical_interval(count):
    from research.experiments._substrate import mean_ci
    values = list(range(count))
    result = ensemble_from_values(values)
    legacy = summarize(values)
    assert legacy['mean'] == result.mean
    assert legacy['ci95_lo'] == result.low
    assert legacy['ci95_hi'] == result.high
    assert mean_ci(values) == (result.mean, result.ci)


def test_legacy_summary_cannot_turn_smoke_into_an_interval():
    with pytest.raises(ValueError, match='confidence interval'):
        summarize([.5])


def test_fingerprint_detects_same_size_edit_with_preserved_timestamp(tmp_path, monkeypatch):
    import os
    from neural_assemblies.assembly_calculus.emergent.evaluation import sweep

    source = tmp_path / 'source.py'
    source.write_bytes(b'x = 1\n')
    before = source.stat()
    monkeypatch.delenv('ASSEMBLIES_IGNORE_CODE_FINGERPRINT', raising=False)
    monkeypatch.setattr(sweep, 'fingerprint_source_files', lambda: (source.as_posix(),))
    monkeypatch.setattr(sweep, '_CODE_FINGERPRINT', None)
    first = sweep.training_code_fingerprint()
    source.write_bytes(b'x = 2\n')
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    # A fresh process would have an empty memoized fingerprint.
    sweep._CODE_FINGERPRINT = None
    assert sweep.training_code_fingerprint() != first


def test_study_records_digest_instead_of_file_count(monkeypatch):
    from types import SimpleNamespace
    from neural_assemblies.assembly_calculus.emergent.evaluation import sweep
    from research.harness import _provenance_snapshot

    monkeypatch.setattr(sweep, 'training_code_fingerprint', lambda: 'a' * 64)
    monkeypatch.setattr(sweep, 'get_parser_cache', lambda: SimpleNamespace(stats=lambda: {}))
    assert _provenance_snapshot()[0] == 'a' * 64
