"""Paired analysis contract and descriptive bootstrap consistency."""
import copy
import pytest
from research.multimodal.analyze_unknown_drift import _paired_interval, analyze
from research.multimodal.unknown_drift import Protocol, run


@pytest.fixture(scope='module')
def report():
    return run(Protocol(n=48, k=4, seeds=(0, 1, 2)))


def test_exact_paired_grid(report):
    result = analyze(report)
    assert len(result['paired_corruption_effects']) == 8 * 2
    assert len(result['change_profiles']) == 8 * 3 * 6
    assert all(row['cells'] == 3 * 3 for row in result['change_profiles'])


def test_deterministic_bootstrap(report):
    assert analyze(report) == analyze(report)
    zero = _paired_interval([0, 0, 0])
    assert zero['mean_difference'] == 0.
    assert zero['descriptive_bootstrap_95pct'] == [0., 0.]


def test_detect_incomplete_grid(report):
    bad = copy.deepcopy(report)
    bad['trials'].pop()
    with pytest.raises(ValueError, match='incomplete'):
        analyze(bad)


def test_detect_mismatched_hidden_data(report):
    bad = copy.deepcopy(report)
    bad['trials'][2]['probe_cues_sha256'] = 'corrupted'
    with pytest.raises(AssertionError, match='paired scenario mismatch'):
        analyze(bad)


def test_cannot_compare_different_change_points(report):
    bad = copy.deepcopy(report)
    for scenario in bad['protocol']['scenarios']:
        if scenario['name'] == 'three_shifts_burst':
            scenario['changes'] = [12, 27, 39]
    with pytest.raises(ValueError, match='change points differ'):
        analyze(bad)


def test_reject_invalid_interval():
    with pytest.raises(ValueError):
        _paired_interval([])
