"""Analysis must reject fake-perfect nulls and keep the primary noise level fixed."""
from copy import deepcopy
import pytest
from research.experiments.context_noise import BARS, judge, parameters, summarize


def fixture():
    cells = []
    for cell in parameters()['cells']:
        trained = cell['id'] in ('noise-0', 'noise-1')
        value = 1. if trained else .1
        rows = [dict(seed=s, accuracy=value, target_overlap=value, joint_success=value) for s in (1, 2, 3)]
        cells.append(dict(id=cell['id'], summary=summarize(rows, [1, 2, 3])))
    return cells


def test_constructed_responsive_instrument_passes():
    assert all(judge(fixture(), BARS).values())


@pytest.mark.parametrize('cell_id', ['zero-coupling', 'context-disabled', 'noise-1000'])
def test_perfect_null_cannot_pass(cell_id):
    cells = fixture()
    perfect = deepcopy(cells[0]['summary'])
    next(cell for cell in cells if cell['id'] == cell_id)['summary'] = perfect
    assert not all(judge(cells, BARS).values())


def test_another_noise_level_cannot_rescue_primary_failure():
    cells = fixture()
    next(cell for cell in cells if cell['id'] == 'noise-1')['summary'] = deepcopy(cells[-1]['summary'])
    assert not judge(cells, BARS)['noise-1-accuracy']


def test_bounds_not_bare_means_determine_verdict():
    cells = fixture()
    cells[0]['summary']['accuracy'].update(mean=1., low=.89)
    assert not judge(cells, BARS)['noise-0-accuracy']


def test_duplicate_cell_and_brain_identities_fail():
    cells = fixture()
    with pytest.raises(ValueError, match='duplicate cell'):
        judge(cells + [cells[0]], BARS)
    rows = [dict(seed=1, accuracy=1., target_overlap=1., joint_success=1.)] * 3
    with pytest.raises(ValueError, match='unique'):
        summarize(rows, [1, 1, 1])


def test_missing_or_nonfinite_metrics_cannot_be_summarized():
    rows = [dict(seed=s, accuracy=1., target_overlap=1., joint_success=1.) for s in (1,2,3)]
    with pytest.raises(ValueError, match='identities'):
        summarize(rows, [3, 2, 1])
    rows[0]['target_overlap'] = float('nan')
    with pytest.raises(ValueError, match='finite proportions'):
        summarize(rows, [1, 2, 3])


def test_smoke_grid_cannot_be_judged_as_primary_study():
    assert [cell['noise'] for cell in parameters(True)['cells'][:2]] == [0., 1000.]
    with pytest.raises(KeyError):
        judge([cell for cell in fixture() if cell['id'] != 'noise-1'], BARS)
