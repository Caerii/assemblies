import numpy as np
import pytest
from research.multimodal.continual_twenty import Protocol, run
from research.multimodal.analyze_twenty import analyze, paired_ci, _per_seed


def test_paired_ci_is_seed_level_and_deterministic():
    vals=np.array([-.1,.2,.3,.6])
    a=paired_ci(vals,draws=500,seed=11)
    assert a == paired_ci(vals,draws=500,seed=11)
    assert a['paired_mean_delta'] == pytest.approx(.25)
    assert len(a['percentile_95_ci']) == 2


def test_learned_recent_vs_global_has_positive_post_change_delta():
    result=run(Protocol(seeds=tuple(range(12))))
    r=analyze(result,draws=500)
    assert r['paired_recent_minus_global']['paired_mean_delta'] > 0
    assert r['paired_recent_minus_hebbian']['paired_mean_delta'] > 0
    assert r['per_model_post_change_mean_exact']['hebbian_pairs'] == 0
    assert r['paired_recent_minus_global']['seeds'] == 12


def test_refuse_per_round_pseudoreplication_and_unfinished_study():
    r=run(Protocol(seeds=(1,2)))
    del r['trials'][0]['rows'][-1]
    with pytest.raises(ValueError,match='incomplete'):
        analyze(r)
    with pytest.raises(ValueError):
        analyze(run(Protocol(seeds=(1,))))
    with pytest.raises(ValueError):
        _per_seed(run(Protocol(seeds=(1,2))), 'recent_rule', 0,9)


@pytest.mark.parametrize('values,draws', [([],200),([1],200),([float('nan'),1],200),([1,2],10)])
def test_refuse_invalid_bootstrap(values,draws):
    with pytest.raises(ValueError):paired_ci(values,draws=draws)
