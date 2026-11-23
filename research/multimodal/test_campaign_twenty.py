import pytest
from dataclasses import replace
from research.multimodal.continual_twenty import Protocol
from research.multimodal.campaign_twenty import plan, campaign


def test_exact_twenty_unique_experiments():
    stages=plan(Protocol(seeds=(0,1)))
    assert len(stages)==20
    assert len(set(name for name, _ in stages))==20
    assert all(p.rounds==20 for _,p in stages)
    assert stages[0][0]=='00_baseline'
    assert stages[-1][0]=='19_transition_poison'
    assert stages[12][1].n==64


def test_campaign_is_replayable_and_preserves_all_twenty_stage_hashes():
    # Two-seed smoke run. Full 20-seed campaign is a separate executed artifact.
    spec=Protocol(seeds=(0,1))
    a=campaign(spec,draws=100)
    b=campaign(spec,draws=100)
    assert a==b
    assert a['stage_count']==20
    assert a['total_learning_updates']==800
    assert all(len(x['raw_sha256'])==64 for x in a['stages'])
    for stage in a['stages']:
        assert stage['round_20_current']['hebbian_pairs']==0.


@pytest.mark.parametrize('spec',[
    Protocol(rounds=19),Protocol(change_after=9),
])
def test_cannot_silently_change_campaign_definition(spec):
    with pytest.raises(ValueError):campaign(spec)
