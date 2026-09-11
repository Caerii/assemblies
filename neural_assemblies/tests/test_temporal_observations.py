"""The corrected measurement cannot turn agreement-word signal into distractor carry."""
from copy import deepcopy

import pytest
from research.experiments.temporal_observations import chain_observation_manifest, chain_arc_contrasts

SENTENCES = [
    ['does','dog','sees','cat','it','bird','doesnt'],
    ['is','dogs','likes','cats','itself','birds','isnt'],
    ['do','dog','see','cat','they','bird','dont'],
    ['are','dogs','like','cats','themselves','birds','arent'],
]


def frames(carry=False):
    result=[]
    for expected in chain_observation_manifest(SENTENCES,gap=1).values():
        frame={key:value for key,value in expected.items() if key!='role'}
        group = 3 if expected['subject_number']=='pl' and (carry or expected['role']!='NOUN') else 0
        frame['neurons']=list(range(group,group+3))
        result.append(frame)
    return result


def measure(observations):
    return chain_arc_contrasts(SENTENCES,observations,gap=1,n=6,k=3)


def test_agreement_signal_is_separate_from_absent_distractor_signal():
    report=measure(frames())
    assert report['frame_count']==24 and report['sentence_count']==4
    assert [row['contrast'] for row in report['positions']]==[1.,0.,1.,0.,1.,0.]
    assert [row['position'] for row in report['positions'] if row['is_distractor']]==[1,3,5]
    assert all(row['same_pair_count']==2 and row['different_pair_count']==4 for row in report['positions'])


def test_real_distractor_signal_moves_each_position_and_frame_order_is_irrelevant():
    observed=frames(carry=True)
    expected=measure(observed)
    assert all(row['contrast']==1. for row in expected['positions'])
    assert measure(list(reversed(observed)))==expected


@pytest.mark.parametrize('change',['missing','duplicate','sentence','position','token','subject','neurons','range','bool','extra'])
def test_corrupt_or_incomplete_frames_fail_instead_of_silent_pooling(change):
    observed=frames()
    if change=='missing':
        observed.pop()
    elif change=='duplicate':
        observed.append(deepcopy(observed[0]))
    elif change=='sentence':
        observed[0]['sentence_id']=99
    elif change=='position':
        observed[0]['position']=True
    elif change=='token':
        observed[0]['token']='do'
    elif change=='subject':
        observed[0]['subject_number']='pl'
    elif change=='neurons':
        observed[0]['neurons']=[0,0,1]
    elif change=='range':
        observed[0]['neurons']=[0,1,6]
    elif change=='bool':
        observed[0]['neurons']=[False,1,2]
    else:
        observed[0]['is_distractor']=True
    with pytest.raises(ValueError):
        measure(observed)


@pytest.mark.parametrize('change',['gap','role','agreement','unknown','empty'])
def test_invalid_corpus_cannot_define_a_mislabelled_manifest(change):
    corpus=deepcopy(SENTENCES)
    gap=1
    if change=='gap': gap=2
    elif change=='role': corpus[0][1]='sees'
    elif change=='agreement': corpus[0][2]='see'
    elif change=='unknown': corpus[0][1]='unknown'
    else: corpus=[]
    with pytest.raises(ValueError):
        chain_observation_manifest(corpus,gap=gap)


def test_single_subject_population_has_no_contrast_and_is_not_dropped():
    corpus=SENTENCES[:2]
    observed=[row for row in frames() if row['sentence_id']<2]
    with pytest.raises(ValueError,match='same- and different-subject'):
        chain_arc_contrasts(corpus,observed,gap=1,n=6,k=3)


@pytest.mark.parametrize('gap',[1,2,3,6])
def test_manifest_matches_real_generated_corpus_without_global_mode_changes(gap,monkeypatch):
    from research.experiments.study4 import ntp_agree
    monkeypatch.setattr(ntp_agree,'GAP',gap)
    corpus=ntp_agree.generate_chain(12,123)
    manifest=chain_observation_manifest(corpus,gap=gap)
    assert len(manifest)==12*(3+3*gap)
    assert sum(row['role']=='NOUN' for row in manifest.values())==12*3*gap
    observed=[]
    for row in manifest.values():
        frame={key:value for key,value in row.items() if key!='role'}
        frame['neurons']=[0,1,2]
        observed.append(frame)
    report=chain_arc_contrasts(corpus,observed,gap=gap,n=6,k=3)
    assert all(row['contrast']==0 for row in report['positions'])
    assert sum(row['is_distractor'] for row in report['positions'])==3*gap
