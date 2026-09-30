"""Replay historical merge dynamics and falsify misleading metric/reset interpretations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from research.experiments.primitives import test_merge as study

CASES = [json.loads(line) for line in (Path(__file__).parent / 'data/historical_merge_trials.jsonl').read_text().splitlines()]


def trace_brains(monkeypatch):
    brains=[]
    class TracedBrain(study.Brain):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.trace=[]
            self.a_to_c=[]
            brains.append(self)
        def project(self,*args,**kwargs):
            value=super().project(*args,**kwargs)
            self.trace.append(dict(args=list(args),kwargs=kwargs,winners={name:area.winners.tolist() for name,area in self.areas.items()}))
            conn=self.connectomes.get('A',{}).get('C')
            self.a_to_c.append(None if conn is None else conn.weights.copy())
            return value
    monkeypatch.setattr(study,'Brain',TracedBrain)
    return brains


@pytest.mark.parametrize('expected',CASES,ids=lambda row:f"{row['trial']}-{row['seed']}")
def test_distinct_trial_schedules_and_weights_are_preserved(monkeypatch,expected):
    brains=trace_brains(monkeypatch)
    cfg=study.MergeConfig(60,6,.2,.1,20.,3,3)
    result=(study.run_merge_trial if expected['trial']=='composition' else study.run_recovery_trial)(cfg,expected['seed'])
    brain=brains[0]
    owner=brain._engine_for(brain.areas['A'])
    weights={}
    for kind in ('_area_conns','_stim_conns'):
        for source,targets in getattr(owner,kind).items():
            for target,conn in targets.items():
                weights[f'{kind}/{source}/{target}']=hashlib.sha256(np.ascontiguousarray(conn.weights).tobytes()).hexdigest()
    assert brain.trace==expected['trace']
    assert weights==expected['weights']
    assert result==expected['result']
    assert (expected['engine'],expected['owner'])==('numpy_sparse','numpy_explicit')
    assert brain._engine.name==owner.name=='numpy_explicit'
    assert owner is brain._engine


def test_perfect_maximum_can_retain_only_one_disjoint_parent():
    a,b=np.arange(6),np.arange(6,12)
    report=study._merge_overlaps(a,a,b)
    assert report['composition_score']==1.
    assert report['merge_quality']==.5
    assert report['overlap_cab_cb']==0.
    balanced=study._merge_overlaps(np.array([0,1,2,6,7,8]),a,b)
    assert balanced['merge_quality']==report['merge_quality']
    assert balanced['composition_score']<report['composition_score']
    assert balanced['overlap_cab_ca']==balanced['overlap_cab_cb']==.5


@pytest.mark.parametrize('trial,offset',[(study.run_merge_trial,88889),(study.run_recovery_trial,77778)])
def test_disjoint_winner_replacements_do_not_change_driven_readout(monkeypatch,trial,offset):
    brains=trace_brains(monkeypatch)
    original=np.random.default_rng
    results=[]
    for replacement in (np.arange(6),np.arange(54,60)):
        class Replacement:
            def choice(self,*args,replacement=replacement,**kwargs):
                return replacement.copy()
        monkeypatch.setattr(np.random,'default_rng',lambda seed=None: Replacement() if seed==offset else original(seed))
        results.append(trial(study.MergeConfig(60,6,.2,.1,20.,3,3),1))
    assert results[0]==results[1]
    assert brains[0].trace==brains[1].trace


def test_composition_keeps_prior_weights_and_joint_phase_continues_learning(monkeypatch):
    brains=trace_brains(monkeypatch)
    study.run_merge_trial(study.MergeConfig(60,6,.2,.1,20.,3,3),1)
    weights=brains[0].a_to_c
    assert np.max(weights[8]) > 1.  # Potentiated, not merely unchanged initial edges.
    assert np.array_equal(weights[8],weights[11])  # A->C survives B-only training.
    assert not np.array_equal(weights[11],weights[14])  # Joint training changes it.


def test_recovery_evaluation_changes_learned_weights(monkeypatch):
    brains=trace_brains(monkeypatch)
    study.run_recovery_trial(study.MergeConfig(60,6,.2,.1,20.,3,3),1)
    weights=brains[0].a_to_c
    assert not np.array_equal(weights[8],weights[28])


def test_configured_harness_preserves_both_parent_vectors_and_seed_order(monkeypatch,tmp_path):
    calls=[]
    def merge(cfg,seed):
        calls.append(('merge',cfg,seed))
        return {'merge_quality':.5, 'composition_score':1., 'overlap_cab_ca':1.,
                'overlap_cab_cb':0., 'overlap_ca_cb':seed/10}
    def recovery(cfg,seed):
        calls.append(('recovery',cfg,seed))
        return {'recovery_from_A':seed/10,'recovery_from_B':0.}
    monkeypatch.setattr(study,'run_merge_trial',merge)
    monkeypatch.setattr(study,'run_recovery_trial',recovery)
    result=study.MergeExperiment(seed=100,verbose=False,results_dir=tmp_path).run(
        n=60,k=6,p=.2,seed_ids=[3,1,2],establish_rounds=2,merge_rounds=4,
        test_rounds=5,round_values=[1,2],h4_sizes=[80])
    assert [seed for _,_,seed in calls]==[3,1,2]*5
    assert all(cfg.establish_rounds==2 and cfg.test_rounds==5 for _,cfg,_ in calls)
    assert {cfg.merge_rounds for _,cfg,_ in calls}=={1,2,4}
    assert {(cfg.n,cfg.k) for _,cfg,_ in calls}=={(60,6),(80,8)}
    assert result.raw_data['seed_ids']==[3,1,2]
    assert len(result.raw_data['cells'])==5
    values=result.raw_data['cells'][0]['values']
    assert values['mean_parent_overlap']==[.5]*3
    assert values['max_parent_overlap']==[1.]*3
    assert values['overlap_cab_cb']==[0.]*3
    assert 'composition_score' not in values and 'merge_quality' not in values
    test=result.metrics['parent_overlaps']['tests_vs_chance']['mean_parent_overlap']
    assert test['p'] is None and test['degenerate']=='zero_variance'
    json.dumps(result.to_dict(),allow_nan=False)


@pytest.mark.parametrize('kwargs',[
    {'n_seeds':2},{'seed_ids':[1,1,2]}, {'seed_ids':[1,2,-1]},
    {'round_values':[]},{'round_values':[1,1]},{'round_values':[0]},
    {'h4_sizes':[]},{'h4_sizes':[60,60]},{'h4_sizes':[1.5]},
    {'test_rounds':0},{'merge_rounds':0},{'establish_rounds':False},
    {'p':float('nan')},{'beta':-1},{'w_max':float('inf')},
])
def test_invalid_merge_configuration_stops_before_timer(kwargs):
    producer=study.MergeExperiment.__new__(study.MergeExperiment)
    producer.seed=42
    producer._start_timer=lambda:pytest.fail('invalid configuration reached timer')
    with pytest.raises(ValueError):
        producer.run(**kwargs)


def test_recovery_round_configuration_reaches_each_readout(monkeypatch):
    brains=trace_brains(monkeypatch)
    study.run_recovery_trial(study.MergeConfig(60,6,.2,.1,20.,3,3,test_rounds=2),1)
    calls=brains[0].trace
    assert len(calls)==13
    assert all(row['args']==[{'sa':['A']},{'A':['C']}] for row in calls[9:11])
    assert all(row['args']==[{'sb':['B']},{'B':['C']}] for row in calls[11:13])


def test_legacy_merge_cli_requires_tag_and_forwards_to_shared_adapter(monkeypatch):
    from research.experiments import historical_merge as adapter
    calls=[]
    monkeypatch.setattr(adapter,'run_experiment',lambda **kwargs:calls.append(kwargs))
    with pytest.raises(SystemExit):
        study.main(['--quick'])
    assert calls==[]
    study.main(['--quick','--tag','fixture','--seeds','9','2','7'])
    assert calls[0]['parameters']==adapter.parameters(True)
    assert calls[0]['seeds']==[9,2,7] and calls[0]['smoke'] is True
    assert calls[0]['engine']=='numpy_explicit'
