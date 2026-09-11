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
    assert (brain._engine.name,owner.name)==(expected['engine'],expected['owner'])


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
            def choice(self,*args,**kwargs):
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
