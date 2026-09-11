"""Historical association schedules and controls for misleading readout interpretations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from research.experiments.primitives import test_association as study

CASES = [json.loads(line) for line in (Path(__file__).parent / "data/historical_association_trials.jsonl").read_text().splitlines()]


def trace_brains(monkeypatch):
    brains = []
    class TracedBrain(study.Brain):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = []
            self.evaluation_weights = []
            brains.append(self)
        def project(self, *args, **kwargs):
            result = super().project(*args, **kwargs)
            self.trace.append(dict(args=list(args), kwargs=kwargs,
                                   winners={name: area.winners.tolist() for name, area in self.areas.items()}))
            if len(self.trace) >= 9:
                self.evaluation_weights.append(self.connectomes["A"]["B"].weights.copy())
            return result
    monkeypatch.setattr(study, "Brain", TracedBrain)
    return brains


@pytest.mark.parametrize("expected", CASES, ids=lambda row: f"{row['trial']}-{row['seed']}")
def test_pre_refactor_dynamics_and_values_are_preserved(monkeypatch, expected):
    brains = trace_brains(monkeypatch)
    cfg = study.AssocConfig(60, 6, .2, .1, 20., 3, 3, 3)
    if expected['trial'] == 'identity':
        result = study.run_identity_trial(cfg, expected['seed'])
    else:
        result = study.run_association_trial(cfg, expected['seed'], expected['trial'] == 'bidirectional')
    result = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}
    brain = brains[0]
    owner = brain._engine_for(brain.areas['A'])
    weights = {}
    for kind in ('_area_conns', '_stim_conns'):
        for source, targets in getattr(owner, kind).items():
            for target, connection in targets.items():
                weights[f'{kind}/{source}/{target}'] = hashlib.sha256(np.ascontiguousarray(connection.weights).tobytes()).hexdigest()
    assert brain.trace == expected['trace']
    assert weights == expected['weights']
    assert result == expected['result']
    assert (brain._engine.name, owner.name) == (expected['engine'], expected['owner'])


@pytest.mark.parametrize('bidirectional', [False, True])
def test_disjoint_b_corruptions_are_not_read_and_evaluation_learns(monkeypatch, bidirectional):
    brains = trace_brains(monkeypatch)
    cfg = study.AssocConfig(60, 6, .2, .1, 20., 3, 3, 3)
    outcomes = []
    for replacement in (np.arange(6), np.arange(54, 60)):
        class Corruption:
            def choice(self, *args, **kwargs):
                return replacement.copy()
        outcomes.append(study.run_association_trial(cfg, 1, bidirectional, rng=Corruption())['recovery'])
    assert outcomes[0] == outcomes[1]
    assert brains[0].trace[9:] == brains[1].trace[9:]
    assert any(not np.array_equal(brains[0].evaluation_weights[0], weights)
               for weights in brains[0].evaluation_weights[1:])


@pytest.mark.parametrize('mode', ['False', 0, 1, None])
def test_invalid_directionality_fails_before_construction(monkeypatch, mode):
    monkeypatch.setattr(study, 'Brain', lambda **kwargs: pytest.fail('invalid direction constructed a brain'))
    with pytest.raises(ValueError, match='bidirectional'):
        study.run_association_trial(study.AssocConfig(60,6,.2,.1,20.), 1, mode)
