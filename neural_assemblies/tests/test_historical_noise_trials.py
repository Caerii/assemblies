"""Replay the 5206558 implementation; these fixtures are not scientific evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from research.experiments.stability import test_noise_robustness as study

CASES = [json.loads(line) for line in (Path(__file__).parent / 'data/historical_noise_trials.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('expected', CASES, ids=lambda row: f"{row['trial']}-{row['seed']}-{row['fraction']}")
def test_historical_trial_retains_schedule_winners_weights_and_owners(monkeypatch, expected):
    brains = []
    class TracedBrain(study.Brain):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = []
            brains.append(self)
        def project(self, *args, **kwargs):
            result = super().project(*args, **kwargs)
            self.trace.append(dict(args=list(args), kwargs=kwargs,
                                   winners={name: area.winners.tolist() for name, area in self.areas.items()}))
            return result
    monkeypatch.setattr(study, 'Brain', TracedBrain)
    config = study.NoiseConfig(60, 6, .2, .1, 20., establish_rounds=3, recovery_rounds=3)
    result = getattr(study, 'run_' + expected['trial'] + '_recovery_trial')(config, expected['fraction'], expected['seed'])
    brain = brains[0]
    owner = brain._engine_for(brain.areas['A'])
    weights = {}
    for kind in ('_area_conns', '_stim_conns'):
        for source, targets in getattr(owner, kind).items():
            for target, connection in targets.items():
                weights[f'{kind}/{source}/{target}'] = hashlib.sha256(np.ascontiguousarray(connection.weights).tobytes()).hexdigest()
    assert result == expected['result']
    assert brain.trace == expected['trace']
    assert weights == expected['weights']
    assert (brain._engine.name, owner.name) == (expected['engine'], expected['owner'])


def test_study_retains_each_seed_and_every_cell(monkeypatch, tmp_path):
    def scalar(cfg, noise_frac, seed):
        return (seed % 10) / 10
    monkeypatch.setattr(study, 'run_stimulus_recovery_trial', scalar)
    monkeypatch.setattr(study, 'run_autonomous_recovery_trial', scalar)
    monkeypatch.setattr(study, 'run_association_recovery_trial',
                        lambda cfg, noise_frac, seed: {'b_recovery': scalar(cfg, noise_frac, seed), 'a_intact': 1.})
    result = study.NoiseRobustnessExperiment(results_dir=tmp_path, seed=42, verbose=False).run(n=60, k=6, n_seeds=3)
    raw = result.raw_data
    assert raw['seeds'] == [42, 43, 44]
    assert len(raw['cells']) == 43
    identities = [(c['arm'], c['n'], c['k'], c['noise_frac']) for c in raw['cells']]
    assert len(set(identities)) == len(identities)
    for cell in raw['cells']:
        expected = {'b_recovery': [.2, .3, .4], 'a_intact': [1., 1., 1.]} if cell['arm'] == 'h3' else [.2, .3, .4]
        assert cell['values'] == expected
    assert result.parameters['primary_engine'] == 'numpy_sparse'
    assert result.parameters['area_engine'] == 'numpy_explicit'
    assert result.parameters['recovery_learning'] is True
    assert result.parameters['association_reference'] == 'pre_association'


@pytest.mark.parametrize('count', [0, 1, 2, True, 3.5])
def test_too_few_seeds_fail_before_timer_or_compute(count):
    experiment = object.__new__(study.NoiseRobustnessExperiment)
    with pytest.raises(ValueError, match='at least three'):
        experiment.run(n_seeds=count)
