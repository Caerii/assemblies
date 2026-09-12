"""Historical association schedules and controls for misleading readout interpretations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from neural_assemblies import describe_brain_model
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
    assert (expected['engine'], expected['owner']) == ('numpy_sparse', 'numpy_explicit')
    assert brain._engine.name == owner.name == 'numpy_explicit'
    assert owner is brain._engine


@pytest.mark.parametrize('bidirectional', [False, True])
def test_disjoint_b_corruptions_are_not_read_and_evaluation_learns(monkeypatch, bidirectional):
    brains = trace_brains(monkeypatch)
    cfg = study.AssocConfig(60, 6, .2, .1, 20., 3, 3, 3)
    outcomes = []
    for replacement in (np.arange(6), np.arange(54, 60)):
        class Corruption:
            def choice(self, *args, replacement=replacement, **kwargs):
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


def test_recorded_model_mismatch_fails_before_topology(monkeypatch):
    wrong = describe_brain_model(
        "numpy_sparse", p=.2, seed=0, w_max=20., norm_init=False,
    )
    monkeypatch.setattr(
        study.Brain, "add_area",
        lambda *args, **kwargs: pytest.fail("semantic mismatch reached topology"),
    )
    with pytest.raises(ValueError, match="model_semantics mismatch"):
        study.run_association_trial(
            study.AssocConfig(60, 6, .2, .1, 20.), 1,
            model_semantics=wrong,
        )


def test_configured_harness_consumes_explicit_inputs_and_retains_paired_values(monkeypatch, tmp_path):
    calls = []
    def association(cfg, seed, bidirectional=True, rng=None):
        calls.append((cfg, seed, bidirectional))
        return {'recovery': seed / 8 + (.125 if bidirectional else 0.)}
    monkeypatch.setattr(study, 'run_association_trial', association)
    monkeypatch.setattr(study, 'run_identity_trial', lambda cfg, seed: {'recovery_a': seed / 10, 'recovery_b': .5})
    result = study.AssociationExperiment(seed=100, verbose=False, results_dir=tmp_path).run(
        n=60, k=6, p=.2, seed_ids=[3,1,2], establish_rounds=2, assoc_rounds=4,
        test_rounds=5, round_values=[0,2], h1e_sizes=[80])
    assert result.parameters['seed_ids'] == [3,1,2]
    assert result.raw_data['seed_ids'] == [3,1,2]
    assert result.raw_data['values']['basic'] == [.5,.25,.375]
    assert [seed for cfg,seed,mode in calls[:3]] == [3,1,2]
    assert {cfg.assoc_rounds for cfg,seed,mode in calls} == {0,2,4}
    assert {(cfg.n,cfg.k) for cfg,seed,mode in calls} == {(60,6),(80,8)}
    assert all(cfg.establish_rounds == 2 and cfg.test_rounds == 5 for cfg,seed,mode in calls)
    assert result.metrics['directionality']['paired_difference']['mean'] == pytest.approx(.125)
    assert len(result.raw_data['values']) == 9


def test_constant_nonzero_paired_difference_is_not_reported_as_p_one(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'run_association_trial', lambda cfg, seed, bidirectional=True, rng=None: {'recovery': .75 if bidirectional else .25})
    monkeypatch.setattr(study, 'run_identity_trial', lambda cfg, seed: {'recovery_a': .5, 'recovery_b': .5})
    result = study.AssociationExperiment(verbose=False, results_dir=tmp_path).run(
        n=60,k=6,seed_ids=[1,2,3],round_values=[0],h1e_sizes=[60])
    assert result.raw_data['values']['paired_difference'] == [.5,.5,.5]
    test = result.metrics['directionality']['paired_test']
    assert test['p'] is None and test['t'] is None and test['d'] is None
    assert test['degenerate'] == 'zero_variance' and test['significant'] is False
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize('kwargs', [
    {'n_seeds': 2}, {'seed_ids': [1,1,2]}, {'seed_ids': [1,2,-1]},
    {'round_values': []}, {'round_values': [1,1]}, {'round_values': [True]},
    {'h1e_sizes': []}, {'h1e_sizes': [60,60]}, {'h1e_sizes': [1.5]},
    {'test_rounds': 0}, {'establish_rounds': 0}, {'assoc_rounds': -1},
    {'p': float('nan')}, {'beta': -1}, {'w_max': float('inf')},
])
def test_invalid_configuration_fails_before_timer_or_trials(kwargs):
    producer = study.AssociationExperiment.__new__(study.AssociationExperiment)
    producer.seed = 42
    producer._start_timer = lambda: pytest.fail('invalid configuration reached timer')
    with pytest.raises(ValueError):
        producer.run(**kwargs)


def test_old_cli_requires_tag_before_producer_construction(monkeypatch):
    monkeypatch.setattr(study, 'AssociationExperiment', lambda **kwargs: pytest.fail('missing tag constructed producer'))
    with pytest.raises(SystemExit) as exc:
        study.main(['--quick'])
    assert exc.value.code == 2


def test_shared_association_adapter_preserves_parameters_and_void_status(monkeypatch):
    from research.experiments import historical_association as adapter
    calls = []
    monkeypatch.setattr(adapter, 'run_experiment', lambda **kwargs: calls.append(kwargs))
    adapter.main(['--tag','fixture','--smoke','--seeds','9','2','7'])
    sent = calls[0]
    assert sent['seeds'] == [9,2,7] and sent['parameters'] == adapter.parameters(True)
    assert sent['engine'] == 'numpy_explicit' and sent['smoke'] is True
    assert sent['protocol'] == 'memory.historical-association'
