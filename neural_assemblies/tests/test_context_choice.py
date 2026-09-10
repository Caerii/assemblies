"""Context coupling and independent noise have separate constructed controls."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from neural_assemblies import Brain
from neural_assemblies.assembly_calculus import (
    AttractorConfig, ContextAttractorChoice, ContextChoiceProtocol, SeedMixtureChoice, SoftmaxContextCoin,
)

PROTOCOL = ContextChoiceProtocol(400, 100, ('left', 'right'), ((4, 0), (0, 4)),
                                 AttractorConfig(2000, 200, 3., rounds_train=10), 1., 3, 0.)


def build(seed, noise=0., engine='numpy_sparse', **changes):
    if engine == 'torch_sparse':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('torch_sparse requires CUDA')
    brain = Brain(p=.05, seed=seed, engine=engine)
    return ContextAttractorChoice(brain, protocol=replace(PROTOCOL, noise_std=noise, **changes))


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('engine', ['numpy_sparse', 'torch_sparse'])
def test_context_teaching_is_read_without_relearning_or_overwriting(seed, engine):
    model = build(seed, engine=engine)
    brain = model.brain
    connection = brain._engine._area_conns[model.context_area][model.outcome_area]
    def snapshot_weights():
        if engine == 'torch_sparse':
            assert brain._engine._device.type == 'cuda'
            return connection._val.detach().float().cpu().numpy().copy()
        return np.asarray(connection.weights).copy()
    weights = snapshot_weights()
    activity = {name: area.winners.copy() for name, area in brain.areas.items()}
    for context, target in [('left', 0), ('right', 1)]:
        observed = model.observe(context, seed=700)
        assert observed.label == target
        assert observed.overlaps[target] > .8
        assert observed.margin > .5
    np.testing.assert_array_equal(snapshot_weights(), weights)
    for name, area in brain.areas.items():
        np.testing.assert_array_equal(area.winners, activity[name])
    disabled = [model.observe(context, seed=700, context_enabled=False) for context in PROTOCOL.contexts]
    assert disabled[0] == disabled[1]
    assert disabled[0].label is None and disabled[0].overlaps == (0., 0.)


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('engine', ['numpy_sparse', 'torch_sparse'])
def test_large_noise_moves_the_overlap_measurement_and_is_not_discarded(seed, engine):
    model = build(seed, noise=1000., engine=engine)
    observed = [model.observe(context, seed=700) for context in PROTOCOL.contexts]
    assert all(max(row.overlaps) < .4 for row in observed)
    noise_only = [model.observe(context, seed=700, context_enabled=False,
                                recurrence_enabled=False) for context in PROTOCOL.contexts]
    assert noise_only[0] == noise_only[1]
    assert max(noise_only[0].overlaps) > 0, 'native noise was ignored at zero synaptic drive'


def test_seeded_context_read_repeats_after_an_intervening_read():
    model = build(1, noise=50.)
    first = model.observe('left', seed=701)
    model.observe('right', seed=702)
    assert model.observe('left', seed=701) == first


def test_legacy_entry_refuses_before_brain_access():
    with pytest.raises(NotImplementedError, match='ContextAttractorChoice'):
        SoftmaxContextCoin(object())


@pytest.mark.parametrize('changes', [
    {'n': 0}, {'k': True}, {'contexts': ('same', 'same')}, {'contexts': 'left'},
    {'contexts': ('too', 'many', 'for', 'the', 'population')},
    {'presentations': ((1, 2),)}, {'presentations': ((True, 0), (0, 1))},
    {'coupling_beta': -1}, {'coupling_beta': float('nan')}, {'read_rounds': 0},
    {'noise_std': -1}, {'attractors': SeedMixtureChoice(100, 10, .1)},
])
def test_protocol_rejects_ambiguous_or_invalid_controls(changes):
    with pytest.raises(ValueError):
        replace(PROTOCOL, **changes)


def test_record_and_fixed_configuration_are_explicit():
    model = build(1)
    record = json.loads(json.dumps(model.parameters))
    assert record['attractors'] == asdict(PROTOCOL.attractors)
    assert record['presentations'] == [[4, 0], [0, 4]]
    assert record['rng_policy'] == 'read-only-seed-v1'
    assert model.brain.areas[model.outcome_area].beta_by_area[model.context_area] == 1.
    with pytest.raises(AttributeError):
        model.protocol = replace(PROTOCOL, read_rounds=1)


def test_unknown_context_and_nonboolean_gates_fail_before_read():
    model = build(1)
    with pytest.raises(KeyError):
        model.observe('unknown')
    with pytest.raises(ValueError, match='booleans'):
        model.observe('left', context_enabled='false')



def test_unsupported_native_noise_is_rejected_before_allocation():
    brain = Brain(engine='numpy_exact', seed=1)
    with pytest.raises(NotImplementedError, match='native input noise support'):
        ContextAttractorChoice(brain, protocol=replace(PROTOCOL, noise_std=1.))
    assert not brain.areas


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('engine', ['numpy_sparse', 'torch_sparse'])
@pytest.mark.parametrize('control', [
    {'coupling_beta': 0.}, {'presentations': ((0, 0), (0, 0))},
])
def test_disabled_teaching_fails_the_trained_readout_contract(seed, engine, control):
    model = build(seed, engine=engine, **control)
    reads = [model.observe(context, seed=700) for context in PROTOCOL.contexts]
    trained_contract = all(row.label == target and row.overlaps[target] > .8
                           and row.margin > .5 for target, row in enumerate(reads))
    assert not trained_contract, f'initialized teaching null passed: {reads}'
