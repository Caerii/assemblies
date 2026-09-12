"""Separate the branch selector, neural transition readout and learning null."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from neural_assemblies import Brain
from neural_assemblies.assembly_calculus import SeedMixtureChoice
from neural_assemblies.programs.arc_markov import ArcMarkovNetwork, ArcMarkovProtocol
from neural_assemblies.programs.markov_coin import MarkovChainModel
from neural_assemblies.programs.nemo_fsm import NemoArcFSM

STATES = ['q0', 'q1', 'q2']
TRANSITIONS = [(f'q{i}', 'flip', f'q{(i+j+1)%3}', weight)
               for i in range(3) for j, weight in enumerate((.25, .75))]
PROTOCOL = ArcMarkovProtocol(1000, 50, .1, .3, .1, 20)
CHOICE = SeedMixtureChoice(500, 50, 3., rounds_train=5, rounds=5)


def build(seed, presentations=20, engine='numpy_sparse'):
    brain = Brain(p=.3, seed=seed, engine=engine)
    return ArcMarkovNetwork(brain, STATES, TRANSITIONS, 'q0',
                            protocol=replace(PROTOCOL, presentations=presentations), choice=CHOICE)


@pytest.fixture(params=[1, 2, 3])
def trained(request):
    return build(request.param)


def probe_branches(network):
    observed, expected = [], []
    for state in STATES:
        for bit in (0, 1):
            network._current = state
            network.coin = SimpleNamespace(flip=lambda bit=bit, **kw: bit)
            observed.append(network.sample_step(seed=11))
            expected.append(f'q{(int(state[1])+bit+1)%3}')
    return observed, expected


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_transition_teaching_moves_the_readout_against_initialized_null(seed):
    null, expected = probe_branches(build(seed, 0))
    learned, expected_again = probe_branches(build(seed))
    assert expected == expected_again
    assert null != expected, 'initialized untrained organ reproduced every transition'
    assert learned == expected


def test_real_selector_composes_with_readout_and_preserves_probe_state(trained):
    network = trained
    brain, machine = network.brain, network.transition_machine
    owner = brain._engine_for(brain.areas[machine.arc_area])
    activity = {name: area.winners.copy() for name, area in brain.areas.items()}
    bias = np.asarray(owner._areas[machine.arc_area]._cumulative_bias).copy()
    coin_flip = network.coin.flip
    choices = []
    def observed_flip(**kwargs):
        label = coin_flip(**kwargs)
        choices.append((label, kwargs['bias']))
        return label
    network.coin.flip = observed_flip
    machine._table = None  # Training lookup must not be consulted by observation.
    current = 0
    for seed in range(6):
        label = network.sample_step(seed=seed)
        bit, weight = choices[-1]
        current = (current + bit + 1) % 3
        assert weight == .25
        assert label == f'q{current}'
    for name, area in brain.areas.items():
        np.testing.assert_array_equal(area.winners, activity[name])
    np.testing.assert_array_equal(owner._areas[machine.arc_area]._cumulative_bias, bias)
    network.reset()
    assert network.current_state == 'q0'
    np.testing.assert_array_equal(owner._areas[machine.arc_area]._cumulative_bias, bias)


def test_neural_readout_disagreement_is_not_replaced_by_expected_target(trained, monkeypatch):
    trained.coin = SimpleNamespace(flip=lambda **kw: 0)
    monkeypatch.setattr(trained.transition_machine, 'run', lambda *a, **kw: ['q0'])
    # Branch zero from q0 was taught q1. The returned neural label is authoritative.
    assert trained.sample_step(seed=1) == trained.current_state == 'q0'


@pytest.mark.parametrize('overrides', [
    {'n': 0}, {'k': True}, {'beta': -1}, {'organ_p': 0}, {'organ_p': 1.1},
    {'refracted_strength': float('nan')}, {'presentations': True}, {'presentations': -1},
])
def test_protocol_rejects_invalid_controls(overrides):
    with pytest.raises(ValueError):
        replace(PROTOCOL, **overrides)


def test_protocol_record_carries_domain_and_consumed_settings(trained):
    record = trained.parameters
    assert record['organ'] == asdict(PROTOCOL)
    assert record['choice'] == asdict(CHOICE)
    assert record['transitions'] == TRANSITIONS
    assert json.loads(json.dumps(record))['feedback'] == 'decoded_state'
    record['states'].append('not-a-state')
    assert trained.parameters['states'] == STATES


@pytest.mark.parametrize('transitions', [TRANSITIONS[:-2], TRANSITIONS + [('q0','other','q0',1.)]])
def test_incomplete_or_other_symbol_rows_fail_before_brain_access(transitions):
    with pytest.raises(ValueError):
        ArcMarkovNetwork(object(), STATES, transitions, 'q0', protocol=PROTOCOL, choice=CHOICE)


def test_implicit_choice_and_invalid_protocol_stop_before_brain_access():
    with pytest.raises(ValueError, match='SeedMixtureChoice'):
        ArcMarkovNetwork(object(), STATES, TRANSITIONS, 'q0', protocol=PROTOCOL)
    with pytest.raises(ValueError, match='ArcMarkovProtocol'):
        ArcMarkovNetwork(object(), STATES, TRANSITIONS, 'q0', protocol=None)


def test_deterministic_graph_allocates_no_coin():
    brain = Brain(p=.3, seed=1, engine='numpy_sparse')
    transitions = [(f'q{i}', 'flip', f'q{(i+1)%3}') for i in range(3)]
    network = ArcMarkovNetwork(brain, STATES, transitions, 'q0', protocol=PROTOCOL)
    assert network.coin is None
    assert [network.sample_step() for _ in range(3)] == ['q1','q2','q0']


def test_trace_wrapper_uses_same_protocol():
    brain = Brain(p=.3, seed=1, engine='numpy_sparse')
    traces = [(f'q{i}', 'flip', f'q{(i+1)%3}') for i in range(3)]
    model = MarkovChainModel(brain, traces, 'q0', protocol=PROTOCOL)
    assert model.parameters['protocol_version'] == 'arc-symbol-feedback-v1'
    assert model.run(3) == ['q1','q2','q0']


def test_invalid_arc_domain_fails_before_population_allocation():
    with pytest.raises(ValueError, match='outside declared domain'):
        NemoArcFSM(object(), ['q0'], ['a'], [('q0', 'a', 'missing')])



def test_configuration_cannot_be_relabelled_after_training(trained):
    with pytest.raises(AttributeError):
        trained.protocol = replace(PROTOCOL, presentations=0)
    with pytest.raises(AttributeError):
        trained.choice = replace(CHOICE, rounds=0)


@pytest.mark.parametrize('weights', [[], [.2], [.2, True], [float('nan'), 1.], [-.1, 1.]])
def test_invalid_conditional_schedule_stops_before_coin_access(weights):
    with pytest.raises(ValueError, match='conditional weights'):
        CHOICE.select_index(object(), weights, seed=1)


@pytest.mark.parametrize('label', [2, True, None, .0])
def test_invalid_coin_labels_do_not_become_fallback_targets(label):
    with pytest.raises(ValueError, match='label zero or one'):
        CHOICE.select_index(SimpleNamespace(flip=lambda **kw: label), [.2, 1.], seed=1)


def test_step_exception_preserves_decoded_feedback(trained, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('constructed readout failure')
    monkeypatch.setattr(trained.transition_machine, 'run', fail)
    with pytest.raises(RuntimeError, match='constructed'):
        trained.sample_step(seed=1)
    assert trained.current_state == 'q0'



@pytest.mark.parametrize('seed', [1, 2, 3])
def test_torch_transition_readout_has_its_own_initialized_null(seed):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('torch_sparse arc composition requires CUDA')
    null_network = build(seed, 0, engine='torch_sparse')
    assert null_network.brain._engine._device.type == 'cuda'
    null, expected = probe_branches(null_network)
    learned, expected_again = probe_branches(build(seed, engine='torch_sparse'))
    assert expected == expected_again
    assert null != expected, 'untrained CUDA organ reproduced every transition'
    assert learned == expected
