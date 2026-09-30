"""Symbolic automata errors must stop before a neural side effect."""
from fractions import Fraction
from itertools import product
from types import SimpleNamespace

import pytest

from neural_assemblies.assembly_calculus import FSMNetwork, PFANetwork, SeedMixtureChoice, Transition, TransitionMap
from neural_assemblies.assembly_calculus import fsm as fsm_module, pfa as pfa_module


@pytest.mark.parametrize('weight', [True, False, '0.5', None, float('nan'), float('inf'), 0, -1, 1.1])
def test_object_and_tuple_weights_share_strict_validation(weight):
    for build in (lambda: Transition('q0', 'a', 'q1', weight),
                  lambda: Transition.from_value(('q0', 'a', 'q1', weight))):
        with pytest.raises(ValueError, match='probability'):
            build()


@pytest.mark.parametrize('field', range(3))
@pytest.mark.parametrize('value', ['', None, 1, []])
def test_edge_labels_are_validated_before_hashing(field, value):
    row = ['q0', 'a', 'q1']
    row[field] = value
    with pytest.raises(ValueError, match='nonempty string'):
        TransitionMap([tuple(row)])


@pytest.mark.parametrize('machine', [FSMNetwork, PFANetwork])
@pytest.mark.parametrize('override', [
    {'states': ['q0', 'q0']}, {'symbols': ['a', 'a']}, {'states': []},
    {'states': ['q0', None]}, {'symbols': 'a'}, {'initial_state': 'missing'},
    {'transitions': [('q0', 'a', 'missing')]},
    {'transitions': [('missing', 'a', 'q0')]},
    {'transitions': [('q0', 'missing', 'q0')]},
    {'transitions': [('q0', 'a', 'q0'), ('q0', 'a', 'q0')]},
])
def test_invalid_domain_stops_before_any_brain_access(machine, override):
    settings = dict(states=['q0'], symbols=['a'], transitions=[('q0', 'a', 'q0')], initial_state='q0')
    settings.update(override)
    with pytest.raises(ValueError):
        machine(object(), **settings)


def test_branching_target_cannot_escape_deterministic_subset_validation():
    with pytest.raises(ValueError, match='outside declared domain'):
        PFANetwork(object(), ['q0'], ['a'],
                   [('q0', 'a', 'q0', .5), ('q0', 'a', 'missing', .5)],
                   'q0', choice=SeedMixtureChoice(100, 10, .1))


@pytest.mark.parametrize('machine,module', [(FSMNetwork, fsm_module), (PFANetwork, pfa_module)])
def test_missing_edge_has_no_projection_side_effect(machine, module, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('missing transition must not project')
    monkeypatch.setattr(module, 'project', forbidden)
    network = object.__new__(machine)
    network.brain = object()
    network._current_state = 'q0'
    network._transition_table = {}
    network._branch_schedules = {}
    network._fsm = SimpleNamespace(_st_stim={'q0': 'stim'}, state_area='state', rounds=1)
    network._sym_stim = {'a': 'stim'}
    network.symbol_area = 'symbol'
    network.rounds = 1
    with pytest.raises(KeyError):
        network.step('a')
    assert network._current_state == 'q0'


def test_conditional_factorization_matches_independent_rational_path_masses():
    # All 81 four-target integer weight vectors, with exact rational oracle.
    for masses in product(range(1, 4), repeat=4):
        total = sum(masses)
        transitions = TransitionMap([('q', 'a', str(i), float(Fraction(mass, total)))
                                     for i, mass in enumerate(masses)])
        schedule = transitions.branch_schedule('q', 'a')
        surviving = 1.
        for i, (target, conditional) in enumerate(schedule):
            assert target == str(i)
            assert surviving * conditional == pytest.approx(float(Fraction(masses[i], total)), abs=1e-15)
            surviving *= 1. - conditional
        assert surviving == 0
        assert schedule[-1][1] == 1.


def test_small_tail_is_not_lost_by_subtracting_rounded_prefix():
    transitions = TransitionMap([('q', 'a', 'first', 1.),
                                 ('q', 'a', 'second', 1e-20), ('q', 'a', 'third', 1e-20)])
    assert 1. - transitions.targets('q', 'a')[0].probability == 0.
    assert transitions.branch_schedule('q', 'a') == (('first', 1.), ('second', .5), ('third', 1.))


@pytest.mark.parametrize('tol', [-1, True, float('nan'), float('inf'), '0.1'])
def test_invalid_mass_tolerance_cannot_disable_validation(tol):
    with pytest.raises(ValueError, match='tolerance'):
        TransitionMap([('q', 'a', 'q', .2)]).validate_probability_mass(tol)


def test_partial_tables_and_empty_alphabets_are_explicitly_allowed():
    assert TransitionMap([]).validate_domain(['q'], [], 'q').deterministic_table() == {}
    with pytest.raises(KeyError):
        TransitionMap([]).branch_schedule('q', 'missing')


def test_positive_exact_weight_cannot_silently_underflow():
    with pytest.raises(ValueError, match='underflows'):
        Transition('q', 'a', 'q', Fraction(1, 10**1000))


@pytest.mark.parametrize('machine', [FSMNetwork, PFANetwork])
def test_constructor_preserves_single_pass_domain_declarations(machine, monkeypatch):
    if machine is FSMNetwork:
        monkeypatch.setattr(machine, '_setup_areas', lambda self: None)
        monkeypatch.setattr(machine, '_train', lambda self: None)
        monkeypatch.setattr(machine, 'reset', lambda self: None)
    else:
        monkeypatch.setattr(pfa_module, 'FSMNetwork', lambda brain, states, symbols, *a, **kw:
                            SimpleNamespace(states=list(states), symbols=list(symbols)))
    network = machine(object(), iter(['q']), iter(['a']), [('q', 'a', 'q')], 'q')
    encoded = network if machine is FSMNetwork else network._fsm
    assert encoded.states == ['q'] and encoded.symbols == ['a']
