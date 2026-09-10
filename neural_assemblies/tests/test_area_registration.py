"""Rejected registration must preserve existing populations and random streams."""
import copy

import numpy as np
import pytest

from neural_assemblies import Brain


@pytest.fixture(params=[(name, direct) for name in
                        ('numpy_sparse', 'numpy_exact', 'numpy_explicit') for direct in (False, True)])
def surface(request):
    name, direct = request.param
    brain = Brain(p=.1, seed=31, norm_init=False, engine=name)
    brain.add_area('A', 20, 2)
    return brain, brain._engine if direct else brain


def rng_states(brain):
    owners = [brain.rng, brain._conn_rng]
    if hasattr(brain._engine, '_rng'):
        owners.append(brain._engine._rng)
    return [copy.deepcopy(rng.bit_generator.state) for rng in owners]


@pytest.mark.parametrize('name,n,k', [('', 20, 2), ('B', 0, 1), ('B', 20, 0),
                                    ('B', 2, 3), ('B', 2.5, 1), ('B', 20, True), ('B', False, 1)])
def test_invalid_registration_preserves_population_and_rng(surface, name, n, k):
    brain, caller = surface
    original = brain._engine._areas['A']
    before = rng_states(brain)
    with pytest.raises(ValueError):
        caller.add_area(name, n, k, .1)
    assert list(brain.areas) == list(brain._engine._areas) == ['A']
    assert brain._engine._areas['A'] is original
    assert rng_states(brain) == before


def test_duplicate_area_cannot_replace_existing_population(surface):
    brain, caller = surface
    original = brain._engine._areas['A']
    descriptor = brain.areas['A']
    before = rng_states(brain)
    with pytest.raises(ValueError, match='already registered'):
        caller.add_area('A', 40, 4, .2)
    assert brain._engine._areas['A'] is original
    assert brain.areas['A'] is descriptor
    assert rng_states(brain) == before


def test_valid_numpy_dimensions_are_canonical(surface):
    brain, caller = surface
    caller.add_area('B', np.int64(20), np.int64(2), .1)
    state = brain._engine._areas['B']
    assert type(state.n) is int and type(state.k) is int
    assert (state.n, state.k) == (20, 2)



def test_dimension_limit_matches_uint32_neuron_ids_without_allocation():
    from neural_assemblies.core.registration import validate_area_registration
    assert validate_area_registration('A', 2**32, 1) == (2**32, 1)
    with pytest.raises(ValueError):
        validate_area_registration('A', 2**32 + 1, 1)


def test_standalone_area_uses_the_same_dimension_contract():
    from neural_assemblies import Area
    with pytest.raises(ValueError):
        Area('A', 2, 3, .1)
    area = Area('A', np.int64(20), np.int64(2), .1)
    assert type(area.n) is int and type(area.k) is int


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('threshold,expected', [(5, [0]), (12, [])])
def test_winner_policy_controls_primary_and_auxiliary_dense_selection(explicit, threshold, expected):
    from neural_assemblies import ThresholdPolicy
    policy = ThresholdPolicy(k=2, threshold=threshold)
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse' if explicit else 'numpy_explicit')
    brain.add_area('T', 4, 2, explicit=explicit, winner_policy=policy)
    brain.project(external_drive={'T': [9, 4, 3, 1]})
    assert brain.areas['T'].winners.tolist() == expected
    assert brain._engine_for(brain.areas['T'])._areas['T'].winner_policy == policy


def test_lazy_explicit_registration_preserves_descriptor_policy():
    from neural_assemblies import Area, ThresholdPolicy
    policy = ThresholdPolicy(k=2, threshold=5)
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse')
    # Exercise the existing lazy-registration branch for pending descriptors.
    brain.areas['T'] = Area('T', 4, 2, .1, explicit=True, winner_policy=policy)
    pending = Area('new', 4, 2, .1, explicit=True)
    owner = brain._engine_for(pending)
    result = owner.project_into('T', [], [], external_drive=[9, 4, 3, 1])
    assert result.winners.tolist() == [0]
    assert owner._areas['T'].winner_policy == policy


@pytest.mark.parametrize('surface', ['primary', 'auxiliary', 'direct'])
@pytest.mark.parametrize('fault', ['policy', 'fractional', 'uneven', 'too_many'])
def test_invalid_slot_configuration_stops_before_registration(surface, fault):
    from neural_assemblies import ThresholdPolicy
    from neural_assemblies.core.numpy_engine import NumpyExplicitEngine
    n, slots = (5, 2) if fault == 'uneven' else (4, 1.5) if fault == 'fractional' else (4, 5) if fault == 'too_many' else (4, 2)
    policy = ThresholdPolicy(k=2, threshold=5) if fault == 'policy' else None
    if surface == 'direct':
        caller = NumpyExplicitEngine(p=.1)
        kwargs = {}
    else:
        caller = Brain(p=.1, norm_init=False, engine='numpy_sparse' if surface == 'auxiliary' else 'numpy_explicit')
        kwargs = {'explicit': surface == 'auxiliary'}
    with pytest.raises((ValueError, NotImplementedError)):
        caller.add_area('T', n, 2, .1, slot_count=slots, winner_policy=policy, **kwargs)
    assert not (caller.areas if isinstance(caller, Brain) else caller._areas)
    if isinstance(caller, Brain):
        assert not caller._engine._areas and caller._explicit_engine is None


@pytest.mark.parametrize('explicit', [False, True])
def test_slot_selection_uses_best_partition_on_both_dense_paths(explicit):
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse' if explicit else 'numpy_explicit')
    brain.add_area('T', 4, 2, explicit=explicit, slot_count=2)
    brain.project(external_drive={'T': [10, 0, 6, 5]})
    # Global top-k would pick {0, 2}; the best whole slot is {2, 3}.
    assert brain.areas['T'].winners.tolist() == [2, 3]


def test_sparse_owner_cannot_silently_ignore_slots():
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse')
    with pytest.raises(NotImplementedError, match='slot'):
        brain.add_area('T', 4, 2, slot_count=2)
    assert not brain.areas



def test_standalone_slot_selection_cannot_discard_trailing_neurons():
    from neural_assemblies.compute.winner_selection import select_slot_winners
    with pytest.raises(ValueError, match='partition'):
        select_slot_winners(np.array([0, 0, 0, 0, 100]), 2, 2)
