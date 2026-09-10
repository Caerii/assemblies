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
