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


@pytest.mark.parametrize('name,size', [('', 2), ('B', -1), ('B', 1.5), ('B', True), ('B', '2'), ('A', 2)])
def test_invalid_stimulus_registration_preserves_state(surface, name, size):
    brain, caller = surface
    before = rng_states(brain)
    with pytest.raises(ValueError):
        caller.add_stimulus(name, size)
    assert not brain.stimuli and not brain._engine._stimuli
    assert rng_states(brain) == before


def test_duplicate_stimulus_preserves_source_learning_rate(surface):
    brain, caller = surface
    caller.add_stimulus('s', 2)
    state = brain._engine._stimuli['s']
    brain._engine.set_beta('A', 's', .37)
    before = rng_states(brain)
    with pytest.raises(ValueError, match='already registered'):
        caller.add_stimulus('s', 3)
    assert brain._engine._stimuli['s'] is state
    assert brain._engine.get_beta('A', 's') == .37
    assert rng_states(brain) == before


def test_stimulus_name_cannot_be_reused_for_an_area(surface):
    brain, caller = surface
    caller.add_stimulus('s', 2)
    with pytest.raises(ValueError):
        caller.add_area('s', 20, 2, .1)
    assert 's' not in brain.areas and 's' not in brain._engine._areas


def test_empty_stimulus_is_valid_and_numpy_size_is_canonical(surface):
    brain, caller = surface
    caller.add_stimulus('empty', np.int64(0))
    assert type(brain._engine._stimuli['empty'].size) is int
    assert brain._engine._stimuli['empty'].size == 0


@pytest.mark.parametrize('option', ['custom_inner_p', 'custom_out_p', 'custom_in_p'])
@pytest.mark.parametrize('probability', [0.0, 0.9])
def test_explicit_probability_override_cannot_be_silently_ignored(option, probability):
    brain = Brain(p=.1, seed=31, norm_init=False)
    brain.add_area('A', 20, 2)
    before = rng_states(brain)
    with pytest.raises(NotImplementedError, match=option):
        brain.add_explicit_area('B', 20, 2, **{option: probability})
    assert list(brain.areas) == list(brain._engine._areas) == ['A']
    assert brain._explicit_engine is None
    assert 'B' not in brain.connectomes
    assert rng_states(brain) == before


def test_explicit_area_default_probability_uses_brain_configuration():
    brain = Brain(p=1.0, seed=31, norm_init=False)
    brain.add_explicit_area('B', 4, 2)
    owner = brain._engine_for(brain.areas['B'])
    assert owner.p == brain.p
    assert np.all(owner._area_conns['B']['B'].weights == 1)



def test_runtime_policy_reaches_registered_engine(surface):
    from neural_assemblies import ThresholdPolicy
    brain, caller = surface
    policy = ThresholdPolicy(k=2, threshold=5)
    caller.set_competition_policy('A', policy)
    assert brain._engine._areas['A'].winner_policy is policy


@pytest.mark.parametrize('explicit', [False, True])
def test_runtime_policy_changes_dense_selection_and_can_be_reset(explicit):
    from neural_assemblies import ThresholdPolicy
    from neural_assemblies.diagnostics import read_assembly
    brain = Brain(p=.1, norm_init=False,
                  engine='numpy_sparse' if explicit else 'numpy_explicit')
    brain.add_area('A', 4, 2, explicit=explicit)
    drive = {'A': np.array([9, 4, 3, 1], dtype=np.float32)}
    brain.set_competition_policy('A', ThresholdPolicy(k=2, threshold=5))
    brain.project({}, {}, external_drive=drive)
    assert list(read_assembly(brain, 'A')) == [0]
    brain.set_competition_policy('A', None)
    brain.project({}, {}, external_drive=drive)
    assert set(read_assembly(brain, 'A')) == {0, 1}


def test_competition_policy_rejects_unknown_type_before_registration_or_mutation():
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse')
    with pytest.raises(TypeError, match='competition policy'):
        brain.add_area('bad', 4, 2, winner_policy=object())
    assert 'bad' not in brain.areas
    brain.add_area('A', 4, 2)
    with pytest.raises(TypeError, match='competition policy'):
        brain.set_competition_policy('A', object())
    assert brain.areas['A'].winner_policy is None


def test_competition_policy_cap_cannot_exceed_population():
    from neural_assemblies import TopKPolicy
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse')
    with pytest.raises(ValueError, match='population'):
        brain.add_area('bad', 4, 2, winner_policy=TopKPolicy(k=5))


@pytest.mark.parametrize('beta', [-1.0, float('nan'), float('inf'), True, '0.1'])
def test_area_beta_rejects_nonfinite_negative_and_boolean(beta):
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse')
    with pytest.raises(ValueError):
        brain.add_area('bad', 4, 2, beta=beta)
    assert 'bad' not in brain.areas


@pytest.mark.parametrize('engine_name', ['numpy_sparse', 'numpy_exact', 'numpy_explicit'])
def test_direct_numpy_engines_share_beta_registration_validation(engine_name):
    from neural_assemblies.core.numpy_engine import (
        NumpyExactEngine, NumpyExplicitEngine, NumpySparseEngine,
    )
    engine = {
        'numpy_sparse': NumpySparseEngine,
        'numpy_exact': NumpyExactEngine,
        'numpy_explicit': NumpyExplicitEngine,
    }[engine_name](p=.1)
    with pytest.raises(ValueError):
        engine.add_area('bad', 4, 2, beta=float('nan'))
    assert not engine._areas


@pytest.mark.parametrize('path', ['primary', 'auxiliary', 'direct'])
def test_runtime_policy_cannot_bypass_slot_contract(path):
    from neural_assemblies import ThresholdPolicy
    brain = Brain(p=.1, norm_init=False,
                  engine='numpy_sparse' if path == 'auxiliary' else 'numpy_explicit')
    brain.add_area('A', 4, 2, explicit=path == 'auxiliary', slot_count=2)
    owner = brain._engine_for(brain.areas['A'])
    caller = owner if path == 'direct' else brain
    with pytest.raises(NotImplementedError):
        caller.set_competition_policy('A', ThresholdPolicy(k=2, threshold=5))
    assert brain.areas['A'].winner_policy is None
    assert owner._areas['A'].winner_policy is None



@pytest.mark.parametrize('std', [-1, float('nan'), float('inf'), True, '0.5'])
def test_invalid_runtime_noise_preserves_descriptor_and_owner(surface, std):
    brain, caller = surface
    owner = brain._engine
    before = rng_states(brain)
    with pytest.raises(ValueError):
        caller.set_input_noise('A', std)
    assert brain.areas['A'].input_noise_std == 0
    assert getattr(owner._areas['A'], 'input_noise_std', 0) == 0
    assert rng_states(brain) == before


@pytest.mark.parametrize('engine,explicit', [('numpy_exact', False),
                                            ('numpy_explicit', False), ('numpy_sparse', True)])
def test_unsupported_noise_preserves_registration_and_runtime_state(engine, explicit):
    brain = Brain(p=.1, seed=31, norm_init=False, engine=engine)
    brain.add_area('A', 20, 2, explicit=explicit)
    before = rng_states(brain)
    with pytest.raises(NotImplementedError):
        brain.set_input_noise('A', .5)
    assert brain.areas['A'].input_noise_std == 0
    with pytest.raises(NotImplementedError):
        brain.add_area('B', 20, 2, explicit=explicit, input_noise_std=.5)
    assert 'B' not in brain.areas and 'B' not in brain._engine._areas
    assert rng_states(brain) == before
    brain.set_input_noise('A', 0)


@pytest.mark.parametrize('direct', [False, True])
def test_supported_noise_can_be_enabled_and_disabled(direct):
    brain = Brain(p=.1, seed=31, norm_init=False)
    brain.add_area('A', 20, 2)
    caller = brain._engine if direct else brain
    caller.set_input_noise('A', np.float64(.5))
    assert brain._engine._areas['A'].input_noise_std == .5
    assert type(brain._engine._areas['A'].input_noise_std) is float
    caller.set_input_noise('A', 0)
    assert brain._engine._areas['A'].input_noise_std == 0



def test_runtime_noise_changes_materialized_selection():
    from neural_assemblies.diagnostics import read_assembly
    brain = Brain(p=1, seed=31, norm_init=False)
    brain.add_stimulus('s', 2)
    brain.add_area('A', 20, 2)
    brain.materialize_area('A')
    noisy = brain.clone()
    noisy.set_input_noise('A', 100)
    brain.project({'s': ['A']}, {})
    noisy.project({'s': ['A']}, {})
    baseline = read_assembly(brain, 'A')
    perturbed = read_assembly(noisy, 'A')
    assert len(baseline) == len(perturbed) == 2
    assert set(perturbed) != set(baseline)


@pytest.mark.parametrize('std', [-1, float('nan'), True])
def test_invalid_noise_registration_is_nonmutating(surface, std):
    brain, caller = surface
    before = rng_states(brain)
    with pytest.raises(ValueError):
        caller.add_area('B', 20, 2, .1, input_noise_std=std)
    assert 'B' not in brain.areas and 'B' not in brain._engine._areas
    assert rng_states(brain) == before
