"""Invalid inputs and misleading execution modes are visible at the boundary."""
import warnings
from pathlib import Path
import runpy

import numpy as np
import pytest

from neural_assemblies.assembly_calculus.assembly import Assembly
from neural_assemblies.assembly_calculus.ops import activate_assembly
from neural_assemblies.core.brain import Brain
from neural_assemblies.core.index_spaces import to_neuron_ids


@pytest.mark.parametrize('bad', [[-1], [1.5], [2 ** 32], [[1, 2]]])
def test_assembly_does_not_coerce_invalid_ids(bad):
    with pytest.raises(ValueError):
        Assembly('A', np.asarray(bad))


def test_mapping_cannot_drop_a_winner():
    with pytest.raises(ValueError, match='compact indices'):
        to_neuron_ids(np.array([0, 2]), [77, 88])
    assert to_neuron_ids(np.array([1, 0]), [77, 88]).tolist() == [88, 77]


def test_explicit_activation_rejects_neuron_outside_area():
    brain = Brain(engine='numpy_exact')
    brain.add_area('A', 64, 8)
    with pytest.raises(ValueError, match='assembly neuron IDs'):
        activate_assembly(brain, Assembly('A', np.array([64])))


def test_sampled_recurrence_warns_once_and_names_audit():
    brain = Brain(engine='numpy_sparse', p=.1)
    brain.add_area('A', 300, 10)
    brain.add_stimulus('s', 10)
    brain.project({'s': ['A']}, {})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for _ in range(2):
            brain.project({'s': ['A']}, {'A': ['A']})
    messages = [str(w.message) for w in captured if 'sampled numpy connectome' in str(w.message)]
    assert len(messages) == 1
    assert 'PREREG_sampler_audit.md' in messages[0]


def test_materialized_recurrence_does_not_claim_to_be_sampled():
    brain = Brain(engine='numpy_sparse', p=.1)
    brain.add_area('A', 100, 10)
    brain.add_stimulus('s', 10)
    brain.materialize_area('A')
    brain.project({'s': ['A']}, {})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        brain.project({'s': ['A']}, {'A': ['A']})
    assert not any('sampled numpy connectome' in str(w.message) for w in captured)


def test_teaching_example_has_a_working_learning_disabled_control():
    # This is a regression check on the instructional fixture, not a newly
    # preregistered scientific bar or a general claim about capacity.
    example = Path(__file__).resolve().parents[2] / 'examples' / '01_basic_assembly_calculus.py'
    recovery = runpy.run_path(str(example))['recovery']
    null = recovery(1, beta=0)
    trained = recovery(1, beta=.2)
    assert null < .3
    assert trained > .8


@pytest.mark.parametrize('engine_name', ['numpy_sparse', 'numpy_exact', 'numpy_explicit'])
@pytest.mark.parametrize('field,value', [('p', .2), ('seed', 42), ('w_max', 9)])
def test_supplied_engine_identity_conflict_stops_before_adoption(engine_name, field, value, monkeypatch):
    from neural_assemblies.core.engine import create_engine
    engine = create_engine(engine_name, p=.1, seed=41, w_max=8)
    requested = dict(p=.1, seed=41, w_max=8, norm_init=False, engine=engine)
    requested[field] = value
    calls = []
    monkeypatch.setattr(engine, "set_projection_fidelity", lambda value: calls.append(value), raising=False)
    with pytest.raises(ValueError, match=field):
        Brain(**requested)
    assert calls == []
    assert not engine._areas


@pytest.mark.parametrize('engine_name', ['numpy_sparse', 'numpy_exact', 'numpy_explicit'])
@pytest.mark.parametrize("clip", [8, None])
def test_supplied_matching_engine_uses_the_same_auxiliary_identity(engine_name, clip):
    from neural_assemblies.core.engine import create_engine
    engine = create_engine(engine_name, p=.1, seed=41, w_max=clip)
    brain = Brain(p=.1, seed=41, w_max=clip, norm_init=False, engine=engine)
    assert brain._engine is engine
    brain.add_area('A', 20, 2, explicit=True)
    owner = brain._engine_for(brain.areas['A'])
    assert owner.p == brain.p == .1
    assert owner.w_max == brain.w_max == clip
    assert owner.seed == brain._seed == 41



def test_supplied_engine_with_unavailable_seed_cannot_claim_identity(monkeypatch):
    from neural_assemblies.core.numpy_engine import NumpyExactEngine
    engine = NumpyExactEngine(p=.1, seed=41, w_max=8)
    monkeypatch.delattr(engine, 'seed')
    with pytest.raises(ValueError, match='cannot validate Brain seed'):
        Brain(p=.1, seed=41, w_max=8, norm_init=False, engine=engine)


@pytest.mark.parametrize('settings', [
    {'synaptic_scaling': 'AREA'}, {'synaptic_scaling': {'AREA': True}},
    {'synaptic_scaling': [1]}, {'synaptic_scaling': ['']},
    {'norm_init': 'false'}, {'synaptic_scaling_deferred': 'false'},
    {'synaptic_scaling_deferred': True},
])
@pytest.mark.parametrize('surface', ['brain', 'engine'])
def test_invalid_homeostasis_stops_at_construction(settings, surface):
    from neural_assemblies.core.numpy_engine import NumpySparseEngine
    constructor = Brain if surface == 'brain' else NumpySparseEngine
    with pytest.raises(ValueError):
        constructor(p=.1, **settings)


def test_scaling_scope_is_detached_from_callers_collection():
    scope = {'AREA'}
    brain = Brain(p=.1, engine='numpy_sparse', synaptic_scaling=scope)
    scope.clear()
    assert brain._synaptic_scaling == brain._engine.synaptic_scaling == frozenset({'AREA'})


@pytest.mark.parametrize('engine_settings,brain_settings', [
    ({'norm_init': True}, {'norm_init': False}),
    ({'norm_init': False}, {'norm_init': True}),
    ({'synaptic_scaling': True}, {'synaptic_scaling': False}),
    ({'synaptic_scaling': {'A'}}, {'synaptic_scaling': {'B'}}),
    ({'synaptic_scaling': True, 'synaptic_scaling_deferred': True},
     {'synaptic_scaling': True, 'synaptic_scaling_deferred': False}),
])
def test_supplied_engine_rejects_conflicting_homeostasis(engine_settings, brain_settings):
    from neural_assemblies.core.numpy_engine import NumpySparseEngine
    identity = dict(p=.1, seed=41, w_max=8)
    engine = NumpySparseEngine(**identity, **engine_settings)
    request = dict(norm_init=False, **identity)
    request.update(brain_settings)
    with pytest.raises(ValueError, match='homeostasis'):
        Brain(engine=engine, **request)


def test_supplied_engine_accepts_equivalent_scope_spellings():
    from neural_assemblies.core.numpy_engine import NumpySparseEngine
    identity = dict(p=.1, seed=41, w_max=8)
    engine = NumpySparseEngine(**identity, synaptic_scaling={'A', 'B'})
    brain = Brain(engine=engine, **identity, norm_init=False, synaptic_scaling=['B', 'A', 'A'])
    assert brain._synaptic_scaling == engine.synaptic_scaling == frozenset({'A', 'B'})



def test_exact_engine_normalization_uses_the_shared_boolean_contract():
    from neural_assemblies.core.numpy_engine import NumpyExactEngine
    with pytest.raises(ValueError, match='booleans'):
        NumpyExactEngine(p=.1, norm_init='false')


def test_homeostasis_config_is_immutable_and_composes_with_constructors():
    from dataclasses import FrozenInstanceError
    from neural_assemblies import HomeostasisConfig
    from neural_assemblies.core.numpy_engine import NumpySparseEngine
    scope = ['A']
    config = HomeostasisConfig(norm_init=True, synaptic_scaling=scope, synaptic_scaling_deferred=True)
    scope.append('B')
    with pytest.raises(FrozenInstanceError):
        config.norm_init = False
    identity = dict(p=.1, seed=41, w_max=8)
    engine = NumpySparseEngine(**identity, **config.as_kwargs())
    brain = Brain(engine=engine, **identity, **config.as_kwargs())
    assert HomeostasisConfig.from_engine(engine) == config
    assert brain._synaptic_scaling == frozenset({'A'})


def test_empty_scaling_scope_canonicalizes_to_disabled():
    from neural_assemblies import HomeostasisConfig
    assert HomeostasisConfig(synaptic_scaling=[]) == HomeostasisConfig(synaptic_scaling=False)


@pytest.mark.parametrize('scope', [True, {'A'}])
def test_direct_engine_refraction_cannot_bypass_scaling_conflict(scope):
    from neural_assemblies.core.numpy_engine import NumpySparseEngine
    from neural_assemblies.core._homeostasis import HomeostasisConflict
    engine = NumpySparseEngine(p=.1, synaptic_scaling=scope)
    engine.add_area('A', 20, 2, .1)
    with pytest.raises(HomeostasisConflict):
        engine.set_refracted('A', True, .1)
    assert not engine._areas['A'].refracted
    assert engine._areas['A'].refracted_strength == 0


@pytest.mark.parametrize('surface', ['brain', 'engine'])
def test_refracted_target_rejects_explicit_normalization(surface):
    from neural_assemblies.core._homeostasis import HomeostasisConflict
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse')
    brain.add_area('A', 20, 2, .1)
    brain.materialize_area('A')
    weights = np.full((20, 20), 2, dtype=np.float32)
    brain.connectomes['A']['A'].weights = weights
    brain.set_refracted('A', True, .1)
    before = weights.copy()
    caller = brain if surface == 'brain' else brain._engine
    with pytest.raises(HomeostasisConflict):
        caller.normalize_weights('A')
    np.testing.assert_array_equal(brain.connectomes['A']['A'].weights, before)


@pytest.mark.parametrize('primary', ['numpy_explicit', 'numpy_sparse'])
def test_unsupported_refraction_does_not_change_brain_descriptor(primary):
    brain = Brain(p=.1, norm_init=False, engine=primary)
    brain.add_area('A', 20, 2, .1, explicit=True)
    with pytest.raises(NotImplementedError):
        brain.set_refracted('A', True, .1)
    assert not brain.areas['A'].refracted
    assert brain.areas['A'].refracted_strength == 0


def test_unscaled_refraction_and_nonrefracted_normalization_remain_available():
    brain = Brain(p=.1, norm_init=False, engine='numpy_sparse', synaptic_scaling={'B'})
    brain.add_area('A', 20, 2, .1)
    brain.add_area('B', 20, 2, .1)
    brain.materialize_area('A')
    brain.set_refracted('A', True, .1)
    assert brain._engine._areas['A'].refracted
    brain.set_refracted('A', False)
    weights = np.full((20, 20), 2, dtype=np.float32)
    brain.connectomes['A']['A'].weights = weights
    brain.normalize_weights('A')
    np.testing.assert_allclose(brain.connectomes['A']['A'].weights.sum(axis=0), 1, atol=1e-6)


@pytest.mark.parametrize('engine_name,explicit', [('numpy_exact', False), ('numpy_explicit', False), ('numpy_sparse', True)])
def test_unsupported_runtime_lri_leaves_descriptor_unchanged(engine_name, explicit):
    brain = Brain(engine=engine_name, norm_init=False)
    brain.add_area('A', 20, 2, explicit=explicit)
    with pytest.raises(NotImplementedError, match='LRI'):
        brain.set_lri('A', 3, .2)
    assert brain.areas['A'].refractory_period == 0
    assert brain.areas['A'].inhibition_strength == 0
    brain.set_lri('A', 0, 0)


@pytest.mark.parametrize('method', ['clear_refractory', 'clear_refracted_bias'])
def test_history_clear_reaches_area_owner(method, monkeypatch):
    brain = Brain(engine='numpy_sparse', norm_init=False)
    brain.add_area('A', 20, 2, explicit=True)
    calls = []
    owner = brain._engine_for(brain.areas['A'])
    monkeypatch.setattr(owner, method, lambda name: calls.append(name))
    def wrong_owner(name):
        pytest.fail('history clear reached primary mirror instead of owner')
    monkeypatch.setattr(brain._engine, method, wrong_owner)
    getattr(brain, method)('A')
    assert calls == ['A']


def test_supported_lri_updates_both_states_and_resets_history():
    brain = Brain(engine='numpy_sparse', norm_init=False)
    brain.add_area('A', 20, 2)
    state = brain._engine._areas['A']
    brain.set_lri('A', 3, .2)
    assert state.refractory_period == brain.areas['A'].refractory_period == 3
    assert state.inhibition_strength == brain.areas['A'].inhibition_strength == .2
    state._refractory_history.append([1])
    brain.clear_refractory('A')
    assert not state._refractory_history


@pytest.mark.parametrize('period,strength', [
    (-1, .2), (1.5, .2), (True, .2), ('3', .2), (2**100, .2),
    (2, -.1), (2, float('nan')), (2, float('inf')), (2, True), (2, '.2'),
])
@pytest.mark.parametrize('surface', ['brain_set', 'engine_set', 'brain_add', 'engine_add'])
def test_invalid_lri_parameters_preserve_state(period, strength, surface):
    brain = Brain(p=.1, engine='numpy_sparse', norm_init=False)
    brain.add_area('A', 20, 2)
    brain.set_lri('A', 3, .2)
    engine = brain._engine
    state = engine._areas['A']
    history = state._refractory_history
    history.append({1})
    rng_before = repr(engine._rng.bit_generator.state)
    with pytest.raises(ValueError):
        if surface == 'brain_set':
            brain.set_lri('A', period, strength)
        elif surface == 'engine_set':
            engine.set_lri('A', period, strength)
        elif surface == 'brain_add':
            brain.add_area('B', 20, 2, refractory_period=period, inhibition_strength=strength)
        else:
            engine.add_area('B', 20, 2, .1, refractory_period=period, inhibition_strength=strength)
    assert state.refractory_period == brain.areas['A'].refractory_period == 3
    assert state.inhibition_strength == brain.areas['A'].inhibition_strength == .2
    assert state._refractory_history is history and list(history) == [{1}]
    assert 'B' not in brain.areas and 'B' not in engine._areas
    assert repr(engine._rng.bit_generator.state) == rng_before


def test_numpy_lri_scalars_are_canonicalized_without_changing_window():
    brain = Brain(p=.1, engine='numpy_sparse', norm_init=False)
    brain.add_area('A', 20, 2)
    brain.set_lri('A', np.int64(2), np.float64(.25))
    state = brain._engine._areas['A']
    assert type(state.refractory_period) is int
    assert type(brain.areas['A'].refractory_period) is int
    state._refractory_history.extend([{1}, {2}, {3}])
    assert list(state._refractory_history) == [{2}, {3}]


@pytest.mark.parametrize('engine_name', ['numpy_exact', 'numpy_explicit'])
@pytest.mark.parametrize('period,strength', [(False, 0), (0, float('nan'))])
def test_unsupported_lri_backend_still_validates_numeric_inputs(engine_name, period, strength):
    from neural_assemblies.core.engine import create_engine
    engine = create_engine(engine_name, p=.1)
    engine.add_area('A', 20, 2, .1)
    with pytest.raises(ValueError):
        engine.set_lri('A', period, strength)
