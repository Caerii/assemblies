"""Cue replacement controls; development fixtures, not a tolerance-range study.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery
The historical single-seed sampled protocol is recorded in SEMANTIC_CARDS.md.
"""
import pickle
import numpy as np
import pytest

from neural_assemblies import Brain
from neural_assemblies.assembly_calculus import (
    Assembly, AttractorConfig, replace_neurons, observe_recovery,
)
from neural_assemblies.core.index_spaces import NeuronIds


def ids(values):
    return NeuronIds(np.asarray(values, dtype=np.uint32))


def build(seed, beta, engine):
    if engine == 'torch_sparse':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
    brain = Brain(p=.05, seed=seed, engine=engine)
    attractors = AttractorConfig(2000, 200, beta, rounds_train=10).build(brain, prefix='recovery')
    return brain, attractors.asm0


@pytest.mark.parametrize('engine', ['numpy_sparse', 'torch_sparse'])
@pytest.mark.parametrize('seed', [1, 2, 3])
def test_recovery_improves_cue_and_fails_learning_and_dynamics_nulls(engine, seed):
    recovered = {}
    for beta in (3., 0.):
        brain, reference = build(seed, beta, engine)
        cue = replace_neurons(reference, population=ids(range(2000)), count=100, seed=700)
        area = brain.areas[reference.area]
        connection = brain._engine._area_conns[reference.area][reference.area]
        def weights():
            if engine == 'torch_sparse':
                assert brain._engine._device.type == 'cuda'
                return connection._val.float().cpu().numpy().copy()
            return np.asarray(connection.weights).copy()
        before_weights = weights()
        before_activity = area.winners.copy()
        before_rng = pickle.dumps(brain._engine._rng.bit_generator.state)
        area.fixed_assembly = True  # Observation must release and restore the clamp.
        result = observe_recovery(brain, reference, cue, rounds=5, seed=701)
        null = observe_recovery(brain, reference, cue, rounds=5, seed=701, recurrence_enabled=False)
        assert null.cue_overlap == null.recovered_overlap == .5
        assert null.improvement == 0
        assert result.cue_overlap == .5
        assert area.fixed_assembly
        np.testing.assert_array_equal(area.winners, before_activity)
        np.testing.assert_array_equal(weights(), before_weights)
        assert pickle.dumps(brain._engine._rng.bit_generator.state) == before_rng
        recovered[beta] = result
    assert recovered[3.].recovered_overlap > .9
    assert recovered[3.].improvement > .4
    assert recovered[0.].improvement <= 0, 'initialized untrained control improved the cue'


def test_replacement_is_exact_order_independent_and_nested():
    reference = Assembly('A', ids([50, 10, 30, 20]))
    population = ids([90, 20, 10, 30, 50, 70, 80, 60])
    prior_removed, prior_added = set(), set()
    for count in range(5):
        cue = replace_neurons(reference, population=population, count=count, seed=12)
        reordered = replace_neurons(Assembly('A', reference.neuron_ids[::-1]),
                                    population=population[::-1], count=count, seed=12)
        assert cue == reordered
        removed = set(reference.neuron_ids) - set(cue.neuron_ids)
        added = set(cue.neuron_ids) - set(reference.neuron_ids)
        assert len(removed) == len(added) == count
        assert prior_removed <= removed and prior_added <= added
        assert len(cue) == len(reference) and not cue.neuron_ids.flags.writeable
        prior_removed, prior_added = removed, added


@pytest.mark.parametrize('count', [-1, True, .5, 3])
def test_impossible_replacement_never_clamps(count):
    with pytest.raises(ValueError):
        replace_neurons(Assembly('A', ids([0, 1])), population=ids([0, 1, 2]), count=count, seed=1)


@pytest.mark.parametrize('population', [[0, 0, 1], [0, 2, 3]])
def test_population_must_be_unique_and_contain_reference(population):
    with pytest.raises(ValueError):
        replace_neurons(Assembly('A', ids([0, 1])), population=ids(population), count=1, seed=1)


def test_subset_does_not_score_as_full_recovery_and_exception_restores(monkeypatch):
    brain, reference = build(1, 3., 'numpy_sparse')
    cue = Assembly(reference.area, reference.neuron_ids[:100])
    result = observe_recovery(brain, reference, cue, rounds=1, recurrence_enabled=False)
    assert result.recovered_overlap == .5
    area = brain.areas[reference.area]
    area.fixed_assembly = True
    before = area.winners.copy()
    def fail(*args, **kwargs):
        raise RuntimeError('constructed')
    monkeypatch.setattr(brain, 'project', fail)
    with pytest.raises(RuntimeError, match='constructed'):
        observe_recovery(brain, reference, cue, rounds=1)
    assert area.fixed_assembly
    np.testing.assert_array_equal(area.winners, before)


def test_partial_population_refuses_before_activating_cue():
    brain = Brain(engine='numpy_sparse', seed=1)
    brain.add_area('A', 100, 10, .1)
    reference = Assembly('A', ids(range(10)))
    with pytest.raises(ValueError, match='fully materialized'):
        observe_recovery(brain, reference, reference, rounds=1)
    assert len(brain.areas['A'].winners) == 0


@pytest.mark.parametrize('engine', ['numpy_sparse', 'torch_sparse'])
def test_materialization_accessor_distinguishes_cold_from_full(engine):
    if engine == 'torch_sparse':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
    brain = Brain(engine=engine, seed=1)
    brain.add_area('A', 100, 10, .1)
    assert brain._engine.materialized_count('A') == 0
    assert brain._engine.materialized_count('missing') is None
    brain.materialize_area('A')
    assert brain._engine.materialized_count('A') == 100



def test_oversized_cue_fails_before_brain_access():
    with pytest.raises(ValueError, match='cue winner count'):
        observe_recovery(object(), Assembly('A', ids([0])), Assembly('A', ids([0, 1])), rounds=1)


def test_oversized_recovery_cannot_manufacture_full_reference_coverage():
    brain = Brain(p=1., seed=1, engine='numpy_sparse')
    brain.add_area('A', 20, 10, .1)
    brain.materialize_area('A')
    reference = Assembly('A', ids(range(5)))
    before = brain.areas['A'].winners.copy()
    with pytest.raises(ValueError, match='recovery winner count'):
        observe_recovery(brain, reference, reference, rounds=1)
    np.testing.assert_array_equal(brain.areas['A'].winners, before)
