"""Seeded observations restore streams and do not erase independent input noise."""
import pickle

import numpy as np
import pytest

from neural_assemblies import Brain


def make(engine='numpy_sparse', materialized=True):
    if engine == 'torch_sparse':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('torch_sparse requires CUDA')
    brain = Brain(p=.1, seed=12, engine=engine)
    brain.add_area('A', 100, 10, .1)
    brain.add_stimulus('zero', 0)
    if materialized:
        brain.materialize_area('A')
    else:
        brain.add_stimulus('train', 10)
        brain.project({'train': ['A']}, {})
    brain.set_input_noise('A', 1.)
    return brain


@pytest.mark.parametrize('engine, compiled', [
    ('numpy_sparse', False), ('numpy_sparse', True), ('torch_sparse', False),
])
def test_noise_only_drive_replays_and_changes_with_seed(engine, compiled):
    brain = make(engine)
    if compiled:
        brain.projection_fidelity = 'compiled'
        target = brain._engine._areas['A']
        target._freeze_connectome_growth = True
        assert brain._engine._use_compiled_projection(target)
    rng = brain._engine._rng
    prior_rng = pickle.dumps(rng.bit_generator.state)
    prior_winners = brain.areas['A'].winners.copy()
    outputs = []
    for seed in (701, 702, 701):
        with brain.read_only(seed=seed):
            brain.project({'zero': ['A']}, {})
            outputs.append(brain.areas['A'].winners.copy())
        assert pickle.dumps(rng.bit_generator.state) == prior_rng
        np.testing.assert_array_equal(brain.areas['A'].winners, prior_winners)
    assert all(len(output) == 10 for output in outputs)
    np.testing.assert_array_equal(outputs[0], outputs[2])
    assert not np.array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize('engine', ['numpy_sparse', 'torch_sparse'])
def test_partial_population_noise_only_read_fails_explicitly(engine):
    brain = make(engine, materialized=False)
    with pytest.raises(ValueError, match='fully materialized'):
        with brain.read_only(seed=1):
            brain.project({'zero': ['A']}, {})


@pytest.mark.parametrize('bit_generator', [np.random.PCG64, np.random.MT19937])
def test_exception_and_nested_scope_restore_original_generator(bit_generator):
    brain = Brain(engine='numpy_sparse', seed=1)
    rng = brain._engine._rng = np.random.Generator(bit_generator(12))
    original = pickle.dumps(rng.bit_generator.state)
    with brain.read_only(seed=20):
        first = rng.integers(0, 2**31, size=5)
        outer = pickle.dumps(rng.bit_generator.state)
        with pytest.raises(RuntimeError):
            with brain.read_only(seed=30):
                rng.integers(0, 2**31, size=5)
                raise RuntimeError('constructed')
        assert pickle.dumps(rng.bit_generator.state) == outer
    assert brain._engine._rng is rng
    assert pickle.dumps(rng.bit_generator.state) == original
    with brain.read_only(seed=20):
        np.testing.assert_array_equal(first, rng.integers(0, 2**31, size=5))


@pytest.mark.parametrize('seed', [-1, True, 1.5, '1'])
def test_invalid_seed_fails_before_brain_access(seed):
    brain = object.__new__(Brain)
    with pytest.raises(ValueError, match='observation seed'):
        with brain.read_only(seed=seed):
            raise AssertionError('must not enter')


def test_mixed_owner_streams_are_distinct_and_restored():
    brain = Brain(engine='numpy_sparse', seed=1)
    brain.add_explicit_area('E', 20, 4, .1)
    generators = [engine._rng for engine in brain._all_engines()]
    assert len(generators) == 2
    before = [pickle.dumps(rng.bit_generator.state) for rng in generators]
    with brain.read_only(seed=7):
        values = [rng.integers(0, 2**31, size=5) for rng in generators]
        assert not np.array_equal(values[0], values[1])
    assert [pickle.dumps(rng.bit_generator.state) for rng in generators] == before
