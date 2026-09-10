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
