"""A second vocabulary source must not replace a word's existing input."""
import copy

import pytest

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import NOUN_CORE


@pytest.fixture
def parser():
    return EmergentParser(n=100, k=10, seed=13, rounds=1,
                          engine="numpy_sparse", fast_training=True)


def test_repeated_phon_registration_preserves_source_and_learning_rate(parser):
    name = parser.add_phon_stimulus('wug')
    brain = parser.brain
    original = brain.stimuli[name]
    connections = dict(brain._engine._stim_conns[name])
    brain._engine.set_beta(NOUN_CORE, name, .37)
    before = copy.deepcopy(brain._engine._rng.bit_generator.state)
    assert parser.add_phon_stimulus('wug') == name
    assert brain.stimuli[name] is original
    assert all(brain._engine._stim_conns[name][target] is conn
               for target, conn in connections.items())
    assert brain._engine.get_beta(NOUN_CORE, name) == .37
    assert brain._engine._rng.bit_generator.state == before


def test_existing_phon_cannot_be_resized_by_configuration_change(parser):
    name = parser.add_phon_stimulus('wug')
    original = parser.brain.stimuli[name]
    parser.phon_weight *= 2
    with pytest.raises(ValueError, match='size'):
        parser.add_phon_stimulus('wug')
    assert parser.brain.stimuli[name] is original
    assert parser.stim_map['wug'] == name
