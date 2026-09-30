"""The retired Markov instrument must reject old calls before brain mutation.

Old tests checked only both-label coverage and reset. Those cannot establish
neural transition readout; replacement controls live in test_arc_markov_contract.
Historical goldens are retained without reinterpreting them as this new protocol.
"""
import pytest
from neural_assemblies.programs.nemo_fsm import NemoMarkovPFA, AlternatingMarkovNetwork
from neural_assemblies.programs.markov_coin import MarkovChainModel


@pytest.mark.parametrize('constructor', [NemoMarkovPFA, AlternatingMarkovNetwork])
def test_invalid_legacy_markov_path_stops_before_brain_access(constructor):
    with pytest.raises(NotImplementedError, match='ArcMarkovNetwork'):
        constructor(object(), ['q0'], [('q0', 'flip', 'q0', 1.)], 'q0')


def test_trace_wrapper_requires_the_new_protocol_explicitly():
    with pytest.raises(ValueError, match='explicit ArcMarkovProtocol'):
        MarkovChainModel(object(), [('q0', 'flip', 'q0')], 'q0')
