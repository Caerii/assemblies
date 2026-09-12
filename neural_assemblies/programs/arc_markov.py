"""Neural branch selection composed with the existing refracted transition organ."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Sequence

from neural_assemblies.assembly_calculus.coin_config import SeedMixtureChoice
from neural_assemblies.assembly_calculus.transitions import TransitionMap
from neural_assemblies.core.registration import validate_area_registration
from .nemo_fsm import NemoArcFSM


@dataclass(frozen=True)
class ArcMarkovProtocol:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-arc-markov

    Explicit organ geometry, learning rule and presentation schedule.
    Zero presentations/beta/refraction are permitted for constructed controls.
    """
    n: int
    k: int
    beta: float
    organ_p: float
    refracted_strength: float
    presentations: int

    def __post_init__(self):
        n, k = validate_area_registration('arc_markov', self.n, self.k)
        object.__setattr__(self, 'n', n)
        object.__setattr__(self, 'k', k)
        for name in ('beta', 'organ_p', 'refracted_strength'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be a finite nonnegative real number')
            object.__setattr__(self, name, float(value))
        if not 0 < self.organ_p <= 1:
            raise ValueError('organ_p must be in (0, 1]')
        if (isinstance(self.presentations, bool) or not isinstance(self.presentations, Integral)
                or self.presentations < 0):
            raise ValueError('presentations must be a nonnegative integer')
        object.__setattr__(self, 'presentations', int(self.presentations))


class ArcMarkovNetwork:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-arc-markov

    A decoded-state feedback experiment, not an alternating-area Markov port.
    Coin labels select branch stimuli; the arc machine decodes the next state.
    Target weights parameterize seed mixtures, not calibrated probabilities.
    Each step observes inside a probe, then retains only the decoded state label.
    """
    def __init__(self, brain: Any, states: Sequence[str], transitions: Sequence[tuple[str, str, str]], initial_state: str, *,
                 protocol: ArcMarkovProtocol, choice: SeedMixtureChoice | None = None,
                 symbol='flip', prefix='_arc_markov'):
        if not isinstance(protocol, ArcMarkovProtocol):
            raise ValueError('ArcMarkovNetwork requires an explicit ArcMarkovProtocol')
        if choice is not None and not isinstance(choice, SeedMixtureChoice):
            raise ValueError('choice must be an explicit SeedMixtureChoice')
        states = tuple(states)
        table = TransitionMap(transitions).validate_domain(states, [symbol], initial_state)
        table.validate_probability_mass()
        if set(table.keys()) != {(state, symbol) for state in states}:
            raise ValueError('Markov transitions must declare an outgoing group for every state')
        schedules = {state: table.branch_schedule(state, symbol) for state in states}
        branching = any(len(schedule) > 1 for schedule in schedules.values())
        if branching and choice is None:
            raise ValueError('branching requires explicit SeedMixtureChoice; weights are not calibrated probabilities')
        branch_symbols = tuple(f'branch_{i}' for i in range(max(map(len, schedules.values()))))
        training: list[tuple[str, str, str]] = [(branch_symbols[i], state, target)
                    for state, schedule in schedules.items()
                    for i, (target, _) in enumerate(schedule)]
        self._weights = {state: tuple(weight for _, weight in schedule)
                         for state, schedule in schedules.items()}
        self._branch_symbols = branch_symbols
        self.brain, self._protocol, self._choice = brain, protocol, choice
        self._states, self._symbol = tuple(states), symbol
        self._transitions = tuple(table)
        self._initial_state = self._current = initial_state
        self.transition_machine = NemoArcFSM(
            brain, list(states), list(branch_symbols), [(fr, sym, to) for sym, fr, to in training],
            n=protocol.n, k=protocol.k, beta=protocol.beta, organ_p=protocol.organ_p,
            refracted_strength=protocol.refracted_strength, prefix=f'{prefix}_transition')
        # Fixed population in both trained and zero-presentation controls.
        brain.materialize_area(self.transition_machine.arc_area)
        self.transition_machine.train_from_list(training, presentations=protocol.presentations)
        self.coin = choice.build(brain, prefix=f'{prefix}_coin') if branching and choice is not None else None

    @property
    def protocol(self):
        return self._protocol

    @property
    def choice(self):
        return self._choice

    @property
    def current_state(self):
        return self._current

    @property
    def parameters(self):
        return {'protocol_version': 'arc-symbol-feedback-v1', 'organ': asdict(self.protocol),
                'choice': asdict(self.choice) if self.choice is not None else None,
                'feedback': 'decoded_state', 'connectome': 'materialized',
                'states': list(self._states), 'symbol': self._symbol,
                'initial_state': self._initial_state,
                'transitions': [edge.as_tuple(include_probability=True) for edge in self._transitions]}

    def reset(self):
        """Reset decoded feedback; preserve all learned weights and refraction."""
        self._current = self._initial_state

    def sample_step(self, seed=None):
        weights = self._weights[self._current]
        with self.brain.probe():
            choice = self._choice
            if len(weights) > 1 and (choice is None or self.coin is None):
                raise RuntimeError('branching arc state is missing its configured coin choice')
            if len(weights) > 1:
                assert choice is not None and self.coin is not None
                index = choice.select_index(self.coin, weights, seed=seed)
            else:
                index = 0
            decoded = self.transition_machine.run([self._branch_symbols[index]], start_state=self._current)[0]
        self._current = decoded
        return decoded
