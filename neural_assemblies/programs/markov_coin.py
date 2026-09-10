"""Coin and Markov experiment wrappers with explicit neural selector settings.

Trace frequencies configure seed mixtures, not calibrated outcome laws. The new
arc Markov composition does not reproduce historical coin2024 parity protocols.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from neural_assemblies.assembly_calculus.pfa import FlipMode, PFANetwork
from neural_assemblies.assembly_calculus.coin_config import SeedMixtureChoice
from neural_assemblies.compute import EPercentPolicy


def _balanced_coin_traces() -> list:
    """50/50 q0/q1 branches with ergodic q1 return paths."""
    return [
        ("q0", "flip", "q0"),
        ("q0", "flip", "q1"),
        ("q1", "flip", "q0"),
        ("q1", "flip", "q1"),
    ]


TransitionTrace = Tuple[str, str, str]


def train_markov_from_sequences(
    traces: Sequence[TransitionTrace],
) -> List[Tuple[str, str, str, float]]:
    """Estimate probabilistic transitions from ``(state, symbol, next)`` traces."""
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for state, symbol, nxt in traces:
        counts[(state, symbol)][nxt] += 1

    transitions = []
    for (state, symbol), dest_counts in sorted(counts.items()):
        total = sum(dest_counts.values())
        for dest, c in sorted(dest_counts.items()):
            transitions.append((state, symbol, dest, c / total))
    return transitions


class CoinFlipModel:
    """PFA seed-mixture experiment plus an independently trained sampling coin.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-pfa-choice
    Trace frequencies set seed mixtures; neither coin is probability-calibrated.
    input_noise_std affects the separate sampling coin, not the PFA selector.
    """

    def __init__(
        self,
        brain,
        traces: Sequence[TransitionTrace],
        initial_state: str,
        n: int = 5000,
        k: int = 50,
        beta: float = 0.08,
        rounds: int = 8,
        input_noise_std: float = 0.0,
        flip_mode: FlipMode | None = None,
        *,
        choice: SeedMixtureChoice | None = None,
    ):
        if not isinstance(choice, SeedMixtureChoice):
            raise ValueError("CoinFlipModel requires explicit SeedMixtureChoice; "
                             "trace frequencies are not calibrated neural probabilities")
        if flip_mode is not None and flip_mode != choice.mode:
            raise ValueError("flip_mode must agree with choice.mode")
        self.choice = choice
        transitions = train_markov_from_sequences(traces)
        states = sorted({s for fr, _, to in traces for s in (fr, to)})
        symbols = sorted({sym for _, sym, _ in traces})
        self.flip_mode = choice.mode
        self.pfa = PFANetwork(
            brain, states, symbols, transitions, initial_state,
            n=n, k=k, beta=beta, rounds=rounds, prefix="_coin_pfa",
            choice=choice,
        )
        self.coin = choice.build(brain, prefix="_coin_rc", area_name="_coin")
        self.input_noise_std = input_noise_std
        if input_noise_std > 0:
            brain.set_input_noise(self.coin.area_name, input_noise_std)
            brain.set_competition_policy(
                self.coin.area_name, EPercentPolicy(fraction_of_max=0.5, min_winners=1)
            )

    def sample_branch(self, bias: float = 0.5, seed: int | None = None) -> int:
        return self.coin.flip(bias=bias, seed=seed, mode=self.choice.mode,
                              rounds=self.choice.rounds)

    def empirical_flip_counts(
        self,
        n_flips: int,
        bias: float = 0.5,
        seed_base: int = 0,
    ) -> tuple[int, int]:
        """Run ``n_flips`` and return ``(count_0, count_1)``."""
        counts = {0: 0, 1: 0}
        for i in range(n_flips):
            counts[self.sample_branch(bias=bias, seed=seed_base + i * 17)] += 1
        return counts[0], counts[1]

    def step_symbol(self, symbol: str, seed: int | None = None) -> str:
        return self.pfa.step(symbol, seed=seed)


class MarkovChainModel:
    """Trace-frequency wrapper for the explicit decoded-state arc experiment.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-arc-markov
    Frequency weights become neural seed mixtures, not calibrated probabilities.
    """

    def __init__(self, brain, traces: Sequence[TransitionTrace], initial_state: str, *,
                 protocol=None, choice: SeedMixtureChoice | None = None,
                 symbol: str = 'flip', prefix: str = '_markov_chain'):
        from .arc_markov import ArcMarkovNetwork, ArcMarkovProtocol
        if not isinstance(protocol, ArcMarkovProtocol):
            raise ValueError('MarkovChainModel requires an explicit ArcMarkovProtocol; '
                             'historical alternating-area results are not reproduced')
        traces = tuple(traces)
        transitions = train_markov_from_sequences(traces)
        states = sorted({s for fr, _, to in traces for s in (fr, to)})
        self._pfa = ArcMarkovNetwork(brain, states, transitions, initial_state,
                                    symbol=symbol, protocol=protocol, choice=choice, prefix=prefix)

    @property
    def parameters(self):
        return self._pfa.parameters

    @property
    def current_state(self) -> str:
        return self._pfa.current_state

    def reset(self) -> None:
        self._pfa.reset()

    def step(self, seed: int | None = None) -> str:
        return self._pfa.sample_step(seed=seed)

    def run(self, n_steps: int, *, seed_base: int = 0) -> List[str]:
        """Sample ``n_steps`` and return the state trajectory."""
        trajectory = []
        for i in range(n_steps):
            trajectory.append(self.step(seed=seed_base + i * 17))
        return trajectory
