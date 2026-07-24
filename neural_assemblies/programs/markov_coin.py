"""
Coin-flipping / Markov protocols (Dabagia et al. 2024).

Trains a ``PFANetwork`` from observed transition frequencies and exposes
``RandomChoiceArea`` for binary sampling with optional input noise.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from neural_assemblies.assembly_calculus.pfa import FlipMode, PFANetwork, RandomChoiceArea
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
    """Markov coin model: PFA for state evolution + neural coin for branches."""

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
        flip_mode: FlipMode = "k_split",
    ):
        transitions = train_markov_from_sequences(traces)
        states = sorted({s for fr, _, to in traces for s in (fr, to)})
        symbols = sorted({sym for _, sym, _ in traces})
        self.flip_mode = flip_mode
        self.pfa = PFANetwork(
            brain, states, symbols, transitions, initial_state,
            n=n, k=k, beta=beta, rounds=rounds, prefix="_coin_pfa",
            flip_mode=flip_mode,
        )
        self.coin = RandomChoiceArea(brain, n=n, k=k, beta=beta, prefix="_coin_rc")
        self.input_noise_std = input_noise_std
        if input_noise_std > 0:
            brain.set_input_noise(self.coin.area_name, input_noise_std)
            brain.set_competition_policy(
                self.coin.area_name, EPercentPolicy(fraction_of_max=0.5, min_winners=1)
            )

    def sample_branch(self, bias: float = 0.5, seed: int | None = None) -> int:
        return self.coin.flip(bias=bias, seed=seed, mode=self.flip_mode)

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
    """Markov chain via refracted arc + alternating state areas (NEMO coinflipping).

    Uses ``NemoMarkovPFA`` instead of ``PFANetwork`` — closer to dabagia.org/nemo
    Markov architecture (refracted arc, current/next inhibition schedule).
    """

    def __init__(
        self,
        brain,
        traces: Sequence[TransitionTrace],
        initial_state: str,
        *,
        symbol: str = "flip",
        n: int = 5000,
        k: int = 80,
        beta: float = 0.1,
        refracted_strength: float = 0.1,
        flip_mode: FlipMode = "compete",
        prefix: str = "_markov_chain",
    ):
        from neural_assemblies.programs.nemo_fsm import NemoMarkovPFA

        transitions = train_markov_from_sequences(traces)
        states = sorted({s for fr, _, to in traces for s in (fr, to)})
        self._pfa = NemoMarkovPFA(
            brain, states, transitions, initial_state,
            symbol=symbol, n=n, k=k, beta=beta,
            refracted_strength=refracted_strength,
            flip_mode=flip_mode, prefix=prefix,
        )

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
