"""Separate attractor construction from the policy used to observe it."""
from dataclasses import dataclass
import math
from numbers import Integral, Real

from ..core.registration import validate_area_registration, validate_round_count


def _nonnegative_count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


@dataclass(frozen=True)
class AttractorConfig:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-context-choice

    Construction of two recurrent attractors; no observation or probability law.
    """
    n: int
    k: int
    beta: float
    rounds_train: int = 15
    fires: int = 2

    def __post_init__(self):
        n, k = validate_area_registration('coin', self.n, self.k)
        rounds_train = validate_round_count(self.rounds_train)
        if (isinstance(self.beta, bool) or not isinstance(self.beta, Real)
                or not math.isfinite(self.beta) or self.beta < 0):
            raise ValueError('coin beta must be a finite nonnegative real number')
        for name, value in (('n', n), ('k', k), ('beta', float(self.beta)),
                            ('rounds_train', rounds_train),
                            ('fires', _nonnegative_count(self.fires, 'coin fires'))):
            object.__setattr__(self, name, value)

    def build(self, brain, *, prefix, area_name='flip'):
        from .pfa import RandomChoiceArea
        return RandomChoiceArea(brain, area_name=area_name, prefix=prefix,
                                n=self.n, k=self.k, beta=self.beta,
                                rounds_train=self.rounds_train, fires=self.fires,
                                construction='attractor')


@dataclass(frozen=True)
class SeedMixtureChoice(AttractorConfig):
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-pfa-choice

    Target weights control initial seed mixtures, not outcome probabilities.
    Zero fires/rounds permit mechanism-disabled and seed-only controls.
    """
    rounds: int = 10
    mode: str = 'k_split'

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'rounds', _nonnegative_count(self.rounds, 'coin rounds'))
        if self.mode not in ('k_split', 'compete'):
            raise ValueError("coin mode must be 'k_split' or 'compete'")

    def select_index(self, coin, conditional_weights, *, seed=None) -> int:
        """Execute a validated ordered branch schedule; last index is fallback.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-branch-schedule
        Binary schedules use the caller seed; multiway schedules draw a seed per
        attempted choice. Labels are neural observations, not calibrated draws.
        """
        import numpy as np
        weights = tuple(conditional_weights)
        if (not weights or weights[-1] != 1.0
                or any(isinstance(w, bool) or not isinstance(w, Real)
                       or not math.isfinite(w) or not 0 <= w <= 1 for w in weights)):
            raise ValueError('conditional weights must be finite in [0, 1] with a final fallback 1')
        rng = np.random.default_rng(seed) if len(weights) > 2 else None
        for index, weight in enumerate(weights[:-1]):
            branch_seed = int(rng.integers(0, 2**31)) if rng is not None else seed
            label = coin.flip(bias=weight, rounds=self.rounds, seed=branch_seed, mode=self.mode)
            if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
                raise ValueError('neural branch selector must return label zero or one')
            if label == 0:
                return index
        return len(weights) - 1
