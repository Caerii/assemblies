"""Explicit configuration for an uncalibrated neural branch experiment."""
from dataclasses import dataclass
import math
from numbers import Integral, Real

from ..core.registration import validate_area_registration, validate_round_count


@dataclass(frozen=True)
class SeedMixtureChoice:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-pfa-choice

    Target weights control initial seed mixtures, not outcome probabilities.
    Settings are independent of the FSM's state-encoding population and training.
    Zero fires/rounds are permitted for mechanism-disabled/seed-only controls.
    """

    n: int
    k: int
    beta: float
    rounds_train: int = 15
    fires: int = 2
    rounds: int = 10
    mode: str = "k_split"

    def __post_init__(self):
        n, k = validate_area_registration("coin", self.n, self.k)
        rounds_train = validate_round_count(self.rounds_train)
        if (isinstance(self.beta, bool) or not isinstance(self.beta, Real)
                or not math.isfinite(self.beta) or self.beta < 0):
            raise ValueError("coin beta must be a finite nonnegative real number")
        for name in ("fires", "rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"coin {name} must be a nonnegative integer")
        if self.mode not in ("k_split", "compete"):
            raise ValueError("coin mode must be 'k_split' or 'compete'")
        # Keep the immutable record directly serializable into a run protocol.
        for name, value in (("n", n), ("k", k), ("beta", float(self.beta)),
                            ("rounds_train", rounds_train), ("fires", int(self.fires)),
                            ("rounds", int(self.rounds))):
            object.__setattr__(self, name, value)

    def build(self, brain, *, prefix, area_name="flip"):
        from .pfa import RandomChoiceArea
        return RandomChoiceArea(brain, area_name=area_name, prefix=prefix,
                                n=self.n, k=self.k, beta=self.beta,
                                rounds_train=self.rounds_train, fires=self.fires,
                                construction="attractor")
