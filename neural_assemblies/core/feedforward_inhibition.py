"""Configuration and admission for inhibitory feedforward synapses."""

from dataclasses import dataclass
import math
from numbers import Real


DEFAULT_INHIBITORY_WEIGHT = -0.2


@dataclass(frozen=True)
class FeedforwardInhibitionConfig:
    """Probability and weight of inhibitory present area-to-area synapses.

    Stimulus afferents are outside this mechanism.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-feedforward-inhibition
    """

    probability: float = 0.0
    weight: float = DEFAULT_INHIBITORY_WEIGHT

    def __post_init__(self):
        values = (self.probability, self.weight)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError("feedforward inhibition values must be finite real numbers")
        probability, weight = map(float, values)
        if not math.isfinite(probability) or not math.isfinite(weight):
            raise ValueError("feedforward inhibition values must be finite real numbers")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("inhibitory probability must be in [0, 1]")
        if weight >= 0.0:
            raise ValueError("inhibitory weight must be negative")
        if probability == 0.0 and weight != DEFAULT_INHIBITORY_WEIGHT:
            raise ValueError(
                "inhibitory weight is inert when inhibitory probability is zero"
            )
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "weight", weight)

    @property
    def enabled(self) -> bool:
        return self.probability > 0.0

    def as_kwargs(self) -> dict[str, float]:
        return {
            "inhibitory_prob": self.probability,
            "inhibitory_weight": self.weight,
        }

    @classmethod
    def from_engine(cls, engine) -> "FeedforwardInhibitionConfig":
        return cls(
            getattr(engine, "inhibitory_prob", 0.0),
            getattr(engine, "inhibitory_weight", DEFAULT_INHIBITORY_WEIGHT),
        )


def validate_feedforward_inhibition_capability(engine_type, config) -> None:
    """Reject an enabled inhibition law unsupported by an engine."""
    if config.enabled and not getattr(
        engine_type, "supports_feedforward_inhibition", False
    ):
        raise ValueError(
            f"{engine_type.__name__} does not support feedforward inhibition; "
            "choose an engine whose declared capability is enabled"
        )
