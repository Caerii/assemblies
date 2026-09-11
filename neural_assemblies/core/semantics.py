"""Validated model semantics and runtime admission policies.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-sampled-recurrence
Specification: neural_assemblies/ir/VERIFICATION.md#contract-model-semantics
"""

from dataclasses import dataclass, fields
from enum import Enum
import math
import numbers
from typing import Mapping


BRAIN_ENGINE_NAMES = frozenset({
    "numpy_sparse",
    "numpy_explicit",
    "numpy_exact",
    "torch_sparse",
    "cuda_implicit",
    "cupy_sparse",
})


class _SemanticEnum(str, Enum):
    @classmethod
    def normalize(cls, value: object):
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"{cls.__name__} must be one of {[x.value for x in cls]}")
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"{cls.__name__} must be one of {[x.value for x in cls]}"
            ) from exc


class ConnectomeMode(_SemanticEnum):
    LAZY_CONTENT_ADDRESSED = "lazy-content-addressed"
    LAZY_STREAM_ADDRESSED = "lazy-stream-addressed"
    FIXED_DENSE_CONTENT_ADDRESSED = "fixed-dense-content-addressed"
    FIXED_HASH_REGENERATED = "fixed-hash-regenerated"


class CandidateDomain(_SemanticEnum):
    ALL_NEURONS = "all-neurons"
    MATERIALIZED_PLUS_ORDER_STATISTICS = "materialized-plus-order-statistics"
    ALL_NEURONS_WITH_SAMPLED_DRIVE = "all-neurons-with-sampled-drive"


class StimulusDriveLaw(_SemanticEnum):
    FIXED_BERNOULLI_AFFERENT_COUNT = "fixed-bernoulli-afferent-count"
    LAZY_CONDITIONED_AFFERENT_COUNT = "lazy-conditioned-afferent-count"


class TieBreakRule(_SemanticEnum):
    LOWEST_NEURON_ID = "lowest-neuron-id"
    PARTITION_ORDER = "partition-order"
    BACKEND_TOPK_ORDER = "backend-topk-order"


class ArithmeticMode(_SemanticEnum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class NormalizationMode(_SemanticEnum):
    NONE = "none"
    INVERSE_INDEGREE = "inverse-indegree"


class PlasticityRule(_SemanticEnum):
    MULTIPLICATIVE_CLIPPED = "multiplicative-clipped"
    MULTIPLICATIVE_UNBOUNDED = "multiplicative-unbounded"


@dataclass(frozen=True)
class ModelSemantics:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-model-semantics

    Backend-independent identity of choices that can change a result.

    The object describes an engine's default k-WTA path. Area-local winner
    policies and operation schedules are separate protocol state.
    """

    connectome: ConnectomeMode
    candidate_domain: CandidateDomain
    stimulus_drive: StimulusDriveLaw
    default_tie_break: TieBreakRule
    arithmetic: ArithmeticMode
    normalization: NormalizationMode
    plasticity: PlasticityRule = PlasticityRule.MULTIPLICATIVE_CLIPPED
    weight_ceiling: float | None = 20.0

    def __post_init__(self):
        enum_types = {
            "connectome": ConnectomeMode,
            "candidate_domain": CandidateDomain,
            "stimulus_drive": StimulusDriveLaw,
            "default_tie_break": TieBreakRule,
            "arithmetic": ArithmeticMode,
            "normalization": NormalizationMode,
            "plasticity": PlasticityRule,
        }
        for name, enum_type in enum_types.items():
            object.__setattr__(self, name, enum_type.normalize(getattr(self, name)))
        ceiling = self.weight_ceiling
        if ceiling is not None:
            if isinstance(ceiling, bool) or not isinstance(ceiling, numbers.Real):
                raise ValueError("weight_ceiling must be a positive finite number or None")
            ceiling = float(ceiling)
            if not math.isfinite(ceiling) or ceiling <= 0:
                raise ValueError("weight_ceiling must be a positive finite number or None")
            object.__setattr__(self, "weight_ceiling", ceiling)
        if (
            self.plasticity is PlasticityRule.MULTIPLICATIVE_CLIPPED
            and ceiling is None
        ):
            raise ValueError("clipped plasticity requires weight_ceiling")
        if (
            self.plasticity is PlasticityRule.MULTIPLICATIVE_UNBOUNDED
            and ceiling is not None
        ):
            raise ValueError("unbounded plasticity requires weight_ceiling=None")

    @classmethod
    def normalize(cls, value: object) -> "ModelSemantics":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("model_semantics must be ModelSemantics or a mapping")
        expected = {field.name for field in fields(cls)}
        supplied = set(value)
        missing = expected - supplied
        extra = supplied - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extra:
                details.append(f"unknown {sorted(extra, key=repr)}")
            raise ValueError("invalid model_semantics mapping: " + "; ".join(details))
        return cls(**dict(value))

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: (
                value.value if isinstance(value, Enum) else value
            )
            for field in fields(self)
            for value in (getattr(self, field.name),)
        }

    def mismatch(self, actual: "ModelSemantics") -> dict[str, tuple[object, object]]:
        actual = self.normalize(actual)
        return {
            field.name: (
                self.to_dict()[field.name],
                actual.to_dict()[field.name],
            )
            for field in fields(self)
            if getattr(self, field.name) != getattr(actual, field.name)
        }


class SampledRecurrencePolicy(str, Enum):
    """Admission policy for recurrence over an unmaterialized connectome."""

    WARN = "warn"
    ACKNOWLEDGED = "acknowledged"
    FORBID = "forbid"

    @classmethod
    def normalize(cls, value: object) -> "SampledRecurrencePolicy":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(
                "sampled_recurrence_policy must be 'warn', 'acknowledged', or 'forbid'"
            )
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                "sampled_recurrence_policy must be 'warn', 'acknowledged', or 'forbid'"
            ) from exc


def describe_brain_model(engine: str, **brain_kwargs) -> ModelSemantics:
    """Resolve a Brain engine profile without registering model topology.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-model-semantics
    """
    if not isinstance(engine, str) or engine == "auto":
        raise ValueError("describe_brain_model requires an explicit engine name")
    from .brain import Brain

    return Brain(engine=engine, **brain_kwargs).model_semantics
