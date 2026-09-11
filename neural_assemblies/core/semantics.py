"""Validated model semantics and runtime admission policies.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-sampled-recurrence
Specification: neural_assemblies/ir/VERIFICATION.md#contract-model-semantics
"""

from dataclasses import dataclass, fields
from enum import Enum
import math
import numbers
from types import MappingProxyType
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
    ZERO_OR_SIZE_AFFERENT_COUNT = "zero-or-size-afferent-count"


class TieBreakRule(_SemanticEnum):
    LOWEST_NEURON_ID = "lowest-neuron-id"
    PARTITION_ORDER = "partition-order"
    BACKEND_TOPK_ORDER = "backend-topk-order"
    DETERMINISTIC_HASH_JITTER = "deterministic-hash-jitter"


class ArithmeticMode(_SemanticEnum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class NormalizationMode(_SemanticEnum):
    NONE = "none"
    INVERSE_INDEGREE = "inverse-indegree"
    INVERSE_INDEGREE_WITH_COLUMN_SCALING = "inverse-indegree-with-column-scaling"


class PlasticityRule(_SemanticEnum):
    MULTIPLICATIVE_CLIPPED = "multiplicative-clipped"
    MULTIPLICATIVE_UNBOUNDED = "multiplicative-unbounded"


class OrganKind(_SemanticEnum):
    ASSEMBLY_MEMORY = "assembly-memory"
    ASSIGNED_STATE_FSM = "assigned-state-fsm"
    SEQUENCE_TRANSDUCER = "sequence-transducer"


ORGAN_ENGINE_KINDS = {
    "hashed_assembly_memory": OrganKind.ASSEMBLY_MEMORY,
    "hashed_arc_fsm": OrganKind.ASSIGNED_STATE_FSM,
    "hashed_transducer": OrganKind.SEQUENCE_TRANSDUCER,
}


class StateCode(_SemanticEnum):
    NONE = "none"
    ASSIGNED_BLOCKS = "assigned-blocks"
    INDUCED_ASSEMBLY = "induced-assembly"
    PREVIOUS_ARC_COPY = "previous-arc-copy"


class TrainingSchedule(_SemanticEnum):
    STIMULUS_PLUS_RECURRENCE = "stimulus-plus-recurrence"
    TEACHER_FORCED_TRANSITION = "teacher-forced-transition"
    GROUNDED_TEACHER_FORCED_TRANSDUCTION = "grounded-teacher-forced-transduction"


class InferenceSchedule(_SemanticEnum):
    FROZEN_RECURRENT_COMPLETION = "frozen-recurrent-completion"
    FROZEN_STATE_ADVANCING_TRANSITION = "frozen-state-advancing-transition"
    FROZEN_STATE_ADVANCING_EMISSION = "frozen-state-advancing-emission"


class ExecutionKind(_SemanticEnum):
    BRAIN = "brain"
    ORGAN = "organ"


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


@dataclass(frozen=True)
class OrganSemantics:
    """Composable substrate and schedule identity for a hashed organ.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-organ-semantics

    Dimensions and study-specific bars remain protocol parameters.  These
    fields name choices that can change the transition relation itself.
    """

    organ: OrganKind
    substrate: ModelSemantics
    state_code: StateCode
    training_schedule: TrainingSchedule
    inference_schedule: InferenceSchedule
    tie_jitter: float
    arc_refraction_charge: float
    state_refraction_charge: float
    horizon: int = 0
    successor_gain: float = 1.0
    prediction_gain: float = 0.0
    feature_register: bool = False
    convergence_gate: bool = False

    def __post_init__(self):
        enum_types = {
            "organ": OrganKind,
            "state_code": StateCode,
            "training_schedule": TrainingSchedule,
            "inference_schedule": InferenceSchedule,
        }
        for name, enum_type in enum_types.items():
            object.__setattr__(self, name, enum_type.normalize(getattr(self, name)))
        object.__setattr__(self, "substrate", ModelSemantics.normalize(self.substrate))
        for name in (
            "tie_jitter", "arc_refraction_charge", "state_refraction_charge",
            "successor_gain", "prediction_gain",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a finite nonnegative number")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
            object.__setattr__(self, name, value)
        if type(self.horizon) is not int or self.horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        if type(self.feature_register) is not bool:
            raise ValueError("feature_register must be boolean")
        if type(self.convergence_gate) is not bool:
            raise ValueError("convergence_gate must be boolean")
        if self.tie_jitter == 0:
            expected_tie = TieBreakRule.LOWEST_NEURON_ID
        else:
            expected_tie = TieBreakRule.DETERMINISTIC_HASH_JITTER
        if self.substrate.default_tie_break is not expected_tie:
            raise ValueError(
                "tie_jitter and substrate.default_tie_break describe different rules"
            )
        if self.organ is OrganKind.ASSEMBLY_MEMORY:
            if any((self.state_code is not StateCode.NONE, self.horizon,
                    self.successor_gain != 1.0, self.prediction_gain,
                    self.feature_register)):
                raise ValueError("assembly memory cannot claim transducer state features")
        elif self.organ is OrganKind.ASSIGNED_STATE_FSM:
            if (self.state_code is not StateCode.ASSIGNED_BLOCKS or self.horizon
                    or self.successor_gain != 1.0 or self.prediction_gain
                    or self.feature_register or self.convergence_gate):
                raise ValueError("assigned-state FSM semantics require assigned blocks only")
        else:
            if self.state_code not in (
                StateCode.INDUCED_ASSEMBLY, StateCode.PREVIOUS_ARC_COPY,
            ):
                raise ValueError("sequence transducer requires induced or copied state")
            if self.convergence_gate:
                raise ValueError("sequence transducer does not implement convergence gating")

    @classmethod
    def normalize(cls, value: object) -> "OrganSemantics":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("organ_semantics must be OrganSemantics or a mapping")
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
            raise ValueError("invalid organ_semantics mapping: " + "; ".join(details))
        return cls(**dict(value))

    def to_dict(self) -> dict[str, object]:
        return {
            "organ": self.organ.value,
            "substrate": self.substrate.to_dict(),
            "state_code": self.state_code.value,
            "training_schedule": self.training_schedule.value,
            "inference_schedule": self.inference_schedule.value,
            "tie_jitter": self.tie_jitter,
            "arc_refraction_charge": self.arc_refraction_charge,
            "state_refraction_charge": self.state_refraction_charge,
            "horizon": self.horizon,
            "successor_gain": self.successor_gain,
            "prediction_gain": self.prediction_gain,
            "feature_register": self.feature_register,
            "convergence_gate": self.convergence_gate,
        }

    def mismatch(self, actual: "OrganSemantics") -> dict[str, tuple[object, object]]:
        actual = self.normalize(actual)
        expected_wire, actual_wire = self.to_dict(), actual.to_dict()
        return {
            name: (expected_wire[name], actual_wire[name])
            for name in expected_wire
            if expected_wire[name] != actual_wire[name]
        }


def _hashed_substrate(*, zero_or_size: bool, tie_jitter: float,
                      norm_init: bool, synaptic_scaling: bool,
                      w_max: float | None) -> ModelSemantics:
    if type(zero_or_size) is not bool or type(norm_init) is not bool:
        raise ValueError("stimulus and normalization switches must be boolean")
    if type(synaptic_scaling) is not bool:
        raise ValueError("synaptic_scaling must be boolean")
    if synaptic_scaling and not norm_init:
        raise ValueError("column scaling requires inverse-indegree initialization")
    if (isinstance(tie_jitter, bool) or not isinstance(tie_jitter, numbers.Real)
            or not math.isfinite(tie_jitter) or tie_jitter < 0):
        raise ValueError("tie_jitter must be a finite nonnegative number")
    normalization = (
        NormalizationMode.INVERSE_INDEGREE_WITH_COLUMN_SCALING
        if synaptic_scaling else
        NormalizationMode.INVERSE_INDEGREE if norm_init else
        NormalizationMode.NONE
    )
    return ModelSemantics(
        connectome=ConnectomeMode.FIXED_HASH_REGENERATED,
        candidate_domain=CandidateDomain.ALL_NEURONS,
        stimulus_drive=(
            StimulusDriveLaw.ZERO_OR_SIZE_AFFERENT_COUNT if zero_or_size
            else StimulusDriveLaw.FIXED_BERNOULLI_AFFERENT_COUNT
        ),
        default_tie_break=(
            TieBreakRule.LOWEST_NEURON_ID if tie_jitter == 0
            else TieBreakRule.DETERMINISTIC_HASH_JITTER
        ),
        arithmetic=ArithmeticMode.FLOAT32,
        normalization=normalization,
        plasticity=(PlasticityRule.MULTIPLICATIVE_CLIPPED
                    if w_max is not None else PlasticityRule.MULTIPLICATIVE_UNBOUNDED),
        weight_ceiling=w_max,
    )


@dataclass(frozen=True)
class ExecutionSemantics:
    """A strict, discriminated collection of model profiles used by one run."""

    kind: ExecutionKind
    profiles: Mapping[str, ModelSemantics | OrganSemantics]

    def __post_init__(self):
        object.__setattr__(self, "kind", ExecutionKind.normalize(self.kind))
        if not isinstance(self.profiles, Mapping) or not self.profiles:
            raise ValueError("execution semantics require at least one named profile")
        normalized = {}
        for name, value in self.profiles.items():
            if (not isinstance(name, str) or not name
                    or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in name)):
                raise ValueError("semantic profile names must be nonempty simple names")
            profile = (ModelSemantics.normalize(value) if self.kind is ExecutionKind.BRAIN
                       else OrganSemantics.normalize(value))
            normalized[name] = profile
        if self.kind is ExecutionKind.BRAIN and set(normalized) != {"default"}:
            raise ValueError("Brain execution semantics require exactly the default profile")
        object.__setattr__(
            self, "profiles", MappingProxyType(dict(sorted(normalized.items())))
        )

    @classmethod
    def normalize(cls, value: object) -> "ExecutionSemantics":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"kind", "profiles"}:
            raise ValueError("execution_semantics requires exactly kind and profiles")
        return cls(kind=value["kind"], profiles=value["profiles"])

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "profiles": {
                name: profile.to_dict() for name, profile in self.profiles.items()
            },
        }
def describe_assembly_memory(*, w_max: float | None = 20.0,
                             norm_init: bool = True,
                             synaptic_scaling: bool = False,
                             strength: float = 0.5,
                             beta: float = 0.1,
                             gate: bool = False) -> OrganSemantics:
    """Describe the transition relation implemented by ``AssemblyMemory``."""
    for name, value in (("strength", strength), ("beta", beta)):
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number")
    return OrganSemantics(
        organ=OrganKind.ASSEMBLY_MEMORY,
        substrate=_hashed_substrate(
            zero_or_size=True, tie_jitter=0.0, norm_init=norm_init,
            synaptic_scaling=synaptic_scaling, w_max=w_max,
        ),
        state_code=StateCode.NONE,
        training_schedule=TrainingSchedule.STIMULUS_PLUS_RECURRENCE,
        inference_schedule=InferenceSchedule.FROZEN_RECURRENT_COMPLETION,
        tie_jitter=0.0,
        arc_refraction_charge=float(strength) * float(beta),
        state_refraction_charge=0.0,
        convergence_gate=gate,
    )


def describe_hashed_arc_fsm(*, w_max: float | None = 20.0,
                            norm_init: bool = False,
                            refracted_strength: float = 0.1,
                            tie_jitter: float = 0.0,
                            zero_or_size: bool = False) -> OrganSemantics:
    """Describe the transition relation implemented by ``HashedArcFSM``."""
    return OrganSemantics(
        organ=OrganKind.ASSIGNED_STATE_FSM,
        substrate=_hashed_substrate(
            zero_or_size=zero_or_size, tie_jitter=tie_jitter,
            norm_init=norm_init, synaptic_scaling=False, w_max=w_max,
        ),
        state_code=StateCode.ASSIGNED_BLOCKS,
        training_schedule=TrainingSchedule.TEACHER_FORCED_TRANSITION,
        inference_schedule=InferenceSchedule.FROZEN_STATE_ADVANCING_TRANSITION,
        tie_jitter=tie_jitter,
        arc_refraction_charge=refracted_strength,
        state_refraction_charge=0.0,
    )


def describe_hashed_transducer(*, w_max: float | None = 20.0,
                               norm_init: bool = True,
                               refracted_strength: float = 0.1,
                               state_refracted_strength: float = 0.0,
                               tie_jitter: float = 1e-6,
                               zero_or_size: bool = True,
                               horizon: int = 0,
                               successor_gain: float = 1.0,
                               state_mode: str = "induced",
                               predict_gain: float = 0.0,
                               feature_register: bool = False) -> OrganSemantics:
    """Describe the transition relation implemented by ``HashedTransducer``."""
    if state_mode == "induced":
        state_code = StateCode.INDUCED_ASSEMBLY
    elif state_mode == "copy":
        state_code = StateCode.PREVIOUS_ARC_COPY
    else:
        raise ValueError("state_mode must be 'induced' or 'copy'")
    return OrganSemantics(
        organ=OrganKind.SEQUENCE_TRANSDUCER,
        substrate=_hashed_substrate(
            zero_or_size=zero_or_size, tie_jitter=tie_jitter,
            norm_init=norm_init, synaptic_scaling=False, w_max=w_max,
        ),
        state_code=state_code,
        training_schedule=TrainingSchedule.GROUNDED_TEACHER_FORCED_TRANSDUCTION,
        inference_schedule=InferenceSchedule.FROZEN_STATE_ADVANCING_EMISSION,
        tie_jitter=tie_jitter,
        arc_refraction_charge=refracted_strength,
        state_refraction_charge=state_refracted_strength,
        horizon=horizon,
        successor_gain=successor_gain,
        prediction_gain=predict_gain,
        feature_register=feature_register,
    )


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
