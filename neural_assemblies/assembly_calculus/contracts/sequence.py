"""Sequence plans and contracts: memorizing a sequence, recalling it in order.

Part of neural_assemblies.assembly_calculus.contracts."""
from dataclasses import dataclass
import math
from numbers import Integral, Real



from .contract import OperationContract
from .schedule import _positive_rounds, _require_name


@dataclass(frozen=True)
class OrderedRecallPlan:
    """Validated schedule for recurrent sequence readout with LRI.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-sequence-memory
    """

    area: str
    cue: str
    max_steps: int = 20
    convergence_threshold: float = 0.9
    rounds_per_step: int = 1
    novelty_threshold: float = 0.3

    def __post_init__(self) -> None:
        _require_name("area", self.area)
        _require_name("cue", self.cue)
        for label, value in (("max_steps", self.max_steps),
                             ("rounds_per_step", self.rounds_per_step)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
            object.__setattr__(self, label, int(value))
        for label, value in (("convergence_threshold", self.convergence_threshold),
                             ("novelty_threshold", self.novelty_threshold)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{label} must be a finite real number in [0, 1]")
            object.__setattr__(self, label, float(value))

    def preflight(self, brain) -> None:
        if self.area not in brain.areas:
            raise KeyError(f"ordered_recall area is unknown: {self.area!r}")
        if self.cue not in brain.stimuli:
            raise KeyError(f"ordered_recall cue stimulus is unknown: {self.cue!r}")
        if brain.areas[self.area].refractory_period == 0:
            raise ValueError(
                f"ordered_recall requires refractory_period > 0 for area {self.area!r}. "
                "Add the area with refractory_period=N to enable LRI."
            )


@dataclass(frozen=True)
class SequenceMemorizePlan:
    """Validated ordered-stimulus training schedule.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-sequence-memory
    """

    stimuli: tuple[str, ...]
    target: str
    rounds_per_step: int = 10
    repetitions: int = 1
    phase_b_ratio: float | None = None
    beta_boost: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stimuli, tuple) or not self.stimuli:
            raise ValueError("stimuli must be a nonempty ordered tuple")
        if any(not isinstance(name, str) or not name for name in self.stimuli):
            raise ValueError("stimuli must contain nonempty names")
        _require_name("target", self.target)
        for label, value in (("rounds_per_step", self.rounds_per_step),
                             ("repetitions", self.repetitions)):
            object.__setattr__(self, label, _positive_rounds(value))
        if self.phase_b_ratio is not None:
            if (isinstance(self.phase_b_ratio, bool)
                    or not isinstance(self.phase_b_ratio, Real)
                    or not math.isfinite(float(self.phase_b_ratio))
                    or not 0.0 <= float(self.phase_b_ratio) <= 1.0):
                raise ValueError("phase_b_ratio must be a finite real number in [0, 1]")
            object.__setattr__(self, "phase_b_ratio", float(self.phase_b_ratio))
        if self.beta_boost is not None:
            if (isinstance(self.beta_boost, bool)
                    or not isinstance(self.beta_boost, Real)
                    or not math.isfinite(float(self.beta_boost))
                    or float(self.beta_boost) < 0.0):
                raise ValueError("beta_boost must be a finite nonnegative real number")
            object.__setattr__(self, "beta_boost", float(self.beta_boost))

    def preflight(self, brain) -> None:
        if self.target not in brain.areas:
            raise KeyError(f"sequence_memorize target area is unknown: {self.target!r}")
        unknown = [name for name in self.stimuli if name not in brain.stimuli]
        if unknown:
            raise KeyError(f"sequence_memorize stimulus name(s) are unknown: {unknown!r}")


ORDERED_RECALL_CONTRACT = OperationContract(
    operation_id="ordered-recall-v1",
    specification=(
        "docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-sequence-memory"
    ),
    plan_type=OrderedRecallPlan,
    inputs=(
        "brain", "area", "cue", "max_steps", "known_assemblies",
        "convergence_threshold", "rounds_per_step", "novelty_threshold",
    ),
    reads=("cue stimulus", "area winners", "recurrent weights", "refractory state"),
    mutates=("area winners", "refractory state", "engine history"),
    regime=("registered area and cue", "positive refractory period", "ordered learned transitions"),
    observed_outcome=("ordered neuron-ID assembly snapshots", "explicit termination at cycle, novelty, or budget"),
    failure_conditions=(
        "invalid schedule", "unknown area or cue", "refractory period disabled",
        "malformed reference assemblies", "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_ordered_recall_plan_requires_lri",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_ordered_recall_plan_requires_lri",
    ),
)


SEQUENCE_MEMORIZE_CONTRACT = OperationContract(
    operation_id="sequence-memorize-v1",
    specification=(
        "docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-sequence-memory"
    ),
    plan_type=SequenceMemorizePlan,
    inputs=("brain", "stimuli", "target", "rounds_per_step", "repetitions", "phase_b_ratio", "beta_boost"),
    reads=("ordered stimuli", "target winners", "recurrent weights", "plasticity state"),
    mutates=("target winners", "transition weights", "engine history", "temporary beta state"),
    regime=("registered nonempty stimulus sequence", "explicit phase schedule", "backend model regime"),
    observed_outcome=("ordered neuron-ID assembly snapshots", "resolved training schedule"),
    failure_conditions=("scalar or empty stimuli", "unknown topology", "invalid schedule", "backend projection rejection"),
    constructed_controls=(
        "neural_assemblies/tests/test_sequence_memorize_contract.py::"
        "test_sequence_memorize_rejects_scalar_stimulus_input",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_sequence_memorize_contract.py::"
        "test_sequence_memorize_rejects_scalar_stimulus_input",
    ),
)
