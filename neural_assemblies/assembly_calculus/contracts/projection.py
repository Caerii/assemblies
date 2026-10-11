"""Projection-family plans and contracts: activation, readout, fiber materialization,
projection, reciprocal projection, convergence.

Part of neural_assemblies.assembly_calculus.contracts."""
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Mapping

import numpy as np

from ..assembly import Assembly

from .contract import OperationContract
from .schedule import ProjectionStep, _execute_schedule, _explicit_bool, _positive_rounds, _require_name, _schedule


@dataclass(frozen=True)
class ActivationPlan:
    """Validated injection of a stable-neuron Assembly snapshot."""

    assembly: Assembly

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, Assembly):
            raise TypeError("assembly must be an Assembly snapshot")

    def preflight(self, brain) -> None:
        area_name = self.assembly.area
        if area_name not in brain.areas:
            raise ValueError(f"Unknown area {area_name!r}")
        area = brain.areas[area_name]
        ids = np.asarray(self.assembly.winners)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("assembly neuron IDs must be a one-dimensional integer array")
        if np.any(ids < 0) or np.any(ids >= area.n):
            raise ValueError("assembly neuron IDs must be within the area")


@dataclass(frozen=True)
class ReadoutPlan:
    """Validated pure decoder query over immutable Assembly snapshots."""

    assembly: Assembly
    lexicon: Mapping[str, Assembly]
    threshold: float = 0.7

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, Assembly):
            raise TypeError("readout assembly must be an Assembly snapshot")
        if not isinstance(self.lexicon, Mapping):
            raise TypeError("readout lexicon must be a mapping")
        if any(not isinstance(word, str) or not word for word in self.lexicon):
            raise ValueError("readout lexicon labels must be nonempty strings")
        if any(not isinstance(reference, Assembly) for reference in self.lexicon.values()):
            raise TypeError("readout lexicon values must be Assembly snapshots")
        if (isinstance(self.threshold, bool) or not isinstance(self.threshold, Real)
                or not math.isfinite(float(self.threshold))
                or not 0.0 <= float(self.threshold) <= 1.0):
            raise ValueError("readout threshold must be a finite real number in [0, 1]")
        object.__setattr__(self, "threshold", float(self.threshold))


@dataclass(frozen=True)
class FiberMaterializationPlan:
    """Validated allocation of a source-to-target fiber."""

    src_area: str
    dst_area: str
    src_assembly: Assembly | None = None

    def __post_init__(self) -> None:
        _require_name("src_area", self.src_area)
        _require_name("dst_area", self.dst_area)
        if self.src_assembly is not None:
            if not isinstance(self.src_assembly, Assembly):
                raise TypeError("src_assembly must be an Assembly snapshot or None")
            if self.src_assembly.area != self.src_area:
                raise ValueError("src_assembly belongs to another source area")

    def preflight(self, brain) -> None:
        if self.src_area not in brain.areas:
            raise KeyError(f"materialize_fiber source area is unknown: {self.src_area!r}")
        if self.dst_area not in brain.areas:
            raise KeyError(f"materialize_fiber target area is unknown: {self.dst_area!r}")


@dataclass(frozen=True)
class ProjectionPlan:
    """Validated schedule for the named stimulus-to-area operation.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-projection
    """

    stimulus: str
    target: str
    rounds: int = 10
    recurrent: bool = False

    def __post_init__(self) -> None:
        for label, value in (("stimulus", self.stimulus), ("target", self.target)):
            _require_name(label, value)
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        _explicit_bool("recurrent", self.recurrent)

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        stimulus = ((self.stimulus, (self.target,)),)
        first = ProjectionStep(stimuli=stimulus, fibers=())
        tail = ProjectionStep(
            stimuli=stimulus,
            fibers=((self.target, (self.target,)),) if self.recurrent else (),
        )
        return _schedule(first, tail, self.rounds)

    def execute(self, brain) -> None:
        """Preflight the named topology, then execute the declared steps."""
        if self.stimulus not in brain.stimuli:
            raise IndexError(f"Not in brain.stimuli: {self.stimulus}")
        if self.target not in brain.areas:
            raise IndexError(f"Not in brain.areas: {self.target}")
        _execute_schedule(self.steps, brain)


@dataclass(frozen=True)
class ReciprocalProjectionPlan:
    """Validated two-area copy and return-edge schedule.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-reciprocal-projection
    """

    source: str
    target: str
    rounds: int = 10
    fix_source: bool = True

    def __post_init__(self) -> None:
        for label, value in (("source", self.source), ("target", self.target)):
            _require_name(label, value)
        if self.source == self.target:
            raise ValueError("reciprocal projection requires distinct source and target")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        _explicit_bool("fix_source", self.fix_source)

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        first = ProjectionStep(stimuli=(), fibers=((self.source, (self.target,)),))
        tail = ProjectionStep(
            stimuli=(),
            fibers=(
                (self.source, (self.target,)),
                (self.target, (self.target, self.source)),
            ),
        )
        return _schedule(first, tail, self.rounds)

    def preflight(self, brain) -> None:
        for name in (self.source, self.target):
            if name not in brain.areas:
                raise IndexError(f"Not in brain.areas: {name}")
        if len(brain.areas[self.source].winners) == 0:
            raise ValueError("reciprocal projection requires an active source assembly")

    def execute_steps(self, brain) -> None:
        self.preflight(brain)
        _execute_schedule(self.steps, brain)


@dataclass(frozen=True)
class ConvergencePlan:
    """Immutable stopping schedule shared by convergent learning helpers."""

    max_epochs: int = 12
    project_rounds: int = 6
    stability_window: int = 2
    threshold: float = 0.90
    recurrent: bool = True

    def __post_init__(self) -> None:
        for label, value in (("max_epochs", self.max_epochs), ("project_rounds", self.project_rounds),
                             ("stability_window", self.stability_window)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if self.stability_window < 2:
            raise ValueError("stability_window must be at least two")
        if (isinstance(self.threshold, bool) or not isinstance(self.threshold, Real)
                or not math.isfinite(float(self.threshold))
                or not 0.0 <= float(self.threshold) <= 1.0):
            raise ValueError("convergence threshold must be a finite real number in [0, 1]")
        _explicit_bool("recurrent", self.recurrent)
        object.__setattr__(self, "max_epochs", int(self.max_epochs))
        object.__setattr__(self, "project_rounds", int(self.project_rounds))
        object.__setattr__(self, "stability_window", int(self.stability_window))
        object.__setattr__(self, "threshold", float(self.threshold))


CONVERGENCE_CONTRACT = OperationContract(
    operation_id="convergence-learning-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-convergence",
    plan_type=ConvergencePlan,
    inputs=("brain", "source", "target", "max_epochs", "project_rounds", "stability_window", "threshold", "recurrent"),
    reads=("source stimulus or pattern", "target winners", "afferent and recurrent weights"),
    mutates=("target winners", "enabled weights", "engine history"),
    regime=("validated finite source pattern or registered stimulus", "consecutive snapshot overlap threshold"),
    observed_outcome=("final assembly snapshot", "epochs used", "persistence"),
    failure_conditions=("unknown areas or stimulus", "invalid source pattern", "invalid schedule", "empty source activation"),
    constructed_controls=(
        "neural_assemblies/tests/test_learning_schedule_contract.py::"
        "test_learning_rejects_invalid_convergence",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_learning_schedule_contract.py::"
        "test_learning_rejects_invalid_convergence",
    ),
)


READOUT_CONTRACT = OperationContract(
    operation_id="fuzzy-readout-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-readout",
    plan_type=ReadoutPlan,
    inputs=("assembly snapshot", "lexicon", "threshold"),
    reads=("stable neuron-ID overlap against each lexicon snapshot",),
    mutates=("nothing; pure decoder observation",),
    regime=("finite threshold in [0, 1]", "deterministic lexical tie break"),
    observed_outcome=("best label or None when below threshold",),
    failure_conditions=("malformed snapshot", "malformed lexicon", "invalid threshold"),
    constructed_controls=(
        "neural_assemblies/tests/test_readout.py::test_readout_ties_are_independent_of_dictionary_order",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_readout.py::test_invalid_threshold_is_rejected",
    ),
)


FIBER_MATERIALIZATION_CONTRACT = OperationContract(
    operation_id="fiber-materialization-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-fiber-materialization",
    plan_type=FiberMaterializationPlan,
    inputs=("brain", "source area", "target area", "optional source snapshot"),
    reads=("source activity", "target fiber allocation state"),
    mutates=("target fiber columns", "temporary source/target activity"),
    regime=("known source and target areas", "frozen allocation with plasticity off"),
    observed_outcome=("boolean indicating whether source traffic was materialized",),
    failure_conditions=("unknown topology", "snapshot/source mismatch", "stale snapshot"),
    constructed_controls=(
        "neural_assemblies/tests/test_materialize_fiber_contract.py::"
        "test_materialize_fiber_rejects_unknown_topology",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_materialize_fiber_contract.py::"
        "test_materialize_fiber_reports_inactive_source_explicitly",
    ),
)


ACTIVATION_CONTRACT = OperationContract(
    operation_id="assembly-activation-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-activation",
    plan_type=ActivationPlan,
    inputs=("brain", "assembly snapshot"),
    reads=("stable neuron IDs", "target area index mapping"),
    mutates=("target area winners", "engine activity state"),
    regime=("snapshot belongs to target area", "IDs are valid in current population"),
    observed_outcome=("no return value; target activity is updated",),
    failure_conditions=("unknown area", "invalid neuron IDs", "stale snapshot mapping"),
    constructed_controls=(
        "neural_assemblies/tests/test_public_model_boundaries.py::"
        "test_explicit_activation_rejects_neuron_outside_area",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_public_model_boundaries.py::"
        "test_explicit_activation_rejects_neuron_outside_area",
    ),
)


PROJECTION_CONTRACT = OperationContract(
    operation_id="projection-v1",
    specification=(
        "docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-projection"
    ),
    plan_type=ProjectionPlan,
    inputs=("brain", "stimulus", "target", "rounds", "recurrent"),
    reads=("stimulus", "target", "afferent weights", "plasticity state"),
    mutates=("target winners", "enabled incoming weights", "engine history"),
    regime=("registered stimulus and target", "backend-defined model regime"),
    observed_outcome=("final target neuron-ID snapshot",),
    failure_conditions=(
        "invalid schedule",
        "unknown stimulus",
        "unknown target",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_operation_semantic_cards.py::"
        "test_p3_operation_owns_recurrence_schedule",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_topology_rejects_before_the_first_mutation",
    ),
)


RECIPROCAL_PROJECTION_CONTRACT = OperationContract(
    operation_id="reciprocal-projection-v1",
    specification=(
        "docs/reviews/whole-codebase/SEMANTIC_CARDS.md#"
        "contract-reciprocal-projection"
    ),
    plan_type=ReciprocalProjectionPlan,
    inputs=("brain", "source", "target", "rounds", "fix_source"),
    reads=("source winners", "forward and return fibers", "plasticity state"),
    mutates=("target winners", "participating weights", "engine history", "clamps"),
    regime=("distinct registered areas", "active source assembly"),
    observed_outcome=("final target neuron-ID snapshot",),
    failure_conditions=(
        "invalid schedule",
        "unknown area",
        "empty source assembly",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_pnas_roundtrip_contract.py::"
        "test_roundtrip_responds_to_learning_disabled_control",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_reciprocal_preflight_rejects_before_the_first_mutation",
    ),
)
