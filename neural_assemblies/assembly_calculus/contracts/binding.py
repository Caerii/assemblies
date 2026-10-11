"""Binding plans and contracts: binding, its read, input drive, binding strength, source
binding, binding recall.

Part of neural_assemblies.assembly_calculus.contracts."""
from dataclasses import dataclass
from numbers import Integral


from ..assembly import Assembly

from .contract import OperationContract
from .schedule import _explicit_bool, _positive_rounds, _require_name


@dataclass(frozen=True)
class BindingPlan:
    """Immutable schedule for storing one source assembly in a shared area."""

    source_area: str
    target_area: str
    project_rounds: int = 10
    tail_rounds: int = 1
    fix_source: bool = True

    def __post_init__(self) -> None:
        _require_name("source_area", self.source_area)
        _require_name("target_area", self.target_area)
        if self.source_area == self.target_area:
            raise ValueError("bind requires distinct source and target areas")
        if isinstance(self.project_rounds, bool) or not isinstance(self.project_rounds, Integral) or self.project_rounds < 1:
            raise ValueError("bind project_rounds must be a positive integer")
        if isinstance(self.tail_rounds, bool) or not isinstance(self.tail_rounds, Integral) or self.tail_rounds < 0:
            raise ValueError("bind tail_rounds must be a nonnegative integer")
        _explicit_bool("fix_source", self.fix_source)
        object.__setattr__(self, "project_rounds", int(self.project_rounds))
        object.__setattr__(self, "tail_rounds", int(self.tail_rounds))

    def preflight(self, brain) -> None:
        for label, area in (("source_area", self.source_area), ("target_area", self.target_area)):
            if area not in brain.areas:
                raise KeyError(f"bind {label} is unknown: {area!r}")


@dataclass(frozen=True)
class BindingReadPlan:
    """Immutable read-only traversal schedule for one stored binding."""

    source_area: str
    target_area: str
    tail_rounds: int = 1

    def __post_init__(self) -> None:
        _require_name("source_area", self.source_area)
        _require_name("target_area", self.target_area)
        if self.source_area == self.target_area:
            raise ValueError("binding read requires distinct source and target areas")
        if isinstance(self.tail_rounds, bool) or not isinstance(self.tail_rounds, Integral) or self.tail_rounds < 0:
            raise ValueError("binding read tail_rounds must be a nonnegative integer")
        object.__setattr__(self, "tail_rounds", int(self.tail_rounds))

    def preflight(self, brain) -> None:
        for label, area in (("source", self.source_area), ("target", self.target_area)):
            if area not in brain.areas:
                raise KeyError(f"binding read {label} area is unknown: {area!r}")


@dataclass(frozen=True)
class InputDrivePlan:
    """Immutable multi-target drive observation schedule."""

    sources: tuple[str, ...]
    target_areas: tuple[str, ...]
    metric: str = "pre_kwta"

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("input drive requires at least one source area")
        if not isinstance(self.target_areas, tuple) or not self.target_areas:
            raise ValueError("input drive requires at least one target area")
        for label, values in (("sources", self.sources), ("target_areas", self.target_areas)):
            if any(not isinstance(name, str) or not name for name in values):
                raise ValueError(f"input drive {label} must contain nonempty names")
            if len(set(values)) != len(values):
                raise ValueError(f"input drive {label} must be distinct")
        if self.metric not in {"pre_kwta", "winners"}:
            raise ValueError("input drive metric must be 'pre_kwta' or 'winners'")

    def preflight(self, brain) -> None:
        unknown_sources = [name for name in self.sources if name not in brain.areas]
        unknown_targets = [name for name in self.target_areas if name not in brain.areas]
        if unknown_sources:
            raise KeyError(f"input_drive source area(s) are unknown: {unknown_sources!r}")
        if unknown_targets:
            raise KeyError(f"input_drive target area(s) are unknown: {unknown_targets!r}")


@dataclass(frozen=True)
class BindingStrengthPlan:
    """Immutable overlap readout against a stored target assembly."""

    sources: tuple[str, ...]
    target_area: str
    target_assembly: Assembly

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("binding strength requires at least one source area")
        if any(not isinstance(name, str) or not name for name in self.sources):
            raise ValueError("binding strength sources must be nonempty names")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("binding strength sources must be distinct")
        _require_name("target_area", self.target_area)
        if not isinstance(self.target_assembly, Assembly):
            raise TypeError("binding strength target_assembly must be an Assembly")
        if self.target_assembly.area != self.target_area:
            raise ValueError("binding strength target_assembly belongs to another area")

    def preflight(self, brain) -> None:
        unknown = [name for name in (*self.sources, self.target_area)
                   if name not in brain.areas]
        if unknown:
            raise KeyError(f"binding strength area name(s) are unknown: {unknown!r}")


@dataclass(frozen=True)
class SourceBindingPlan:
    """Immutable schedule for multi-source teacher-driven binding."""

    sources: tuple[str, ...]
    target_area: str
    teachers: tuple[str, ...] = ()
    rounds: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("source binding requires at least one source area")
        if any(not isinstance(name, str) or not name for name in self.sources):
            raise ValueError("source binding sources must be nonempty names")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("source binding sources must be distinct")
        _require_name("target_area", self.target_area)
        if not isinstance(self.teachers, tuple):
            raise ValueError("source binding teachers must be a tuple")
        if any(not isinstance(name, str) or not name for name in self.teachers):
            raise ValueError("source binding teachers must be nonempty names")
        if len(set(self.teachers)) != len(self.teachers):
            raise ValueError("source binding teachers must be distinct")
        if self.target_area in self.sources or self.target_area in self.teachers:
            raise ValueError("source binding target must be distinct from sources and teachers")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))

    def preflight(self, brain) -> None:
        unknown = [name for name in (*self.sources, *self.teachers, self.target_area)
                   if name not in brain.areas]
        if unknown:
            raise KeyError(f"source binding area name(s) are unknown: {unknown!r}")


@dataclass(frozen=True)
class BindingRecallPlan:
    """Immutable readout schedule for a multi-source binding."""

    sources: tuple[str, ...]
    target_area: str
    clear_target: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("binding recall requires at least one source area")
        if any(not isinstance(name, str) or not name for name in self.sources):
            raise ValueError("binding recall sources must be nonempty names")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("binding recall sources must be distinct")
        _require_name("target_area", self.target_area)
        _explicit_bool("clear_target", self.clear_target)

    def preflight(self, brain) -> None:
        unknown_sources = [name for name in self.sources if name not in brain.areas]
        if unknown_sources:
            raise KeyError(f"binding recall source area(s) are unknown: {unknown_sources!r}")
        if self.target_area not in brain.areas:
            raise KeyError(f"binding recall target area is unknown: {self.target_area!r}")


BINDING_CONTRACT = OperationContract(
    operation_id="binding-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-binding",
    plan_type=BindingPlan,
    inputs=("brain", "source_area", "target_area", "source_assembly", "source_stimulus", "project_rounds", "tail_rounds", "fix_source"),
    reads=("source assembly or live source winners", "source-to-target fiber", "target recurrent fiber"),
    mutates=("target winners", "binding weights", "temporary source clamp", "engine history"),
    regime=("distinct registered source and shared target areas", "one explicit source of activity", "feed-forward round then optional recurrent tail"),
    observed_outcome=("bound target neuron-ID snapshot",),
    failure_conditions=("unknown areas", "missing source activity", "invalid schedule", "stale injected snapshot"),
    constructed_controls=(
        "neural_assemblies/tests/test_bind_input_contract.py::test_bind_rejects_empty_implicit_source",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_bind_input_contract.py::test_bind_rejects_ambiguous_schedule_before_source_resolution",
    ),
)


BINDING_READ_CONTRACT = OperationContract(
    operation_id="binding-read-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-binding-read",
    plan_type=BindingReadPlan,
    inputs=("brain", "source_area", "target_area", "source_assembly", "tail_rounds"),
    reads=("source snapshot", "source-to-target weights", "target winners"),
    mutates=("nothing persistent; activity is restored by read_only",),
    regime=("distinct source and target areas", "plasticity and recruitment disabled", "optional recurrent tail"),
    observed_outcome=("target neuron-ID snapshot",),
    failure_conditions=("unknown areas", "invalid tail schedule", "stale source snapshot"),
    constructed_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_completion_observation_modes_have_distinct_mutation_contracts",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_reciprocal_plan_rejects_a_self_projection",
    ),
)


INPUT_DRIVE_CONTRACT = OperationContract(
    operation_id="input-drive-v1",
    specification="neural_assemblies/ir/VERIFICATION.md#contract-pre-kwta-observation",
    plan_type=InputDrivePlan,
    inputs=("brain", "sources", "target_areas", "source_assemblies", "metric"),
    reads=("source winners", "source-to-target weights", "pre-k-WTA or winner drive"),
    mutates=("nothing persistent; activity is restored by probe scope",),
    regime=("nonempty distinct source and target lists", "one shared projection", "explicit drive metric"),
    observed_outcome=("per-target comparable drive scores",),
    failure_conditions=("unknown areas", "empty or duplicate topology", "invalid metric", "no active source"),
    constructed_controls=(
        "neural_assemblies/tests/test_binding_area_contract.py::"
        "test_input_drive_rejects_unknown_areas_instead_of_returning_empty_mapping",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_binding_area_contract.py::"
        "test_input_drive_rejects_unknown_areas_instead_of_returning_empty_mapping",
    ),
)


BINDING_STRENGTH_CONTRACT = OperationContract(
    operation_id="binding-strength-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-binding-strength",
    plan_type=BindingStrengthPlan,
    inputs=("brain", "sources", "target_area", "target_assembly", "source_assemblies"),
    reads=("source winners", "binding pathway", "target Assembly neuron IDs"),
    mutates=("nothing persistent; delegates to read-only recall",),
    regime=("nonempty active source", "target snapshot in target area", "stable neuron-ID overlap"),
    observed_outcome=("bounded target recovery overlap",),
    failure_conditions=("unknown or empty sources", "target snapshot area mismatch", "inactive source"),
    constructed_controls=(
        "neural_assemblies/tests/test_bind_strength_contract.py::"
        "test_bind_strength_rejects_snapshot_from_wrong_area",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_bind_strength_contract.py::"
        "test_bind_strength_rejects_inactive_source",
    ),
)


SOURCE_BINDING_CONTRACT = OperationContract(
    operation_id="source-binding-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-source-binding",
    plan_type=SourceBindingPlan,
    inputs=("brain", "sources", "target_area", "teachers", "source_assemblies", "rounds"),
    reads=("source and teacher winners", "target fibers", "plasticity state"),
    mutates=("target winners", "source-to-target weights", "engine history"),
    regime=("one or more active source areas", "optional teacher co-drive", "free target competition"),
    observed_outcome=("boolean indicating whether a pairing was applied",),
    failure_conditions=("unknown areas", "duplicate source or teacher names", "invalid rounds", "no live source activity"),
    constructed_controls=(
        "neural_assemblies/tests/test_binding_operator_contract.py::"
        "test_bind_rejects_unknown_area_names",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_binding_operator_contract.py::"
        "test_bind_rejects_ambiguous_round_schedule",
    ),
)


BINDING_RECALL_CONTRACT = OperationContract(
    operation_id="binding-recall-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-binding-recall",
    plan_type=BindingRecallPlan,
    inputs=("brain", "sources", "target_area", "source_assemblies", "clear_target"),
    reads=("source winners", "source-to-target weights", "target winners"),
    mutates=("temporary activity only under read-only scope",),
    regime=("at least one active source", "plasticity and recruitment disabled", "optional target clearing"),
    observed_outcome=("target Assembly snapshot or no result",),
    failure_conditions=("unknown areas", "duplicate source names", "invalid clear_target flag", "no active source"),
    constructed_controls=(
        "neural_assemblies/tests/test_binding_area_contract.py::"
        "test_recall_rejects_unknown_area_instead_of_returning_none",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_binding_area_contract.py::"
        "test_recall_rejects_unknown_area_instead_of_returning_none",
    ),
)
