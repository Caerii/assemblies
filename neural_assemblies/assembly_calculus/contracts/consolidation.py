"""Consolidation plans and contracts: consolidation, its protocol, context accumulation,
and one accumulation step.

Part of neural_assemblies.assembly_calculus.contracts."""
from dataclasses import dataclass
from numbers import Integral


from ..assembly import Assembly

from .contract import OperationContract
from .schedule import _explicit_bool, _positive_rounds, _require_name


@dataclass(frozen=True)
class ConsolidationPlan:
    """Immutable bidirectional sleep-replay schedule for two assemblies."""

    area_a: str
    assembly_a: Assembly
    area_b: str
    assembly_b: Assembly
    rounds: int = 5
    a_to_b: bool = True
    b_to_a: bool = True

    def __post_init__(self) -> None:
        _require_name("area_a", self.area_a)
        _require_name("area_b", self.area_b)
        if self.area_a == self.area_b:
            raise ValueError("consolidation requires distinct areas")
        if not isinstance(self.assembly_a, Assembly) or self.assembly_a.area != self.area_a:
            raise ValueError("assembly_a must belong to area_a")
        if not isinstance(self.assembly_b, Assembly) or self.assembly_b.area != self.area_b:
            raise ValueError("assembly_b must belong to area_b")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        _explicit_bool("a_to_b", self.a_to_b)
        _explicit_bool("b_to_a", self.b_to_a)
        if not self.a_to_b and not self.b_to_a:
            raise ValueError("consolidation requires at least one replay direction")

    def preflight(self, brain) -> None:
        for area in (self.area_a, self.area_b):
            if area not in brain.areas:
                raise KeyError(f"consolidation area is unknown: {area!r}")


@dataclass(frozen=True)
class ConsolidationProtocolPlan:
    """Immutable ordered replay protocol for generic consolidation."""

    steps: tuple
    passes: int = 1
    clear_activity: bool = True
    prepare_areas: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("consolidation requires a nonempty step tuple")
        if isinstance(self.passes, bool) or not isinstance(self.passes, Integral) or self.passes < 1:
            raise ValueError("consolidation passes must be a positive integer")
        _explicit_bool("clear_activity", self.clear_activity)
        _explicit_bool("prepare_areas", self.prepare_areas)
        object.__setattr__(self, "passes", int(self.passes))


@dataclass(frozen=True)
class ContextAccumulationPlan:
    """Immutable ordered word-to-context accumulation schedule."""

    word_steps: tuple[tuple[str, str], ...]
    context_area: str
    core_assemblies: tuple[Assembly | None, ...] | None = None
    rounds: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.word_steps, tuple) or not self.word_steps:
            raise ValueError("context accumulation requires a nonempty word schedule")
        if any(not isinstance(step, tuple) or len(step) != 2
               or any(not isinstance(name, str) or not name for name in step)
               for step in self.word_steps):
            raise ValueError("context accumulation steps must be (phon, core_area) tuples")
        _require_name("context_area", self.context_area)
        if self.core_assemblies is not None:
            if not isinstance(self.core_assemblies, tuple) or len(self.core_assemblies) != len(self.word_steps):
                raise ValueError("core_assemblies must match word_steps length")
            if any(assembly is not None and not isinstance(assembly, Assembly)
                   for assembly in self.core_assemblies):
                raise TypeError("core_assemblies entries must be Assembly or None")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))

    def preflight(self, brain) -> None:
        if self.context_area not in brain.areas:
            raise KeyError(f"context accumulation area is unknown: {self.context_area!r}")
        for phon, core_area in self.word_steps:
            if core_area not in brain.areas:
                raise KeyError(f"context accumulation core area is unknown: {core_area!r}")
            if self.core_assemblies is None and phon not in brain.stimuli:
                raise KeyError(f"context accumulation stimulus is unknown: {phon!r}")
        if self.core_assemblies is not None:
            for (_, core_area), assembly in zip(self.word_steps, self.core_assemblies, strict=True):
                if assembly is not None and assembly.area != core_area:
                    raise ValueError("context accumulation core assembly belongs to another area")


@dataclass(frozen=True)
class ContextAccumulationStepPlan:
    """Immutable one-word context transition schedule."""

    core_area: str
    context_area: str
    phon: str | None = None
    core_assembly: Assembly | None = None
    rounds: int = 10

    def __post_init__(self) -> None:
        _require_name("core_area", self.core_area)
        _require_name("context_area", self.context_area)
        if self.core_area == self.context_area:
            raise ValueError("context step requires distinct core and context areas")
        if self.phon is not None and (not isinstance(self.phon, str) or not self.phon):
            raise ValueError("context step phon must be a nonempty stimulus name or None")
        if self.core_assembly is not None:
            if not isinstance(self.core_assembly, Assembly):
                raise TypeError("context step core_assembly must be an Assembly or None")
            if self.core_assembly.area != self.core_area:
                raise ValueError("context step core_assembly belongs to another area")
        if self.phon is not None and self.core_assembly is not None:
            raise ValueError("context step requires exactly one source representation")
        if self.phon is None and self.core_assembly is None:
            raise ValueError("context step requires phon or core_assembly")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))

    def preflight(self, brain) -> None:
        for label, area in (("core", self.core_area), ("context", self.context_area)):
            if area not in brain.areas:
                raise KeyError(f"context step {label} area is unknown: {area!r}")
        if self.core_assembly is None and self.phon not in brain.stimuli:
            raise KeyError(f"context step stimulus is unknown: {self.phon!r}")


CONSOLIDATION_CONTRACT = OperationContract(
    operation_id="consolidation-pair-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-consolidation",
    plan_type=ConsolidationPlan,
    inputs=("brain", "area_a", "assembly_a", "area_b", "assembly_b", "rounds", "a_to_b", "b_to_a"),
    reads=("stored assembly snapshots", "forward and return fibers", "plasticity state"),
    mutates=("participating weights", "area winners", "engine history"),
    regime=("distinct areas with current assembly snapshots", "one or both explicit replay directions"),
    observed_outcome=("post-replay snapshots for both areas",),
    failure_conditions=("unknown areas", "snapshot/area mismatch", "invalid rounds or direction flags", "no replay direction"),
    constructed_controls=(
        "neural_assemblies/tests/test_engine_e2_overlap.py::test_init_reciprocal_connectome_and_consolidate_pair",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::test_consolidation_plan_rejects_empty_direction_before_mutation",
    ),
)


CONSOLIDATION_PROTOCOL_CONTRACT = OperationContract(
    operation_id="consolidation-protocol-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-consolidation-protocol",
    plan_type=ConsolidationProtocolPlan,
    inputs=("brain", "ordered replay steps", "passes", "clear_activity", "prepare_areas"),
    reads=("step source assemblies and stimuli", "existing area-to-area weights"),
    mutates=("replayed weights", "area activity", "area index mappings when preparation is enabled"),
    regime=("nonempty ordered protocol", "positive replay passes", "optional destructive area preparation"),
    observed_outcome=("set of strengthened pathway edges",),
    failure_conditions=("empty protocol", "invalid pass count or flags", "malformed replay step"),
    constructed_controls=(
        "neural_assemblies/tests/test_consolidation.py::"
        "test_consolidate_strengthens_pathway_without_reset",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_consolidation.py::"
        "test_consolidate_rejects_empty_protocol_or_invalid_passes",
    ),
)


CONTEXT_ACCUMULATION_CONTRACT = OperationContract(
    operation_id="context-accumulation-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-context-accumulation",
    plan_type=ContextAccumulationPlan,
    inputs=("brain", "ordered word steps", "context_area", "core_assemblies", "rounds"),
    reads=("phonological stimuli or core snapshots", "core and context winners", "context recurrence"),
    mutates=("core/context winners", "core-to-context weights", "engine history"),
    regime=("nonempty ordered schedule", "one context area", "fixed round budget"),
    observed_outcome=("final context Assembly snapshot",),
    failure_conditions=("empty/malformed schedule", "unknown topology", "stimulus or snapshot mismatch", "invalid rounds"),
    constructed_controls=(
        "neural_assemblies/tests/test_consolidation.py::"
        "test_accumulate_context_matches_manual_steps",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_consolidation.py::"
        "test_accumulate_context_rejects_empty_schedule",
    ),
)


CONTEXT_STEP_CONTRACT = OperationContract(
    operation_id="context-accumulation-step-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-context-accumulation-step",
    plan_type=ContextAccumulationStepPlan,
    inputs=("brain", "phon", "core_area", "context_area", "core_assembly", "rounds"),
    reads=("phonological stimulus or core snapshot", "core/context winners", "context recurrence"),
    mutates=("core/context winners", "core-to-context weights", "engine history"),
    regime=("exactly one source representation", "distinct core and context areas", "fixed round budget"),
    observed_outcome=("post-step context Assembly snapshot",),
    failure_conditions=("missing source", "unknown topology", "snapshot/area mismatch", "invalid rounds"),
    constructed_controls=(
        "neural_assemblies/tests/test_consolidation.py::"
        "test_accumulate_context_matches_manual_steps",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_consolidation.py::"
        "test_accumulate_context_step_rejects_missing_source",
    ),
)
