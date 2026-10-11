"""Association-family plans and contracts: association, merge, separation.

Part of neural_assemblies.assembly_calculus.contracts."""
from dataclasses import dataclass
from numbers import Integral
from typing import cast



from .contract import OperationContract
from .schedule import ProjectionStep, _execute_schedule, _explicit_bool, _positive_rounds, _require_name, _schedule


@dataclass(frozen=True)
class AssociationPlan:
    """Validated sequential-pathway and joint-coactivation schedule.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-association
    """

    source_a: str
    source_b: str
    target: str
    stim_a: str | None = None
    stim_b: str | None = None
    rounds: int = 10
    cofire_rounds: int | None = None

    def __post_init__(self) -> None:
        names = (
            ("source_a", self.source_a),
            ("source_b", self.source_b),
            ("target", self.target),
        )
        for label, value in names:
            _require_name(label, value)
        if len({value for _, value in names}) != len(names):
            raise ValueError("association requires three distinct areas")
        if (self.stim_a is None) != (self.stim_b is None):
            raise ValueError("association stimuli must be both present or both absent")
        for label, value in (("stim_a", self.stim_a), ("stim_b", self.stim_b)):
            if value is not None:
                _require_name(label, value)
        if self.stim_a is not None and self.stim_a == self.stim_b:
            raise ValueError("association requires distinct source stimuli")
        rounds = _positive_rounds(self.rounds)
        cofire = rounds if self.cofire_rounds is None else self.cofire_rounds
        if (
            isinstance(cofire, bool)
            or not isinstance(cofire, Integral)
            or cofire < 0
        ):
            raise ValueError("cofire_rounds must be a nonnegative integer or None")
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "cofire_rounds", int(cofire))

    @property
    def fix_sources(self) -> bool:
        return self.stim_a is None

    def _single_source_steps(
        self, source: str, stimulus: str | None,
    ) -> tuple[ProjectionStep, ...]:
        stimuli = ((stimulus, (source,)),) if stimulus is not None else ()
        source_targets = (self.target,) if self.fix_sources else (source, self.target)
        steps = []
        for index in range(self.rounds):
            fibers = ((source, source_targets),)
            if index:
                fibers += ((self.target, (self.target,)),)
            steps.append(ProjectionStep(stimuli=stimuli, fibers=fibers))
        return tuple(steps)

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        joint_stimuli = () if self.fix_sources else (
            (cast(str, self.stim_a), (self.source_a,)),
            (cast(str, self.stim_b), (self.source_b,)),
        )
        source_a_targets = (
            (self.target,) if self.fix_sources else (self.source_a, self.target)
        )
        source_b_targets = (
            (self.target,) if self.fix_sources else (self.source_b, self.target)
        )
        joint = ProjectionStep(
            stimuli=joint_stimuli,
            fibers=(
                (self.source_a, source_a_targets),
                (self.source_b, source_b_targets),
                (self.target, (self.target,)),
            ),
        )
        return (
            self._single_source_steps(self.source_a, self.stim_a)
            + self._single_source_steps(self.source_b, self.stim_b)
            + (joint,) * self.cofire_count
        )

    @property
    def cofire_count(self) -> int:
        """Normalized cofire count after validation."""
        return int(self.cofire_rounds or 0)

    def preflight(self, brain) -> None:
        for name in (self.source_a, self.source_b, self.target):
            if name not in brain.areas:
                raise IndexError(f"Not in brain.areas: {name}")
        for stimulus in (self.stim_a, self.stim_b):
            if stimulus is not None and stimulus not in brain.stimuli:
                raise IndexError(f"Not in brain.stimuli: {stimulus}")
        if self.fix_sources:
            empty = [
                name for name in (self.source_a, self.source_b)
                if len(brain.areas[name].winners) == 0
            ]
            if empty:
                raise ValueError(f"association requires active fixed sources: {empty}")

    def execute_steps(self, brain) -> None:
        self.preflight(brain)
        _execute_schedule(self.steps, brain)


_UNSTIMULATED_SOURCE_MODES = frozenset({"require-fixed", "fix-current", "evolving"})


@dataclass(frozen=True)
class MergePlan:
    """Validated simultaneous two-parent merge schedule.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-merge
    """

    source_a: str
    source_b: str
    target: str
    stim_a: str | None = None
    stim_b: str | None = None
    rounds: int = 10
    parent_self: bool = True
    target_self: bool = True
    back_project: bool = True
    unstimulated_source_mode: str | None = None

    def __post_init__(self) -> None:
        names = (
            ("source_a", self.source_a),
            ("source_b", self.source_b),
            ("target", self.target),
        )
        for label, value in names:
            _require_name(label, value)
        if len({value for _, value in names}) != len(names):
            raise ValueError("merge requires three distinct areas")
        for label, value in (("stim_a", self.stim_a), ("stim_b", self.stim_b)):
            if value is not None:
                _require_name(label, value)
        if self.stim_a is not None and self.stim_a == self.stim_b:
            raise ValueError("merge requires distinct parent stimuli")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        for label in ("parent_self", "target_self", "back_project"):
            _explicit_bool(label, getattr(self, label))

        partial = (self.stim_a is None) != (self.stim_b is None)
        mode = self.unstimulated_source_mode
        if partial:
            if mode not in _UNSTIMULATED_SOURCE_MODES:
                raise ValueError(
                    "partial-stimulus merge requires unstimulated_source_mode "
                    "'require-fixed', 'fix-current', or 'evolving'"
                )
        elif mode is not None:
            raise ValueError(
                "unstimulated_source_mode is only valid for a partial-stimulus merge"
            )

    @property
    def unstimulated_source(self) -> str | None:
        if self.stim_a is None and self.stim_b is not None:
            return self.source_a
        if self.stim_b is None and self.stim_a is not None:
            return self.source_b
        return None

    @property
    def fixed_sources(self) -> tuple[str, ...]:
        if self.stim_a is None and self.stim_b is None:
            return (self.source_a, self.source_b)
        source = self.unstimulated_source
        if self.unstimulated_source_mode == "fix-current" and source is not None:
            return (source,)
        return ()

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        stimuli = tuple(
            (stimulus, (source,))
            for source, stimulus in (
                (self.source_a, self.stim_a),
                (self.source_b, self.stim_b),
            )
            if stimulus is not None
        )
        def parent_targets(source: str) -> tuple[str, ...]:
            return (source, self.target) if self.parent_self else (self.target,)
        parents = (
            (self.source_a, parent_targets(self.source_a)),
            (self.source_b, parent_targets(self.source_b)),
        )
        target_targets = (
            ((self.target,) if self.target_self else ())
            + ((self.source_a, self.source_b) if self.back_project else ())
        )
        first = ProjectionStep(stimuli=stimuli, fibers=parents)
        tail_fibers = parents + (
            ((self.target, target_targets),) if target_targets else ()
        )
        tail = ProjectionStep(stimuli=stimuli, fibers=tail_fibers)
        return _schedule(first, tail, self.rounds)

    def preflight(self, brain) -> None:
        for name in (self.source_a, self.source_b, self.target):
            if name not in brain.areas:
                raise IndexError(f"Not in brain.areas: {name}")
        for stimulus in (self.stim_a, self.stim_b):
            if stimulus is not None and stimulus not in brain.stimuli:
                raise IndexError(f"Not in brain.stimuli: {stimulus}")
        unstimulated = [
            source for source, stimulus in (
                (self.source_a, self.stim_a),
                (self.source_b, self.stim_b),
            )
            if stimulus is None
        ]
        empty = [name for name in unstimulated if len(brain.areas[name].winners) == 0]
        if empty:
            raise ValueError(f"merge requires active unstimulated sources: {empty}")
        source = self.unstimulated_source
        if source is not None and self.unstimulated_source_mode == "require-fixed":
            if not brain.areas[source].fixed_assembly:
                raise ValueError(f"merge requires source {source!r} to be fixed")
        if source is not None and self.unstimulated_source_mode == "evolving":
            if brain.areas[source].fixed_assembly:
                raise ValueError(f"merge requires source {source!r} to be evolving")

    def execute_steps(self, brain) -> None:
        self.preflight(brain)
        _execute_schedule(self.steps, brain)


@dataclass(frozen=True)
class SeparationPlan:
    """Validated two-stimulus separation measurement schedule.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-separation
    """

    stim_a: str
    stim_b: str
    target: str
    rounds: int = 10

    def __post_init__(self) -> None:
        for label, value in (("stim_a", self.stim_a),
                             ("stim_b", self.stim_b),
                             ("target", self.target)):
            _require_name(label, value)
        if self.stim_a == self.stim_b:
            raise ValueError("separate requires distinct stimuli")
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))

    def preflight(self, brain) -> None:
        for stimulus in (self.stim_a, self.stim_b):
            if stimulus not in brain.stimuli:
                raise KeyError(f"separate stimulus is unknown: {stimulus!r}")
        if self.target not in brain.areas:
            raise KeyError(f"separate target area is unknown: {self.target!r}")


ASSOCIATION_CONTRACT = OperationContract(
    operation_id="association-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-association",
    plan_type=AssociationPlan,
    inputs=(
        "brain", "source_a", "source_b", "target", "stim_a", "stim_b",
        "rounds", "cofire_rounds",
    ),
    reads=("source winners", "optional source stimuli", "pathway weights"),
    mutates=("target winners", "participating weights", "engine history", "clamps"),
    regime=(
        "three distinct registered areas",
        "two distinct registered stimuli or two active fixed sources",
    ),
    observed_outcome=("final jointly-driven target neuron-ID snapshot",),
    failure_conditions=(
        "invalid phase schedule",
        "partial stimulus specification",
        "unknown topology",
        "empty fixed source",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_ac_conformance.py::"
        "test_association_grows_with_coactivation",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_association_preflight_rejects_before_the_first_mutation",
    ),
)


MERGE_CONTRACT = OperationContract(
    operation_id="merge-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-merge",
    plan_type=MergePlan,
    inputs=(
        "brain", "source_a", "source_b", "target", "stim_a", "stim_b",
        "rounds", "parent_self", "target_self", "back_project",
        "unstimulated_source_mode",
    ),
    reads=("parent winners", "optional parent stimuli", "participating weights"),
    mutates=("target winners", "participating weights", "engine history", "clamps"),
    regime=(
        "three distinct registered areas",
        "active unstimulated parents",
        "explicit mode for exactly one unstimulated parent",
    ),
    observed_outcome=("final jointly-driven target neuron-ID snapshot",),
    failure_conditions=(
        "invalid schedule switch",
        "ambiguous partial-stimulus protocol",
        "source clamp contradicts declared mode",
        "unknown topology",
        "empty unstimulated source",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_merge_back_projection_switch_changes_the_declared_schedule",
        "neural_assemblies/tests/test_ac_conformance.py::"
        "test_merge_creates_two_way_connectivity_with_bounded_support",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_merge_preflight_rejects_a_source_state_that_contradicts_its_mode",
    ),
)


SEPARATION_CONTRACT = OperationContract(
    operation_id="separation-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-separation",
    plan_type=SeparationPlan,
    inputs=("brain", "stim_a", "stim_b", "target", "rounds"),
    reads=("stimulus fibers", "target recurrent weights", "target winners"),
    mutates=("target winners", "target recurrent weights"),
    regime=("two distinct registered stimuli", "stimulus-driven target", "destructive recurrent-reset measurement"),
    observed_outcome=("two neuron-ID assembly snapshots", "normalized pairwise overlap"),
    failure_conditions=("identical stimuli", "unknown topology", "invalid rounds", "backend projection rejection"),
    constructed_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_separation_plan_rejects_identical_stimuli",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_separation_plan_preflight_rejects_unknown_topology",
    ),
)
