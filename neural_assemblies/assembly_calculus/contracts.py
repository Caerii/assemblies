"""Inspectable contracts and immutable schedules for calculus operations.

Specification: neural_assemblies/ir/VERIFICATION.md#contract-operation-objects
"""

from dataclasses import dataclass, is_dataclass
from contextlib import nullcontext
import math
from numbers import Integral, Real
import random
from types import MappingProxyType
from typing import Callable

import numpy as np

from .assembly import Assembly


def _require_name(label: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty name")


def _positive_rounds(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("rounds must be a positive integer")
    return int(value)


def _explicit_bool(label: str, value: object) -> None:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an explicit boolean")


@dataclass(frozen=True)
class ProjectionStep:
    """One simultaneous projection step in immutable tuple form."""

    stimuli: tuple[tuple[str, tuple[str, ...]], ...]
    fibers: tuple[tuple[str, tuple[str, ...]], ...]

    def stimuli_dict(self) -> dict[str, list[str]]:
        return {source: list(targets) for source, targets in self.stimuli}

    def fibers_dict(self) -> dict[str, list[str]]:
        return {source: list(targets) for source, targets in self.fibers}


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
        return (first,) + (tail,) * (self.rounds - 1)

    def execute(self, brain) -> None:
        """Preflight the named topology, then execute the declared steps."""
        if self.stimulus not in brain.stimuli:
            raise IndexError(f"Not in brain.stimuli: {self.stimulus}")
        if self.target not in brain.areas:
            raise IndexError(f"Not in brain.areas: {self.target}")
        for step in self.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())


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
        return (first,) + (tail,) * (self.rounds - 1)

    def preflight(self, brain) -> None:
        for name in (self.source, self.target):
            if name not in brain.areas:
                raise IndexError(f"Not in brain.areas: {name}")
        if len(brain.areas[self.source].winners) == 0:
            raise ValueError("reciprocal projection requires an active source assembly")

    def execute_steps(self, brain) -> None:
        self.preflight(brain)
        for step in self.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())


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
            (self.stim_a, (self.source_a,)),
            (self.stim_b, (self.source_b,)),
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
            + (joint,) * self.cofire_rounds
        )

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
        for step in self.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())


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
        return (first,) + (tail,) * (self.rounds - 1)

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
        for step in self.steps:
            brain.project(step.stimuli_dict(), step.fibers_dict())


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


_COMPLETION_OBSERVATION_MODES = frozenset({"plastic", "frozen", "read-only"})


@dataclass(frozen=True)
class PreparedCompletion:
    """A reference assembly and its exact sampled compact-index cue."""

    plan: "CompletionPlan"
    reference: Assembly
    entry_compact: tuple[int, ...]
    compact_cue: tuple[int, ...]
    brain_identity: int

    def inject_cue(self, brain) -> None:
        if id(brain) != self.brain_identity:
            raise ValueError("prepared completion belongs to a different brain")
        current = tuple(int(value) for value in brain.areas[self.plan.area].winners)
        if current != self.entry_compact:
            raise ValueError("completion source changed after cue preparation")
        brain.areas[self.plan.area].winners = np.asarray(
            self.compact_cue, dtype=np.uint32,
        )


@dataclass(frozen=True)
class CompletionPlan:
    """Validated partial-cue construction and recurrent recovery schedule.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-completion
    """

    area: str
    fraction: float = 0.5
    rounds: int = 5
    seed: int | None = None
    observation_mode: str | None = None

    def __post_init__(self) -> None:
        _require_name("area", self.area)
        if (
            isinstance(self.fraction, bool)
            or not isinstance(self.fraction, Real)
            or not math.isfinite(float(self.fraction))
            or not 0 < self.fraction <= 1
        ):
            raise ValueError("fraction must be finite and in (0, 1]")
        object.__setattr__(self, "fraction", float(self.fraction))
        object.__setattr__(self, "rounds", _positive_rounds(self.rounds))
        if (
            self.seed is None
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
        ):
            raise ValueError("seed must be an explicit integer")
        object.__setattr__(self, "seed", int(self.seed))
        if self.observation_mode not in _COMPLETION_OBSERVATION_MODES:
            raise ValueError(
                "observation_mode must be 'plastic', 'frozen', or 'read-only'"
            )

    @property
    def steps(self) -> tuple[ProjectionStep, ...]:
        step = ProjectionStep(stimuli=(), fibers=((self.area, (self.area,)),))
        return (step,) * self.rounds

    def observation_scope(self, brain):
        """Return the one state policy named by this protocol."""
        if self.observation_mode == "plastic":
            return nullcontext(brain)
        if self.observation_mode == "frozen":
            return brain.frozen()
        return brain.read_only()

    def prepare(self, brain) -> PreparedCompletion:
        if self.area not in brain.areas:
            raise IndexError(f"Not in brain.areas: {self.area}")
        reference = Assembly.from_area(brain, self.area)
        compact = tuple(int(value) for value in brain.areas[self.area].winners)
        if not compact:
            raise ValueError(f"area {self.area!r} has no assembly to complete")
        if len(compact) != len(reference):
            raise ValueError("completion reference and compact cue source disagree")
        cue_size = int(len(reference) * self.fraction)
        if cue_size < 1:
            raise ValueError(
                "fraction retains no neurons at the current assembly size"
            )
        cue = tuple(random.Random(self.seed).sample(compact, cue_size))
        return PreparedCompletion(
            plan=self,
            reference=reference,
            entry_compact=compact,
            compact_cue=cue,
            brain_identity=id(brain),
        )


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


@dataclass(frozen=True)
class AttentionPlan:
    """Immutable readout schedule for sparse assembly attention."""

    query: Assembly
    keys: tuple[tuple[str, Assembly], ...]
    values: tuple[tuple[str, Assembly], ...]
    top_k: int = 1
    output_size: int | None = None
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.query, Assembly) or not self.query:
            raise ValueError("attention query must be a nonempty Assembly")
        if not self.keys or not self.values:
            raise ValueError("attention requires nonempty keys and values")
        if tuple(label for label, _ in self.keys) != tuple(label for label, _ in self.values):
            raise ValueError("attention keys and values must have identical ordered labels")
        if len({label for label, _ in self.keys}) != len(self.keys):
            raise ValueError("attention labels must be unique")
        if any(not isinstance(label, str) or not label or not isinstance(assembly, Assembly)
               or not assembly for label, assembly in (*self.keys, *self.values)):
            raise ValueError("attention entries must be nonempty labeled Assemblies")
        if any(assembly.area != self.query.area for _, assembly in self.keys):
            raise ValueError("attention keys must share the query area")
        if len({assembly.area for _, assembly in self.values}) != 1:
            raise ValueError("attention values must share one area")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, Integral) or not 1 <= self.top_k <= len(self.keys):
            raise ValueError("attention top_k must be between one and the number of keys")
        if self.output_size is not None and (isinstance(self.output_size, bool)
                                              or not isinstance(self.output_size, Integral)
                                              or self.output_size < 1):
            raise ValueError("attention output_size must be a positive integer")
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, Real)
                or not math.isfinite(float(self.temperature)) or self.temperature <= 0):
            raise ValueError("attention temperature must be finite and positive")


@dataclass(frozen=True)
class OperationContract:
    """Reviewable scientific surface attached to an executable operation."""

    operation_id: str
    specification: str
    plan_type: type
    inputs: tuple[str, ...]
    reads: tuple[str, ...]
    mutates: tuple[str, ...]
    regime: tuple[str, ...]
    observed_outcome: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    constructed_controls: tuple[str, ...]
    true_negative_controls: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation contract requires a nonempty string ID")
        if not isinstance(self.specification, str) or "#contract-" not in self.specification:
            raise ValueError("operation contract requires an ID and specification anchor")
        plan_params = getattr(self.plan_type, "__dataclass_params__", None)
        if not is_dataclass(self.plan_type) or not getattr(plan_params, "frozen", False):
            raise ValueError("operation contract plan_type must be a frozen dataclass")
        surfaces = {
            "inputs": self.inputs,
            "reads": self.reads,
            "mutates": self.mutates,
            "regime": self.regime,
            "observed outcome": self.observed_outcome,
            "failure conditions": self.failure_conditions,
            "constructed controls": self.constructed_controls,
            "true-negative controls": self.true_negative_controls,
        }
        invalid = []
        for name, values in surfaces.items():
            if (
                not isinstance(values, tuple)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)
            ):
                invalid.append(name)
        if invalid:
            raise ValueError(f"operation contract has invalid surfaces: {invalid}")


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


COMPLETION_CONTRACT = OperationContract(
    operation_id="pattern-completion-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-completion",
    plan_type=CompletionPlan,
    inputs=("brain", "area", "fraction", "rounds", "seed", "observation_mode"),
    reads=("entry assembly", "compact winners", "recurrent weights"),
    mutates=(
        "area winners",
        "recurrent weights in plastic mode",
        "recruitment outside read-only mode",
        "engine history outside read-only mode",
    ),
    regime=("registered active area", "nonempty retained cue"),
    observed_outcome=(
        "final recovered neuron-ID snapshot",
        "min-normalized overlap with immutable entry reference",
    ),
    failure_conditions=(
        "invalid cue protocol",
        "missing seed or observation mode",
        "unknown or empty area",
        "cue rounds to zero neurons",
        "backend projection rejection",
    ),
    constructed_controls=(
        "neural_assemblies/tests/test_public_model_boundaries.py::"
        "test_teaching_example_has_a_working_learning_disabled_control",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_operation_contract_objects.py::"
        "test_completion_prepare_rejects_before_mutation",
    ),
)


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


ATTENTION_CONTRACT = OperationContract(
    operation_id="assembly-attention-v1",
    specification="neural_assemblies/ir/VERIFICATION.md#contract-assembly-attention",
    plan_type=AttentionPlan,
    inputs=("query", "labeled keys", "labeled values", "top_k", "output_size", "temperature"),
    reads=("query and key neuron IDs", "value neuron IDs"),
    mutates=("nothing; pure readout",),
    regime=("nonempty immutable assemblies", "shared query/key area", "shared value area"),
    observed_outcome=("ranked compatibility weights", "selected labels", "bounded sparse value assembly"),
    failure_conditions=("mismatched labels", "mixed areas", "invalid top_k/output_size/temperature"),
    constructed_controls=(
        "neural_assemblies/tests/test_attention_operator.py::test_attention_multi_key_output_is_deterministic_and_bounded",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_attention_operator.py::test_attention_rejects_key_value_mismatch_and_mixed_value_areas",
    ),
)


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


OPERATION_CONTRACTS = MappingProxyType({
    "projection": PROJECTION_CONTRACT,
    "reciprocal_projection": RECIPROCAL_PROJECTION_CONTRACT,
    "association": ASSOCIATION_CONTRACT,
    "merge": MERGE_CONTRACT,
    "pattern_completion": COMPLETION_CONTRACT,
    "ordered_recall": ORDERED_RECALL_CONTRACT,
    "sequence_memorize": SEQUENCE_MEMORIZE_CONTRACT,
    "separate": SEPARATION_CONTRACT,
    "attention": ATTENTION_CONTRACT,
    "bind": BINDING_CONTRACT,
    "read_binding": BINDING_READ_CONTRACT,
    "source_binding": SOURCE_BINDING_CONTRACT,
    "binding_recall": BINDING_RECALL_CONTRACT,
    "consolidate_pair": CONSOLIDATION_CONTRACT,
    "consolidate": CONSOLIDATION_PROTOCOL_CONTRACT,
    "learn_assembly": CONVERGENCE_CONTRACT,
    "learn_assembly_from_pattern": CONVERGENCE_CONTRACT,
})


def implements(contract: OperationContract) -> Callable:
    """Attach the exact contract object to its public implementation."""
    def decorate(operation: Callable) -> Callable:
        operation.operation_contract = contract
        return operation
    return decorate
