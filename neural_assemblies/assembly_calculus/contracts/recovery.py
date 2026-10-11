"""Recovery plans and contracts: recovery, cue replacement, pattern completion.

Part of neural_assemblies.assembly_calculus.contracts."""
from dataclasses import dataclass
from contextlib import nullcontext
import math
from numbers import Integral, Real
import random

import numpy as np

from ..assembly import Assembly
from ...core.index_spaces import NeuronIds
from ...core.semantics import ObservationPolicy

from .contract import OperationContract
from .schedule import ProjectionStep, _explicit_bool, _positive_rounds, _require_name, _schedule


@dataclass(frozen=True)
class RecoveryPlan:
    """Validated state-preserving cue-recovery observation."""

    reference: Assembly
    cue: Assembly
    rounds: int
    seed: int | None = None
    recurrence_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Assembly) or not len(self.reference):
            raise ValueError("recovery requires a nonempty Assembly reference")
        if not isinstance(self.cue, Assembly) or self.cue.area != self.reference.area:
            raise ValueError("cue must be an Assembly in the reference area")
        if len(self.cue) > len(self.reference):
            raise ValueError("cue winner count exceeds reference size")
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, Integral) or self.rounds < 1:
            raise ValueError("recovery rounds must be a positive integer")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0):
            raise ValueError("recovery seed must be a nonnegative integer or None")
        _explicit_bool("recurrence_enabled", self.recurrence_enabled)
        object.__setattr__(self, "rounds", int(self.rounds))
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))

    def preflight(self, brain) -> None:
        if self.reference.area not in brain.areas:
            raise KeyError(f"recovery area is unknown: {self.reference.area!r}")
        area = brain.areas[self.reference.area]
        for assembly in (self.reference, self.cue):
            if np.any(np.asarray(assembly.neuron_ids) >= area.n):
                raise ValueError("recovery neuron IDs exceed area population")
        owner = brain.engine_for(self.reference.area)
        count = owner.materialized_count(self.reference.area)
        if count is not None and count != area.n:
            raise ValueError("recovery observation requires a fully materialized population")


@dataclass(frozen=True)
class CueReplacementPlan:
    """Validated deterministic replacement cue construction."""

    reference: Assembly
    population: NeuronIds
    count: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Assembly) or not len(self.reference):
            raise ValueError("cue replacement requires a nonempty Assembly reference")
        if isinstance(self.count, bool) or not isinstance(self.count, Integral) or self.count < 0:
            raise ValueError("count must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        universe = np.asarray(self.population)
        if universe.ndim != 1 or not np.issubdtype(universe.dtype, np.integer):
            raise ValueError("population must be a one-dimensional integer array")
        if len(np.unique(universe)) != len(universe):
            raise ValueError("population must contain unique neuron IDs")
        if not np.isin(self.reference.neuron_ids, universe).all():
            raise ValueError("reference must be contained in population")
        alternatives = len(universe) - len(self.reference)
        if self.count > min(len(self.reference), alternatives):
            raise ValueError("population cannot deliver the requested replacement count")
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "seed", int(self.seed))


# The three policies completion can honour, spelled by the one closed enum
# every run record uses (ObservationPolicy); `probe` and `none` do not apply
# to a recurrent recovery schedule.
_COMPLETION_OBSERVATION_MODES = frozenset({
    ObservationPolicy.PLASTIC.value,
    ObservationPolicy.FROZEN.value,
    ObservationPolicy.READ_ONLY.value,
})


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
        return _schedule(step, step, self.rounds)

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
        # Card C3: a clamped (fixed) target cannot masquerade as free recall.
        # With winners pinned the schedule below cannot change them, so any
        # overlap it reports is the clamp, not recovery. Refuse before the
        # cue is even drawn.
        area = brain.areas[self.area]
        if getattr(area, "fixed_assembly", False) or brain._engine_for(area).is_fixed(self.area):
            raise ValueError(
                f"area {self.area!r} has a fixed assembly; completion measures free "
                "recall and a clamped target cannot report it (unfix the area first)"
            )
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


RECOVERY_CONTRACT = OperationContract(
    operation_id="cue-recovery-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-cue-recovery",
    plan_type=RecoveryPlan,
    inputs=("brain", "reference Assembly", "cue Assembly", "rounds", "seed", "recurrence_enabled"),
    reads=("fully materialized recurrent area", "reference/cue neuron IDs"),
    mutates=("temporary winners only under read-only scope",),
    regime=("nonempty reference", "cue in reference area", "full materialization", "explicit recurrence control"),
    observed_outcome=("cue overlap, recovered overlap, improvement",),
    failure_conditions=("unknown area", "invalid IDs", "partial population", "invalid rounds/seed"),
    constructed_controls=(
        "neural_assemblies/tests/test_noise_robustness.py::test_recovery_improves_cue_and_fails_learning_and_dynamics_nulls",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_noise_robustness.py::test_partial_population_refuses_before_activating_cue",
    ),
)


CUE_REPLACEMENT_CONTRACT = OperationContract(
    operation_id="cue-replacement-v1",
    specification="docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-cue-replacement",
    plan_type=CueReplacementPlan,
    inputs=("reference Assembly", "eligible population", "replacement count", "seed"),
    reads=("reference neuron IDs", "eligible population IDs"),
    mutates=("nothing; pure deterministic cue construction",),
    regime=("distinct alternatives", "count bounded by reference and complement", "seeded draw"),
    observed_outcome=("cue Assembly with requested replacements",),
    failure_conditions=("empty/malformed reference", "invalid population", "count overflow", "invalid seed"),
    constructed_controls=(
        "neural_assemblies/tests/test_noise_robustness.py::test_population_must_be_unique_and_contain_reference",
    ),
    true_negative_controls=(
        "neural_assemblies/tests/test_noise_robustness.py::test_population_must_be_unique_and_contain_reference",
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
