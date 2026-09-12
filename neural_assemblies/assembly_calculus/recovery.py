"""Explicit cue perturbations and state-preserving recurrent observations."""
from dataclasses import dataclass
import numpy as np
from neural_assemblies.assembly_calculus.metrics import recall_fraction

from ..core.index_spaces import NeuronIds, validated_indices
from .assembly import Assembly
from .ops import activate_assembly, _snap
from .contracts import CUE_REPLACEMENT_CONTRACT, CueReplacementPlan, RECOVERY_CONTRACT, RecoveryPlan, implements


@implements(CUE_REPLACEMENT_CONTRACT)
def replace_neurons(reference: Assembly, *, population: NeuronIds, count: int, seed: int) -> Assembly:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery

    Replace exactly count members by distinct IDs outside the reference. Population
    explicitly determines eligible IDs; input ordering does not change the draw.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-cue-replacement
    """
    plan = CueReplacementPlan(reference, population, count, seed)
    reference = plan.reference
    population_values = plan.population
    count, seed = plan.count, plan.seed
    original = np.sort(validated_indices(reference.neuron_ids, unique=True))
    universe = np.sort(validated_indices(population_values, unique=True))
    alternatives = np.setdiff1d(universe, original, assume_unique=True)
    rng = np.random.default_rng(int(seed))
    positions = rng.permutation(len(original))[:int(count)]
    replacements = rng.permutation(alternatives)[:int(count)]
    cue = original.copy()
    cue[positions] = replacements
    return Assembly(reference.area, NeuronIds(np.sort(cue)))


def _validate_recovery_members(reference, **snapshots):
    if not isinstance(reference, Assembly) or not len(reference):
        raise ValueError('recovery requires a nonempty Assembly reference')
    validated_indices(reference.neuron_ids, unique=True)
    for label, snapshot in snapshots.items():
        if not isinstance(snapshot, Assembly) or snapshot.area != reference.area:
            raise ValueError(f'{label} must be an Assembly in the reference area')
        if len(snapshot) > len(reference):
            raise ValueError(f'{label} winner count exceeds reference size')
        validated_indices(snapshot.neuron_ids, unique=True)


@dataclass(frozen=True)
class RecoveryObservation:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery"""
    reference: Assembly
    cue: Assembly
    recovered: Assembly

    def __post_init__(self):
        _validate_recovery_members(self.reference, cue=self.cue, recovery=self.recovered)

    @property
    def cue_overlap(self):
        return recall_fraction(self.cue.neuron_ids, self.reference.neuron_ids)

    @property
    def recovered_overlap(self):
        return recall_fraction(self.recovered.neuron_ids, self.reference.neuron_ids)

    @property
    def improvement(self):
        return self.recovered_overlap - self.cue_overlap


@implements(RECOVERY_CONTRACT)
def observe_recovery(brain, reference: Assembly, cue: Assembly, *, rounds: int,
                     seed: int | None = None, recurrence_enabled: bool = True) -> RecoveryObservation:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery

    Read a fully materialized recurrent area without retaining learning or activity.
    Overlap uses reference size, so dropping neurons cannot manufacture completion.

    Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-cue-recovery
    """
    plan = RecoveryPlan(reference, cue, rounds, seed, recurrence_enabled)
    plan.preflight(brain)
    reference, cue = plan.reference, plan.cue
    rounds, seed, recurrence_enabled = plan.rounds, plan.seed, plan.recurrence_enabled
    area = brain.areas[reference.area]
    for assembly in (reference, cue):
        validated_indices(assembly.neuron_ids, upper=area.n, unique=True)
    with brain.read_only(seed=seed):
        area.fixed_assembly = False
        activate_assembly(brain, cue)
        if recurrence_enabled:
            for _ in range(rounds):
                brain.project({}, {reference.area: [reference.area]})
        recovered = _snap(brain, reference.area)
        return RecoveryObservation(reference, cue, recovered)
