"""Explicit cue perturbations and state-preserving recurrent observations."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np

from ..core.index_spaces import NeuronIds, validated_indices
from ..core.registration import validate_round_count
from .assembly import Assembly
from .ops import activate_assembly, _snap


def replace_neurons(reference: Assembly, *, population: NeuronIds, count: int, seed: int) -> Assembly:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery

    Replace exactly count members by distinct IDs outside the reference. Population
    explicitly determines eligible IDs; input ordering does not change the draw.
    """
    original = np.sort(validated_indices(reference.neuron_ids, unique=True))
    universe = np.sort(validated_indices(population, unique=True))
    for name, value in (('count', count), ('seed', seed)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if not len(original) or not np.isin(original, universe).all():
        raise ValueError('nonempty reference must be contained in the population')
    alternatives = np.setdiff1d(universe, original, assume_unique=True)
    if count > min(len(original), len(alternatives)):
        raise ValueError('population cannot deliver the requested replacement count')
    rng = np.random.default_rng(int(seed))
    positions = rng.permutation(len(original))[:int(count)]
    replacements = rng.permutation(alternatives)[:int(count)]
    cue = original.copy()
    cue[positions] = replacements
    return Assembly(reference.area, NeuronIds(np.sort(cue)))


@dataclass(frozen=True)
class RecoveryObservation:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery"""
    reference: Assembly
    cue: Assembly
    recovered: Assembly

    @property
    def cue_overlap(self):
        return len(np.intersect1d(self.reference.neuron_ids, self.cue.neuron_ids)) / len(self.reference)

    @property
    def recovered_overlap(self):
        return len(np.intersect1d(self.reference.neuron_ids, self.recovered.neuron_ids)) / len(self.reference)

    @property
    def improvement(self):
        return self.recovered_overlap - self.cue_overlap


def observe_recovery(brain, reference: Assembly, cue: Assembly, *, rounds: int,
                     seed: int | None = None, recurrence_enabled: bool = True) -> RecoveryObservation:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery

    Read a fully materialized recurrent area without retaining learning or activity.
    Overlap uses reference size, so dropping neurons cannot manufacture completion.
    """
    rounds = validate_round_count(rounds)
    if type(recurrence_enabled) is not bool:
        raise ValueError('recurrence_enabled must be boolean')
    if reference.area != cue.area or not len(reference):
        raise ValueError('reference and cue require the same area and a nonempty reference')
    if len(cue) > len(reference):
        raise ValueError('cue winner count exceeds reference size')
    area = brain.areas[reference.area]
    for assembly in (reference, cue):
        validated_indices(assembly.neuron_ids, upper=area.n, unique=True)
    owner = brain._engine_for(area)
    count = owner.materialized_count(reference.area)
    # ComputeEngine uses None for dense engines whose population is always present.
    if count is not None and count != area.n:
        raise ValueError('recovery observation requires a fully materialized population')
    with brain.read_only(seed=seed):
        area.fixed_assembly = False
        activate_assembly(brain, cue)
        if recurrence_enabled:
            for _ in range(rounds):
                brain.project({}, {reference.area: [reference.area]})
        recovered = _snap(brain, reference.area)
        if len(recovered) > len(reference):
            raise ValueError('recovery winner count exceeds reference size')
        return RecoveryObservation(reference, cue, recovered)
