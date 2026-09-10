"""Context-conditioned attractor observation with explicit teaching and noise."""
from dataclasses import asdict, dataclass
import math
from numbers import Real

import numpy as np

from ..core.index_spaces import NeuronIds
from ..core.registration import validate_area_registration, validate_round_count, validate_input_noise
from .assembly import Assembly, overlap
from .coin_config import AttractorConfig, _nonnegative_count
from .ops import _snap, activate_assembly


@dataclass(frozen=True)
class ContextChoiceProtocol:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-context-choice

    Assigned disjoint context codes and integer teaching counts for two outcomes.
    Coupling beta is independent of recurrent-attractor training beta.
    """
    n: int
    k: int
    contexts: tuple[str, ...]
    presentations: tuple[tuple[int, int], ...]
    attractors: AttractorConfig
    coupling_beta: float
    read_rounds: int
    noise_std: float

    def __post_init__(self):
        n, k = validate_area_registration('context', self.n, self.k)
        if isinstance(self.contexts, (str, bytes)):
            raise ValueError('contexts must be a sequence of names')
        contexts = tuple(self.contexts)
        if (not contexts or any(not isinstance(name, str) or not name for name in contexts)
                or len(set(contexts)) != len(contexts) or len(contexts) * k > n):
            raise ValueError('contexts require unique nonempty names and disjoint k-sized codes within n')
        counts = tuple(tuple(_nonnegative_count(x, 'presentation count') for x in row)
                       for row in self.presentations)
        if len(counts) != len(contexts) or any(len(row) != 2 for row in counts):
            raise ValueError('presentations must supply two counts for each context')
        if type(self.attractors) is not AttractorConfig:
            raise ValueError('attractors must be AttractorConfig, without unused seed-mixture settings')
        if (isinstance(self.coupling_beta, bool) or not isinstance(self.coupling_beta, Real)
                or not math.isfinite(self.coupling_beta) or self.coupling_beta < 0):
            raise ValueError('coupling_beta must be finite and nonnegative')
        for name, value in (('n', n), ('k', k), ('contexts', contexts), ('presentations', counts),
                            ('coupling_beta', float(self.coupling_beta)),
                            ('read_rounds', validate_round_count(self.read_rounds)),
                            ('noise_std', validate_input_noise(self.noise_std))):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ContextChoiceObservation:
    label: int | None
    overlaps: tuple[float, float]

    @property
    def margin(self):
        return abs(self.overlaps[0] - self.overlaps[1])


class ContextAttractorChoice:
    """Specification: neural_assemblies/ir/VERIFICATION.md#contract-context-choice

    Context drive and optional recurrence compete with native Gaussian input noise.
    Observation preserves brain state and exposes both overlaps, not probabilities.
    """
    def __init__(self, brain, *, protocol: ContextChoiceProtocol, prefix='_context_choice'):
        if not isinstance(protocol, ContextChoiceProtocol):
            raise ValueError('ContextAttractorChoice requires an explicit ContextChoiceProtocol')
        if protocol.noise_std > 0 and not brain._engine.supports_input_noise:
            raise NotImplementedError('positive noise requires an engine with native input noise support')
        self.brain, self._protocol = brain, protocol
        self.context_area = f'{prefix}_context'
        brain.add_area(self.context_area, protocol.n, protocol.k, 0.)
        brain.materialize_area(self.context_area)
        self._contexts = {name: Assembly(self.context_area, NeuronIds(
            np.arange(i * protocol.k, (i+1) * protocol.k, dtype=np.uint32)))
            for i, name in enumerate(protocol.contexts)}
        self.attractors = protocol.attractors.build(brain, prefix=prefix, area_name='outcomes')
        self.outcome_area = self.attractors.area_name
        brain.update_plasticity(self.context_area, self.outcome_area, protocol.coupling_beta)
        self._zero_stimulus = f'{prefix}_zero'
        brain.add_stimulus(self._zero_stimulus, 0)
        areas = [brain.areas[self.context_area], brain.areas[self.outcome_area]]
        fixed = [area.fixed_assembly for area in areas]
        try:
            for context, counts in zip(protocol.contexts, protocol.presentations):
                if not any(counts):
                    continue
                activate_assembly(brain, self._contexts[context])
                areas[0].fix_assembly()
                for target, count in zip((self.attractors.asm0, self.attractors.asm1), counts):
                    if not count:
                        continue
                    activate_assembly(brain, target)
                    areas[1].fix_assembly()
                    for _ in range(count):
                        brain.project({}, {self.context_area: [self.outcome_area]})
        finally:
            for area, value in zip(areas, fixed):
                area.fixed_assembly = value
        brain.set_input_noise(self.outcome_area, protocol.noise_std)

    @property
    def protocol(self):
        return self._protocol

    @property
    def parameters(self):
        return {'protocol_version': 'context-attractor-v1', **asdict(self.protocol),
                'context_codes': 'assigned_disjoint', 'noise': 'backend_input_noise', 'rng_policy': 'read-only-seed-v1'}

    def observe(self, context: str, *, seed=None, context_enabled=True,
                recurrence_enabled=True) -> ContextChoiceObservation:
        cue = self._contexts[context]  # Validate before touching brain activity.
        if type(context_enabled) is not bool or type(recurrence_enabled) is not bool:
            raise ValueError('context_enabled and recurrence_enabled must be booleans')
        brain = self.brain
        with brain.read_only(seed=seed):
            brain.inhibit_areas([self.context_area, self.outcome_area])
            activate_assembly(brain, cue)
            brain.areas[self.context_area].fix_assembly()
            sources = {}
            if context_enabled:
                sources[self.context_area] = [self.outcome_area]
            if recurrence_enabled:
                sources[self.outcome_area] = [self.outcome_area]
            for _ in range(self.protocol.read_rounds):
                # A zero-sized stimulus schedules even the noise-only control.
                brain.project({self._zero_stimulus: [self.outcome_area]}, sources)
            observed = _snap(brain, self.outcome_area)
            scores = tuple(float(overlap(observed, target))
                           for target in (self.attractors.asm0, self.attractors.asm1))
            label = None if scores[0] == scores[1] else (0 if scores[0] > scores[1] else 1)
            return ContextChoiceObservation(label, scores)
