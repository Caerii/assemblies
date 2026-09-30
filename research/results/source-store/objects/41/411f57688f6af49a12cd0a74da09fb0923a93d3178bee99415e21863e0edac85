"""Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-round

Executable, deliberately restricted projection IR. Numerical semantics remain
owned by NumpyExplicitEngine; this module validates and lowers instructions.
"""
from dataclasses import dataclass
import math

import numpy as np

from .protocol import validate_schema_document


def validate_explicit_round_document(document):
    """Validate the complete, backend-independent wire representation."""
    return validate_schema_document(document, "explicit-round.schema.json")


@dataclass(frozen=True)
class ExplicitRound:
    """One sequential round, with stable neuron IDs and explicit learning intent.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-round
    """

    target: str
    from_areas: tuple[str, ...]
    plasticity: bool
    external_drive: tuple[float, ...] = ()

    def __post_init__(self):
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a nonempty area name")
        if type(self.from_areas) not in (tuple, list) or type(self.external_drive) not in (tuple, list):
            raise ValueError("sources and drive must be sequences, not strings or iterators")
        sources = tuple(self.from_areas)
        if any(not isinstance(s, str) or not s for s in sources):
            raise ValueError("sources must be nonempty area names")
        if len(set(sources)) != len(sources):
            raise ValueError("duplicate sources would count drive and learning twice")
        if type(self.plasticity) is not bool:
            raise ValueError("plasticity must be an explicit boolean")
        drive = tuple(self.external_drive)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in drive):
            raise ValueError("external drive must contain finite JSON numbers")
        if not sources and not drive:
            raise ValueError("a round needs area input or explicit external drive")
        object.__setattr__(self, "from_areas", sources)
        object.__setattr__(self, "external_drive", drive)

    def to_document(self):
        document = {"profile": "explicit-area-round-v1", "target": self.target,
                    "from_areas": list(self.from_areas), "plasticity": self.plasticity,
                    "external_drive": list(self.external_drive)}
        errors = validate_explicit_round_document(document)
        if errors:
            raise ValueError(f"invalid explicit round: {errors}")
        return document

    @classmethod
    def from_document(cls, document):
        errors = validate_explicit_round_document(document)
        if errors:
            raise ValueError(f"invalid explicit round: {errors}")
        return cls(**{key: value for key, value in document.items() if key != "profile"})

    def validate(self, engine):
        """Check profile and numerical inputs without executing or synchronizing state."""
        from ..core.backend import get_xp
        from ..core.numpy_engine import NumpyExplicitEngine

        if type(engine) is not NumpyExplicitEngine or get_xp() is not np:
            raise ValueError("explicit-area-round-v1 requires the NumPy explicit CPU engine")
        names = (self.target, *self.from_areas)
        if any(name not in engine._areas for name in names):
            raise ValueError("instruction references an unregistered area")
        engine.validate_projection_inputs(self.target, [], self.from_areas,
                                          self.external_drive or None)
        target = engine._areas[self.target]
        for name in names:
            area = engine._areas[name]
            if area.fixed_assembly or area.slot_count or area.winner_policy is not None:
                raise ValueError("clamps, slots and custom winner policies are unsupported")
        if self.plasticity and not engine._plasticity_enabled_global:
            raise ValueError("global plasticity disable contradicts instruction")
        if engine.w_max is not None and (not math.isfinite(engine.w_max) or engine.w_max <= 0):
            raise ValueError("weight clip must be finite and positive or None")
        for name in self.from_areas:
            if self.plasticity and not engine.fiber_learning_allowed(name, self.target):
                raise ValueError("Engine learning mask contradicts instruction")
            conn = engine._area_conns.get(name, {}).get(self.target)
            if conn is None:
                raise ValueError("instruction references a missing fiber")
            weights = conn.weights
            if (not isinstance(weights, np.ndarray) or weights.dtype != np.float32
                    or weights.shape != (engine._areas[name].n, target.n)
                    or not np.isfinite(weights).all() or (weights < 0).any()):
                raise ValueError("fiber must be a finite nonnegative dense float32 matrix")
            beta = engine.get_beta(self.target, name)
            if not math.isfinite(beta) or beta < 0:
                raise ValueError("Hebbian beta must be finite and nonnegative")
    def execute(self, engine):
        """Execute on a standalone engine; use execute_on_brain for a Brain."""
        self.validate(engine)
        return self._execute_validated(engine)

    def _execute_validated(self, engine):
        """Execute after validation has already established the profile."""
        return engine.project_into(self.target, [], list(self.from_areas),
                                   plasticity_enabled=self.plasticity,
                                   external_drive=self.external_drive or None)

    def execute_on_brain(self, brain):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-brain-round

        Lower through ordinary projection so descriptors and history stay coherent.
        Returns a detached winner array. No full-program rollback is promised.
        """
        self.validate_on_brain(brain)
        return self._execute_on_brain_validated(brain)

    def validate_on_brain(self, brain) -> None:
        """Validate Brain lowering without changing activity or learning state."""
        names = (self.target, *self.from_areas)
        if any(name not in brain.areas for name in names):
            raise ValueError("instruction references an unregistered Brain area")
        if (brain._mutual_inhibition_groups or brain.plasticity_mask
                or (brain._inhibition is not None and brain._inhibition.any_closed())):
            raise ValueError("Brain inhibition and fiber plasticity overrides are unsupported")
        if self.plasticity and brain.disable_plasticity:
            raise ValueError("Brain learning disable contradicts instruction")
        engine = brain.engine_for(self.target)
        self.validate(engine)
        for name in names:
            area = brain.areas[name]
            if brain.engine_for(name) is not engine:
                raise ValueError("all instruction areas must use the same dense engine")
            if (area.fixed_assembly or area.slot_count or area.winner_policy is not None
                    or area.n != engine._areas[name].n or area.k != engine._areas[name].k):
                raise ValueError("Brain area configuration differs from the supported profile")
            engine._validated_winners(name, area.winners)
        for name in self.from_areas:
            if brain.connectomes[name][self.target] is not engine._area_conns[name][self.target]:
                raise ValueError("Brain and engine disagree on fiber ownership")

    def _execute_on_brain_validated(self, brain):
        """Execute after :meth:`validate_on_brain` has admitted the round."""
        saved = brain.disable_plasticity
        brain.disable_plasticity = not self.plasticity
        try:
            brain.project({}, {name: [self.target] for name in self.from_areas},
                          external_drive=({self.target: self.external_drive}
                                          if self.external_drive else None))
        finally:
            brain.disable_plasticity = saved
        return brain.areas[self.target].winners.copy()


@dataclass(frozen=True)
class ExplicitProgram:
    """Immutable composition of explicit rounds.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-program

    A program is only an ordered value until a backend chooses to execute it.
    Keeping composition here makes order and round validation inspectable and
    prevents callers from mutating a list after it has been checked.
    """

    rounds: tuple[ExplicitRound, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.rounds, tuple):
            raise TypeError("program rounds must be a tuple")
        if any(not isinstance(round_, ExplicitRound) for round_ in self.rounds):
            raise TypeError("program rounds must contain ExplicitRound values")

    def then(self, other: "ExplicitProgram") -> "ExplicitProgram":
        """Compose programs in execution order, preserving both sequences."""
        if not isinstance(other, ExplicitProgram):
            raise TypeError("program composition requires another ExplicitProgram")
        return ExplicitProgram(self.rounds + other.rounds)

    def __len__(self) -> int:
        return len(self.rounds)

    def __iter__(self):
        return iter(self.rounds)

    def to_documents(self) -> tuple[dict, ...]:
        """Return canonical wire documents without exposing mutable state."""
        return tuple(round_.to_document() for round_ in self.rounds)

    @classmethod
    def from_documents(cls, documents) -> "ExplicitProgram":
        """Validate and decode every round before constructing the program."""
        if not isinstance(documents, (tuple, list)):
            raise TypeError("program documents must be a sequence")
        return cls(tuple(ExplicitRound.from_document(document) for document in documents))

    def execute(self, engine):
        """Execute rounds in order and return the final round's observation."""
        self.validate(engine)
        result = None
        for round_ in self.rounds:
            result = round_._execute_validated(engine)
        return result

    def validate(self, engine) -> None:
        """Admit every round before the first round can mutate the engine."""
        for round_ in self.rounds:
            round_.validate(engine)

    def execute_on_brain(self, brain):
        """Lower rounds in order through the coherent Brain boundary."""
        for round_ in self.rounds:
            round_.validate_on_brain(brain)
        result = None
        for round_ in self.rounds:
            result = round_._execute_on_brain_validated(brain)
        return result
