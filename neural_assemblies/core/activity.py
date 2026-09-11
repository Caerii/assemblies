"""Scoped dynamical state, owned by the area that defines its fields.

Activity can evolve during a probe without becoming training history. Learned
weights are protected separately by the plasticity gate; derived drive caches
and the latest measurement remain available to the caller.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class PopulationCounts:
    """Three noninterchangeable sizes for one area.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-population-counts
    """

    active: int
    ever_fired: int
    materialized: int | None


@dataclass(frozen=True)
class PreKwtaObservation:
    """A pre-selection sum paired with the candidates it covers.

    Specification: neural_assemblies/ir/VERIFICATION.md#contract-pre-kwta-observation
    """

    total: float
    candidate_count: int

    def __post_init__(self):
        if isinstance(self.total, bool) or not isinstance(self.total, Real):
            raise ValueError("pre-k-WTA total must be a finite real number")
        if not math.isfinite(self.total):
            raise ValueError("pre-k-WTA total must be finite")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, Integral)
            or self.candidate_count <= 0
        ):
            raise ValueError("pre-k-WTA candidate_count must be a positive integer")
        object.__setattr__(self, "total", float(self.total))
        object.__setattr__(self, "candidate_count", int(self.candidate_count))

    @property
    def mean(self) -> float:
        return self.total / self.candidate_count


@dataclass
class ActivitySnapshot:
    owner: object
    fields: dict

    def restore(self):
        for name, (original, saved) in self.fields.items():
            # Preserve references held by callers when restoring mutable buffers.
            if isinstance(original, list):
                original[:] = saved
            elif hasattr(original, "shape") and hasattr(original, "__setitem__"):
                original[...] = saved
            elif hasattr(original, "clear") and hasattr(original, "extend"):
                original.clear()
                original.extend(saved)
            else:
                original = saved
            setattr(self.owner, name, original)


class ActivityState:
    """Declare activity fields beside their owner, rather than in Brain."""
    __slots__ = ()
    _activity_fields = ()
    _activity_history_fields = ()

    def snapshot_activity(self):
        fields = {}
        for name in self._activity_fields:
            if hasattr(self, name):
                original = getattr(self, name)
                fields[name] = (original, deepcopy(original))
        # Projection histories are append-only. Preserve the existing records
        # by reference rather than recopying an entire training run per probe.
        for name in self._activity_history_fields:
            original = getattr(self, name)
            fields[name] = (original, list(original))
        return ActivitySnapshot(self, fields)
