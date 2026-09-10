"""Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-round

Executable, deliberately restricted projection IR. Numerical semantics remain
owned by NumpyExplicitEngine; this module validates and lowers instructions.
"""
from dataclasses import dataclass
import math

import numpy as np

from ..core.index_spaces import validated_indices


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
        return {"profile": "explicit-area-round-v1", "target": self.target,
                "from_areas": list(self.from_areas), "plasticity": self.plasticity,
                "external_drive": list(self.external_drive)}

    @classmethod
    def from_document(cls, document):
        expected = {"profile", "target", "from_areas", "plasticity", "external_drive"}
        if (type(document) is not dict or set(document) != expected
                or document["profile"] != "explicit-area-round-v1"):
            raise ValueError("expected a complete explicit-area-round-v1 instruction")
        return cls(**{key: value for key, value in document.items() if key != "profile"})

    def execute(self, engine):
        """Validate this instruction, then use the existing dense CPU kernel.

        This is an engine-level API: it does not synchronize a Brain facade.
        No program-level rollback or formal backend certification is implied.
        """
        from ..core.backend import get_xp
        from ..core.numpy_engine import NumpyExplicitEngine

        if type(engine) is not NumpyExplicitEngine or get_xp() is not np:
            raise ValueError("explicit-area-round-v1 requires the NumPy explicit CPU engine")
        names = (self.target, *self.from_areas)
        if any(name not in engine._areas for name in names):
            raise ValueError("instruction references an unregistered area")
        target = engine._areas[self.target]
        for name in names:
            area = engine._areas[name]
            if area.fixed_assembly or area.slot_count or area.winner_policy is not None:
                raise ValueError("clamps, slots and custom winner policies are unsupported")
            if not 0 < area.k <= area.n:
                raise ValueError("area requires 0 < k <= n")
            ids = validated_indices(area.winners, upper=area.n, label=f"{name} neuron IDs")
            if len(np.unique(ids)) != len(ids):
                raise ValueError("duplicate winners are not an assembly")
        if self.plasticity and not engine._plasticity_enabled_global:
            raise ValueError("global plasticity disable contradicts instruction")
        if engine.w_max is not None and (not math.isfinite(engine.w_max) or engine.w_max <= 0):
            raise ValueError("weight clip must be finite and positive or None")
        if self.external_drive and len(self.external_drive) != target.n:
            raise ValueError("external drive length must equal target population")
        for name in self.from_areas:
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
        drive = None
        if self.external_drive:
            with np.errstate(over="ignore"):
                drive = np.asarray(self.external_drive, dtype=np.float32)
            if not np.isfinite(drive).all():
                raise ValueError("external drive is not representable as float32")
        return engine.project_into(self.target, [], list(self.from_areas),
                                   plasticity_enabled=self.plasticity, external_drive=drive)
