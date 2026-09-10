"""Legacy projection selection modes; these do not certify model fidelity.

On the sampled numpy engine, ``exact`` permits candidate sampling and
connectome growth. It does not request a fixed connectome or select the
separate ``numpy_exact`` engine, and implies no statistical equivalence proof.

``compiled`` (alias ``fuzzy``) selects only existing columns when growth is
frozen and at least k columns exist. Otherwise the backend uses its ordinary
projection path. Restricting the candidate population changes the dynamics;
readout agreement alone does not establish preservation of an experiment.

Per-area ``_plasticity_only_mode`` can select the compiled path despite a global
``exact`` setting. ``_force_exact_projection`` overrides it on the sampled numpy
engine. A resolved model contract must record these choices before lowering.
"""

from __future__ import annotations

from enum import Enum


class ProjectionFidelity(str, Enum):
    """Engine projection fidelity."""

    EXACT = "exact"
    COMPILED = "compiled"

    @classmethod
    def normalize(cls, value: str | "ProjectionFidelity") -> "ProjectionFidelity":
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        if key in ("compiled", "fuzzy", "optimized", "quantized"):
            return cls.COMPILED
        if key in ("exact", "microscopic", "full"):
            return cls.EXACT
        raise ValueError(
            f"Unknown projection fidelity {value!r}; "
            f"use {cls.EXACT.value!r} or {cls.COMPILED.value!r}",
        )
