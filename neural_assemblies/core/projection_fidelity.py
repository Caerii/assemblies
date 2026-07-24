"""Projection fidelity modes for assembly calculus engines.

**Exact** mode runs the full microscopic sparse simulation: truncated-normal
candidate sampling, connectome expansion, and k-WTA on the combined input
vector.  This is the default and preserves the statistical dynamics of the
NEMO / Assembly Calculus model.

**Compiled** mode (also accepted as ``"fuzzy"``) is our topology-quantization
shortcut: when an area's connectome is frozen and already pregrown to at
least ``k`` columns, projection skips sampling and expansion and selects
winners by top-k on the existing weight columns only.  Plasticity still
applies on the fixed topology.  Accuracy is gated by readout overlap rather
than microscopic step identity — suitable for large-scale training after
pathways are stabilized (lexicon, bridge pregrow, role pregrow).

Per-area ``_plasticity_only_mode`` flags still force compiled dynamics for
that area even when global fidelity is ``exact`` (used by
:class:`~neural_assemblies.assembly_calculus.emergent.training.compiled.CompiledTopologySession`).
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
