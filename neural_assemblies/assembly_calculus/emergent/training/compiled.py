"""
Compiled topology sessions for emergent training.

A **compiled topology** pass replays projections on pregrown connectome
columns: freeze growth, optional ring reuse, and top-k-only dynamics
(see ``numpy_engine._sparse.project_into``).  This is an engine-level
optimization of the same calculus protocols — not a different learning rule.

Behavior contract:
    - **Training** may use compiled sessions when pathways are pregrown.
    - **Inference** (``predict_next``, ``parse_incremental``) never enables
      compiled mode; readout parity is tested in ``test_training_perf``.
    - Microscopic step-by-step identity is not guaranteed; overlap/readout
      gates define "preserved behavior."
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterator, Sequence, Tuple

if TYPE_CHECKING:
    from ..parser_mixins.core import CoreParserMixin


@dataclass(frozen=True)
class CompiledTopologySpec:
    """Areas and ring capacities for one compiled training episode."""

    areas: Tuple[str, ...]
    ring_capacity: Dict[str, int] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return bool(self.areas)


class CompiledTopologySession:
    """Context manager: enable freeze + ring + compiled projection on areas."""

    def __init__(self, parser: "CoreParserMixin", spec: CompiledTopologySpec):
        self._parser = parser
        self._spec = spec
        self._entered = False
        self._prev_fidelity: str = "exact"

    def __enter__(self) -> "CompiledTopologySession":
        if not self._spec.ready:
            return self
        p = self._parser
        areas = self._spec.areas
        self._prev_fidelity = p.brain.projection_fidelity
        p.brain.projection_fidelity = "compiled"
        p._set_freeze_connectome_growth(areas, enabled=True)
        p._set_compiled_topology_mode(areas, enabled=True)
        for area_name, cols in self._spec.ring_capacity.items():
            if area_name in areas and cols >= p.k:
                p._enable_area_ring_mode(area_name, cols)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._entered:
            return
        p = self._parser
        areas = self._spec.areas
        p.brain.projection_fidelity = self._prev_fidelity
        p._set_freeze_connectome_growth(areas, enabled=False)
        p._set_compiled_topology_mode(areas, enabled=False)
        for area_name in self._spec.ring_capacity:
            if area_name in areas:
                p._disable_area_ring_mode(area_name)
        self._entered = False


@contextmanager
def compiled_topology(
    parser: "CoreParserMixin",
    spec: CompiledTopologySpec,
) -> Iterator[CompiledTopologySession]:
    """Functional wrapper around :class:`CompiledTopologySession`."""
    session = CompiledTopologySession(parser, spec)
    with session:
        yield session


def prediction_topology_spec(parser: "CoreParserMixin") -> CompiledTopologySpec:
    """Spec for PREDICTION lexicon build after pregrow."""
    from ..core.areas import PREDICTION

    pred_cols = max(
        int(getattr(parser, "_prediction_ring_capacity_cols", 0)),
        parser.k,
    )
    ring: Dict[str, int] = {}
    if pred_cols >= parser.k:
        ring[PREDICTION] = pred_cols
    return CompiledTopologySpec(
        areas=(PREDICTION,),
        ring_capacity=ring,
    )


def bridge_topology_spec(parser: "CoreParserMixin") -> CompiledTopologySpec:
    """Spec for CONTEXT/PREDICTION bridge training after pregrow."""
    from ..core.areas import CONTEXT, PREDICTION

    ring: Dict[str, int] = {}
    ctx_cols = int(getattr(parser, "_context_ring_capacity_cols", 0))
    pred_cols = max(
        int(getattr(parser, "_prediction_ring_capacity_cols", 0)),
        parser.k,
    )
    if ctx_cols >= parser.k:
        ring[CONTEXT] = ctx_cols
    if pred_cols >= parser.k:
        ring[PREDICTION] = pred_cols
    return CompiledTopologySpec(
        areas=(CONTEXT, PREDICTION),
        ring_capacity=ring,
    )


def role_topology_spec(parser: "CoreParserMixin") -> CompiledTopologySpec:
    """Spec for ROLE_AGENT / ROLE_PATIENT unsupervised training."""
    from ..core.areas import ROLE_AGENT, ROLE_PATIENT

    caps = dict(getattr(parser, "_role_ring_capacity_cols", {}))
    ring = {
        area: cols
        for area, cols in caps.items()
        if cols >= parser.k
    }
    areas = tuple(a for a in (ROLE_AGENT, ROLE_PATIENT) if a in ring)
    return CompiledTopologySpec(areas=areas, ring_capacity=ring)


def lexicon_topology_spec(
    parser: "CoreParserMixin",
    core_areas: Sequence[str],
) -> CompiledTopologySpec:
    """Spec for core-area lexicon training after pregrow."""
    caps = dict(getattr(parser, "_core_ring_capacity_cols", {}))
    ring: Dict[str, int] = {}
    for area in core_areas:
        cols = int(caps.get(area, 0))
        if cols >= parser.k:
            ring[area] = cols
    areas = tuple(a for a in core_areas if a in ring)
    return CompiledTopologySpec(areas=areas, ring_capacity=ring)


def link_bridge_topology_legacy(parser: "CoreParserMixin") -> CompiledTopologySpec:
    """Backward-compatible alias; prefer ``topology_linker.link_bridge_topology``."""
    from ..training.linker import link_preallocate_stim_targets

    spec = bridge_topology_spec(parser)
    link_preallocate_stim_targets(parser, spec.areas)
    return spec
