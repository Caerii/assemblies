"""
ComputeEngine: Abstract interface for assembly calculus computation.

The engine owns ALL compute state (connectivity, activation buffers, winner
tracking) and implements the core projection cycle.  Brain delegates here
and only manages logical topology and routing.

Different engines implement different hardware strategies:
  - NumpySparseEngine:  CPU, statistical sparse simulation (default)
  - NumpyExplicitEngine: CPU, dense matrix simulation
  - CudaImplicitEngine:  GPU, hash-based implicit connectivity + CUDA kernels

Usage::

    from src.core.engine import create_engine

    engine = create_engine("numpy_sparse", p=0.05, seed=0, w_max=20.0)
    engine.add_area("A", n=10000, k=100, beta=0.05)
    engine.add_stimulus("s", size=100)
    engine.add_connectivity("s", "A", p=0.05)
    result = engine.project_into("A", from_stimuli=["s"], from_areas=[])
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ProjectionResult:
    """Result of projecting into one area.

    All arrays are CPU numpy -- the engine converts internally.
    This keeps Brain backend-agnostic.
    """
    winners: np.ndarray     # uint32, shape (k,) -- new winner indices
    num_first_winners: int  # how many fired for the first time this step
    num_ever_fired: int     # total w (ever-fired count) after this step


class ComputeEngine(ABC):
    """Abstract base for all compute backends.

    The engine owns ALL compute state: connectome weights, activation
    buffers, per-area tracking (ever-fired counts, winner histories).
    Brain holds only logical topology and delegates all math here.

    Contract:
    - All public methods accept and return CPU numpy arrays.
    - Internal computation may use any device or format.
    - The engine is responsible for its own memory management.
    """

    # -- Area / stimulus registration --

    @abstractmethod
    def add_area(self, name: str, n: int, k: int, beta: float) -> None:
        """Register a new area and initialise internal compute state."""

    @abstractmethod
    def add_stimulus(self, name: str, size: int) -> None:
        """Register a new stimulus."""

    @abstractmethod
    def add_connectivity(self, source: str, target: str, p: float) -> None:
        """Declare that *source* can project to *target* with probability *p*.

        Called after both source and target are registered.
        The engine creates connectivity in its native format.
        """

    # -- Projection (the core operation) --

    @abstractmethod
    def project_into(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
    ) -> ProjectionResult:
        """Execute one full projection cycle into the target area.

        Performs (in engine-specific order):
        1. Accumulate inputs from source stimuli and areas
        2. Select top-k winners
        3. Apply Hebbian plasticity (if enabled)
        4. Expand connectivity for first-time winners

        Returns a :class:`ProjectionResult` with CPU numpy arrays.
        """

    # -- State accessors --

    @abstractmethod
    def get_winners(self, area: str) -> np.ndarray:
        """Return current winners for *area* (CPU numpy uint32)."""

    @abstractmethod
    def set_winners(self, area: str, winners: np.ndarray) -> None:
        """Inject winners into *area* (for external_inputs API)."""

    @abstractmethod
    def get_num_ever_fired(self, area: str) -> int:
        """Return *w* (number of neurons that have ever fired) for *area*."""

    # -- Plasticity control --

    @abstractmethod
    def set_beta(self, target: str, source: str, beta: float) -> None:
        """Set plasticity rate for the *source* -> *target* connection."""

    @abstractmethod
    def get_beta(self, target: str, source: str) -> float:
        """Get plasticity rate for the *source* -> *target* connection."""

    # -- Assembly fixation --

    @abstractmethod
    def fix_assembly(self, area: str) -> None:
        """Freeze the current assembly so projection returns it unchanged."""

    @abstractmethod
    def unfix_assembly(self, area: str) -> None:
        """Allow the assembly to change in future projections."""

    @abstractmethod
    def is_fixed(self, area: str) -> bool:
        """Return whether the area's assembly is currently fixed."""

    # -- Identity --

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name (e.g. ``'numpy_sparse'``)."""


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------

_ENGINE_REGISTRY: Dict[str, type] = {}
_ENGINES_LOADED = False


def _ensure_engines_loaded():
    """Import built-in engine modules so they register themselves."""
    global _ENGINES_LOADED
    if _ENGINES_LOADED:
        return
    _ENGINES_LOADED = True
    try:
        from . import numpy_engine  # noqa: F401 — registers numpy_sparse, numpy_explicit
    except ImportError:
        pass
    try:
        from . import cuda_engine  # noqa: F401 — registers cuda_implicit (if cupy available)
    except ImportError:
        pass


def register_engine(engine_name: str, cls: type) -> None:
    """Register an engine class under the given name."""
    _ENGINE_REGISTRY[engine_name] = cls


def list_engines() -> List[str]:
    """Return names of all registered engines."""
    _ensure_engines_loaded()
    return list(_ENGINE_REGISTRY.keys())


def create_engine(engine_name: str, **kwargs) -> ComputeEngine:
    """Instantiate a registered engine by name.

    Extra *kwargs* are forwarded to the engine constructor.

    Example::

        engine = create_engine("numpy_sparse", p=0.05, seed=42, w_max=20.0)
    """
    _ensure_engines_loaded()
    if engine_name not in _ENGINE_REGISTRY:
        available = ", ".join(_ENGINE_REGISTRY.keys()) or "(none)"
        raise ValueError(
            f"Unknown engine {engine_name!r}. Available: {available}"
        )
    return _ENGINE_REGISTRY[engine_name](**kwargs)
