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

    from neural_assemblies.core.engine import create_engine

    engine = create_engine("numpy_sparse", p=0.05, seed=0, w_max=20.0)
    engine.add_area("A", n=10000, k=100, beta=0.05)
    engine.add_stimulus("s", size=100)
    engine.add_connectivity("s", "A", p=0.05)
    result = engine.project_into("A", from_stimuli=["s"], from_areas=[])
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
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
    total_activation: float = 0.0  # sum of input signals to winners

    # -- Pre-k-WTA activation data (only populated when record_activation=True) --
    pre_kwta_inputs: Optional[np.ndarray] = None      # float32, full all_inputs before topk
    pre_kwta_prev_only: Optional[np.ndarray] = None   # float32, prev_winner_inputs before penalties
    pre_kwta_total: float = 0.0                        # sum of all_inputs (scalar)
    #: How many candidates ``pre_kwta_total`` was summed over, i.e.
    #: ``len(all_inputs)``. REPORTED BECAUSE A SUM WITHOUT ITS COUNT IS NOT A
    #: MEASUREMENT: every consumer that wanted a per-candidate figure had to
    #: guess a divisor, and both of them guessed `area.w` -- the MATERIALISED
    #: count, which is not the candidate set and grows with training history.
    #: That one mis-guess sets the entire scale of the P600 (#104): `w` grows
    #: 7.8x across probe arms and the measured gap falls 12.5x. Same defect
    #: class as `.w` meaning two things -- see
    #: research/notes/language/erp_scale_is_an_implementation_detail.md.
    pre_kwta_count: int = 0


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

    def validate_brain_identity(self, *, p, seed, w_max, homeostasis=None) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-engine-identity

        Reject conflicting parameters before an existing engine is adopted.
        Backends with other seed storage must override this nonmutating check.
        This checks shared construction identity, not all model semantics.
        """
        missing = object()
        actual = {"p": getattr(self, "p", missing),
                  "seed": getattr(self, "seed", getattr(self, "_seed", missing)),
                  "w_max": getattr(self, "w_max", missing)}
        requested = {"p": p, "seed": seed, "w_max": w_max}
        for name, value in actual.items():
            if value is missing:
                raise ValueError(f"{type(self).__name__} cannot validate Brain {name}; "
                                 "implement validate_brain_identity for this backend")
            if requested[name] != value:
                raise ValueError(f"Brain {name}={requested[name]!r} conflicts with supplied "
                                 f"engine {name}={value!r}; pass matching parameters")

        if homeostasis is not None:
            from ._homeostasis import HomeostasisConfig
            if HomeostasisConfig.from_engine(self) != homeostasis:
                raise ValueError("Brain homeostasis conflicts with supplied engine; pass matching settings")

    # -- Area / stimulus registration --

    supports_slots = False
    supports_refraction = False
    supports_fiber_learning_masks = False

    @contextmanager
    def suppress_fiber_learning(self, fibers):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-fiber-learning

        Scoped suppression is additive under nesting and never changes drive.
        Backends must opt in after implementing the learning-only predicate.
        """
        fibers = frozenset(fibers)
        if fibers and not self.supports_fiber_learning_masks:
            raise NotImplementedError(f"{type(self).__name__} does not support fiber learning masks")
        previous = getattr(self, "_suppressed_learning_fibers", None)
        self._suppressed_learning_fibers = (previous or frozenset()) | fibers
        try:
            yield self
        finally:
            if previous is None:
                del self._suppressed_learning_fibers
            else:
                self._suppressed_learning_fibers = previous

    def fiber_learning_allowed(self, source, target):
        return (source, target) not in getattr(self, "_suppressed_learning_fibers", ())

    @abstractmethod
    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0) -> None:
        """Register a new area and initialise internal compute state.

        Parameters *refractory_period* and *inhibition_strength* control
        Long-Range Inhibition (LRI) for sequence operations.  When
        ``refractory_period > 0``, recently-fired neurons receive a penalty
        during winner selection so that sequences advance instead of
        oscillating between consecutive assemblies.
        """

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
        record_activation: bool = False,
    ) -> ProjectionResult:
        """Execute one full projection cycle into the target area.

        Performs (in engine-specific order):
        1. Accumulate inputs from source stimuli and areas
        2. Select top-k winners
        3. Apply Hebbian plasticity (if enabled)
        4. Expand connectivity for first-time winners

        If *record_activation* is True, the result includes pre-k-WTA
        activation snapshots (``pre_kwta_inputs``, ``pre_kwta_prev_only``,
        ``pre_kwta_total``).

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

    # -- Optional overrides (concrete defaults) --

    def snapshot_activity(self):
        """Snapshot declared per-area dynamics, without copying learned fibers."""
        return [state.snapshot_activity() for state in self._areas.values()]

    def probe_target_ready(self, name) -> bool:
        """Whether the target has a population usable without recruitment.

        Engines with a no-recruitment mode need at least k materialized neurons.
        Full-population engines need no such initialization. Readiness says
        nothing about learned fibers, informative drive or successful recall.
        """
        state = self._areas[name]
        return not hasattr(self, "_no_recruitment") or bool(state.w >= state.k)

    def validate_probe_target(self, name):
        """A sampled read cannot initialize a population as a side effect."""
        if getattr(self, "_no_recruitment", False) and not self.probe_target_ready(name):
            raise ValueError(
                f"{name}: read_only requires at least k materialized neurons; "
                "initialize/train the area or materialize it before probing")

    def get_neuron_id_mapping(self, area: str) -> Optional[list]:
        """Return compact-index-to-neuron-ID mapping, or None.

        Engines that use compact indexing (e.g. sparse engines where only
        fired neurons are tracked) override this to return the mapping.
        Default returns None (indices are neuron IDs).
        """
        return None

    def clear_refractory(self, area: str) -> None:
        """Clear refractory history for an area.

        Used to reset LRI state between memorization and recall phases,
        or between independent trials.  Default is a no-op (suitable for
        engines without LRI support).
        """

    def set_competition_policy(self, area: str, policy) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-runtime-policy

        Backends must opt into runtime competition changes. A facade must not
        attach a field to backend storage and assume the selector consumes it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement runtime competition policies")

    def set_lri(self, area: str, refractory_period: int,
                inhibition_strength: float) -> None:
        """Update LRI parameters for an area at runtime.

        Enables or disables Long-Range Inhibition after area creation.
        Useful for enabling LRI only during recall while keeping it
        disabled during memorization. Unsupported nondefault requests raise.
        Specification: neural_assemblies/ir/VERIFICATION.md#contract-area-controls
        """
        from ._homeostasis import validate_lri_parameters
        refractory_period, inhibition_strength = validate_lri_parameters(
            refractory_period, inhibition_strength)
        if refractory_period != 0 or inhibition_strength != 0:
            raise NotImplementedError(f"{type(self).__name__} does not implement LRI")

    def set_refracted(self, area: str, enabled: bool,
                      strength: float = 0.0) -> None:
        """Enable or disable refracted mode for an area.

        Refracted mode accumulates a bias: each time a neuron fires, its bias
        grows, making it progressively harder to fire again.  This is distinct
        from LRI (sliding-window penalty).

        ENABLING IT ON AN ENGINE THAT DOES NOT IMPLEMENT IT RAISES.  This
        default used to be a silent no-op, which is the failure this repo keeps
        rediscovering: a mechanism configured, accepted, and never run, whose
        signature is "X seems to have little effect".  See
        [[silent-no-op-dead-fibers]] and `_reject_unsupported` in
        `numpy_engine/_exact.py`, which applies the same rule to `add_area`.

        Refraction is not a modifier that merely sharpens a result.  Ablated
        from the reference FSM it takes the mod-3 task from 3/3 seeds to 0/3
        and the arc's across-symbol overlap from 0.000 to 0.989 -- the area
        stops being a conjunction at all.  An engine that silently ignores the
        request therefore does not return a slightly different number; it
        returns a different experiment.

        Requesting the DEFAULT (``enabled=False``) is not a request for the
        mechanism and stays a no-op everywhere, matching `_reject_unsupported`.
        """
        if enabled:
            raise NotImplementedError(
                f"{type(self).__name__}.set_refracted({area!r}, enabled=True) "
                f"is not implemented by this engine, so refraction would be "
                f"configured and never applied. Use numpy_sparse (or torch/cuda, "
                f"which inherit it), or leave refracted=False."
            )

    def clear_refracted_bias(self, area: str) -> None:
        """Reset accumulated refracted bias to zero.

        Default is a no-op.
        """

    def normalize_weights(self, target: str, source: str = None) -> None:
        """Column-normalize weights into *target* so each neuron sums to 1.0.

        If *source* is given, only that connection is normalized.
        Otherwise all connections into *target* are normalized.
        Default is a no-op.
        """

    def reset_area_connections(self, area: str) -> None:
        """Reset all area->area connections involving *area* to initial state.

        Preserves stimulus->area connections.  Used by the ``separate``
        operation to give a second stimulus a fresh recurrent landscape.
        Default is a no-op (suitable for engines without persistent
        area->area connectivity, e.g. implicit hash-based engines).
        """

    def project_into_batch(
        self,
        configs: List[tuple],
        plasticity_enabled: bool = True,
        record_activation: bool = False,
    ) -> Dict[str, "ProjectionResult"]:
        """Project into multiple target areas, potentially in parallel.

        Default falls back to sequential :meth:`project_into` calls.
        Engines with batch-capable hardware may override for throughput.
        """
        return {
            target: self.project_into(
                target, stims, areas, plasticity_enabled,
                record_activation=record_activation)
            for target, stims, areas in configs
        }

    def project_rounds(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        rounds: int,
        plasticity_enabled: bool = True,
        record_activation: bool = False,
    ) -> "ProjectionResult":
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-engine-rounds

        Repeat project_into sequentially and return its last ProjectionResult.

        Requires a positive integer count. Edges and plasticity/recording flags
        are forwarded unchanged on every step. This engine-level API does not
        apply Brain routing, inhibition, facade synchronization or histories.
        """
        if isinstance(rounds, bool) or not isinstance(rounds, (int, np.integer)) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        result = None
        for _ in range(rounds):
            result = self.project_into(
                target, from_stimuli, from_areas, plasticity_enabled,
                record_activation=record_activation)
        return result

    # -- Projection fidelity (exact vs compiled topology) -----------------

    def set_projection_fidelity(self, fidelity: str) -> None:
        """Set global projection fidelity (``exact`` or ``compiled``).

        Engines that do not implement compiled topology ignore this call.
        """

    def get_projection_fidelity(self) -> str:
        """Return global projection fidelity (default ``exact``)."""
        return "exact"

    def preallocate_stim_targets(self, target: str, min_columns: int) -> None:
        """Extend stim→*target* 1-D weight vectors to *min_columns* (no-op default)."""

    # -- Materialization --

    def fiber_extent(self, source: str, target: str) -> Optional[int]:
        """How many of *target*'s neurons this fiber has COLUMNS for, or None.

        WHY THIS EXISTS.  Three different numbers are in play for a lazily
        materialized fiber and callers were picking between them by hand::

            area.w            neurons the engine has materialized -- EXCEPT on
                              the explicit engine, where it is len(winners)==k
                              and is not an extent at all
            conn._log_cols    the fiber's logical column watermark, which can
                              lag `w` when the area grew through some OTHER
                              fiber
            conn.weights.shape[1]   PHYSICAL capacity, which over-runs both
                              because growth doubles

        Measured on one recurrent protocol at n=1000, the three gave 247 / 213 /
        367 columns for the same fiber at the same instant, and the quantity
        being measured through them moved 1.87 / 1.65 / 2.84.  Choosing wrong is
        not a rounding error, and nothing in the API made the choice explicit.

        ``None`` means the question does not apply: the fiber is dense, so every
        one of the target's neurons has a column and there is no watermark to
        disagree with.  Callers must treat ``None`` as "not applicable", NOT as
        zero -- that distinction is the whole point of an Optional return.

        Engines that do not materialize lazily inherit this default.
        """
        return None

    def materialized_count(self, area: str) -> Optional[int]:
        """Neurons of *area* that have been given a compact index, or None.

        The companion to `fiber_extent`.  ``None`` for engines that allocate all
        ``n`` up front, where the concept does not apply.  Deliberately NOT
        named ``w``: `w` already means two different things depending on which
        object is asked, and adding a third reader of that name to the ABC is
        how the confusion propagates.
        """
        return None

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
        from . import numpy_engine  # noqa: F401 — numpy_sparse, numpy_explicit, numpy_exact
    except ImportError:
        pass
    try:
        from . import cuda_engine  # noqa: F401 — registers cuda_implicit (if cupy available)
    except ImportError:
        pass
    try:
        from . import cupy_engine  # noqa: F401 — registers cupy_sparse (if cupy available)
    except ImportError:
        pass
    try:
        from . import torch_engine  # noqa: F401 — registers torch_sparse (if torch+CUDA available)
    except ImportError:
        pass


def register_engine(engine_name: str, cls: type) -> None:
    """Register an engine class under the given name."""
    _ENGINE_REGISTRY[engine_name] = cls


# Which module registers which engine, so a caller that already knows the engine
# it wants can import ONLY that one. Loading every engine to answer "is
# numpy_sparse available?" drags in torch (~9s), which profiling showed was 9.1s
# of a 22.6s CPU training run -- paid once per process, including per xdist
# worker. Keep in sync with _ensure_engines_loaded above.
_ENGINE_MODULES = {
    "numpy_sparse": "numpy_engine",
    "numpy_explicit": "numpy_engine",
    "numpy_exact": "numpy_engine",
    "cuda_implicit": "cuda_engine",
    "cupy_sparse": "cupy_engine",
    "torch_sparse": "torch_engine",
}


def ensure_engine(engine_name: str) -> bool:
    """Import just the module providing *engine_name*; True if now registered.

    Narrow counterpart to ``_ensure_engines_loaded``: same registration path and
    same ImportError tolerance, but it does not import engines the caller has
    not asked for. Unknown names return False rather than raising, matching how
    ``list_engines`` membership tests behave.
    """
    if engine_name in _ENGINE_REGISTRY:
        return True
    module = _ENGINE_MODULES.get(engine_name)
    if module is None:
        return False
    try:
        import importlib

        importlib.import_module(f".{module}", __package__)
    except ImportError:
        return False
    return engine_name in _ENGINE_REGISTRY


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
    # Import ONLY the requested engine's module. `_ensure_engines_loaded` pulls
    # in every backend, and `cuda_engine` imports torch at module scope -- so
    # asking for "numpy_sparse" was paying for torch. Profiled on a research
    # trial: ~13s per process of `nt.listdir` / importlib path scanning /
    # torch._register_to_dispatcher, none of which the numpy path uses. That is
    # charged PER PROCESS, so it also capped `parallel_seeds` (measured 1.57x
    # where ~3x was available) and every pytest-xdist worker.
    #
    # The narrow loader and its module map already existed for exactly this
    # reason -- see the comment on `_ENGINE_MODULES` -- but `create_engine`,
    # the main entry point, never used them.
    #
    # Fall back to the broad load only when the narrow one fails, so an unknown
    # or aliased name still reports the full list of what is available.
    if not ensure_engine(engine_name):
        _ensure_engines_loaded()
    if engine_name not in _ENGINE_REGISTRY:
        available = ", ".join(_ENGINE_REGISTRY.keys()) or "(none)"
        raise ValueError(
            f"Unknown engine {engine_name!r}. Available: {available}"
        )
    return _ENGINE_REGISTRY[engine_name](**kwargs)
