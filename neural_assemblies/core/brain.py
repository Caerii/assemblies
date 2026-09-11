# brain.py
"""
Neural Assembly Brain Simulation

This module implements the core Brain class for simulating neural assemblies
based on the Assembly Calculus framework introduced by Papadimitriou et al.
in "Brain Computation by Assemblies of Neurons" (PNAS, 2020).

The Brain class orchestrates the fundamental operations of the Assembly Calculus:
- Projection: Creating new assemblies in downstream areas
- Association: Increasing overlap between assemblies
- Merge: Combining assemblies to form new representations

Biological Context:
- Implements the NEMO model (Mitropolsky et al., 2023) for biological realism
- Models Hebbian plasticity: "neurons that fire together, wire together"
- Simulates sparse neural activity patterns found in biological brains
- Supports both explicit (full simulation) and sparse (statistical) modes
- ``Brain.projection_fidelity``: ``exact`` (microscopic) vs ``compiled`` / ``fuzzy``
  (top-k on frozen pregrown connectomes — topology quantization for scale)

Mathematical Foundation:
- Assembly Calculus operations preserve overlap properties
- Winner-take-all selection implements sparse coding principles
- Synaptic plasticity follows Hebbian learning rules
- Statistical approximations enable scalable simulations
"""

import contextlib
from copy import deepcopy
from numbers import Integral
import os
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from .backend import get_xp, to_cpu, detect_best_engine
from .engine import (
    ComputeEngine,
    create_engine,
    engine_type,
    validate_deterministic_allocation,
    validate_engine_boolean_option,
)
from .registration import validate_round_count, validate_input_noise, validate_plasticity_rate, validate_area_registration, validate_stimulus_registration
from ._homeostasis import (
    HomeostasisConfig,
    check_area_homeostasis,
    validate_homeostasis_capabilities,
    validate_lri_parameters,
)
from .index_spaces import CompactIdx, to_neuron_ids, validated_indices
from .semantics import ModelSemantics, SampledRecurrencePolicy
from .activity import PopulationCounts, PreKwtaObservation
from .feedforward_inhibition import (
    FeedforwardInhibitionConfig,
    validate_feedforward_inhibition_capability,
)
from .projection_fidelity import validate_projection_fidelity_capability

from .area import Area
from .stimulus import Stimulus
from .connectome import Connectome, dense_connectome_or_new, is_dense_connectome

# ImageActivationEngine is used by activate_with_image()
try:
    from ..compute.image_activation import ImageActivationEngine
except ImportError:
    from compute.image_activation import ImageActivationEngine

try:
    from ..constants.default_params import DEFAULT_P, DEFAULT_BETA, DEFAULT_W_MAX
except ImportError:
    # Fallback for when running as script
    from constants.default_params import DEFAULT_P, DEFAULT_BETA, DEFAULT_W_MAX

class Brain:
    """
    Neural Assembly Brain Orchestrator
    
    Manages the simulation of neural assemblies across multiple brain areas,
    implementing the Assembly Calculus framework for neural computation.
    
    This class serves as the central coordinator for:
    - Neural area management and connectivity
    - Assembly projection operations (Assembly Calculus)
    - Synaptic plasticity and learning
    - Stimulus processing and integration
    
    Biological Principles:
    - Sparse neural coding: Only k neurons fire per area per timestep
    - Hebbian plasticity: Synaptic weights strengthen with co-activation
    - Hierarchical processing: Information flows through area hierarchies
    - Statistical efficiency: Sparse simulation for large-scale networks
    
    Assembly Calculus Operations:
    - Projection: A → B (assembly A projects to create assembly B)
    - Association: A + B → A' + B' (assemblies A and B become more similar)
    - Merge: A + B → C (assemblies A and B combine to form assembly C)
    
    References:
    - Papadimitriou, C. H., et al. "Brain Computation by Assemblies of Neurons." 
      Proceedings of the National Academy of Sciences 117.25 (2020): 14464-14472.
    - Mitropolsky, D., et al. "The Architecture of a Biologically Plausible 
      Language Organ." 2023.
    """

    def __init__(
        self,
        p: float = DEFAULT_P,
        save_size: bool = True,
        save_winners: bool = False,
        seed: int = 0,
        w_max: float = DEFAULT_W_MAX,
        engine="auto",
        deterministic: bool = False,
        gpu_sampling: bool | None = None,
        dense_drive: bool | None = None,
        n_hint: int = 0,
        projection_fidelity: str = "exact",
        inhibitory_prob: float = 0.0,
        inhibitory_weight: float = -0.2,
        synaptic_scaling: "bool | frozenset | set | tuple" = False,
        synaptic_scaling_deferred: bool = False,
        recurrent_projection: bool = False,
        norm_init: bool | None = None,
        sampled_recurrence_policy: str = "warn",
        model_semantics=None,
    ):
        """
        Initialize a neural assembly brain simulation.

        Args:
            p (float): Connection probability between neurons (0 < p < 1).
                      Typical values: 0.01-0.1 for large networks.
            seed (int): Random seed for reproducible simulations.
            engine: ComputeEngine instance, engine name string, or ``"auto"``
                   (default) to select the best available backend.
                   Examples: ``"numpy_sparse"``, ``"cuda_implicit"``, or a
                   pre-constructed ComputeEngine instance. With an instance,
                   p, seed and w_max must explicitly match its values (including
                   when Brain defaults are used). Conflicts raise before adoption;
                   normalization and scaling settings must match too. Use
                   HomeostasisConfig.as_kwargs() to share those settings.
                   See ir/VERIFICATION.md#contract-engine-identity.
            deterministic (bool): If True, request exact-fit allocation and the
                   engine's deterministic sampling path. This is an execution
                   policy, not a promise of cross-backend or cross-version bit
                   identity. Unsupported engines reject it. If False (default),
                   capable engines may use amortised growth and faster sampling.
            gpu_sampling (bool | None): Torch-only choice of GPU versus CPU
                   candidate sampling. ``None`` uses the backend default and
                   other engines reject an explicit value.
            dense_drive (bool | None): Torch-only choice to score every neuron
                   candidate instead of sampled order statistics. ``None`` uses
                   the backend default and other engines reject an explicit value.
            n_hint (int): Expected neuron count per area.  When
                   ``engine="auto"``, this guides engine selection: n >= 1M
                   with GPU available selects ``torch_sparse`` (CSR, GPU),
                   otherwise ``numpy_sparse`` (CPU).
            projection_fidelity (str): Legacy selection mode: "exact" or
                   "compiled" / "fuzzy". "exact" does not select a fixed
                   connectome; see core.projection_fidelity for backend behavior.
            norm_init (bool | None): Request per-fiber initialization
                   normalization. ``None`` selects the Brain default: enabled
                   when the chosen engine supports it and disabled otherwise.
                   Implementations differ between sampled and fixed-connectome
                   engines. Pin it in protocols; no scientific equivalence or
                   recurrence-safety claim follows from enabling it.
            recurrent_projection (bool): Legacy project_rounds schedule flag,
                   False by default. Non-explicit targets keep a supplied
                   self-edge only when this and either norm_init or full
                   synaptic_scaling are enabled. Explicit targets keep it.
                   Ordinary project calls use their supplied edge maps directly.
                   ops.project selects recurrence with its own argument.
            sampled_recurrence_policy (str): What to do when recurrence targets
                   an incompletely materialized sampled NumPy area: ``warn``
                   once (default), ``acknowledged`` for deliberate comparison,
                   or ``forbid`` before the engine draws randomness or mutates.
            model_semantics (ModelSemantics | mapping | None): Optional required
                   semantics. Brain compares the complete normalized object to
                   the selected engine and rejects any mismatch before area or
                   stimulus registration. The engine remains an explicit choice.
        """
        sampled_policy = SampledRecurrencePolicy.normalize(
            sampled_recurrence_policy
        )
        requested_semantics = (
            None
            if model_semantics is None
            else ModelSemantics.normalize(model_semantics)
        )
        if isinstance(engine, str) and engine == "auto":
            engine = detect_best_engine(n_hint)
        if not isinstance(engine, (str, ComputeEngine)):
            raise TypeError(
                "engine must be a string name or ComputeEngine instance, "
                f"got {type(engine)}"
            )
        owner_type = engine_type(engine) if isinstance(engine, str) else type(engine)
        deterministic = validate_deterministic_allocation(owner_type, deterministic)
        if gpu_sampling is None:
            gpu_sampling = (
                bool(getattr(engine, "_gpu_sampling", True))
                if isinstance(engine, ComputeEngine)
                and owner_type.supports_gpu_sampling
                else (True if owner_type.supports_gpu_sampling else None)
            )
        else:
            gpu_sampling = validate_engine_boolean_option(
                owner_type, "gpu_sampling", gpu_sampling,
                "supports_gpu_sampling",
            )
        if dense_drive is None:
            dense_drive = (
                bool(getattr(engine, "dense_drive", False))
                if isinstance(engine, ComputeEngine)
                else False
            )
        else:
            dense_drive = validate_engine_boolean_option(
                owner_type, "dense_drive", dense_drive,
                "supports_dense_drive",
            )
        projection_fidelity = validate_projection_fidelity_capability(
            owner_type, projection_fidelity
        )
        supports_norm_init = bool(owner_type.supports_norm_init)
        if norm_init is None:
            norm_init = (
                supports_norm_init
                if isinstance(engine, str)
                else HomeostasisConfig.from_engine(engine).norm_init
            )
        homeostasis = HomeostasisConfig(
            norm_init, synaptic_scaling, synaptic_scaling_deferred
        )
        synaptic_scaling = homeostasis.synaptic_scaling
        validate_homeostasis_capabilities(owner_type, homeostasis)
        feedforward_inhibition = FeedforwardInhibitionConfig(
            inhibitory_prob, inhibitory_weight
        )
        validate_feedforward_inhibition_capability(
            owner_type, feedforward_inhibition
        )
        if isinstance(engine, ComputeEngine):
            engine.validate_brain_identity(
                p=p,
                seed=seed,
                w_max=w_max,
                homeostasis=homeostasis,
                feedforward_inhibition=feedforward_inhibition,
                projection_fidelity=projection_fidelity,
                deterministic=deterministic,
                gpu_sampling=gpu_sampling,
                dense_drive=dense_drive,
            )
        self.p = p
        self.w_max = w_max
        self.save_size = save_size
        self.save_winners = save_winners
        self.deterministic = deterministic
        self.gpu_sampling = gpu_sampling
        self.dense_drive = dense_drive
        self.areas: Dict[str, Area] = {}
        self.stimuli: Dict[str, Stimulus] = {}
        self.connectomes_by_stimulus: Dict[str, Dict[str, Connectome]] = {}
        self.connectomes: Dict[str, Dict[str, Connectome]] = {}
        self.rng = np.random.default_rng(seed)
        self._conn_rng_gen = np.random.default_rng(seed + 0x9E3779B9)
        self.disable_plasticity = False
        # Per-fiber plasticity override: (src, dst) -> enabled.  Missing keys default True.
        self.plasticity_mask: dict[tuple[str, str], bool] = {}

        # Compute engine — required, defaults to auto-detected best backend
        if isinstance(engine, str):
            engine_kwargs = dict(
                p=p, seed=seed, w_max=w_max, deterministic=deterministic,
            )
            # Admission above proves the selected engine implements this pair.
            if feedforward_inhibition.enabled:
                engine_kwargs.update(feedforward_inhibition.as_kwargs())
            # Forward the VALUE: True means every target area (legacy);
            # a collection of area names scopes scaling to those targets
            # only (see NumpySparseEngine._normalize_area_columns).
            if synaptic_scaling:
                engine_kwargs["synaptic_scaling"] = synaptic_scaling
                if synaptic_scaling_deferred:
                    engine_kwargs["synaptic_scaling_deferred"] = True
            # FORWARDED ONLY WHEN TRUE, so `numpy_explicit` -- whose
            # constructor does not accept it -- is unaffected. The invariant
            # that makes this safe: OMISSION MEANS FALSE, so every engine that
            # accepts `norm_init` MUST default it to False. An engine that
            # defaults to True silently upgrades `Brain(norm_init=False)` to
            # the production substrate, which is invisible in every log and
            # wrong in exactly the runs that pinned it off on purpose (paper
            # parity). Pinned by `test_engine_norm_init_contract`.
            if norm_init:
                engine_kwargs["norm_init"] = True
            if gpu_sampling is not None:
                engine_kwargs["gpu_sampling"] = gpu_sampling
            if dense_drive:
                engine_kwargs["dense_drive"] = True
            self._engine: ComputeEngine = create_engine(engine, **engine_kwargs)
        else:
            self._engine = engine

        # Record the effective backend policy. Deterministic Torch execution
        # disables GPU sampling even when its request default was true.
        self.gpu_sampling = (
            bool(self._engine._gpu_sampling)
            if self._engine.supports_gpu_sampling
            else None
        )
        self.dense_drive = (
            bool(self._engine.dense_drive)
            if self._engine.supports_dense_drive
            else False
        )

        actual_semantics = self._engine.describe_model_semantics()
        if requested_semantics is not None:
            mismatch = requested_semantics.mismatch(actual_semantics)
            if mismatch:
                details = ", ".join(
                    f"{name}: requested {requested!r}, engine implements {actual!r}"
                    for name, (requested, actual) in mismatch.items()
                )
                raise ValueError(f"model_semantics mismatch: {details}")
        self._model_semantics = actual_semantics

        if self._engine.supports_sampled_recurrence_policy:
            if isinstance(engine, ComputeEngine):
                actual_policy = getattr(
                    self._engine,
                    "sampled_recurrence_policy",
                    SampledRecurrencePolicy.WARN,
                )
                if actual_policy is not sampled_policy:
                    raise ValueError(
                        "Brain sampled_recurrence_policy conflicts with supplied "
                        "engine; pass matching policies"
                    )
            self._engine._configure_sampled_recurrence_policy(sampled_policy)
        self._sampled_recurrence_policy = sampled_policy
        self.feedforward_inhibition = feedforward_inhibition

        self._engine.set_projection_fidelity(projection_fidelity)

        # Secondary engine for explicit areas (lazily created)
        self._explicit_engine: ComputeEngine = None
        self._seed = seed

        # Inter-area inhibition groups for winner-take-all
        self._mutual_inhibition_groups: List[List[str]] = []
        #: AC area/fiber inhibition. None until something is actually
        #: inhibited, so a Brain that never gates pays nothing for it.
        self._inhibition = None
        # One-time incoming-weight normalization (reference `norm_init`).
        # Legacy schedule inputs; see project_rounds's source-linked contract.
        self.norm_init: bool = norm_init
        self._synaptic_scaling: bool = synaptic_scaling
        self._synaptic_scaling_deferred: bool = synaptic_scaling_deferred
        # Select target self-recurrence in the legacy project_rounds schedule.
        self.recurrent_projection: bool = recurrent_projection
        # Total synaptic drive per target from the most recent projection,
        # summed over the SELECTED winners.
        self.last_activation_scores: Dict[str, float] = {}
        # Opt-in: also record global pre-k-WTA energy (sum of all_inputs over
        # every neuron, before winner selection). Off by default because it
        # copies a length-n vector per projection.
        self.record_activation: bool = False
        self.last_pre_kwta_totals: Dict[str, float] = {}
        #: Candidates each total was summed over. A SUM WITHOUT ITS
        #: COUNT is not a measurement: every consumer wanting a
        #: per-candidate figure had to guess a divisor, and both of
        #: them guessed `area.w` -- the MATERIALISED count, which is
        #: not the candidate set. That guess sets the whole scale of
        #: the P600 (#104).
        self.last_pre_kwta_counts: Dict[str, int] = {}

        # Used by activate_with_image()
        self.image_activation_engine = ImageActivationEngine()

    @property
    def sampled_recurrence_policy(self) -> SampledRecurrencePolicy:
        """Immutable admission policy selected when this Brain was constructed."""
        return getattr(
            self,
            "_sampled_recurrence_policy",
            SampledRecurrencePolicy.WARN,
        )

    @property
    def model_semantics(self) -> ModelSemantics:
        """Immutable, executable description of the primary engine path."""
        return getattr(
            self,
            "_model_semantics",
            self._engine.describe_model_semantics(),
        )

    def set_fiber_plasticity(self, src: str, dst: str, enabled: bool) -> None:
        """Enable or disable Hebbian updates on one directed fiber (E6)."""
        self.plasticity_mask[(src, dst)] = enabled

    def fiber_plasticity_enabled(self, src: str, dst: str) -> bool:
        """Whether Hebbian plasticity is allowed on *src* → *dst*."""
        if self.disable_plasticity:
            return False
        return self.plasticity_mask.get((src, dst), True)

    def init_reciprocal_connectome(
        self,
        forward_src: str,
        forward_dst: str,
        *,
        init: str = "transpose_forward",
    ) -> None:
        """Initialize backward connectome from an existing forward fiber (E4).

        ``init='transpose_forward'`` copies ``forward_src→forward_dst`` weights
        transposed into ``forward_dst→forward_src`` with column normalization.
        """
        if init != "transpose_forward":
            raise ValueError(f"unsupported reciprocal init: {init!r}")
        if forward_src not in self.connectomes:
            raise KeyError(forward_src)
        if forward_dst not in self.connectomes[forward_src]:
            raise KeyError(f"{forward_src}→{forward_dst}")
        w_fwd = self.connectomes[forward_src][forward_dst].weights
        w_back = np.asarray(w_fwd.T, dtype=np.float32).copy()
        col_sums = np.maximum(w_back.sum(axis=0, keepdims=True), 1e-12)
        w_back /= col_sums
        if forward_dst not in self.connectomes:
            self.connectomes[forward_dst] = {}
        if forward_src not in self.connectomes[forward_dst]:
            raise KeyError(f"{forward_dst}→{forward_src}")
        self.connectomes[forward_dst][forward_src].weights = w_back
        if self._explicit_engine is not None:
            econn = self._explicit_engine._area_conns.get(forward_dst, {}).get(forward_src)
            if econn is not None:
                econn.weights = w_back
        self._sync_engine_connectomes()

    @property
    def engine_name(self) -> str:
        """Return the active compute engine name."""
        return self._engine.name

    @property
    def projection_fidelity(self) -> str:
        """Global projection fidelity: ``exact`` or ``compiled`` (fuzzy)."""
        return self._engine.get_projection_fidelity()

    @projection_fidelity.setter
    def projection_fidelity(self, value: str) -> None:
        value = validate_projection_fidelity_capability(type(self._engine), value)
        self._engine.set_projection_fidelity(value)

    @property
    def area_by_name(self) -> Dict[str, Area]:
        """Backward-compatible alias for self.areas."""
        return self.areas

    @property
    def _conn_rng(self):
        """Dedicated seeded stream for connectome wiring draws.

        Kept SEPARATE from ``self.rng`` on purpose. ``self.rng`` is also consumed
        by ``area.neuron_id_pool`` permutations, so drawing wiring from it shifts
        every subsequent permutation and silently changes which neurons each area
        recruits -- measured at 22 otherwise-passing tests broken by exactly that.

        Resolved LAZILY so brains unpickled from an older ``.cache/`` parser
        backbone (which predate this attribute) still load: those pickles have no
        ``_conn_rng_gen``, and raising here surfaced as 11 setup ERRORs in
        TestWordOrderTypology rather than as anything to do with wiring.
        """
        gen = self.__dict__.get("_conn_rng_gen")
        if gen is None:
            gen = np.random.default_rng(getattr(self, "_seed", 0) + 0x9E3779B9)
            self.__dict__["_conn_rng_gen"] = gen
        return gen

    def add_area(self, area_name: str, n: int, k: int, beta: float = DEFAULT_BETA,
                 explicit: bool = False, refractory_period: int = 0,
                 inhibition_strength: float = 0.0,
                 refracted: bool = False, refracted_strength: float = 0.0,
                 winner_policy=None, input_noise_std: float = 0.0,
                 slot_count: int = 0):
        """
        Add a neural area to the brain simulation.

        Args:
            area_name (str): Unique identifier for the brain area.
            n (int): Total number of neurons in the area (population size).
            k (int): Assembly size - number of neurons that fire per timestep.
            beta (float): Synaptic plasticity parameter (0 < beta < 1).
            explicit (bool): Whether to use explicit (full) or sparse simulation.
            refractory_period (int): Number of steps of LRI suppression
                (0 = disabled).  When > 0, recently-fired neurons receive
                a penalty during winner selection so that sequences can
                advance instead of oscillating.
            inhibition_strength (float): Magnitude of the LRI penalty.
            refracted (bool): Whether refracted mode is enabled.  When
                True, a cumulative bias grows each time a neuron fires,
                making repeated firing progressively harder.
            refracted_strength (float): Magnitude of per-firing bias
                increment in refracted mode.
        """
        n, k = validate_area_registration(area_name, n, k, existing=self.areas, reserved=self.stimuli)
        input_noise_std = validate_input_noise(input_noise_std)
        if input_noise_std and (explicit or not self._engine.supports_input_noise):
            owner_name = "NumpyExplicitEngine" if explicit else type(self._engine).__name__
            raise NotImplementedError(f"{owner_name} does not implement input_noise_std")
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-refraction-registration
        if refracted:
            owner_type = type(self._engine)
            if explicit:
                # Determine capability without creating/registering the auxiliary engine.
                from .numpy_engine import NumpyExplicitEngine
                owner_type = NumpyExplicitEngine
            if not owner_type.supports_refraction:
                raise NotImplementedError(f"{owner_type.__name__} does not implement refraction")
            check_area_homeostasis(area_name, refracted=True,
                                   synaptic_scaling=getattr(self._engine, "synaptic_scaling", False))
        if slot_count and not explicit and not self._engine.supports_slots:
            raise NotImplementedError(f"{type(self._engine).__name__} does not implement slots")
        area = Area(area_name, n, k, beta, explicit,
                    refractory_period=refractory_period,
                    inhibition_strength=inhibition_strength,
                    refracted=refracted,
                    refracted_strength=refracted_strength,
                    winner_policy=winner_policy,
                    input_noise_std=input_noise_std,
                    slot_count=slot_count)
        self.areas[area_name] = area
        # Initialize neuron id pool for sparse areas (permute 0..n-1)
        if not explicit:
            area.neuron_id_pool = self.rng.permutation(np.arange(n, dtype=np.uint32))
            area.neuron_id_pool_ptr = 0
        self.connectomes[area_name] = {}
        # Initialize connectomes for the new area
        self._initialize_connectomes_for_area(area)
        # ALWAYS register with the main engine so cross-engine source
        # lookups work (e.g., explicit area as source for a sparse target).
        self._engine.add_area(area_name, n, k, beta,
                              refractory_period=refractory_period,
                              inhibition_strength=inhibition_strength,
                              winner_policy=winner_policy,
                              input_noise_std=input_noise_std,
                              **({"slot_count": area.slot_count} if not explicit and area.slot_count else {}))
        if refracted:
            self._engine.set_refracted(area_name, True, refracted_strength)
        # For explicit areas, ALSO register with a dedicated explicit engine
        # that handles full n×n weight matrices and plasticity correctly.
        if explicit:
            explicit_eng = self._engine_for(area)  # lazily creates it
            if explicit_eng is not self._engine:
                self._register_explicit_area(explicit_eng, area)
        # Share engine's connectome objects so b.connectomes[x][y] is the
        # actual object the engine reads/writes during projection.
        self._sync_engine_connectomes()

    def add_stimulus(self, stimulus_name: str, size: int):
        """
        Adds a stimulus to the brain.

        Args:
            stimulus_name (str): Name of the stimulus.
            size (int): Number of firing neurons in the stimulus.
        """
        size = validate_stimulus_registration(stimulus_name, size,
                                             existing=self.stimuli, reserved=self.areas)
        stimulus = Stimulus(stimulus_name, size)
        self.stimuli[stimulus_name] = stimulus
        self.connectomes_by_stimulus[stimulus_name] = {}
        # Initialize connectomes for the new stimulus
        self._initialize_connectomes_for_stimulus(stimulus)
        self._engine.add_stimulus(stimulus_name, size)
        # Also register in explicit engine if it exists
        if self._explicit_engine is not None:
            self._explicit_engine.add_stimulus(stimulus_name, size)
        self._sync_engine_connectomes()

    def add_explicit_area(self, area_name: str, n: int, k: int, beta: float = DEFAULT_BETA,
                          custom_inner_p=None, custom_out_p=None, custom_in_p=None):
        """Add an explicitly-simulated brain area.

        Convenience wrapper around ``add_area(explicit=True)`` using this
        brain's connection probability. Non-None ``custom_*_p`` overrides
        are unsupported and rejected before registering or allocating an area.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-probability
        """
        overrides = {"custom_inner_p": custom_inner_p, "custom_out_p": custom_out_p,
                     "custom_in_p": custom_in_p}
        requested = [name for name, value in overrides.items() if value is not None]
        if requested:
            raise NotImplementedError(
                "add_explicit_area does not implement probability overrides: "
                + ", ".join(requested)
                + ". Omit them only if the brain-wide p is the intended model.")
        self.add_area(area_name, n, k, beta, explicit=True)

    @staticmethod
    def _register_explicit_area(engine, area):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-registration"""
        engine.add_area(area.name, area.n, area.k, area.beta,
                        refractory_period=area.refractory_period,
                        inhibition_strength=area.inhibition_strength,
                        slot_count=area.slot_count, winner_policy=area.winner_policy,
                        input_noise_std=area.input_noise_std)

    def _engine_for(self, area: Area) -> ComputeEngine:
        """Return the correct engine for an area.

        Explicit areas use a dedicated NumpyExplicitEngine.
        Sparse areas use the main engine.
        """
        if area.explicit:
            # The primary dense engine already owns this contract. A second
            # identical engine hides the real arithmetic owner and duplicates
            # every area and stimulus registration.
            if self._engine.name == "numpy_explicit":
                return self._engine
            if self._explicit_engine is None:
                self._explicit_engine = create_engine(
                    "numpy_explicit", p=self.p, seed=self._seed, w_max=self.w_max,
                )
                # Register existing stimuli so stim→area connectomes exist
                for stim_name, stim in self.stimuli.items():
                    self._explicit_engine.add_stimulus(stim_name, stim.size)
                # Register existing explicit areas (in case areas are added
                # before the first explicit area triggers engine creation)
                for existing_name, existing_area in self.areas.items():
                    if existing_area.explicit and existing_name != area.name:
                        self._register_explicit_area(self._explicit_engine, existing_area)
            return self._explicit_engine
        return self._engine

    def population_counts(self, area_name: str) -> PopulationCounts:
        """Return active, lifetime and materialized sizes without `.w`.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-population-counts
        """
        try:
            area = self.areas[area_name]
        except KeyError:
            raise KeyError(f"unknown area {area_name!r}") from None
        owner = self._engine_for(area)
        materialized = owner.materialized_count(area_name)
        return PopulationCounts(
            active=area.active_count,
            ever_fired=int(owner.get_num_ever_fired(area_name)),
            materialized=(None if materialized is None else int(materialized)),
        )

    def pre_kwta_observation(
        self, area_name: str
    ) -> PreKwtaObservation | None:
        """Return the last pre-selection total and its actual candidate count.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-pre-kwta-observation
        """
        if area_name not in self.areas:
            raise KeyError(f"unknown area {area_name!r}")
        has_total = area_name in self.last_pre_kwta_totals
        has_count = area_name in self.last_pre_kwta_counts
        if not has_total and not has_count:
            return None
        if has_total != has_count:
            raise RuntimeError(
                f"incomplete pre-k-WTA observation for {area_name!r}: "
                f"total={has_total}, candidate_count={has_count}"
            )
        return PreKwtaObservation(
            total=self.last_pre_kwta_totals[area_name],
            candidate_count=self.last_pre_kwta_counts[area_name],
        )

    def _preserve_mixed_connectomes(self) -> None:
        """Install dense cross-connectomes for explicit↔sparse mixed edges."""
        sparse_eng = self._engine
        if sparse_eng is not None and hasattr(sparse_eng, "_area_conns"):
            for src_name, src_area in self.areas.items():
                if not src_area.explicit:
                    continue
                for tgt_name, tgt_area in self.areas.items():
                    if src_name == tgt_name or tgt_area.explicit:
                        continue
                    existing = self.connectomes.get(src_name, {}).get(tgt_name)
                    dense = dense_connectome_or_new(
                        existing,
                        source_size=src_area.n,
                        target_size=tgt_area.n,
                        p=self.p,
                        rng=self._conn_rng,
                    )
                    if not is_dense_connectome(existing):
                        self.connectomes.setdefault(src_name, {})[tgt_name] = dense
                    if hasattr(sparse_eng, "set_dense_area_conn"):
                        sparse_eng.set_dense_area_conn(src_name, tgt_name, dense)
                    else:
                        sparse_eng._area_conns[src_name][tgt_name] = dense

        # Sparse → explicit: dense connectomes live on brain.connectomes for drive merge
        for src_name, src_area in self.areas.items():
            if src_area.explicit:
                continue
            for tgt_name, tgt_area in self.areas.items():
                if src_name == tgt_name or not tgt_area.explicit:
                    continue
                existing = self.connectomes.get(src_name, {}).get(tgt_name)
                if not is_dense_connectome(existing):
                    dense = Connectome(
                        src_area.n, tgt_area.n, self.p, sparse=False,
                        rng=self._conn_rng,
                    )
                    self.connectomes.setdefault(src_name, {})[tgt_name] = dense
                if self._explicit_engine is not None and hasattr(
                    self._explicit_engine, "_area_conns"
                ):
                    dense = self.connectomes[src_name][tgt_name]
                    if src_name not in self._explicit_engine._area_conns:
                        self._explicit_engine._area_conns[src_name] = {}
                    self._explicit_engine._area_conns[src_name][tgt_name] = dense

    def _source_neuron_ids(self, source_name):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-supervised-reinforcement

        Resolve active source positions to stable IDs for full-population fibers.
        """
        source = self.areas[source_name]
        compact = validated_indices(to_cpu(source.winners), upper=source.n,
                                    label=f"{source_name} winners", unique=True)
        mapping = self._engine_for(source).get_neuron_id_mapping(source_name)
        if mapping is not None:
            compact = validated_indices(compact, upper=len(mapping),
                                        label=f"{source_name} compact winners")
            ids = to_neuron_ids(CompactIdx(compact), mapping)
        else:
            ids = compact
        return validated_indices(ids, upper=source.n, label=f"{source_name} neuron IDs",
                                 unique=True)

    def _sparse_sources_drive_to_explicit(
        self, target_name: str, sparse_source_names: List[str],
    ) -> np.ndarray:
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-mixed-drive-indices

        Sum stable-neuron rows, rejecting invalid compact indices and mappings.
        """
        xp = get_xp()
        tgt = self.areas[target_name]
        drive = xp.zeros(tgt.n, dtype=xp.float32)

        for src_name in sparse_source_names:
            real_ids = self._source_neuron_ids(src_name)
            if real_ids.size == 0:
                continue
            conn = self.connectomes.get(src_name, {}).get(target_name)
            if not is_dense_connectome(conn):
                continue
            rows = validated_indices(real_ids, upper=conn.weights.shape[0],
                                     label=f"{src_name}->{target_name} neuron rows")
            drive += conn.weights[rows].sum(axis=0).astype(xp.float32, copy=False)
        return np.array(to_cpu(drive), dtype=np.float32)

    def _sync_engine_connectomes(self):
        """Replace Brain's connectome dicts with references to the engine's objects.

        After this call, ``self.connectomes[src][tgt]`` and
        ``self.connectomes_by_stimulus[stim][area]`` point to the same
        Connectome instances the engine uses for projection and plasticity.
        """
        for engine in (self._engine, self._explicit_engine):
            if engine is None:
                continue
            if hasattr(engine, '_area_conns'):
                for src in engine._area_conns:
                    if src not in self.connectomes:
                        self.connectomes[src] = {}
                    for tgt in engine._area_conns[src]:
                        self.connectomes[src][tgt] = engine._area_conns[src][tgt]
            if hasattr(engine, '_stim_conns'):
                for stim in engine._stim_conns:
                    if stim not in self.connectomes_by_stimulus:
                        self.connectomes_by_stimulus[stim] = {}
                    for area in engine._stim_conns[stim]:
                        self.connectomes_by_stimulus[stim][area] = engine._stim_conns[stim][area]
        self._preserve_mixed_connectomes()

    def reinforce_connectome(
        self,
        src_area: str,
        dst_area: str,
        post_neurons: np.ndarray,
        *,
        beta: float | None = None,
    ) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-supervised-reinforcement

        Supervised dense-fiber update, with stable post IDs and explicit zero seeding.
        Only selected synapses are mutated, including clipping.
        """
        if src_area not in self.areas or dst_area not in self.areas:
            raise KeyError(f"unknown area in reinforce_connectome: {src_area!r} -> {dst_area!r}")
        src, dst = self.areas[src_area], self.areas[dst_area]
        pre = self._source_neuron_ids(src_area)
        post = validated_indices(to_cpu(post_neurons), upper=dst.n,
                                 label=f"{dst_area} post neuron IDs", unique=True)
        b = beta if beta is not None else dst.beta_by_area.get(src_area, dst.beta)
        if not np.isfinite(b) or b < 0:
            raise ValueError("Supervised beta must be finite and nonnegative")
        if self.w_max is not None and (not np.isfinite(self.w_max) or self.w_max <= 0):
            raise ValueError("Weight clip must be finite and positive or None")
        conn = self.connectomes.get(src_area, {}).get(dst_area)
        if not is_dense_connectome(conn) or not isinstance(conn.weights, np.ndarray):
            raise NotImplementedError("Supervised reinforcement requires a dense NumPy fiber")
        w = conn.weights
        if w.shape != (src.n, dst.n) or w.dtype.kind != "f":
            raise ValueError("Supervised fiber must have full-population floating-point axes")
        engine = self._engine_for(dst)
        if (not self.fiber_plasticity_enabled(src_area, dst_area)
                or not engine.fiber_learning_allowed(src_area, dst_area)
                or not getattr(engine, "_plasticity_enabled_global", True)
                or b == 0 or not pre.size or not post.size):
            return
        ix = np.ix_(pre, post)
        block = w[ix]
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError("Selected weights must be finite and nonnegative")
        # Supervision can create an edge; ordinary Hebbian projection cannot.
        block[block == 0] = 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            block *= (1 + b)
        if self.w_max is not None:
            np.clip(block, 0, self.w_max, out=block)
        if not np.isfinite(block).all():
            raise ValueError("Supervised update is not representable in the fiber dtype")
        w[ix] = block
        if self._explicit_engine is not None:
            econn = self._explicit_engine._area_conns.get(src_area, {}).get(dst_area)
            if econn is not None:
                econn.weights = w


    @contextlib.contextmanager
    def frozen(self):
        """Temporarily disable plasticity, restoring the prior state on exit.

        Reading a stored assembly, probing, or measuring must not TRAIN the
        thing being measured -- doing so silently made ``score_corpus`` learn
        its own eval corpus and ``pattern_complete`` strengthen the assembly it
        was reporting on. The save/set/restore for this was hand-rolled ~60
        times across the codebase, and a single missing ``finally`` leaves
        plasticity off for the rest of the session, corrupting all later
        training. Prefer::

            with brain.frozen():
                ...read-only projections...

        The ``finally`` here guarantees restoration even if the body raises.
        """
        saved = self.disable_plasticity
        self.disable_plasticity = True
        try:
            yield self
        finally:
            self.disable_plasticity = saved

    @contextlib.contextmanager
    def probe(self):
        """The context a READ should run in. Prefer this over ``frozen()``.

        There are two read contexts and the difference is not cosmetic.
        ``frozen()`` stops weights changing; it does NOT stop the area growing,
        and growth is the channel by which a measurement changes the measured.
        A single ERP probe was measured adding 750 synapses to a self fiber,
        every one of value 1.0 -- materialization arriving while plasticity was
        off. Running the fast suite with ``NEURAL_ASSEMBLIES_STRICT_PROBES=1``
        raised 214 times across the ERP path, the parser, next_token,
        checkpoint/fork and TACL, so this is not a corner case.

        WHY THIS IS A SWITCH AND NOT A SUBSTITUTION. Suppressing recruitment
        changes what a probe can select from, so the numbers MOVE -- on one
        protocol stored/probe overlap went 0.038 -> 0.180, because an area that
        cannot recruit must answer from the neurons it already has. That is the
        semantics a readout wants, but adopting it is a RE-MEASUREMENT, not a
        rename, and each call site needs its before/after recorded.

        MEASURED, then adopted. On the ERP calibration contrast over 10 seeds
        (research/experiments/task100_erp_probe_isolation.py), paired on the
        same trained parser, RE-RUN once training became reproducible (#80) and
        `fork` stopped cloning a mutated parser (#103)::

            p600 AUC         0.972 +/- 0.063  ->  0.917 +/- 0.057
            p600 median gap  0.002 +/- 0.001  ->  0.003 +/- 0.000   CHANGED
            p600 Cohen's d   1.910 +/- 0.534  ->  1.876 +/- 0.446   no change

        Isolation costs nothing and slightly WIDENS the raw separation. An
        earlier reading of this table reported a "~15% P600 attenuation"
        (d 2.192 -> 1.853, delta -0.338 +/- 0.327); that is RETRACTED -- on a
        deterministic substrate the same delta is -0.034 +/- 0.581. It was
        parser variation, not probe contamination.

        WHAT THE DIFFERENCE ACTUALLY IS. A three-arm run
        (research/experiments/erp_growth_neutrality.py) splits this context's
        two interventions apart: suppressing RECRUITMENT accounts for all of
        it, and restoring winners for none -- the no-recruit and read_only arms
        are bit-identical on every metric and all 10 seeds. So the winner
        snapshot is host hygiene with no effect on the measurement, and the
        numbers above are a recruitment effect. Whether that effect is a defect
        is a separate question (the AC has no growth at all; see
        research/experiments/erp_full_substrate.py).

        Isolation is also ~3x faster on that suite, because a probe that does
        not grow the brain has less to do.

        ``NEURAL_ASSEMBLIES_ISOLATED_PROBES=0`` restores the old ``frozen()``
        behaviour for A/B work. The flag is read per call, not cached at import,
        so a comparison can flip it inside one process and get PAIRED trials.

        NOT FOR PROTOCOL, and this is the load-bearing distinction. Blocks that
        ADVANCE a parse or build structure while plasticity happens to be off
        are not reads. ``binding.materialize_fiber`` is the clearest case: it
        exists to ALLOCATE COLUMNS, which happens as a side effect of the target
        recruiting, so under isolation it would become a silent no-op that still
        returns True.
        """
        legacy = os.environ.get(
            "NEURAL_ASSEMBLIES_ISOLATED_PROBES", "1").strip() in ("0", "false")
        with (self.frozen() if legacy else self.read_only()):
            yield self

    @contextlib.contextmanager
    def read_only(self, *, seed: int | None = None):
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-read-only

        Probe without retaining activity, recruitment, learning, or RNG draws.
        Optional seed contract: neural_assemblies/ir/VERIFICATION.md#contract-seeded-observation
        A supplied seed temporarily installs one child stream per distinct backend
        generator. Original generator objects and states are restored on exit.

        Winners move inside the block. On exit, including exceptions, each
        area restores its declared activity state (including firing counts,
        refractory history and saved winners). Latest drive measurements remain
        available. Plasticity is disabled separately by ``frozen()``.

        Sampled areas must already have at least k materialized neurons: probing
        an empty population raises before projection. Initialize/train it first,
        or materialize its connectome. A sampled probe selects only from its
        recruited population; that is not full-connectome selection.
        """
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
            raise ValueError('observation seed must be a nonnegative integer or None')
        engines = self._all_engines()
        snapshots = [area.snapshot_activity() for area in self.areas.values()]
        for engine in engines:
            snapshots.extend(engine.snapshot_activity())
        distinct_rngs = {id(engine._rng): engine._rng for engine in engines if hasattr(engine, '_rng')}
        generators = [(rng, deepcopy(rng.bit_generator.state)) for rng in distinct_rngs.values()]
        streams = np.random.SeedSequence(int(seed)).spawn(len(generators)) if seed is not None else []
        flags = [(engine, engine._no_recruitment) for engine in engines
                 if hasattr(engine, "_no_recruitment")]
        try:
            for engine, _ in flags:
                engine._no_recruitment = True
            for (rng, _), stream in zip(generators, streams):
                rng.bit_generator.state = type(rng.bit_generator)(stream).state
            with self.frozen():
                yield self
        finally:
            for snapshot in snapshots:
                snapshot.restore()
            for rng, state in generators:
                rng.bit_generator.state = state
            for engine, flag in flags:
                engine._no_recruitment = flag

    def _all_engines(self):
        """Every compute engine backing this brain, primary first."""
        seen, out = set(), []
        for engine in (getattr(self, "_engine", None),
                       getattr(self, "_explicit_engine", None)):
            if engine is not None and id(engine) not in seen:
                seen.add(id(engine))
                out.append(engine)
        return out

    def project(
        self,
        areas_by_stim: Dict[str, List[str]] = None,
        dst_areas_by_src_area: Dict[str, List[str]] = None,
        external_inputs: Dict[str, np.ndarray] = None,
        projections: Dict[str, List[str]] = None,
        external_drive: Dict[str, np.ndarray] = None,
        verbose: int = 0,
    ):
        """
        Execute Assembly Calculus Projection operations.

        Supports two calling conventions:

        1. Legacy API (matches root brain.py / simulations.py):
           brain.project({"stim": ["AreaA"]}, {"AreaA": ["AreaB"]})

        2. Direct-injection API:
           brain.project(external_inputs={"AreaA": winners}, projections={"AreaA": ["AreaB"]})
           Sets area winners explicitly, then projects area-to-area.

        Args:
            areas_by_stim: Maps stimulus names to target area names.
            dst_areas_by_src_area: Maps source area names to target area names.
            external_inputs: Directly injects winner arrays into areas before projecting.
            projections: Maps source area names to target area names (used with external_inputs).
            external_drive: Per-target additive input bias for explicit areas (length n).
            verbose: 0=silent, 1=basic, 2=detailed.
        """
        drive = external_drive or {}
        if areas_by_stim is not None or dst_areas_by_src_area is not None:
            self._project_impl(areas_by_stim or {}, dst_areas_by_src_area or {}, verbose, drive)
        elif external_inputs is not None or projections is not None or external_drive is not None:
            # Validate the complete injection and route before changing any winner state.
            self._projection_inputs({}, projections or {})
            injections = self._validated_winner_inputs(external_inputs or {})
            for area_name, winners in injections.items():
                area = self.areas[area_name]
                area.winners = winners
                self._engine_for(area).set_winners(area_name, winners)
            self._project_impl({}, projections or {}, verbose, drive)
        else:
            raise ValueError("Must provide either legacy API parameters or new API parameters")

    def _validated_winner_inputs(self, inputs):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-winner-inputs

        Validate every raw buffer before casting or synchronizing any area.
        """
        validated = {}
        for name, winners in inputs.items():
            if name not in self.areas:
                raise IndexError(f"Unknown winner input area {name!r}")
            validated[name] = validated_indices(to_cpu(winners), upper=self.areas[name].n,
                                                label=f"{name} winners", unique=True)
        return validated

    def _projection_inputs(self, areas_by_stim, dst_areas_by_src_area):
        """Validate names and resolve incoming edges without mutating state."""
        stim_in = defaultdict(list)
        area_in = defaultdict(list)

        for stim, areas in areas_by_stim.items():
            if stim not in self.stimuli:
                raise IndexError(f"Not in brain.stimuli: {stim}")
            for area_name in areas:
                if area_name not in self.areas:
                    raise IndexError(f"Not in brain.areas: {area_name}")
                stim_in[area_name].append(stim)

        for from_area_name, to_area_names in dst_areas_by_src_area.items():
            if from_area_name not in self.areas:
                raise IndexError(f"Not in brain.areas: {from_area_name}")
            for to_area_name in to_area_names:
                if to_area_name not in self.areas:
                    raise IndexError(f"Not in brain.areas: {to_area_name}")
                area_in[to_area_name].append(from_area_name)

        return stim_in, area_in

    def _project_impl(self, areas_by_stim, dst_areas_by_src_area, verbose=0,
                      external_drive=None):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-fiber-learning"""
        with contextlib.ExitStack() as stack:
            if not self.disable_plasticity and any(not value for value in self.plasticity_mask.values()):
                stim_in, area_in = self._projection_inputs(areas_by_stim, dst_areas_by_src_area)
                if self._inhibition is not None and self._inhibition.any_closed():
                    stim_in, area_in = self._apply_inhibition(stim_in, area_in)
                by_engine = defaultdict(set)
                for incoming in (stim_in, area_in):
                    for target, sources in incoming.items():
                        engine = self._engine_for(self.areas[target])
                        for source in sources:
                            if not self.fiber_plasticity_enabled(source, target):
                                by_engine[engine].add((source, target))
                for engine, fibers in by_engine.items():
                    stack.enter_context(engine.suppress_fiber_learning(fibers))
            return self._project_unscoped(areas_by_stim, dst_areas_by_src_area, verbose, external_drive)

    def _project_unscoped(self, areas_by_stim, dst_areas_by_src_area, verbose=0,
                          external_drive=None):
        """
        Specification: neural_assemblies/ir/VERIFICATION.md#contract-explicit-inputs

        Core projection implementation. Builds input mappings from stimuli and areas, then delegates to the
        compute engine for all projection, winner selection, and plasticity.
        """
        external_drive = external_drive or {}
        stim_in, area_in = self._projection_inputs(
            areas_by_stim, dst_areas_by_src_area)

        # AC AREA/FIBER INHIBITION. The calculus has exactly two control
        # primitives and this is where they act: an inhibited area neither
        # fires nor is fired into, and a closed fiber carries nothing. Skipped
        # entirely unless something is actually inhibited, so the default Brain
        # pays nothing and behaves exactly as before.
        if self._inhibition is not None and self._inhibition.any_closed():
            stim_in, area_in = self._apply_inhibition(stim_in, area_in)

        # ORDER IS LOAD-BEARING, so this must not be a set. It drives the batch
        # config order below and the sequential projection loop, and projecting
        # an area materializes neurons -- so the order determines how the seeded
        # RNG stream is consumed. `stim_in.keys() | area_in.keys()` is a set of
        # str, whose iteration order changes with PYTHONHASHSEED from process to
        # process; identical seeds gave different results across runs while
        # being perfectly stable within one run. dict.fromkeys dedupes while
        # keeping the deterministic insertion order of the two dicts.
        to_update_area_names = dict.fromkeys(
            list(stim_in.keys()) + list(area_in.keys()) + list(external_drive)
        )
        if external_drive:
            from .numpy_engine import NumpyExplicitEngine
            for name in external_drive:
                if name not in self.areas:
                    raise ValueError(f"Unknown external drive target {name!r}")
                if self._inhibition is not None and not self._inhibition.area_open(name):
                    raise ValueError(f"External drive target {name!r} is inhibited")
                if not isinstance(self._engine_for(self.areas[name]), NumpyExplicitEngine):
                    raise ValueError("External drive requires a dense explicit engine")

        # Preflight every target before any projection in a batched probe.
        for name in to_update_area_names:
            self._engine_for(self.areas[name]).validate_probe_target(name)

        # Sync source and held-target caps from public descriptors to engines.
        # A held target must use the requested cap, not a stale backend cap.
        # Source synchronization is needed for two reasons:
        # 1. External code may set area.winners directly (pattern completion)
        # 2. Cross-engine projections: an explicit area's winners must be
        #    visible to the sparse engine when used as a source.
        sync_areas = dict.fromkeys(
            [src for sources in area_in.values() for src in sources]
            + [name for name in to_update_area_names if self.areas[name].fixed_assembly]
        )
        source_winners = self._validated_winner_inputs(
            {name: self.areas[name].winners for name in sync_areas})
        for area_name, winners_arr in source_winners.items():
            area = self.areas[area_name]
            # Empty activity is a state update too; otherwise a cleared public
            # source silently reuses its previous backend winners.
            self._engine.set_winners(area_name, winners_arr)
            eng_st = self._engine._areas.get(area_name)
            if eng_st is not None:
                eng_st.explicit_source = area.explicit
            if self._explicit_engine is not None and area.explicit:
                self._explicit_engine.set_winners(area_name, winners_arr)

        # Sync fixed_assembly state from Area descriptors to engine
        for area_name in to_update_area_names:
            area = self.areas[area_name]
            engine = self._engine_for(area)
            if area.fixed_assembly and not engine.is_fixed(area_name):
                engine.fix_assembly(area_name)
            elif not area.fixed_assembly and engine.is_fixed(area_name):
                engine.unfix_assembly(area_name)

        # Track activation scores for mutual inhibition
        activation_scores = {}
        pre_kwta = {}
        pre_kwta_n = {}

        # Batched path: process multiple targets in one kernel launch
        # (only for non-explicit areas on the main engine)
        non_explicit = [n for n in to_update_area_names
                        if not self.areas[n].explicit and n not in external_drive]
        if len(non_explicit) > 1:
            configs = [(name, stim_in[name], area_in[name])
                       for name in non_explicit]
            batch_results = self._engine.project_into_batch(
                configs, plasticity_enabled=not self.disable_plasticity,
                record_activation=getattr(self, 'record_activation', False))
            for area_name, result in batch_results.items():
                self._apply_result(area_name, result, stim_in, area_in)
                activation_scores[area_name] = result.total_activation
                if getattr(self, 'record_activation', False):
                    pre_kwta[area_name] = float(result.pre_kwta_total or 0.0)
                    pre_kwta_n[area_name] = int(result.pre_kwta_count or 0)

        # Sequential path: one target at a time
        # List comprehension, not a set difference: preserves the deterministic
        # order established above (a set difference would re-randomize it).
        _batched = set(non_explicit) if len(non_explicit) > 1 else ()
        remaining = [n for n in to_update_area_names if n not in _batched]
        for area_name in remaining:
            engine = self._engine_for(self.areas[area_name])
            eng_area_names = getattr(engine, "_areas", {})
            from_area_list = [
                a for a in area_in[area_name] if a in eng_area_names
            ]
            cross_sparse = [
                a for a in area_in[area_name]
                if a not in eng_area_names and not self.areas[a].explicit
            ]
            external_drive_vec = external_drive.get(area_name)
            if external_drive_vec is None and cross_sparse and self.areas[area_name].explicit:
                external_drive_vec = self._sparse_sources_drive_to_explicit(
                    area_name, cross_sparse,
                )
            drive_kwargs = ({"external_drive": external_drive_vec}
                            if external_drive_vec is not None else {})
            result = engine.project_into(
                area_name,
                from_stimuli=stim_in[area_name],
                from_areas=from_area_list,
                plasticity_enabled=not self.disable_plasticity,
                record_activation=getattr(self, 'record_activation', False),
                **drive_kwargs,
            )
            self._apply_result(area_name, result, stim_in, area_in,
                               had_external_drive=external_drive_vec is not None)
            activation_scores[area_name] = result.total_activation
            if getattr(self, 'record_activation', False):
                pre_kwta[area_name] = float(result.pre_kwta_total or 0.0)
                pre_kwta_n[area_name] = int(result.pre_kwta_count or 0)

        # Total synaptic drive each target received, kept for the caller. This
        # is the quantity area-level competition is decided on, so exposing it
        # lets callers score a competition the same way the brain does.
        self.last_activation_scores = dict(activation_scores)
        if getattr(self, 'record_activation', False):
            self.last_pre_kwta_totals = dict(pre_kwta)
            self.last_pre_kwta_counts = dict(pre_kwta_n)

        # Post-projection: apply mutual inhibition (area-level WTA)
        if self._mutual_inhibition_groups:
            self._apply_mutual_inhibition(activation_scores)

    def _apply_mutual_inhibition(self, activation_scores):
        """Suppress non-winning areas in mutual inhibition groups.

        For each group, the area with the highest total_activation
        retains its winners; all others are silenced.
        """
        for group in self._mutual_inhibition_groups:
            active = [(name, activation_scores.get(name, 0.0))
                      for name in group if name in activation_scores]
            if len(active) <= 1:
                continue
            # Area with highest total synaptic drive wins
            winner_name = max(active, key=lambda x: x[1])[0]
            for name, _ in active:
                if name != winner_name:
                    area = self.areas[name]
                    area.winners = np.array([], dtype=np.uint32)
                    area.w = 0
                    engine = self._engine_for(area)
                    engine.set_winners(name, np.array([], dtype=np.uint32))
                    if engine is not self._engine:
                        self._engine.set_winners(
                            name, np.array([], dtype=np.uint32))

    def _apply_result(self, area_name, result, stim_in, area_in, *, had_external_drive=False):
        """Apply a ProjectionResult back to the Area descriptor and save history."""
        area = self.areas[area_name]
        area._new_winners = result.winners
        area._new_w = result.num_ever_fired
        area.num_first_winners = result.num_first_winners
        had_inputs = bool(stim_in[area_name] or area_in[area_name] or had_external_drive)

        if self.save_winners and had_inputs:
            mapping = self._engine.get_neuron_id_mapping(area_name)
            if mapping:
                saved = np.array([mapping[idx] if idx < len(mapping) else np.uint32(idx)
                                  for idx in result.winners], dtype=np.uint32)
            else:
                saved = result.winners.copy()
            area.saved_winners.append(saved)

        if self.save_size:
            area.saved_w.append(result.num_ever_fired)

        area.winners = result.winners
        area.w = result.num_ever_fired
        # Keep the recruitment reading alive across a later `winners`
        # assignment, which clobbers `w`. See Area.get_num_ever_fired.
        area._num_ever_fired = int(result.num_ever_fired)

        # Sync explicit-area tracking fields
        if area.explicit:
            area.ever_fired[result.winners] = True
            area.num_ever_fired = result.num_ever_fired

    def clear_refractory(self, area_name: str) -> None:
        """Clear LRI refractory history for an area.

        Resets the refractory buffer so the next projection applies no
        suppression penalty.  Call between memorization and recall phases,
        or between independent trials.
        """
        self._engine_for(self.areas[area_name]).clear_refractory(area_name)

    def reset_area_connections(self, area_name: str) -> None:
        """Forget learned area-to-area weights through the owning engine.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-reset-area-connections

        Areas may be owned by different engines in one Brain: explicit areas
        use the dense owner while sparse areas use the primary owner. A caller
        must not reach through ``_engine`` and accidentally reset the wrong
        storage object.
        """
        try:
            area = self.areas[area_name]
        except KeyError:
            raise KeyError(f"unknown area {area_name!r}") from None
        self._engine_for(area).reset_area_connections(area_name)

    def set_lri(self, area_name: str, refractory_period: int,
                inhibition_strength: float) -> None:
        """Update LRI parameters for an area at runtime.

        Enables or disables Long-Range Inhibition after area creation.
        Typical workflow: add area without LRI, memorize sequences,
        then enable LRI for recall.
        """
        refractory_period, inhibition_strength = validate_lri_parameters(
            refractory_period, inhibition_strength)
        area = self.areas[area_name]
        self._engine_for(area).set_lri(area_name, refractory_period, inhibition_strength)
        area.refractory_period = refractory_period
        area.inhibition_strength = inhibition_strength

    def set_refracted(self, area_name: str, enabled: bool,
                      strength: float = 0.0) -> None:
        """Enable or disable refracted mode for an area at runtime.

        Refracted mode accumulates a permanent bias: each time a neuron
        fires, its bias grows, making it progressively harder to fire
        again.  Distinct from LRI (sliding-window penalty).
        """
        area = self.areas[area_name]
        engine = self._engine_for(area)
        check_area_homeostasis(area_name, refracted=enabled,
                               synaptic_scaling=getattr(engine, "synaptic_scaling", False))
        engine.set_refracted(area_name, enabled, strength)
        area.refracted = enabled
        area.refracted_strength = strength

    def clear_refracted_bias(self, area_name: str) -> None:
        """Reset accumulated refracted bias to zero for an area."""
        self._engine_for(self.areas[area_name]).clear_refracted_bias(area_name)

    def set_masked_readout(self, area_name: str, enabled: bool = True) -> None:
        """Read a refracted area with its bias MASKED.

        A refracted memory is read through the veto or not at all: with the
        bias subtracted a half-cue recall reads chance, with it masked the
        stored assembly returns ([[REFRACTION-ANTI-MERGING]]). The flag is
        honoured only on reads -- projections with plasticity off, as under
        ``probe()`` or ``frozen()``; a write always sees and charges the
        bias, because the bias is what keeps items apart while they are
        written. This is the mode switch the memory needs; nothing in the
        substrate flips it on its own.
        """
        st = self._engine._areas.get(area_name) if hasattr(self._engine, "_areas") else None
        if st is None:
            raise KeyError(f"unknown area {area_name!r}")
        st.masked_readout = bool(enabled)

    def materialize_area(self, area_name: str, storage: str = "csr") -> int:
        """Bring all ``n`` neurons into existence AND resync the descriptor.

        WHY THIS EXISTS RATHER THAN `brain._engine.materialize_area(...)`.
        The engine updates its OWN area state; the `Area` descriptor's `w` is a
        plain attribute that nothing syncs, so after materialising directly on
        the engine `brain.areas[x].w` keeps whatever the last projection left
        there. Measured at n=3000: `area.w` read **42** while the candidate
        vector was **3000**.

        That is not cosmetic. `erp/adapters.py` normalises pre-k-WTA energy by
        `area.w`, so on a fully materialised area it divided a sum over 3000
        candidates by 42 -- inflating the number ~70x in precisely the arm that
        is supposed to be ground truth (#104,
        research/notes/language/erp_scale_is_an_implementation_detail.md).

        Returns the number of neurons newly materialised.
        """
        area = self.areas[area_name]
        engine = self._engine_for(area)
        added = int(engine.materialize_area(area_name, storage=storage) or 0)
        count = engine.materialized_count(area_name)
        if count is not None:
            # `materialized_count` is the SANCTIONED accessor (#69) and says
            # what it returns; `w` does not, which is how they drifted apart.
            area.w = int(count)
        return added

    # ---- AC inhibition: areas and fibers, persistent, gating ---------------
    #
    # THREE THINGS IN THIS CLASS ARE CALLED INHIBITION AND THEY ARE DIFFERENT.
    # Keeping them apart is the whole point of this block:
    #
    #   inhibit_area / inhibit_fiber   the CALCULUS. Persistent gating: an
    #       (this section)             inhibited area neither fires nor is
    #                                  fired into; a closed fiber carries
    #                                  nothing. Two primitives, and the AC has
    #                                  no others for control.
    #
    #   clear_activity                 NOT the calculus. A one-shot wipe of an
    #   (alias: inhibit_areas)         area's winners. The area is fully live
    #                                  again on the very next projection. Named
    #                                  "inhibit_areas" historically, which is
    #                                  [[same-name-two-meanings]] across 20
    #                                  files, so the honest name is primary now.
    #
    #   add_mutual_inhibition          NOT the calculus either. Post-hoc
    #                                  winner-take-all BETWEEN areas, applied
    #                                  after projecting. Measured dormant
    #                                  (#24): 1373 project() calls, zero
    #                                  co-targeting a group.

    @property
    def inhibition(self):
        """The AC inhibition state, created fully OPEN on first use.

        Lazily built and extended, so areas added later are registered
        automatically -- and open, because a new area is not gated until
        somebody gates it.
        """
        from .inhibition import InhibitionState

        if self._inhibition is None:
            self._inhibition = InhibitionState.all_open(list(self.areas))
        else:
            self._inhibition.ensure_areas(list(self.areas), open_new=True)
        return self._inhibition

    def inhibit_area(self, area_name: str, index: int = 0) -> None:
        """Prevent *area_name* from firing or being fired into, until released.

        The AC's area inhibition. Unlike ``clear_activity`` this PERSISTS: the
        area is skipped by every projection until ``disinhibit_area`` releases
        it. Its winners are left alone, which is what makes the parser's
        "hold this role while the next word settles" work.

        ``index`` gives independent rules independent channels on the same
        area; it reopens only when every holder has released it.
        """
        self.inhibition.inhibit_area(area_name, index)

    def disinhibit_area(self, area_name: str, index: int = 0) -> None:
        """Release this holder's claim on *area_name*."""
        self.inhibition.disinhibit_area(area_name, index)

    def inhibit_fiber(self, a1: str, a2: str, index: int = 0) -> None:
        """Close the fiber between *a1* and *a2*. SYMMETRIC, as in the reference."""
        self.inhibition.inhibit_fiber(a1, a2, index)

    def disinhibit_fiber(self, a1: str, a2: str, index: int = 0) -> None:
        """Release this holder's claim on the *a1* <-> *a2* fiber."""
        self.inhibition.disinhibit_fiber(a1, a2, index)

    def _apply_inhibition(self, stim_in, area_in):
        """Drop everything the gating state closes. Returns filtered maps.

        Three rules, all from Algorithm 2's ``project*``: a stimulus cannot
        drive an inhibited area; an inhibited area is not a target; and an edge
        needs BOTH endpoints open and its fiber open.
        """
        state = self._inhibition
        stim_out = defaultdict(list)
        for target, stims in stim_in.items():
            if state.area_open(target):
                stim_out[target] = list(stims)

        area_out = defaultdict(list)
        for target, sources in area_in.items():
            if not state.area_open(target):
                continue
            kept = [s for s in sources
                    if state.area_open(s) and state.fiber_open(s, target)]
            if kept:
                area_out[target] = kept
        return stim_out, area_out

    def clear_activity(self, area_names: List[str]) -> None:
        """Wipe the winners of these areas. NOT the AC's inhibition.

        Clears winners so the next projection step sees no active neurons from
        those areas. Connectome weights are preserved, so re-stimulation
        recovers the original assembly -- and the area is fully live again
        immediately, which is the difference from ``inhibit_area``.

        Typical usage: call before switching patterns in a recurrent loop to
        clear residual activity from the previous pattern.

        Exposed as ``inhibit_areas`` too, which is what ~20 modules call it.
        That name is misleading -- nothing is inhibited, an assembly is erased
        -- and it collided with the calculus's own term for a different
        mechanism, so the descriptive name is the primary one now.
        """
        for name in area_names:
            area = self.areas[name]
            area.winners = np.array([], dtype=np.uint32)
            engine = self._engine_for(area)
            engine.set_winners(name, np.array([], dtype=np.uint32))
            # Also sync to main engine for cross-engine visibility
            if engine is not self._engine:
                self._engine.set_winners(name, np.array([], dtype=np.uint32))

    #: Historical name for ``clear_activity``. Kept because ~20 modules use it
    #: and a mass rename is a separate, reviewable change; see the block
    #: comment above for why it is no longer the primary name.
    inhibit_areas = clear_activity

    def add_mutual_inhibition(self, area_names: List[str]) -> None:
        """Post-hoc winner-take-all between areas. NOT the AC's inhibition.

        When areas in this group receive simultaneous input via
        ``project()``, only the area with the highest total synaptic
        drive retains its winners; all others are silenced (winners
        cleared to empty).

        THIS IS NOT WHAT THE PAPERS MEAN BY INHIBITION, and the difference is
        architectural rather than terminological. This runs AFTER projecting
        and compares total drive across areas; the calculus inhibits BEFORE, by
        gating which projections happen at all (``inhibit_area`` /
        ``inhibit_fiber``). Gating needs no cross-area comparison, which
        matters because that comparison is measurably unreliable here -- a 7x
        pre-k-WTA separation collapses to a ~7% margin.

        Measured DORMANT (#24): across a full parser run, 1373 ``project()``
        calls, ZERO of which co-target a group, so this has never fired. Prefer
        the gating primitives.

        Persistent: applies to all future projections until
        ``remove_mutual_inhibition`` is called.
        """
        self._mutual_inhibition_groups.append(list(area_names))

    def remove_mutual_inhibition(self, area_names: List[str]) -> None:
        """Remove a previously-added mutual inhibition group."""
        target = set(area_names)
        self._mutual_inhibition_groups = [
            g for g in self._mutual_inhibition_groups
            if set(g) != target
        ]

    def normalize_weights(self, target: str, source: str = None) -> None:
        """Column-normalize weights into *target* so each neuron sums to 1.0.

        If *source* is given, only that connection is normalized.
        Otherwise all connections into *target* are normalized.
        """
        self._engine_for(self.areas[target]).normalize_weights(target, source)

    def project_rounds(self, target, areas_by_stim, dst_areas_by_src_area, rounds):
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-projection-rounds

        Repeat the resolved inputs to one target through ordinary projection.

        All supplied names must exist. Other destinations are excluded; no
        effective inputs and nonpositive/noninteger rounds are errors before
        execution. Every round obeys inhibition, clamp/plasticity state,
        read-only preflight, activation recording and per-round history rules.

        Legacy schedule policy: non-explicit targets drop their self-edge unless
        recurrent_projection and (norm_init or full synaptic_scaling) are set.
        Explicit targets retain supplied self-edges. To specify recurrence
        independently of this compatibility policy, use ordinary project calls
        or assembly_calculus.ops.project's explicit recurrent argument.
        """
        rounds = validate_round_count(rounds)
        if target not in self.areas:
            raise IndexError(f"Not in brain.areas: {target}")
        stim_in, area_in = self._projection_inputs(
            areas_by_stim, dst_areas_by_src_area)
        # Preserve legacy caller schedules; this gate is not a scientific
        # assertion that normalization makes recurrence safe. See the card.
        allow_self = self.areas[target].explicit or (
            self.recurrent_projection and (self.norm_init or self._synaptic_scaling is True))
        stimuli = {s: [target] for s in stim_in[target]}
        sources = {s: [target] for s in area_in[target] if allow_self or s != target}
        if not stimuli and not sources:
            raise ValueError(f"no inputs remain for target {target!r} after schedule selection")
        for _ in range(rounds):
            self.project(stimuli, sources)

    def project_legacy(self, areas_by_stim, dst_areas_by_src_area, verbose=0):
        """Alias for backward compatibility."""
        self._project_impl(areas_by_stim, dst_areas_by_src_area, verbose)

    def _initialize_connectomes_for_area(self, area: Area):
        """
        Initializes connectomes related to a new area.

        Args:
            area (Area): The new area.
        """
        xp = get_xp()
        # Initialize connectomes from stimuli to this area
        for stim_name, stim in self.stimuli.items():
            if area.explicit:
                # For explicit areas, create actual connectome matrices
                connectome = Connectome(stim.size, area.n, self.p, sparse=False,
                                        rng=self._conn_rng,
                                        pair_seed=self._fiber_seed(stim_name, area.name))
            else:
                # For sparse areas, start with empty 1D vector of length area.w (0)
                connectome = Connectome(stim.size, area.n, self.p, sparse=True, rng=self._conn_rng)
                connectome.weights = xp.empty(0, dtype=xp.float32)
            self.connectomes_by_stimulus[stim_name][area.name] = connectome
            area.beta_by_stimulus[stim_name] = area.beta

        # Initialize self-connection for the area
        if area.explicit:
            self_connectome = Connectome(area.n, area.n, self.p, sparse=False,
                                         rng=self._conn_rng,
                                         pair_seed=self._fiber_seed(area.name, area.name))
        else:
            self_connectome = Connectome(area.n, area.n, self.p, sparse=True, rng=self._conn_rng)
            # For sparse, represent area-to-area as 2D with 0 columns
            self_connectome.weights = xp.empty((area.n, 0), dtype=xp.float32)
        self.connectomes[area.name][area.name] = self_connectome

        # Initialize connectomes from existing areas to this area
        for other_area_name, other_area in self.areas.items():
            if other_area_name != area.name:
                if area.explicit or other_area.explicit:
                    # Create actual connectome matrices if either area is explicit
                    connectome = Connectome(
                        other_area.n, area.n, self.p, sparse=False,
                        rng=self._conn_rng,
                        pair_seed=self._fiber_seed(other_area_name, area.name))
                    connectome_rev = Connectome(
                        area.n, other_area.n, self.p, sparse=False,
                        rng=self._conn_rng,
                        pair_seed=self._fiber_seed(area.name, other_area_name))
                else:
                    # Both areas are sparse, represent compactly with 0x0 matrices initially
                    connectome = Connectome(other_area.n, area.n, self.p, sparse=True, rng=self._conn_rng)
                    connectome.weights = xp.empty((0, 0), dtype=xp.float32)
                    connectome_rev = Connectome(area.n, other_area.n, self.p, sparse=True, rng=self._conn_rng)
                    connectome_rev.weights = xp.empty((0, 0), dtype=xp.float32)
                
                self.connectomes[other_area_name][area.name] = connectome
                self.connectomes[area.name][other_area_name] = connectome_rev
                self.connectomes[area.name][other_area_name] = connectome_rev
                # Set beta values
                area.beta_by_area[other_area_name] = area.beta
                other_area.beta_by_area[area.name] = area.beta

    def _fiber_seed(self, source: str, target: str) -> int:
        """Content-addressed identity for one fiber's initial wiring.

        Dense connectomes used to draw from a STREAM, so a fiber's wiring
        depended on how many draws preceded it -- i.e. on the order areas and
        stimuli were created. Two Brains with the SAME seed and the same two
        areas, built in opposite order, agreed on X->X wiring at 0.905, which
        is exactly chance for p=0.05. Door 5 of
        [[content-addressed-synapse-init]].

        Keying on (global seed, source name, target name) makes the wiring a
        function of WHICH fiber it is, so construction order cannot reach it.
        Same primitive `numpy_exact` is built on.
        """
        from .numpy_engine._seeding import fnv1a_pair_seed
        return fnv1a_pair_seed(self._seed, source, target)

    def _initialize_connectomes_for_stimulus(self, stimulus: Stimulus):
        """
        Initializes connectomes related to a new stimulus.

        Args:
            stimulus (Stimulus): The new stimulus.
        """
        xp = get_xp()
        # Initialize connectomes from stimulus to all areas
        for area_name, area in self.areas.items():
            if area.explicit:
                connectome = Connectome(stimulus.size, area.n, self.p, sparse=False,
                                        rng=self._conn_rng,
                                        pair_seed=self._fiber_seed(stimulus.name, area_name))
            else:
                connectome = Connectome(stimulus.size, area.n, self.p, sparse=True, rng=self._conn_rng)
                connectome.weights = xp.empty(0, dtype=xp.float32)
            self.connectomes_by_stimulus[stimulus.name][area_name] = connectome
            area.beta_by_stimulus[stimulus.name] = area.beta
            
    def update_plasticity(self, from_area: str, to_area: str, new_beta: float):
        """Update one area's directed incoming-fiber plasticity rate.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-plasticity-rate

        Args:
            from_area (str): Name of the area that the synapses come from.
            to_area (str): Name of the area that the synapses project to.
            new_beta (float): The new synaptic plasticity parameter.
        """
        if from_area not in self.areas:
            raise KeyError(f"unknown plasticity source area {from_area!r}")
        if to_area not in self.areas:
            raise KeyError(f"unknown plasticity target area {to_area!r}")
        new_beta = validate_plasticity_rate(new_beta)
        # Validation above is deliberately complete before either authority is
        # touched. The descriptor supports the legacy dense computation path;
        # every current engine reads its own beta store.
        self.areas[to_area].beta_by_area[from_area] = new_beta
        self._engine.set_beta(to_area, from_area, new_beta)
        if self._explicit_engine is not None and self.areas[to_area].explicit:
            self._explicit_engine.set_beta(to_area, from_area, new_beta)

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        """Set one fiber's connection probability, overriding the global `p`.

        The counterpart of `update_plasticity`, which does the same for beta.
        Mitropolsky & Papadimitriou (2025) need both: four of their fibers
        carry "increased parameters beta AND p", and that asymmetry is what
        makes the noun/verb split emerge without a label.

        Only `numpy_exact` implements it. The others RAISE rather than ignore
        -- this method was `pass` in every engine while the interface
        advertised it, so a caller that set a per-fiber density silently got
        the global one. Requesting the value already in force is not a change
        and stays a no-op everywhere.

        Connectivity is structural: it decides which synapses exist, so it must
        be set before the fiber carries any traffic.
        """
        self._engine.add_connectivity(source, target, p)
        # Gated on the TARGET being explicit, exactly as `update_plasticity`
        # gates `set_beta`. The first version forwarded unconditionally and
        # raised on any explicit-source -> sparse-target fiber -- e.g. the
        # paper's PHON -> LEX1 -- because the explicit engine refuses per-fiber
        # p and does not own that fiber anyway. The main engine does.
        if (self._explicit_engine is not None
                and target in self.areas and self.areas[target].explicit):
            self._explicit_engine.add_connectivity(source, target, p)

    def update_plasticities(
        self,
        area_update_map: Dict[str, List[Tuple[str, float]]] | None = None,
        stim_update_map: Dict[str, List[Tuple[str, float]]] | None = None,
    ):
        """
        Updates the synaptic plasticity parameter between multiple areas and stimuli.

        Args:
            area_update_map (Dict[str, List[Tuple[str, float]]]):
                A dictionary where the keys are the names of areas that the synapses project to.
                The values are lists of tuples, where each tuple contains the name of an area that the synapses come from
                and the new synaptic plasticity parameter.
            stim_update_map (Dict[str, List[Tuple[str, float]]]):
                A dictionary where the keys are the names of areas.
                The values are lists of tuples, where each tuple contains the name of a stimulus and the new synaptic plasticity parameter.
        """
        area_update_map = area_update_map or {}
        stim_update_map = stim_update_map or {}
        # Resolve the whole request before applying its first update. A bad late
        # entry must not leave an apparently successful partial schedule.
        area_updates = []
        for to_area, update_rules in area_update_map.items():
            for from_area, new_beta in update_rules:
                if from_area not in self.areas:
                    raise KeyError(f"unknown plasticity source area {from_area!r}")
                if to_area not in self.areas:
                    raise KeyError(f"unknown plasticity target area {to_area!r}")
                area_updates.append((from_area, to_area, validate_plasticity_rate(new_beta)))
        stim_updates = []
        for area_name, update_rules in stim_update_map.items():
            if area_name not in self.areas:
                raise KeyError(f"unknown plasticity target area {area_name!r}")
            for stim_name, new_beta in update_rules:
                if stim_name not in self.stimuli:
                    raise KeyError(f"unknown plasticity source stimulus {stim_name!r}")
                stim_updates.append((stim_name, area_name, validate_plasticity_rate(new_beta)))
        for from_area, to_area, new_beta in area_updates:
            self.update_plasticity(from_area, to_area, new_beta)
        for stim_name, area_name, new_beta in stim_updates:
                area = self.areas[area_name]
                area.beta_by_stimulus[stim_name] = new_beta
                self._engine.set_beta(area_name, stim_name, new_beta)
                if self._explicit_engine is not None and area.explicit:
                    self._explicit_engine.set_beta(area_name, stim_name, new_beta)

    def set_competition_policy(self, area_name: str, policy) -> None:
        """Set the executing owner's policy, then publish the accepted setting.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-runtime-policy
        """
        area = self.areas[area_name]
        self._engine_for(area).set_competition_policy(area_name, policy)
        area.winner_policy = policy

    def set_input_noise(self, area_name: str, std: float) -> None:
        """Set Gaussian pre-selection noise on the executing owner.

        Specification: neural_assemblies/ir/VERIFICATION.md#contract-input-noise
        """
        area = self.areas[area_name]
        std = validate_input_noise(std)
        self._engine_for(area).set_input_noise(area_name, std)
        area.input_noise_std = std

    def activate(self, area_name: str, index: int):
        """
        Activates a specific assembly in an area.

        Args:
            area_name (str): Name of the area to activate.
            index (int): Index of the assembly to activate.

        Notes:
            This function is a shortcut for activating a specific assembly in an area.
            It is equivalent to calling `area.fix_assembly()` after setting the winners to the desired assembly.
        """
        xp = get_xp()
        area = self.areas[area_name]
        k = area.k
        assembly_start = k * index
        area.winners = xp.arange(assembly_start, assembly_start + k, dtype=xp.uint32)
        area.fix_assembly()
    
    def activate_with_image(self, area_name: str, image: np.ndarray):
        """
        Activates neurons in the given area using raw image data.
        
        Uses the extracted image activation engine for sophisticated image processing
        including normalization, cropping, padding, and top-k selection.
        
        Args:
            area_name (str): The name of the brain area to activate.
            image (np.ndarray): The raw image data (flattened or 2D).
        """
        area = self.areas[area_name]
        
        # Use extracted image activation engine
        image_flat = self.image_activation_engine.preprocess_image(image, area.n)
        winners, _ = self.image_activation_engine.normalize_and_select_topk(image_flat, area.k)
        
        # Set the winners in the area
        area.winners = winners
        area.w = len(area.winners)

    def clone(self) -> "Brain":
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-brain-clone

        Copy the state graph, preserving internal aliases without reconstructing
        configuration from defaults. Specialized copies must meet this contract.
        """
        import copy

        return copy.deepcopy(self)

    # Comprehensive usage example
    @staticmethod
    def example_assembly_calculus_demo():
        """
        Demonstrate Assembly Calculus operations with a complete example.
        
        This example shows how to use the Brain class to implement the
        fundamental operations of the Assembly Calculus framework.
        
        Assembly Calculus Operations Demonstrated:
        1. Projection: Visual → Semantic (A → B)
        2. Association: Semantic + Motor (A + B → A' + B')
        3. Merge: Semantic + Motor → Action (A + B → C)
        
        Biological Context:
        - Visual area processes sensory input
        - Semantic area represents concepts
        - Motor area controls actions
        - Integration areas combine information
        
        Returns:
            Brain: Configured brain instance ready for Assembly Calculus operations
        """
        # Initialize brain with sparse connectivity
        brain = Brain(p=0.05, seed=42)
        
        # Add brain areas representing different functional regions
        brain.add_area("visual", n=1000, k=100, beta=0.1, explicit=True)
        brain.add_area("semantic", n=800, k=80, beta=0.1, explicit=True)
        brain.add_area("motor", n=600, k=60, beta=0.1, explicit=True)
        brain.add_area("integration", n=500, k=50, beta=0.1, explicit=True)
        
        # Add external stimuli
        brain.add_stimulus("image", size=200)
        brain.add_stimulus("sound", size=150)
        
        # Example 1: Projection (Visual → Semantic)
        # External image activates visual area
        visual_assembly = np.random.choice(1000, 100, replace=False)
        external_inputs = {"visual": visual_assembly}
        projections = {"visual": ["semantic"]}
        brain.project(external_inputs, projections)
        
        # Example 2: Association (Semantic + Motor)
        # Both areas activate simultaneously to strengthen association
        semantic_assembly = brain.areas["semantic"].winners
        motor_assembly = np.random.choice(600, 60, replace=False)
        external_inputs = {"semantic": semantic_assembly, "motor": motor_assembly}
        projections = {"semantic": ["motor"], "motor": ["semantic"]}
        brain.project(external_inputs, projections)
        
        # Example 3: Merge (Semantic + Motor → Integration)
        # Both areas project to integration area to form combined representation
        external_inputs = {"semantic": semantic_assembly, "motor": motor_assembly}
        projections = {"semantic": ["integration"], "motor": ["integration"]}
        brain.project(external_inputs, projections)
        
        return brain
