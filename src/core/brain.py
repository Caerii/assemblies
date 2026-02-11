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

Mathematical Foundation:
- Assembly Calculus operations preserve overlap properties
- Winner-take-all selection implements sparse coding principles
- Synaptic plasticity follows Hebbian learning rules
- Statistical approximations enable scalable simulations
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from .backend import get_xp, to_cpu, detect_best_engine
from .engine import ComputeEngine, create_engine

from .area import Area
from .stimulus import Stimulus
from .connectome import Connectome

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

    def __init__(self, p: float = DEFAULT_P, save_size: bool = True, save_winners: bool = False, seed: int = 0, w_max: float = DEFAULT_W_MAX, engine="auto", deterministic: bool = False, n_hint: int = 0):
        """
        Initialize a neural assembly brain simulation.

        Args:
            p (float): Connection probability between neurons (0 < p < 1).
                      Typical values: 0.01-0.1 for large networks.
            seed (int): Random seed for reproducible simulations.
            engine: ComputeEngine instance, engine name string, or ``"auto"``
                   (default) to select the best available backend.
                   Examples: ``"numpy_sparse"``, ``"cuda_implicit"``, or a
                   pre-constructed ComputeEngine instance.
            deterministic (bool): If True, use legacy code paths that preserve
                   bit-identical RNG sequences for a given seed. Slower (~1.5-2x)
                   but ensures exact reproducibility across code versions.
                   If False (default), use optimized paths (amortised buffer
                   growth, fast inverse-CDF sampling) that are statistically
                   equivalent but produce different RNG sequences.
            n_hint (int): Expected neuron count per area.  When
                   ``engine="auto"``, this guides engine selection: n >= 1M
                   with GPU available selects ``torch_sparse`` (CSR, GPU),
                   otherwise ``numpy_sparse`` (CPU).
        """
        self.p = p
        self.w_max = w_max
        self.save_size = save_size
        self.save_winners = save_winners
        self.deterministic = deterministic
        self.areas: Dict[str, Area] = {}
        self.stimuli: Dict[str, Stimulus] = {}
        self.connectomes_by_stimulus: Dict[str, Dict[str, Connectome]] = {}
        self.connectomes: Dict[str, Dict[str, Connectome]] = {}
        self.rng = np.random.default_rng(seed)
        self.disable_plasticity = False

        # Compute engine — required, defaults to auto-detected best backend
        if engine == "auto":
            engine = detect_best_engine(n_hint)
        if isinstance(engine, str):
            self._engine: ComputeEngine = create_engine(
                engine, p=p, seed=seed, w_max=w_max, deterministic=deterministic,
            )
        elif isinstance(engine, ComputeEngine):
            self._engine = engine
        else:
            raise TypeError(f"engine must be a string name or ComputeEngine instance, got {type(engine)}")

        # Secondary engine for explicit areas (lazily created)
        self._explicit_engine: ComputeEngine = None
        self._seed = seed

        # Inter-area inhibition groups for winner-take-all
        self._mutual_inhibition_groups: List[List[str]] = []

        # Used by activate_with_image()
        self.image_activation_engine = ImageActivationEngine()

    @property
    def engine_name(self) -> str:
        """Return the active compute engine name."""
        return self._engine.name

    @property
    def area_by_name(self) -> Dict[str, Area]:
        """Backward-compatible alias for self.areas."""
        return self.areas

    def add_area(self, area_name: str, n: int, k: int, beta: float = DEFAULT_BETA,
                 explicit: bool = False, refractory_period: int = 0,
                 inhibition_strength: float = 0.0,
                 refracted: bool = False, refracted_strength: float = 0.0):
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
        area = Area(area_name, n, k, beta, explicit,
                    refractory_period=refractory_period,
                    inhibition_strength=inhibition_strength,
                    refracted=refracted,
                    refracted_strength=refracted_strength)
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
                              inhibition_strength=inhibition_strength)
        if refracted:
            self._engine.set_refracted(area_name, True, refracted_strength)
        # For explicit areas, ALSO register with a dedicated explicit engine
        # that handles full n×n weight matrices and plasticity correctly.
        if explicit:
            explicit_eng = self._engine_for(area)  # lazily creates it
            explicit_eng.add_area(area_name, n, k, beta,
                                  refractory_period=refractory_period,
                                  inhibition_strength=inhibition_strength)
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

        Convenience wrapper around ``add_area(explicit=True)``.  Accepts
        (and currently ignores) ``custom_*_p`` parameters for backward
        compatibility with legacy callers such as the parser.
        """
        self.add_area(area_name, n, k, beta, explicit=True)

    def _engine_for(self, area: Area) -> ComputeEngine:
        """Return the correct engine for an area.

        Explicit areas use a dedicated NumpyExplicitEngine.
        Sparse areas use the main engine.
        """
        if area.explicit:
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
                        self._explicit_engine.add_area(
                            existing_name, existing_area.n, existing_area.k, existing_area.beta,
                        )
            return self._explicit_engine
        return self._engine

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

    def project(
        self,
        areas_by_stim: Dict[str, List[str]] = None,
        dst_areas_by_src_area: Dict[str, List[str]] = None,
        external_inputs: Dict[str, np.ndarray] = None,
        projections: Dict[str, List[str]] = None,
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
            verbose: 0=silent, 1=basic, 2=detailed.
        """
        if areas_by_stim is not None or dst_areas_by_src_area is not None:
            self._project_impl(areas_by_stim or {}, dst_areas_by_src_area or {}, verbose)
        elif external_inputs is not None or projections is not None:
            # Inject external activations, then route through the same projection path
            xp = get_xp()
            for area_name, input_winners in (external_inputs or {}).items():
                area = self.areas[area_name]
                area.winners = xp.asarray(input_winners, dtype=xp.uint32)
                self._engine_for(area).set_winners(
                    area_name, np.asarray(to_cpu(input_winners), dtype=np.uint32))
            self._project_impl({}, projections or {}, verbose)
        else:
            raise ValueError("Must provide either legacy API parameters or new API parameters")

    def _project_impl(self, areas_by_stim, dst_areas_by_src_area, verbose=0):
        """
        Core projection implementation.

        Builds input mappings from stimuli and areas, then delegates to the
        compute engine for all projection, winner selection, and plasticity.
        """
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

        to_update_area_names = stim_in.keys() | area_in.keys()

        # Sync winner state from Area descriptors to ALL engines for source
        # areas.  This is needed for two reasons:
        # 1. External code may set area.winners directly (pattern completion)
        # 2. Cross-engine projections: an explicit area's winners must be
        #    visible to the sparse engine when used as a source.
        all_source_areas = set()
        for sources in area_in.values():
            all_source_areas.update(sources)
        for area_name in all_source_areas:
            area = self.areas[area_name]
            if len(area.winners) > 0:
                winners_arr = np.asarray(to_cpu(area.winners), dtype=np.uint32)
                self._engine.set_winners(area_name, winners_arr)
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

        # Batched path: process multiple targets in one kernel launch
        # (only for non-explicit areas on the main engine)
        non_explicit = [n for n in to_update_area_names
                        if not self.areas[n].explicit]
        if len(non_explicit) > 1:
            configs = [(name, stim_in[name], area_in[name])
                       for name in non_explicit]
            batch_results = self._engine.project_into_batch(
                configs, plasticity_enabled=not self.disable_plasticity)
            for area_name, result in batch_results.items():
                self._apply_result(area_name, result, stim_in, area_in)
                activation_scores[area_name] = result.total_activation

        # Sequential path: one target at a time
        remaining = (to_update_area_names - set(non_explicit)
                     if len(non_explicit) > 1
                     else to_update_area_names)
        for area_name in remaining:
            engine = self._engine_for(self.areas[area_name])
            result = engine.project_into(
                area_name,
                from_stimuli=stim_in[area_name],
                from_areas=area_in[area_name],
                plasticity_enabled=not self.disable_plasticity,
            )
            self._apply_result(area_name, result, stim_in, area_in)
            activation_scores[area_name] = result.total_activation

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

    def _apply_result(self, area_name, result, stim_in, area_in):
        """Apply a ProjectionResult back to the Area descriptor and save history."""
        area = self.areas[area_name]
        area._new_winners = result.winners
        area._new_w = result.num_ever_fired
        area.num_first_winners = result.num_first_winners
        had_inputs = bool(stim_in[area_name] or area_in[area_name])

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
        self._engine.clear_refractory(area_name)

    def set_lri(self, area_name: str, refractory_period: int,
                inhibition_strength: float) -> None:
        """Update LRI parameters for an area at runtime.

        Enables or disables Long-Range Inhibition after area creation.
        Typical workflow: add area without LRI, memorize sequences,
        then enable LRI for recall.
        """
        self.areas[area_name].refractory_period = refractory_period
        self.areas[area_name].inhibition_strength = inhibition_strength
        self._engine.set_lri(area_name, refractory_period, inhibition_strength)

    def set_refracted(self, area_name: str, enabled: bool,
                      strength: float = 0.0) -> None:
        """Enable or disable refracted mode for an area at runtime.

        Refracted mode accumulates a permanent bias: each time a neuron
        fires, its bias grows, making it progressively harder to fire
        again.  Distinct from LRI (sliding-window penalty).
        """
        self.areas[area_name].refracted = enabled
        self.areas[area_name].refracted_strength = strength
        self._engine.set_refracted(area_name, enabled, strength)

    def clear_refracted_bias(self, area_name: str) -> None:
        """Reset accumulated refracted bias to zero for an area."""
        self._engine.clear_refracted_bias(area_name)

    def inhibit_areas(self, area_names: List[str]) -> None:
        """Suppress all activity in specified areas.

        Clears winners in each named area so the next projection step
        sees no active neurons from those areas.  Connectome weights are
        preserved, so re-stimulation recovers the original assembly.

        Typical usage: call before switching patterns in a recurrent loop
        to clear residual activity from the previous pattern.
        """
        for name in area_names:
            area = self.areas[name]
            area.winners = np.array([], dtype=np.uint32)
            engine = self._engine_for(area)
            engine.set_winners(name, np.array([], dtype=np.uint32))
            # Also sync to main engine for cross-engine visibility
            if engine is not self._engine:
                self._engine.set_winners(name, np.array([], dtype=np.uint32))

    def add_mutual_inhibition(self, area_names: List[str]) -> None:
        """Enable winner-take-all competition between areas.

        When areas in this group receive simultaneous input via
        ``project()``, only the area with the highest total synaptic
        drive retains its winners; all others are silenced (winners
        cleared to empty).

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
        self._engine.normalize_weights(target, source)

    def project_rounds(self, target, areas_by_stim, dst_areas_by_src_area, rounds):
        """Multi-round projection with engine fast path.

        Executes *rounds* projection steps into *target*.  When the engine
        supports ``project_rounds`` (CUDA), the entire loop runs in a tight
        GPU-side path with pre-resolved references and no per-round Brain
        dispatch.  Otherwise falls back to sequential ``self.project()`` calls.
        """
        area = self.areas[target]
        if area.explicit:
            for _ in range(rounds):
                self.project(areas_by_stim, dst_areas_by_src_area)
            return

        # Resolve which stimuli / areas project into target
        from_stims = [s for s, areas in areas_by_stim.items()
                      if target in areas]
        from_areas_list = [a for a, tgts in dst_areas_by_src_area.items()
                           if target in tgts and a != target]

        # Sync source area winners to engine ONCE
        for area_name in from_areas_list:
            src_area = self.areas[area_name]
            if len(src_area.winners) > 0:
                self._engine.set_winners(
                    area_name, np.asarray(to_cpu(src_area.winners), dtype=np.uint32))

        # Sync target area winners (needed for recurrence / Hebbian prev)
        if area.winners is not None and len(area.winners) > 0:
            self._engine.set_winners(
                target, np.asarray(to_cpu(area.winners), dtype=np.uint32))

        result = self._engine.project_rounds(
            target=target,
            from_stimuli=from_stims,
            from_areas=from_areas_list,
            rounds=rounds,
            plasticity_enabled=not self.disable_plasticity,
        )

        area.winners = result.winners
        area.w = result.num_ever_fired
        if self.save_winners:
            area.saved_winners.append(result.winners.copy())
        if self.save_size:
            area.saved_w.append(result.num_ever_fired)

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
                connectome = Connectome(stim.size, area.n, self.p, sparse=False)
            else:
                # For sparse areas, start with empty 1D vector of length area.w (0)
                connectome = Connectome(stim.size, area.n, self.p, sparse=True)
                connectome.weights = xp.empty(0, dtype=xp.float32)
            self.connectomes_by_stimulus[stim_name][area.name] = connectome
            area.beta_by_stimulus[stim_name] = area.beta

        # Initialize self-connection for the area
        if area.explicit:
            self_connectome = Connectome(area.n, area.n, self.p, sparse=False)
        else:
            self_connectome = Connectome(area.n, area.n, self.p, sparse=True)
            # For sparse, represent area-to-area as 2D with 0 columns
            self_connectome.weights = xp.empty((area.n, 0), dtype=xp.float32)
        self.connectomes[area.name][area.name] = self_connectome

        # Initialize connectomes from existing areas to this area
        for other_area_name, other_area in self.areas.items():
            if other_area_name != area.name:
                if area.explicit or other_area.explicit:
                    # Create actual connectome matrices if either area is explicit
                    connectome = Connectome(other_area.n, area.n, self.p, sparse=False)
                    connectome_rev = Connectome(area.n, other_area.n, self.p, sparse=False)
                else:
                    # Both areas are sparse, represent compactly with 0x0 matrices initially
                    connectome = Connectome(other_area.n, area.n, self.p, sparse=True)
                    connectome.weights = xp.empty((0, 0), dtype=xp.float32)
                    connectome_rev = Connectome(area.n, other_area.n, self.p, sparse=True)
                    connectome_rev.weights = xp.empty((0, 0), dtype=xp.float32)
                
                self.connectomes[other_area_name][area.name] = connectome
                self.connectomes[area.name][other_area_name] = connectome_rev
                self.connectomes[area.name][other_area_name] = connectome_rev
                # Set beta values
                area.beta_by_area[other_area_name] = area.beta
                other_area.beta_by_area[area.name] = area.beta

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
                connectome = Connectome(stimulus.size, area.n, self.p, sparse=False)
            else:
                connectome = Connectome(stimulus.size, area.n, self.p, sparse=True)
                connectome.weights = xp.empty(0, dtype=xp.float32)
            self.connectomes_by_stimulus[stimulus.name][area_name] = connectome
            area.beta_by_stimulus[stimulus.name] = area.beta
            
    def update_plasticity(self, from_area: str, to_area: str, new_beta: float):
        """
        Updates the synaptic plasticity parameter between two areas.

        Args:
            from_area (str): Name of the area that the synapses come from.
            to_area (str): Name of the area that the synapses project to.
            new_beta (float): The new synaptic plasticity parameter.
        """
        self.areas[to_area].beta_by_area[from_area] = new_beta
        self._engine.set_beta(to_area, from_area, new_beta)
        if self._explicit_engine is not None and self.areas[to_area].explicit:
            self._explicit_engine.set_beta(to_area, from_area, new_beta)

    def update_plasticities(
        self,
        area_update_map: Dict[str, List[Tuple[str, float]]] = {},
        stim_update_map: Dict[str, List[Tuple[str, float]]] = {},
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
        for to_area, update_rules in area_update_map.items():
            for from_area, new_beta in update_rules:
                self.update_plasticity(from_area, to_area, new_beta)
        for area_name, update_rules in stim_update_map.items():
            area = self.areas[area_name]
            for stim_name, new_beta in update_rules:
                area.beta_by_stimulus[stim_name] = new_beta
                self._engine.set_beta(area_name, stim_name, new_beta)
                if self._explicit_engine is not None and area.explicit:
                    self._explicit_engine.set_beta(area_name, stim_name, new_beta)

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
