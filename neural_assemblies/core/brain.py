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
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from .backend import get_xp, to_cpu, detect_best_engine
from .engine import ComputeEngine, create_engine

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

    def __init__(self, p: float = DEFAULT_P, save_size: bool = True, save_winners: bool = False, seed: int = 0, w_max: float = DEFAULT_W_MAX, engine="auto", deterministic: bool = False, n_hint: int = 0, projection_fidelity: str = "exact", inhibitory_prob: float = 0.0, inhibitory_weight: float = -0.2, synaptic_scaling: bool = False, recurrent_projection: bool = False, norm_init: bool = True):
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
            projection_fidelity (str): ``"exact"`` for full microscopic
                   sparse simulation, or ``"compiled"`` / ``"fuzzy"`` for
                   top-k-only dynamics on frozen pregrown connectomes.
            norm_init (bool): One-time normalization of each postsynaptic
                   neuron's incoming weights, per fiber, to sum 1 -- the
                   reference implementation's ``norm_init``
                   (``.reference/mdabagia-nemo/brain.py``).  ON by default.
                   Literature and parity reproductions must pin it FALSE to
                   match un-normalized paper goldens.  It is the prerequisite
                   for ``recurrent_projection``: without it, self-recurrence
                   collapses independent assemblies into a shared attractor
                   (see ``project_rounds``).  ``numpy_sparse`` only; see
                   ``NumpySparseEngine._norm_scale`` for how it is realized
                   under lazy neuron materialization.
            recurrent_projection (bool): Apply target self-recurrence in the
                   ``project_rounds`` fast path.  Gated on ``norm_init``:
                   with ``norm_init=False`` it is ignored, so parity
                   reproductions are unaffected either way.

                   OFF by default, and that default is a KNOWN-WRONG
                   COMPROMISE rather than a modelling choice.  Off, this path
                   diverges from the documented ``project()`` protocol: it runs
                   stimulus-only projection with NO target self-recurrence, so
                   nothing built through it is an assembly in the defining
                   sense (Dabagia et al. 2024: a set of k neurons whose
                   INTERNAL weights have been strengthened).  ``ops.project``
                   used to route through here and therefore inherited that; it
                   no longer does, and runs the protocol directly.

                   Turning this ON is still the right end state and is blocked
                   on a real bug, not on taste.  Measured 2026-07-28, flipping
                   the default to True gives 10 test failures AND TWO HARD
                   SEGFAULTS (Windows access violation) in the batched
                   subsystem -- ``batched_next_token._scores`` and
                   ``batched_trainer._rec`` -- which evidently assume the
                   recurrence-free projection map.  That is a latent
                   memory-safety bug this flag merely exposes.  Fix it there
                   first, then flip this.

                   Parity when it IS on, parents re-cued by their own stimulus
                   (3 seeds x 8 items, rank-1 ID chance 0.125)::

                       n=2000 k=45 p=0.01   SELF 0.6426  ID 1.0000
                       n=1000 k=50 p=0.05   SELF 0.8992  ID 1.0000

                   matching an explicit per-round loop EXACTLY (0.6426 and
                   0.8992), which is what confirms the dropped self-recurrence
                   is the ONLY divergence between fast path and protocol.
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
        self._conn_rng_gen = np.random.default_rng(seed + 0x9E3779B9)
        self.disable_plasticity = False
        # Per-fiber plasticity override: (src, dst) -> enabled.  Missing keys default True.
        self.plasticity_mask: dict[tuple[str, str], bool] = {}

        # Compute engine — required, defaults to auto-detected best backend
        if engine == "auto":
            engine = detect_best_engine(n_hint)
        if isinstance(engine, str):
            engine_kwargs = dict(
                p=p, seed=seed, w_max=w_max, deterministic=deterministic,
            )
            # Feedforward inhibition is a numpy_sparse feature; only forward it
            # when engaged, so other engines' constructors are unaffected.
            if inhibitory_prob > 0.0:
                engine_kwargs["inhibitory_prob"] = inhibitory_prob
                engine_kwargs["inhibitory_weight"] = inhibitory_weight
            if synaptic_scaling:
                engine_kwargs["synaptic_scaling"] = True
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
            self._engine: ComputeEngine = create_engine(engine, **engine_kwargs)
        elif isinstance(engine, ComputeEngine):
            self._engine = engine
        else:
            raise TypeError(f"engine must be a string name or ComputeEngine instance, got {type(engine)}")

        if hasattr(self._engine, "set_projection_fidelity"):
            self._engine.set_projection_fidelity(projection_fidelity)

        # Secondary engine for explicit areas (lazily created)
        self._explicit_engine: ComputeEngine = None
        self._seed = seed

        # Inter-area inhibition groups for winner-take-all
        self._mutual_inhibition_groups: List[List[str]] = []
        # One-time incoming-weight normalization (reference `norm_init`).
        # Prerequisite for self-recurrence; see project_rounds.
        self.norm_init: bool = norm_init
        self._synaptic_scaling: bool = synaptic_scaling
        # Apply target self-recurrence in the project_rounds fast path.
        # Only safe together with norm_init (see project_rounds).
        self.recurrent_projection: bool = recurrent_projection
        # Total synaptic drive per target from the most recent projection,
        # summed over the SELECTED winners.
        self.last_activation_scores: Dict[str, float] = {}
        # Opt-in: also record global pre-k-WTA energy (sum of all_inputs over
        # every neuron, before winner selection). Off by default because it
        # copies a length-n vector per projection.
        self.record_activation: bool = False
        self.last_pre_kwta_totals: Dict[str, float] = {}

        # Used by activate_with_image()
        self.image_activation_engine = ImageActivationEngine()

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
                              input_noise_std=input_noise_std)
        if refracted:
            self._engine.set_refracted(area_name, True, refracted_strength)
        # For explicit areas, ALSO register with a dedicated explicit engine
        # that handles full n×n weight matrices and plasticity correctly.
        if explicit:
            explicit_eng = self._engine_for(area)  # lazily creates it
            explicit_eng.add_area(area_name, n, k, beta,
                                  refractory_period=refractory_period,
                                  inhibition_strength=inhibition_strength,
                                  slot_count=slot_count)
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
                            slot_count=getattr(existing_area, "slot_count", 0),
                        )
            return self._explicit_engine
        return self._engine

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

    def _sparse_sources_drive_to_explicit(
        self, target_name: str, sparse_source_names: List[str],
    ) -> np.ndarray:
        """Sum dense drive from sparse source assemblies into an explicit target."""
        xp = get_xp()
        tgt = self.areas[target_name]
        drive = xp.zeros(tgt.n, dtype=xp.float32)
        mapping_fn = getattr(self._engine, "get_neuron_id_mapping", None)

        for src_name in sparse_source_names:
            src = self.areas[src_name]
            if len(src.winners) == 0:
                continue
            conn = self.connectomes.get(src_name, {}).get(target_name)
            if not is_dense_connectome(conn):
                continue
            compact = np.asarray(to_cpu(src.winners), dtype=np.int64)
            if mapping_fn is not None:
                neuron_map = mapping_fn(src_name)
                if neuron_map:
                    real_ids = np.array(
                        [
                            int(neuron_map[c]) if c < len(neuron_map) else int(c)
                            for c in compact
                        ],
                        dtype=np.int64,
                    )
                else:
                    real_ids = compact
            else:
                real_ids = compact
            valid = real_ids[real_ids < conn.weights.shape[0]]
            if len(valid) == 0:
                continue
            drive += conn.weights[valid].sum(axis=0).astype(xp.float32, copy=False)
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
        """Supervised Hebbian update: strengthen active src winners → post neurons.

        Used for fixed slot targets (e.g. CLASS digit slots) without fixing the
        target assembly during ``project`` (which would skip plasticity).
        """
        if not self.fiber_plasticity_enabled(src_area, dst_area):
            return
        if src_area not in self.areas or dst_area not in self.areas:
            raise KeyError(f"unknown area in reinforce_connectome: {src_area!r} -> {dst_area!r}")
        src = self.areas[src_area]
        dst = self.areas[dst_area]
        pre = np.asarray(to_cpu(src.winners), dtype=np.intp)
        post = np.asarray(to_cpu(post_neurons), dtype=np.intp)
        if pre.size == 0 or post.size == 0:
            return
        b = beta if beta is not None else dst.beta_by_area.get(src_area, dst.beta)
        if b == 0:
            return
        conn = self.connectomes[src_area][dst_area]
        w = conn.weights
        valid_pre = pre[pre < w.shape[0]]
        valid_post = post[post < w.shape[1]]
        if valid_pre.size == 0 or valid_post.size == 0:
            return
        ix = np.ix_(valid_pre, valid_post)
        block = w[ix]
        # Zero-init connectomes (supervised slots) need a non-zero seed before *= (1+b).
        unset = block == 0
        if np.any(unset):
            block = block.copy()
            block[unset] = 1.0
        block *= (1 + b)
        w[ix] = block
        if self.w_max is not None:
            np.clip(w, 0, self.w_max, out=w)
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
    def read_only(self):
        """``frozen()`` plus no RECRUITMENT and no RNG advance.

        ``frozen()`` stops weights from changing. It does not stop the area
        from GROWING, and growth turned out to be the channel that actually
        made measurement change the measured. Parsing three items as [X,Y,Z]
        and as [Z,Y,X] under ``frozen()`` left ROLE_ACTION at w=647 in one and
        w=650 in the other -- structurally different brains, whose later
        synapses cannot agree however init is seeded. That is the mechanism
        behind probes contaminating each other.

        Inside this block an area answers "which of the neurons I already have
        respond best?" rather than "what would I become?" -- which is the
        semantics a READOUT wants anyway: measure the trained brain, not one
        that grows while being read. Areas still below ``k`` materialised
        neurons are exempt, since there is nothing there to select from.

        The generator's state and the areas' winners are restored too, so
        dynamics draws (input noise, tie-breaks, subsampling) cost the host
        nothing either. Winners still MOVE inside the block -- a probe that
        could not respond would be measuring nothing -- but the host is
        unchanged on exit, which is the whole contract.

        Restoring winners here rather than at each call site is deliberate.
        ``frozen()`` exists because its save/set/restore had been hand-rolled
        about sixty times and one missing ``finally`` poisons the rest of the
        session; a winners snapshot every caller must remember is the same
        trap one level up.
        """
        engines, saved_flags, saved_states = [], [], []
        for engine in self._all_engines():
            if not hasattr(engine, "_no_recruitment"):
                continue
            engines.append(engine)
            saved_flags.append(engine._no_recruitment)
            saved_states.append(engine._rng.bit_generator.state)
            engine._no_recruitment = True
        winners = {name: (area.winners.copy(), area.w, area.fixed_assembly)
                   for name, area in self.areas.items()}
        try:
            with self.frozen():
                yield self
        finally:
            for engine, flag, state in zip(engines, saved_flags, saved_states):
                engine._no_recruitment = flag
                engine._rng.bit_generator.state = state
            for name, (won, w, fixed) in winners.items():
                area = self.areas[name]
                area.unfix_assembly()
                area.winners = won.copy()
                area.w = w
                if fixed:
                    area.fix_assembly()
                self._engine_for(area).set_winners(name, won.copy())

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
        elif external_inputs is not None or projections is not None:
            # Inject external activations, then route through the same projection path
            xp = get_xp()
            for area_name, input_winners in (external_inputs or {}).items():
                area = self.areas[area_name]
                area.winners = xp.asarray(input_winners, dtype=xp.uint32)
                self._engine_for(area).set_winners(
                    area_name, np.asarray(to_cpu(input_winners), dtype=np.uint32))
            self._project_impl({}, projections or {}, verbose, drive)
        else:
            raise ValueError("Must provide either legacy API parameters or new API parameters")

    def _project_impl(self, areas_by_stim, dst_areas_by_src_area, verbose=0,
                      external_drive=None):
        """
        Core projection implementation.

        Builds input mappings from stimuli and areas, then delegates to the
        compute engine for all projection, winner selection, and plasticity.
        """
        external_drive = external_drive or {}
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

        # ORDER IS LOAD-BEARING, so this must not be a set. It drives the batch
        # config order below and the sequential projection loop, and projecting
        # an area materializes neurons -- so the order determines how the seeded
        # RNG stream is consumed. `stim_in.keys() | area_in.keys()` is a set of
        # str, whose iteration order changes with PYTHONHASHSEED from process to
        # process; identical seeds gave different results across runs while
        # being perfectly stable within one run. dict.fromkeys dedupes while
        # keeping the deterministic insertion order of the two dicts.
        to_update_area_names = dict.fromkeys(
            list(stim_in.keys()) + list(area_in.keys())
        )

        # Sync winner state from Area descriptors to ALL engines for source
        # areas.  This is needed for two reasons:
        # 1. External code may set area.winners directly (pattern completion)
        # 2. Cross-engine projections: an explicit area's winners must be
        #    visible to the sparse engine when used as a source.
        all_source_areas = dict.fromkeys(
            src for sources in area_in.values() for src in sources
        )
        for area_name in all_source_areas:
            area = self.areas[area_name]
            if len(area.winners) > 0:
                winners_arr = np.asarray(to_cpu(area.winners), dtype=np.uint32)
                self._engine.set_winners(area_name, winners_arr)
                eng_st = self._engine._areas.get(area_name)
                if eng_st is not None:
                    eng_st.explicit_source = area.explicit
                if self._engine is not self._explicit_engine:
                    torch_st = getattr(self._engine, "_areas", {}).get(area_name)
                    if torch_st is not None and hasattr(torch_st, "explicit_source"):
                        torch_st.explicit_source = area.explicit
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

        # Batched path: process multiple targets in one kernel launch
        # (only for non-explicit areas on the main engine)
        non_explicit = [n for n in to_update_area_names
                        if not self.areas[n].explicit]
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
            if external_drive_vec is not None and engine is self._explicit_engine:
                result = engine.project_into(
                    area_name,
                    from_stimuli=stim_in[area_name],
                    from_areas=from_area_list,
                    plasticity_enabled=not self.disable_plasticity,
                    external_drive=external_drive_vec,
                    record_activation=getattr(self, 'record_activation', False),
                )
            else:
                result = engine.project_into(
                    area_name,
                    from_stimuli=stim_in[area_name],
                    from_areas=from_area_list,
                    plasticity_enabled=not self.disable_plasticity,
                    record_activation=getattr(self, 'record_activation', False),
                )
            self._apply_result(area_name, result, stim_in, area_in)
            activation_scores[area_name] = result.total_activation
            if getattr(self, 'record_activation', False):
                pre_kwta[area_name] = float(result.pre_kwta_total or 0.0)

        # Total synaptic drive each target received, kept for the caller. This
        # is the quantity area-level competition is decided on, so exposing it
        # lets callers score a competition the same way the brain does.
        self.last_activation_scores = dict(activation_scores)
        if getattr(self, 'record_activation', False):
            self.last_pre_kwta_totals = dict(pre_kwta)

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
        # NOTE: `a != target` drops target self-recurrence in this fast path.
        # That is a real divergence from the documented project() protocol,
        # which specifies (stimulus + target->target) recurrence on rounds
        # 2..T, and from Assembly Calculus itself -- Dabagia et al. 2024 define
        # an assembly as a set of k neurons whose INTERNAL synaptic weights
        # have been strengthened, so dropping recurrence removes the defining
        # property.  It was left in because recurrence measurably collapsed
        # assemblies: two INDEPENDENT stimuli projected into one area
        # (n=2000, k=50, p=0.05, beta=0.1, 5 seeds) reached overlap 0.240 at
        # 5 rounds and 0.940 at 15 rounds, against a chance level of 0.025.
        #
        # HISTORY, CORRECTED.  The original diagnosis blamed the absence of
        # ongoing homeostatic normalization, and `synaptic_scaling` was written
        # to supply it.  That was wrong on both counts.  Running the reference
        # implementation (.reference/mdabagia-nemo/brain.py, `RecurrentArea`)
        # shows recurrence is ALWAYS on there, with the same multiplicative
        # plasticity and no weight clipping, and that `normalize()` is called
        # ONLY from `reset()` under `norm_init` -- it is a ONE-TIME
        # INITIALIZATION, not ongoing homeostasis.  With norm_init the
        # reference holds chance overlap (0.024 at 15 rounds); without it, it
        # drifts up (0.112).
        #
        # The real mechanism is degree bias.  With every weight initialized to
        # 1, a neuron's drive is essentially its number of active afferents, so
        # k-cap systematically elects the random graph's high-in-degree hubs;
        # recurrence compounds that, and every stimulus converges on the same
        # hubs.  Measured here, the winners' recurrent in-degree z-score rose
        # to +1.66 by round 15.  Normalizing each postsynaptic neuron's
        # incoming weights per fiber to sum 1 removes the degree advantage,
        # after which recurrence is safe (see NumpySparseEngine._norm_scale).
        #
        # Self-recurrence is therefore enabled only when that normalization is
        # active.  `synaptic_scaling` is still accepted as a gate for backward
        # compatibility, but norm_init is the validated one.
        #
        # HOW FAR THAT GENERALISES -- measured 2026-07-28,
        # research/experiments/norm_init_recurrence_limit.py, n=1000 k=50
        # beta=0.1, M items sharing one area, rank-1 identity across all M:
        #
        #     M           2      4      8     16     32     64    128
        #     rec RAW  1.000  1.000  0.667  0.188  0.031  0.016  0.009
        #     rec norm 1.000  1.000  1.000  1.000  1.000  0.039  0.018
        #     ff  norm 1.000  1.000  1.000  1.000  1.000  1.000  1.000
        #
        # THE TABLE ABOVE IS THE SAMPLER'S, NOT THE SUBSTRATE'S -- re-derived
        # 2026-08-02 on `numpy_exact`, which computes the drive instead of
        # inventing one for neurons that have not fired
        # (research/notes/recurrence_ceiling_on_exact_drive.md):
        #
        #     ceiling (acc > 0.90)      numpy_sparse   numpy_exact
        #     recurrent, norm_init ON       M=32          M=16
        #     recurrent, norm_init OFF      M= 4          M=16
        #     feed-forward,        ON       M=64          M=64
        #
        # So norm_init's capacity gain under recurrence is 8.0x on the sampler
        # and 1.0x on exact drive.  It still does something (acc 0.73 vs 0.44
        # at M=32) but it does not move the ceiling, and "moves the ceiling
        # from M=4 to M=32" was an artifact: the sampler's error is a function
        # of LOAD, and norm_init changes which neurons win and therefore how
        # fast the area recruits.  The two arms did not share the error.
        #
        # The "ceiling scales with n" inference is likewise UNSUPPORTED rather
        # than refuted -- it was read off the same instrument, and n at fixed k
        # is load.  Re-running that sweep on exact drive is task #90's
        # remainder.
        #
        # WHAT DID NOT CHANGE, and is why this gate stays: recurrence is the
        # collapse channel and is far worse than feed-forward on BOTH engines
        # at every M, and the exact ceiling with norm_init on is LOWER (16, not
        # 32) than the sampler claimed.  The gate was right; the stated reason
        # was not.
        #
        # SO DO NOT FLIP `recurrent_projection` ON GLOBALLY.  The production
        # lexicon trains through this exact path -- training/batch.py
        # `apply_lexicon_word` passes {core_area: [core_area]}, which the
        # `a != target` filter below silently strips -- with dozens of words
        # per core area.  Enabling self-recurrence there collapses the lexicon
        # into one assembly, and it fails SILENTLY: at M=128 each word still
        # re-cues to overlap 0.68 with what was stored (so any probe reading
        # only self-overlap reports success) while rank-1 identity across the
        # lexicon is 0.018 against a chance of 0.008.
        #
        # Feed-forward has no measured ceiling at all -- 1.0000 up to M=256 in
        # n=1000, i.e. 12.8x oversubscription, with pairwise overlap 0.0510
        # against a random-pair floor of 0.0500 (lexicon_capacity_law.py).  For
        # areas holding many items, that is the regime to be in.
        allow_self = getattr(self, "recurrent_projection", False) and (
            getattr(self, "norm_init", False)
            or getattr(self, "_synaptic_scaling", False)
        )
        from_areas_list = [a for a, tgts in dst_areas_by_src_area.items()
                           if target in tgts and (allow_self or a != target)]

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
        # Keep the recruitment reading alive across a later `winners`
        # assignment, which clobbers `w`. See Area.get_num_ever_fired.
        area._num_ever_fired = int(result.num_ever_fired)
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
                connectome = Connectome(stim.size, area.n, self.p, sparse=False, rng=self._conn_rng)
            else:
                # For sparse areas, start with empty 1D vector of length area.w (0)
                connectome = Connectome(stim.size, area.n, self.p, sparse=True, rng=self._conn_rng)
                connectome.weights = xp.empty(0, dtype=xp.float32)
            self.connectomes_by_stimulus[stim_name][area.name] = connectome
            area.beta_by_stimulus[stim_name] = area.beta

        # Initialize self-connection for the area
        if area.explicit:
            self_connectome = Connectome(area.n, area.n, self.p, sparse=False, rng=self._conn_rng)
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
                    connectome = Connectome(other_area.n, area.n, self.p, sparse=False, rng=self._conn_rng)
                    connectome_rev = Connectome(area.n, other_area.n, self.p, sparse=False, rng=self._conn_rng)
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
                connectome = Connectome(stimulus.size, area.n, self.p, sparse=False, rng=self._conn_rng)
            else:
                connectome = Connectome(stimulus.size, area.n, self.p, sparse=True, rng=self._conn_rng)
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

    def set_competition_policy(self, area_name: str, policy) -> None:
        """Set the winner-selection policy for an area (sparse engine path)."""
        self.areas[area_name].winner_policy = policy
        eng_areas = getattr(self._engine, "_areas", None)
        if eng_areas is not None and area_name in eng_areas:
            eng_areas[area_name].winner_policy = policy

    def set_input_noise(self, area_name: str, std: float) -> None:
        """Add Gaussian noise to pre-k-WTA inputs (coin-flip / sampling)."""
        self.areas[area_name].input_noise_std = std
        eng_areas = getattr(self._engine, "_areas", None)
        if eng_areas is not None and area_name in eng_areas:
            eng_areas[area_name].input_noise_std = std

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
        """Fast structural clone for sweep forks (numpy_sparse engine)."""
        import copy

        eng = self._engine
        if not hasattr(eng, "clone"):
            return copy.deepcopy(self)

        cloned = object.__new__(Brain)
        cloned.p = self.p
        cloned.w_max = self.w_max
        cloned.save_size = self.save_size
        cloned.save_winners = self.save_winners
        cloned.deterministic = self.deterministic
        cloned.areas = copy.deepcopy(self.areas)
        cloned.stimuli = copy.deepcopy(self.stimuli)
        cloned.connectomes = {}
        cloned.connectomes_by_stimulus = {}
        cloned.rng = copy.deepcopy(self.rng)
        cloned.disable_plasticity = self.disable_plasticity
        cloned.plasticity_mask = dict(self.plasticity_mask)
        cloned._mutual_inhibition_groups = copy.deepcopy(self._mutual_inhibition_groups)
        # NOTE: clone() bypasses __init__ via object.__new__, so every Brain
        # attribute must be copied explicitly here or forks lose it.
        cloned.last_activation_scores = dict(
            getattr(self, "last_activation_scores", {}))
        cloned.record_activation = getattr(self, "record_activation", False)
        cloned.last_pre_kwta_totals = dict(
            getattr(self, "last_pre_kwta_totals", {}))
        cloned._seed = self._seed
        cloned._engine = eng.clone()
        cloned._explicit_engine = None
        cloned.image_activation_engine = ImageActivationEngine()
        cloned._sync_engine_connectomes()
        return cloned
    
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
