"""TorchSparseEngine: PyTorch-native GPU engine for assembly calculus.

Uses PyTorch CUDA tensors for all state and computation.  Eliminates
CuPy entirely from the hot path, gaining:
- Lower per-op dispatch overhead (~50us vs ~200us for CuPy)
- torch_ops.topk: single fused CUDA kernel for winner selection
- torch advanced indexing for Hebbian updates
- Zero CuPy<->torch conversion overhead
- Hash-based deterministic initialization (ported from cuda_engine.py)

The same statistical sparse algorithm as NumpySparseEngine: truncated
normal sampling, Hebbian w *= (1+beta), amortised buffer growth, lazy
expansion.  Truncated normal sampling defaults to GPU-native
(torch_ops.erfinv) but can fall back to CPU (scipy) for deterministic mode
or via gpu_sampling=False.

Requires: torch with CUDA support.

Layout.  This module holds the constructor, area / stimulus / fiber
registration and the state accessors; the algorithm lives in mixins beside it,
one concern each, all reading the state the constructor builds:

    _engine_projection.py  ProjectionMixin  project_into, Hebbian update, scaling
    _engine_growth.py      GrowthMixin      synapses for recruits, densify, materialize
    _engine_sampling.py    SamplingMixin    seeded device RNG, candidate drives, bootstrap
    _engine_norm.py        NormInitMixin    norm_init's in-degree per storage format
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional

import torch

from ._torch_ops import torch_ops

# The fixed-target learning semantics and its A/B escape hatch have ONE
# owner; both engines must read the same switch or an A/B on one engine
# silently means something else on the other.
from .._homeostasis import (HomeostasisConfig, check_area_homeostasis, validate_lri_parameters,
                            validate_refraction_strength)
from ..connectome import Connectome
from ..engine import (
    ComputeEngine,
    validate_deterministic_allocation,
    validate_engine_boolean_option,
)
from ..index_spaces import CompactIdx, NeuronIds, validated_indices
from ..registration import (validate_input_noise, validate_stimulus_registration,
                            validate_area_registration, validate_plasticity_rate)
from ..semantics import (
    ArithmeticMode,
    CandidateDomain,
    ConnectomeMode,
    ModelSemantics,
    NormalizationMode,
    PlasticityRule,
    StimulusDriveLaw,
    TieBreakRule,
)

try:
    from ...compute.sparse_simulation import SparseSimulationEngine
    from ...compute.winner_selection import WinnerSelector
    from ...compute.winner_policies import validate_competition_policy
except ImportError:
    from compute.sparse_simulation import SparseSimulationEngine
    from compute.winner_selection import WinnerSelector
    from compute.winner_policies import validate_competition_policy

from ._hash import (
    WEIGHT_DTYPE, fnv1a_pair_seed,
)
from ._csr import (
    CSRConn, DENSE_MIN_P, TorchDenseConn,
)
from ._state import (
    LAZY_ID_THRESHOLD, TorchAreaState, StimulusState, TorchConn,
)

from ._engine_norm import NormInitMixin
from ._engine_sampling import SamplingMixin
from ._engine_projection import ProjectionMixin
from ._engine_growth import GrowthMixin

TorchAreaConn = CSRConn | TorchDenseConn


class TorchSparseEngine(NormInitMixin, SamplingMixin, ProjectionMixin, GrowthMixin, ComputeEngine):
    """PyTorch-native GPU engine with hash-based connectivity.

    Same statistical sparse algorithm as NumpySparseEngine but all arrays
    are torch.cuda tensors.  Key performance advantages:
    - torch_ops.topk: single fused kernel (vs argpartition + argsort)
    - Lower per-op dispatch overhead than CuPy
    - No CuPy<->torch conversion for operations
    - Hash-based deterministic initialization
    - Optional GPU-native truncated normal sampling via torch_ops.erfinv

    Parameters:
        p:             Connection probability.
        seed:          Global random seed.
        w_max:         Hebbian weight ceiling.
        deterministic: If True, use legacy exact-fit expansion.
        gpu_sampling:  If True (default), sample truncated normal on GPU
                       using torch_ops.erfinv instead of CPU scipy.  Falls
                       back to CPU path when deterministic=True.
    """

    supports_norm_init = True
    supports_synaptic_scaling = True
    supports_deterministic_allocation = True
    supports_gpu_sampling = True
    supports_dense_drive = True
    supports_batched_next_token = True
    supports_input_noise = True
    supports_refraction = True

    def describe_model_semantics(self) -> ModelSemantics:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-model-semantics"""
        return ModelSemantics(
            connectome=ConnectomeMode.LAZY_CONTENT_ADDRESSED,
            candidate_domain=(
                CandidateDomain.ALL_NEURONS_WITH_SAMPLED_DRIVE
                if self.dense_drive
                else CandidateDomain.MATERIALIZED_PLUS_ORDER_STATISTICS
            ),
            stimulus_drive=StimulusDriveLaw.LAZY_CONDITIONED_AFFERENT_COUNT,
            default_tie_break=TieBreakRule.BACKEND_TOPK_ORDER,
            arithmetic=ArithmeticMode.FLOAT32,
            normalization=(
                NormalizationMode.INVERSE_INDEGREE
                if self.norm_init
                else NormalizationMode.NONE
            ),
            plasticity=(
                PlasticityRule.MULTIPLICATIVE_UNBOUNDED
                if self.w_max is None
                else PlasticityRule.MULTIPLICATIVE_CLIPPED
            ),
            weight_ceiling=self.w_max,
        )

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False, gpu_sampling: bool = True,
                 **kwargs):
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-option-remainder
        deterministic = validate_deterministic_allocation(type(self), deterministic)
        gpu_sampling = validate_engine_boolean_option(
            type(self), "gpu_sampling", gpu_sampling, "supports_gpu_sampling"
        )
        if "projection_fidelity" in kwargs:
            from ..projection_fidelity import validate_projection_fidelity_capability
            validate_projection_fidelity_capability(
                type(self), kwargs.pop("projection_fidelity")
            )
        if "inhibitory_prob" in kwargs or "inhibitory_weight" in kwargs:
            from ..feedforward_inhibition import (
                FeedforwardInhibitionConfig,
                validate_feedforward_inhibition_capability,
            )
            inhibition = FeedforwardInhibitionConfig(
                probability=kwargs.pop("inhibitory_prob", 0.0),
                weight=kwargs.pop("inhibitory_weight", -0.2),
            )
            validate_feedforward_inhibition_capability(type(self), inhibition)
        self.p = p
        self.seed = int(seed)  # Shared construction identity when adopted by Brain.
        self.w_max = w_max
        self._deterministic = deterministic
        self._gpu_sampling = gpu_sampling and not deterministic
        # One-time incoming-weight normalization (reference `norm_init`): a
        # read-time per-postsynaptic 1/d_j scale, ported from NumpySparseEngine.
        # Previously this kwarg was silently swallowed by **kwargs and ignored,
        # so a Brain(norm_init=True, engine="torch_sparse") got NO normalization.
        homeostasis = HomeostasisConfig(**{
            name: kwargs.pop(name, False)
            for name in HomeostasisConfig.__dataclass_fields__
        })
        self.norm_init = homeostasis.norm_init
        # Dense-drive mode (see docs/gpu_scale_design.md, Lever A): score ALL n
        # candidate neurons each round instead of sampling ~k order statistics.
        # Materialized neurons keep their real CSR drive; the (n-w) unmaterialized
        # get an i.i.d. Binomial(sum(input_sizes), p) draw (the same statistical
        # model the sparse sampler approximates), then a single topk over n.
        # More arithmetic than the sparse path -- deliberately, so it is
        # GPU-parallel and, with a fixed [n] drive, batchable (Lever B).
        self.dense_drive = validate_engine_boolean_option(
            type(self), "dense_drive", kwargs.pop("dense_drive", False),
            "supports_dense_drive",
        )
        # Read-only inference: suppress candidate sampling so a projection never
        # materializes new neurons (select only among already-materialized ones).
        # Inference should not mutate the brain; this also makes prediction
        # deterministic and gives a fixed connectome to batch over (BatchedLM).
        readonly = kwargs.pop("readonly", False)
        if type(readonly) is not bool:
            raise ValueError("readonly must be a bool")
        self.readonly = readonly
        # Per-fiber connection density (`add_connectivity`). Empty until set;
        # every consumer must route through `_p_for` so homogeneous brains
        # keep the scalar fast paths (mirrors NumpySparseEngine._fiber_p).
        self._fiber_p: Dict[tuple, float] = {}
        # Homeostatic synaptic scaling (CSRConn.scale_columns; the law and its
        # measured failure modes live on NumpySparseEngine._normalize_area_
        # columns). bool True scales every target; a collection scopes it to
        # the listed target areas. This kwarg used to be SILENTLY SWALLOWED by
        # **kwargs -- the same silent no-op that once ate norm_init (above),
        # which would have run a homeostasis study with homeostasis off.
        self.synaptic_scaling = homeostasis.synaptic_scaling
        if homeostasis.synaptic_scaling_deferred:
            raise NotImplementedError(
                "synaptic_scaling_deferred is not implemented on "
                "torch_sparse: per-update scaling only. Use engine="
                "'numpy_sparse' for the deferred/flush mode.")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"TorchSparseEngine got unsupported constructor options: {names}"
            )
        # READ-ONLY PROBE SUPPORT. `brain.read_only()` (and `probe()`, which
        # every evaluation harness runs inside) gates recruitment by setting
        # this flag -- but it discovers engines with `hasattr(engine,
        # "_no_recruitment")` and SILENTLY SKIPS any engine lacking the
        # attribute. Without it, probes recruited: a trained Z60 word-problem
        # machine read back at 0.040 trajectory accuracy (chance 0.017)
        # because every evaluation step grew the arc and moved the compact
        # index space out from under the stored assemblies
        # ([[probe-isolation-required]] -- recruitment, not plasticity, is
        # the channel by which a readout changes what it is reading).
        self._no_recruitment = False
        self._rng = np.random.default_rng(seed)
        # Device-side generator for candidate draws. Without it the samplers
        # fall through to torch's PROCESS-GLOBAL stream and Brain(seed=) stops
        # being reproducible across processes -- see `_device_rng`.
        self._torch_gen = None
        self._plasticity_enabled_global = True
        self._global_seed = seed
        self._pair_seeds: Dict[tuple, int] = {}

        self._device = torch_ops.device('cuda')

        # Internal state
        self._areas: Dict[str, TorchAreaState] = {}
        self._stimuli: Dict[str, StimulusState] = {}

        # Connectivity: stim_name -> area_name -> TorchConn (1-D weights)
        self._stim_conns: Dict[str, Dict[str, TorchConn]] = defaultdict(dict)
        # Connectivity: src_area -> tgt_area -> CSRConn (2-D weights)
        self._area_conns: Dict[str, Dict[str, TorchAreaConn]] = defaultdict(dict)
        # Dense explicit→sparse edges (Connectome objects, not CSR)
        self._dense_area_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)

        # Reusable CPU math primitives (truncated normal, input splits)
        self._sparse_sim = SparseSimulationEngine(
            np.random.default_rng(seed))
        self._winner_sel = WinnerSelector(self._rng)

    # -- Pair seed derivation -----------------------------------------------

    def _get_pair_seed(self, source: str, target: str) -> int:
        key = (source, target)
        if key not in self._pair_seeds:
            self._pair_seeds[key] = fnv1a_pair_seed(
                self._global_seed, source, target)
        return self._pair_seeds[key]

    def set_dense_area_conn(self, src: str, tgt: str, conn: Connectome) -> None:
        """Install a dense connectome for explicit→sparse cross-engine edges."""
        self._dense_area_conns[src][tgt] = conn

    def _dense_weights(self, conn: Connectome) -> torch.Tensor:
        w = conn.weights
        if isinstance(w, torch.Tensor):
            return w.float()
        return torch_ops.from_numpy(np.asarray(w, dtype=np.float32)).to(self._device)

    # -- Registration -------------------------------------------------------

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0,
                 winner_policy=None,
                 input_noise_std: float = 0.0) -> None:
        input_noise_std = validate_input_noise(input_noise_std)
        n, k = validate_area_registration(name, n, k, existing=self._areas, reserved=self._stimuli)
        validate_competition_policy(n, winner_policy)
        beta = validate_plasticity_rate(beta)
        refractory_period, inhibition_strength = validate_lri_parameters(
            refractory_period, inhibition_strength)
        area = TorchAreaState(
            name=name, n=n, k=k, beta=beta,
            refractory_period=refractory_period,
            inhibition_strength=inhibition_strength,
            winner_policy=winner_policy,
            input_noise_std=input_noise_std,
        )
        if n > LAZY_ID_THRESHOLD:
            area._lazy_ids = True
            area._used_ids = set()
            area._id_rng = np.random.default_rng(
                self._rng.integers(0, 2**32))
        else:
            area.neuron_id_pool = self._rng.permutation(
                np.arange(n, dtype=np.uint32))
            area.neuron_id_pool_ptr = 0
        self._areas[name] = area

        for stim_name in self._stimuli:
            conn = TorchConn(
                torch_ops.empty(0, dtype=WEIGHT_DTYPE, device=self._device),
                sparse=True)
            self._stim_conns[stim_name][name] = conn
            area.beta_by_source[stim_name] = beta

        for other_name, other in self._areas.items():
            if other_name == name:
                self._area_conns[name][name] = CSRConn(
                    device=self._device)
            else:
                self._area_conns[other_name][name] = CSRConn(
                    device=self._device)
                self._area_conns[name][other_name] = CSRConn(
                    device=self._device)
                area.beta_by_source[other_name] = beta
                other.beta_by_source[name] = beta

    def add_stimulus(self, name: str, size: int) -> None:
        size = validate_stimulus_registration(name, size, existing=self._stimuli,
                                             reserved=self._areas)
        self._stimuli[name] = StimulusState(name=name, size=size)
        for area_name, area in self._areas.items():
            conn = TorchConn(
                torch_ops.empty(0, dtype=WEIGHT_DTYPE, device=self._device),
                sparse=True)
            self._stim_conns[name][area_name] = conn
            area.beta_by_source[name] = area.beta

    def _p_for(self, source: str, target: str) -> float:
        """This fiber's density; the brain's global `p` unless overridden."""
        return self._fiber_p.get((source, target), self.p)

    def heterogeneous(self) -> bool:
        """True once any fiber's density differs from the global `p`."""
        return bool(self._fiber_p)

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        """Set one fiber's density. Mirrors NumpySparseEngine.add_connectivity.

        STRUCTURAL, SO IT MUST PRECEDE TRAFFIC: `p` is baked into the hash
        threshold that decides which synapses exist, so changing it once the
        fiber has initialized entries would leave potentiation on synapses
        that no longer exist. Requesting the value already in force is a
        no-op.

        This used to be `pass` -- a SILENT no-op, so an organ built with
        `organ_p=0.5` inside a p=0.05 brain got p=0.05 fibers and no warning:
        the whole anatomy was quietly wrong ([[silent-no-op-dead-fibers]]).
        """
        key = (source, target)
        if float(p) == float(self._fiber_p.get(key, self.p)):
            return
        if source not in self._areas and source not in self._stimuli:
            raise KeyError(f"unknown source {source!r}")
        if target not in self._areas:
            raise KeyError(f"unknown target area {target!r}")
        if source in self._areas:
            csr = self._area_conns.get(source, {}).get(target)
            carried = csr is not None and (
                csr.nnz > 0 or csr._log_rows > 0 or csr._log_cols > 0)
        else:
            conn = self._stim_conns.get(source, {}).get(target)
            carried = (conn is not None and conn.weights is not None
                       and int(conn.weights.numel()) > 0)
        if carried:
            raise RuntimeError(
                f"add_connectivity({source!r}, {target!r}, p={p}) after the "
                f"fiber has carried traffic. Connectivity is structural: "
                f"changing it now would leave potentiation on synapses that "
                f"no longer exist. Set it before the first projection.")
        self._fiber_p[key] = float(p)
        # Representation follows density: a fiber this dense stores as a
        # plain tensor (`TorchDenseConn` -- see its docstring for the
        # 13-minute CSR-rebuild failure this replaces). Safe exactly because
        # of the guard above: the fiber is provably empty here.
        if (source in self._areas and float(p) >= DENSE_MIN_P
                and not isinstance(self._area_conns[source][target],
                                   TorchDenseConn)):
            self._area_conns[source][target] = TorchDenseConn(
                device=self._device,
                max_rows=int(self._areas[source].n),
                max_cols=int(self._areas[target].n))

    # -- State accessors ----------------------------------------------------

    def get_winners(self, area: str) -> CompactIdx:
        st = self._areas[area]
        return CompactIdx(st.winners.cpu().numpy().astype(np.uint32))

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        if isinstance(winners, NeuronIds) and self.get_neuron_id_mapping(area):
            raise TypeError("torch sparse engine winner inputs require compact indices")
        st = self._areas[area]
        # torch_ops.tensor(uint32_array, dtype=int32, device=cuda) hits a slow
        # element-wise path -- uint32 is not a native torch dtype, so at large k
        # this dominated the whole projection (measured 32ms/round at k=100k).
        # Route through int64 (torch-native) so from_numpy is zero-copy, then a
        # single fused H2D + cast kernel.
        valid = validated_indices(winners, upper=st.n, label=f"{area} winners", unique=True)
        arr = np.ascontiguousarray(valid, dtype=np.int64)
        st.winners = torch_ops.from_numpy(arr).to(
            self._device, dtype=torch_ops.int32, non_blocking=True)

    def materialized_count(self, area: str):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-cue-recovery"""
        state = self._areas.get(area)
        return None if state is None else int(state.w)

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].w

    def get_neuron_id_mapping(self, area: str) -> list:
        return self._areas[area].compact_to_neuron_id

    # -- Plasticity control -------------------------------------------------

    def set_input_noise(self, area: str, std: float) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-input-noise"""
        self._areas[area].input_noise_std = validate_input_noise(std)

    def set_competition_policy(self, area: str, policy) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-runtime-policy"""
        validate_competition_policy(self._areas[area].n, policy)
        self._areas[area].winner_policy = policy

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = validate_plasticity_rate(beta)

    def get_beta(self, target: str, source: str) -> float:
        tgt = self._areas[target]
        return tgt.beta_by_source.get(source, tgt.beta)

    # -- Assembly fixation --------------------------------------------------

    def fix_assembly(self, area: str) -> None:
        st = self._areas[area]
        if st.winners is None or st.winners.numel() == 0:
            raise ValueError(f"Area {area} has no winners to fix.")
        st.fixed_assembly = True

    def unfix_assembly(self, area: str) -> None:
        self._areas[area].fixed_assembly = False

    def is_fixed(self, area: str) -> bool:
        return self._areas[area].fixed_assembly

    # -- Connection reset ---------------------------------------------------

    def reset_area_connections(self, area: str) -> None:
        for src_name in list(self._area_conns.keys()):
            if area not in self._area_conns[src_name]:
                continue
            self._area_conns[src_name][area].reset()

    # -- LRI control --------------------------------------------------------

    def clear_refractory(self, area: str) -> None:
        self._areas[area]._refractory_history.clear()

    def set_lri(self, area: str, refractory_period: int,
                inhibition_strength: float) -> None:
        refractory_period, inhibition_strength = validate_lri_parameters(
            refractory_period, inhibition_strength)
        st = self._areas[area]
        st.refractory_period = refractory_period
        st.inhibition_strength = inhibition_strength
        st._refractory_history = deque(
            maxlen=max(refractory_period, 1))

    # -- Refracted mode control ---------------------------------------------

    def set_refracted(self, area: str, enabled: bool,
                      strength: float = 0.0) -> None:
        if type(enabled) is not bool:
            raise TypeError("refracted enabled flag must be a bool")
        strength = validate_refraction_strength(strength)
        st = self._areas[area]
        check_area_homeostasis(area, refracted=enabled, synaptic_scaling=self.synaptic_scaling)
        st.refracted = enabled
        st.refracted_strength = strength
        if enabled and st._cumulative_bias.numel() == 0:
            st._cumulative_bias = torch_ops.zeros(
                max(st.w, 0), dtype=torch_ops.float32, device=self._device)

    def clear_refracted_bias(self, area: str) -> None:
        st = self._areas[area]
        st._cumulative_bias = torch_ops.zeros(
            max(st.w, 0), dtype=torch_ops.float32, device=self._device)

    # -- Weight normalization -----------------------------------------------

    def normalize_weights(self, target: str, source: Optional[str] = None) -> None:
        check_area_homeostasis(target, refracted=self._areas[target].refracted,
                               synaptic_scaling=True)
        eps = 1e-8

        def _norm_stim(conn):
            w = conn.weights
            if w.ndim == 1 and w.numel() > 0:
                total = w.sum().item()
                if total > eps:
                    conn.weights = w / total

        if source is not None:
            if (source in self._stim_conns
                    and target in self._stim_conns[source]):
                _norm_stim(self._stim_conns[source][target])
            if (source in self._area_conns
                    and target in self._area_conns[source]):
                self._area_conns[source][target].normalize_columns(eps)
            return

        for stim_name in self._stim_conns:
            if target in self._stim_conns[stim_name]:
                _norm_stim(self._stim_conns[stim_name][target])
        for src_name in self._area_conns:
            if target in self._area_conns[src_name]:
                self._area_conns[src_name][target].normalize_columns(eps)


    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "torch_sparse"
