"""TorchSparseEngine: PyTorch-native GPU engine for assembly calculus.

Uses PyTorch CUDA tensors for all state and computation.  Eliminates
CuPy entirely from the hot path, gaining:
- Lower per-op dispatch overhead (~50us vs ~200us for CuPy)
- torch.topk: single fused CUDA kernel for winner selection
- torch advanced indexing for Hebbian updates
- Zero CuPy<->torch conversion overhead
- Hash-based deterministic initialization (ported from cuda_engine.py)

The same statistical sparse algorithm as NumpySparseEngine: truncated
normal sampling, Hebbian w *= (1+beta), amortised buffer growth, lazy
expansion.  Truncated normal sampling defaults to GPU-native
(torch.erfinv) but can fall back to CPU (scipy) for deterministic mode
or via gpu_sampling=False.

Requires: torch with CUDA support.
"""

import math
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List

import torch

from .._pricing import (
    area_fiber_activity, candidate_divisor, inverse_indegree,
)
# The fixed-target learning semantics and its A/B escape hatch have ONE
# owner; both engines must read the same switch or an A/B on one engine
# silently means something else on the other.
from ..numpy_engine._sparse import (
    _fixed_target_plasticity_enabled as _np_fixed_target_plasticity_enabled,
)
from .._homeostasis import (HomeostasisConfig, check_area_homeostasis, validate_lri_parameters, refraction_increment, scaling_applies,
                            scaling_setpoint)
from ..connectome import Connectome
from ..engine import ComputeEngine, ProjectionResult
from ..index_spaces import reserve_initial_neuron_ids
from ..registration import validate_input_noise, validate_stimulus_registration, validate_area_registration

try:
    from ...compute.sparse_simulation import SparseSimulationEngine
    from ...compute.winner_selection import WinnerSelector
    from ...compute.winner_policies import TopKPolicy
except ImportError:
    from compute.sparse_simulation import SparseSimulationEngine
    from compute.winner_selection import WinnerSelector
    from compute.winner_policies import TopKPolicy

from ._hash import (
    WEIGHT_DTYPE, fnv1a_pair_seed, hash_stim_counts,
    hash_bernoulli_coo,
)
from ._csr import (
    CSRConn, DENSE_MIN_P, DENSIFY_MAX_BYTES, DENSIFY_MIN_NNZ,
    TorchDenseConn, densify,
)
from ._state import (
    LAZY_ID_THRESHOLD, TorchAreaState, StimulusState, TorchConn,
)


class TorchSparseEngine(ComputeEngine):
    """PyTorch-native GPU engine with hash-based connectivity.

    Same statistical sparse algorithm as NumpySparseEngine but all arrays
    are torch.cuda tensors.  Key performance advantages:
    - torch.topk: single fused kernel (vs argpartition + argsort)
    - Lower per-op dispatch overhead than CuPy
    - No CuPy<->torch conversion for operations
    - Hash-based deterministic initialization
    - Optional GPU-native truncated normal sampling via torch.erfinv

    Parameters:
        p:             Connection probability.
        seed:          Global random seed.
        w_max:         Hebbian weight ceiling.
        deterministic: If True, use legacy exact-fit expansion.
        gpu_sampling:  If True (default), sample truncated normal on GPU
                       using torch.erfinv instead of CPU scipy.  Falls
                       back to CPU path when deterministic=True.
    """

    supports_input_noise = True
    supports_refraction = True

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False, gpu_sampling: bool = True,
                 **kwargs):
        self.p = p
        self.seed = int(seed)  # Shared construction identity when adopted by Brain.
        self.w_max = w_max
        self._deterministic = deterministic
        self._gpu_sampling = gpu_sampling and not deterministic
        # One-time incoming-weight normalization (reference `norm_init`): a
        # read-time per-postsynaptic 1/d_j scale, ported from NumpySparseEngine.
        # Previously this kwarg was silently swallowed by **kwargs and ignored,
        # so a Brain(norm_init=True, engine="torch_sparse") got NO normalization.
        homeostasis = HomeostasisConfig(**{name: kwargs.get(name, False)
                                           for name in HomeostasisConfig.__dataclass_fields__})
        self.norm_init = homeostasis.norm_init
        # Dense-drive mode (see docs/gpu_scale_design.md, Lever A): score ALL n
        # candidate neurons each round instead of sampling ~k order statistics.
        # Materialized neurons keep their real CSR drive; the (n-w) unmaterialized
        # get an i.i.d. Binomial(sum(input_sizes), p) draw (the same statistical
        # model the sparse sampler approximates), then a single topk over n.
        # More arithmetic than the sparse path -- deliberately, so it is
        # GPU-parallel and, with a fixed [n] drive, batchable (Lever B).
        self.dense_drive = bool(kwargs.get("dense_drive", False))
        # Read-only inference: suppress candidate sampling so a projection never
        # materializes new neurons (select only among already-materialized ones).
        # Inference should not mutate the brain; this also makes prediction
        # deterministic and gives a fixed connectome to batch over (BatchedLM).
        self.readonly = bool(kwargs.get("readonly", False))
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
        if kwargs.get("synaptic_scaling_deferred", False):
            raise NotImplementedError(
                "synaptic_scaling_deferred is not implemented on "
                "torch_sparse: per-update scaling only. Use engine="
                "'numpy_sparse' for the deferred/flush mode.")
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

        self._device = torch.device('cuda')

        # Internal state
        self._areas: Dict[str, TorchAreaState] = {}
        self._stimuli: Dict[str, StimulusState] = {}

        # Connectivity: stim_name -> area_name -> TorchConn (1-D weights)
        self._stim_conns: Dict[str, Dict[str, TorchConn]] = defaultdict(dict)
        # Connectivity: src_area -> tgt_area -> CSRConn (2-D weights)
        self._area_conns: Dict[str, Dict[str, CSRConn]] = defaultdict(dict)
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

    # -- norm_init: read-time incoming-weight normalization -----------------
    # The LAW lives in core/_pricing.py and is shared with the numpy engine.
    # Only in-degree EXTRACTION is backend-specific and stays here, because how
    # you count a column's realized synapses depends on the storage format.
    #
    # This used to be a hand-written "on-device mirror" of the numpy code, and
    # it drifted: two fixes to the numpy copy never arrived here, so the two
    # engines priced k-WTA differently. See core/_pricing.py for the measured
    # divergence table.

    def _norm_candidate_divisor(self, tgt_n: int, input_sizes=None,
                                src_pops=None) -> float:
        """Scale for sampled (unmaterialized) candidate drive.

        See `core._pricing.candidate_divisor`. With ``src_pops`` omitted this
        falls back to ``tgt_n * p``, which is correct only when every source
        population equals the target's ``n``.
        """
        return candidate_divisor(self.p, tgt_n, input_sizes, src_pops)

    def _norm_scale_stim(self, conn, n_pre, stim_size, needed):
        """1/d_j for a 1-D stimulus fiber (numpy _norm_scale, 1-D branch)."""
        w = conn.weights
        if w is None or w.numel() == 0:
            return None
        cols = int(min(needed, int(w.numel())))
        if cols <= 0:
            return None
        # The stored value IS the observed in-degree from the stimulus, but
        # only until plasticity scales it -- snapshot each column the first
        # time it is seen (a fresh column is read before it is potentiated).
        base = getattr(conn, "_norm_deg_base", None)
        have = 0 if base is None else int(base.numel())
        if have < cols:
            add = w[have:cols].detach().float()
            base = add if (base is None or have == 0) else torch.cat([base, add])
            conn._norm_deg_base = base
        deg = base[:cols]
        return inverse_indegree(deg, n_pre, stim_size, self.p)

    def _norm_scale_area(self, csr, n_pre, rows_known, needed):
        """1/d_j for a 2-D area fiber; see `core._pricing.inverse_indegree`."""
        cols = int(min(needed, int(csr._ncols)))
        if cols <= 0:
            return None
        rows = min(int(rows_known), int(csr._nrows))
        deg = csr.column_indegree(cols, nrows=rows)
        return inverse_indegree(deg, n_pre, rows, self.p)

    def _norm_scale_dense(self, weights, n_pre, needed):
        """1/d_j for a DENSE explicit-source fiber.

        Every row of a dense connectome exists, so there is no unknown-row term
        and the in-degree is just the column count of present synapses. This
        path had NO normalization at all until the law was unified: an explicit
        source's incumbents were delivered as raw counts while candidates were
        divided by n*p, leaving the two populations a factor of n*p apart, which
        sealed the target at k. See `_explicit_src_norm_enabled` in the numpy
        engine for the measured numbers.
        """
        cols = int(min(needed, int(weights.shape[1])))
        if cols <= 0:
            return None
        deg = (weights[:, :cols] > 0).sum(dim=0).float()
        return inverse_indegree(deg, n_pre, n_pre, self.p)

    # -- seeded device RNG --------------------------------------------------

    def _device_rng(self, rng):
        """A torch generator whose stream is derived from the seeded ``rng``.

        WHY THIS EXISTS.  Both candidate samplers below took an
        ``rng: np.random.Generator`` argument and then drew from torch's
        PROCESS-GLOBAL stream instead -- ``torch.rand`` and ``torch.normal``
        with no ``generator=``.  ``_sample_truncated_normal_gpu`` declared the
        parameter and never referenced it at all; ``_sample_dense_candidates``
        used it only on the ``_deterministic`` branch.

        The consequence is invisible within one process and fatal across two:
        ``Brain(seed=1)`` reproduces perfectly if you run it twice in the same
        interpreter, because the global stream advances the same way -- and
        diverges between processes, because the global stream's starting point
        is not ours to control.  Measured on one cell (k=100, p=0.05, beta=0.05,
        15 rounds, norm_init, n_src=1000 -> n_tgt=10000): w = 484, 514, 506 on
        three separate invocations at a fixed seed.  Candidate draws land right
        at the k-WTA cut, so a different draw flips borderline winners and the
        difference compounds over rounds.

        This is the fourth appearance of this class in this codebase -- global
        RNG leaking between Brain constructions, ``hash()``-derived seeds
        differing across processes, CUDA's non-Bernoulli init, and now this.
        The pattern each time: **an in-process test cannot catch it**, because
        within one process the global stream is perfectly repeatable.  Any test
        for this must spawn a subprocess and compare.

        Seeding per call off ``rng`` keeps the device stream slaved to the
        numpy stream that the caller already threads through, so one seed still
        determines the whole run.
        """
        gen = self._torch_gen
        if gen is None:
            gen = torch.Generator(device=self._device)
            self._torch_gen = gen
        if rng is not None:
            gen.manual_seed(int(rng.integers(0, 2 ** 63 - 1)))
        return gen

    # -- dense-drive candidate sampling (Lever A) ---------------------------

    def _sample_dense_candidates(self, input_sizes, n_unmat, rng):
        """i.i.d. drive for every unmaterialized neuron (dense-drive mode).

        The drive from ``M = sum(input_sizes)`` active presynaptic units onto an
        unmaterialized neuron is ``Binomial(M, p)``; we approximate it by a
        clamped normal with the matching mean/variance, drawn for all ``n_unmat``
        neurons at once (O(n) GPU work, deliberately). This is the same model the
        sparse sampler draws only the top-k order statistics of -- here we draw
        the full population and let topk over n choose, so selection is exact.
        """
        if n_unmat <= 0:
            return torch.empty(0, dtype=torch.float32, device=self._device)
        M = float(sum(input_sizes))
        mu = M * self.p
        sigma = math.sqrt(max(M * self.p * (1.0 - self.p), 0.0))
        if self._deterministic:
            draw = rng.normal(mu, sigma or 1e-6, size=int(n_unmat))
            cand = torch.from_numpy(
                np.asarray(draw, dtype=np.float32)).to(self._device)
        else:
            # Same defect as `_sample_truncated_normal_gpu` had: without an
            # explicit generator this reads torch's process-global stream while
            # `rng` -- already threaded in, and used on the branch above -- is
            # ignored. See `_device_rng`.
            cand = torch.normal(
                mu, sigma or 1e-6, size=(int(n_unmat),),
                device=self._device, dtype=torch.float32,
                generator=self._device_rng(rng))
        return cand.clamp_(min=0.0)

    def set_dense_area_conn(self, src: str, tgt: str, conn: Connectome) -> None:
        """Install a dense connectome for explicit→sparse cross-engine edges."""
        self._dense_area_conns[src][tgt] = conn

    def _dense_weights(self, conn: Connectome) -> torch.Tensor:
        w = conn.weights
        if isinstance(w, torch.Tensor):
            return w.float()
        return torch.from_numpy(np.asarray(w, dtype=np.float32)).to(self._device)

    # -- Registration -------------------------------------------------------

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0,
                 winner_policy=None,
                 input_noise_std: float = 0.0) -> None:
        input_noise_std = validate_input_noise(input_noise_std)
        n, k = validate_area_registration(name, n, k, existing=self._areas, reserved=self._stimuli)
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
                torch.empty(0, dtype=WEIGHT_DTYPE, device=self._device),
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
                torch.empty(0, dtype=WEIGHT_DTYPE, device=self._device),
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

    # -- GPU truncated normal sampling --------------------------------------

    def _sample_truncated_normal_gpu(
        self,
        input_sizes: list,
        n: int,
        w: int,
        k: int,
        p: float,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Sample new-winner input strengths entirely on GPU.

        Uses torch.erfinv for the inverse-CDF transform, avoiding the
        scipy dependency and CPU-to-GPU transfer.  The binom.ppf
        threshold (alpha) is still computed on CPU via the cached
        scipy call -- it's O(1) per unique parameter set.

        Mathematically equivalent to SparseSimulationEngine
        .sample_new_winner_inputs():
            alpha = binom.ppf((effective_n - k)/effective_n, total_k, p)
            a = (alpha - mu) / std
            u ~ Uniform(Phi(a), 1)
            x = mu + std * Phi_inv(u)

        Where Phi_inv(u) = sqrt(2) * erfinv(2*u - 1).
        """
        from ...compute.sparse_simulation import _binom_ppf_cached

        total_k = sum(input_sizes)
        effective_n = n - w

        # GRACEFUL SATURATION, ported from `sparse_simulation` where it has
        # lived since the numpy engine hit this. `effective_n = n - w` counts
        # the neurons that have never fired -- the only source of brand-new
        # winners. When it drops to k or below the area cannot recruit a full
        # k of fresh ones, so recruit as many as remain and let the caller
        # complete the winner set from already-materialised incumbents (it
        # top-k selects over `prev_winner_inputs` plus these).
        #
        # A biological area at capacity simply stops recruiting; it must not
        # crash mid-training. This engine RAISED instead, which made every
        # saturating workload unrunnable on GPU -- and saturation is not an
        # edge case here, it is where the capacity studies live (an area at
        # rows/n = 1.0 is the normal end state of storing many assemblies).
        #
        # `k_eff == k` whenever `effective_n > k`, so no non-saturated run
        # changes.
        k_eff = min(k, max(0, effective_n - 1))
        if k_eff <= 0:
            return torch.empty(0, dtype=torch.float32, device=self._device)

        alpha = _binom_ppf_cached(effective_n - k_eff, effective_n, total_k, p)

        mu = total_k * p
        std = math.sqrt(total_k * p * (1.0 - p))
        if std == 0:
            return torch.full((k_eff,), mu, dtype=torch.float32,
                              device=self._device)

        a = (alpha - mu) / std

        _SQRT2 = math.sqrt(2.0)
        phi_a = 0.5 * (1.0 + math.erf(a / _SQRT2))

        u = torch.rand(k_eff, dtype=torch.float32, device=self._device,
                       generator=self._device_rng(rng))
        u = phi_a + (1.0 - phi_a) * u
        u.clamp_(phi_a + 1e-12, 1.0 - 1e-12)

        samples = mu + std * _SQRT2 * torch.erfinv(2.0 * u - 1.0)
        samples.round_()
        samples.clamp_(0, total_k)

        return samples

    def _bootstrap_from_explicit_dense(
        self,
        target: str,
        dense_act: torch.Tensor,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        rng=None,
        record_activation: bool = False,
    ) -> ProjectionResult:
        """First assembly in a sparse area driven by explicit-source dense input."""
        tgt = self._areas[target]
        if rng is None:
            rng = np.random.default_rng(self._rng.integers(0, 2**32))

        act = dense_act.float().clone()
        for stim in from_stimuli:
            stim_conn = self._stim_conns[stim][target]
            if not stim_conn.sparse and len(stim_conn.weights) > 0:
                act += stim_conn.weights.float().sum()

        policy = tgt.winner_policy or TopKPolicy(k=tgt.k)
        act_cpu = act.detach().cpu().numpy().astype(np.float64)
        if tgt.input_noise_std > 0:
            act_cpu = act_cpu + rng.normal(
                0.0, tgt.input_noise_std, size=act_cpu.shape,
            )
        neuron_ids = self._winner_sel.select_with_policy(act_cpu, policy)
        neuron_ids = [int(i) for i in neuron_ids]
        compact = list(range(len(neuron_ids)))
        pool = reserve_initial_neuron_ids(tgt.neuron_id_pool, neuron_ids, n=tgt.n)
        tgt.compact_to_neuron_id = list(neuron_ids)
        tgt.neuron_id_pool = pool
        tgt.neuron_id_pool_ptr = len(neuron_ids)

        if plasticity_enabled and self._plasticity_enabled_global:
            for src_name in from_areas:
                dense_conn = self._dense_area_conns.get(src_name, {}).get(target)
                if dense_conn is None:
                    continue
                src = self._areas[src_name]
                if not getattr(src, "explicit_source", False):
                    continue
                beta = tgt.beta_by_source.get(src_name, tgt.beta)
                if beta > 0:
                    valid_rows = src.winners.long()
                    valid_rows = valid_rows[valid_rows < dense_conn.weights.shape[0]]
                    if len(valid_rows) > 0 and len(neuron_ids) > 0:
                        dense_conn.update_weights(
                            valid_rows.cpu().numpy(),
                            neuron_ids,
                            beta,
                        )

        tgt.winners = torch.tensor(
            compact, dtype=torch.int32, device=self._device,
        )
        tgt.w = len(compact)
        total_act = float(act[neuron_ids].sum().item()) if neuron_ids else 0.0

        result = ProjectionResult(
            winners=np.array(compact, dtype=np.uint32),
            num_first_winners=len(compact),
            num_ever_fired=len(compact),
            total_activation=total_act,
        )
        if record_activation:
            result.pre_kwta_inputs = act.detach().cpu().numpy().astype(
                np.float32, copy=True,
            )
            result.pre_kwta_prev_only = np.zeros(0, dtype=np.float32)
            result.pre_kwta_total = float(act.sum().item())
        return result

    # -- Projection (core operation) ----------------------------------------

    def project_into(self, target, from_stimuli, from_areas,
                     plasticity_enabled=True, record_activation=False):
        tgt = self._areas[target]
        self.validate_probe_target(target)
        rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # Filter sourceless areas
        from_areas = [
            a for a in from_areas
            if self._areas[a].winners.numel() > 0
            and (
                self._areas[a].w > 0
                or getattr(self._areas[a], "explicit_source", False)
            )
        ]

        # Fixed assembly -- the winners do not move, but the AFFERENTS learn.
        # This used to be a bare short-circuit (inputs discarded, no
        # plasticity), the exact footgun the numpy engine already fixed: a
        # projection into a fixed area looked like training and wrote nothing.
        # On this engine it presented as an EMPTY arc->state fiber after a
        # full FSM training run -- the fiber is only materialized on demand,
        # and the demand never came -- so every step read state 0. See
        # NumpySparseEngine.project_into's fixed-target branch for the
        # history and `_fixed_target_plasticity_enabled` for the A/B gate.
        if tgt.fixed_assembly:
            learn = (plasticity_enabled and (from_stimuli or from_areas)
                     and _np_fixed_target_plasticity_enabled())
            if learn:
                # Size any never-used source block FIRST -- a multiplicative
                # `w *= 1 + beta` cannot touch entries that do not exist, and
                # `hebbian_update` no-ops on an empty CSR.
                for src_name in from_areas:
                    csr = self._maybe_densify(src_name, target)
                    src = self._areas[src_name]
                    if int(src.w) > 0 and int(tgt.w) > 0:
                        r, c, v = self._hash_grow_parts(
                            csr, self._get_pair_seed(src_name, target),
                            self._p_for(src_name, target),
                            max(int(src.w), csr._log_rows),
                            max(int(tgt.w), csr._log_cols))
                        if r:
                            csr.expand(csr._log_rows, csr._log_cols,
                                       torch.cat(r), torch.cat(c),
                                       torch.cat(v))
                self._apply_plasticity(
                    target, from_stimuli, from_areas, tgt.winners)
            return ProjectionResult(
                winners=tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w)

        # No inputs
        if not from_stimuli and not from_areas:
            return ProjectionResult(
                winners=tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w)

        # --- Accumulate inputs from previous winners ---
        prev_winner_inputs = torch.zeros(
            tgt.w, dtype=torch.float32, device=self._device)
        explicit_dense_act = None
        empty_fibers = []

        limit = tgt.w
        for stim in from_stimuli:
            stim_conn = self._stim_conns[stim][target]
            stim_w = stim_conn.weights
            end = min(limit, len(stim_w))
            if end > 0:
                contrib = stim_w[:end].float()
                if self.norm_init:
                    nscale = self._norm_scale_stim(
                        stim_conn, tgt.n, self._stimuli[stim].size, end)
                    if nscale is not None:
                        contrib = contrib * nscale[:end]
                prev_winner_inputs[:end] += contrib

        for src_name in from_areas:
            src = self._areas[src_name]
            dense_conn = self._dense_area_conns.get(src_name, {}).get(target)
            if dense_conn is not None and getattr(src, "explicit_source", False):
                w = self._dense_weights(dense_conn)
                valid = src.winners.long()
                valid = valid[valid < w.shape[0]]
                if len(valid) == 0:
                    continue
                # norm_init applies to THIS fiber too. The connectome is dense
                # and full-width, so its columns are NEURON IDs rather than
                # compact indices -- scale the whole width and index the result
                # by neuron id. Omitting this left explicit incumbents on the
                # raw-count scale while candidates were divided by n*p, so the
                # target sealed at k and every readout read exactly 1.0000.
                enorm = None
                if self.norm_init:
                    enorm = self._norm_scale_dense(w, src.n, int(w.shape[1]))
                if tgt.w == 0:
                    contrib = w[valid].sum(dim=0)
                    if enorm is not None:
                        contrib = contrib * enorm[:len(contrib)]
                    if explicit_dense_act is None:
                        explicit_dense_act = contrib
                    else:
                        explicit_dense_act += contrib
                    continue
                if tgt.compact_to_neuron_id:
                    id_t = torch.tensor(
                        tgt.compact_to_neuron_id,
                        dtype=torch.long,
                        device=self._device,
                    )
                    valid_cols = id_t[id_t < w.shape[1]]
                    if len(valid_cols) > 0:
                        contrib = w[valid][:, valid_cols].sum(dim=0)
                        if enorm is not None:
                            contrib = contrib * enorm[valid_cols]
                        end = min(limit, len(contrib))
                        if end > 0:
                            prev_winner_inputs[:end] += contrib[:end]
                continue

            csr = self._area_conns[src_name][target]
            if csr.nnz == 0:
                # An unmaterialised fiber delivers zero drive. That is FINE as
                # a transient -- a self-fiber is empty for the one round before
                # recruitment builds it -- but it DEADLOCKS when nothing else
                # drives the target: zero drive, so nothing is recruited, so
                # `_expand_connectomes` never runs, so the fiber stays empty
                # forever. Handled at the zero-signal branch below, which is
                # exactly the condition that separates the two.
                empty_fibers.append((src_name, csr))
                continue
            contrib = csr.accumulate_rows(src.winners.long(), limit)
            end = min(limit, len(contrib))
            if end > 0:
                contrib = contrib[:end]
                if self.norm_init:
                    nscale = self._norm_scale_area(csr, src.n, src.w, end)
                    if nscale is not None:
                        # nscale only covers the CSR's materialized columns
                        # (_ncols may be < end); columns beyond it carry no
                        # synapses, so their drive is 0 and left unscaled.
                        m = min(end, int(nscale.numel()))
                        contrib = contrib.clone()
                        contrib[:m] = contrib[:m] * nscale[:m]
                prev_winner_inputs[:end] += contrib

        if explicit_dense_act is not None and tgt.w == 0:
            return self._bootstrap_from_explicit_dense(
                target,
                explicit_dense_act,
                from_stimuli,
                from_areas,
                plasticity_enabled=plasticity_enabled,
                rng=rng,
                record_activation=record_activation,
            )

        # Zero signal — preserve current assembly
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-noise-only-observation
        zero_signal = prev_winner_inputs.numel() > 0 and not prev_winner_inputs.any()
        if zero_signal and tgt.input_noise_std > 0 and tgt.w < tgt.n:
            raise ValueError('noise-only projection requires a fully materialized population')
        if zero_signal and tgt.input_noise_std == 0:
            # UNLESS the silence is a DEAD FIBER rather than a quiet source.
            # Driving a converged target from a SECOND source left that
            # fiber at nrows=0 ncols=0 nnz=0 for every round while numpy grew
            # the same fiber to (514, 218) / 3837 entries and recruited
            # 109 -> 218; this engine's `w` never moved off 239. It presents
            # as a SEALED area, because a zero-drive projection still returns
            # k winners ([[silent-no-op-dead-fibers]]) and this branch then
            # freezes them.
            #
            # Seeding HERE rather than in the drive loop is what separates the
            # deadlock from the harmless transient: a self-fiber that is empty
            # for one round still has the stimulus driving recruitment, so it
            # never reaches this branch and its construction order is left
            # alone. Same defect the fixed-assembly branch above already fixes.
            grew = False
            for src_name, csr in empty_fibers:
                src = self._areas[src_name]
                if int(src.w) <= 0 or int(tgt.w) <= 0:
                    continue
                if src_name == target:
                    # A SELF-fiber that is silent means the area has nothing
                    # to say to itself yet; seeding it mid-run replaces the
                    # assembly with a fresh random draw, measured at stability
                    # 0.010 == chance (k/n). Preserving the assembly is the
                    # correct answer there. The deadlock is a fiber from
                    # ANOTHER area, which no other input will ever build.
                    continue
                r, c, v = self._hash_grow_parts(
                    csr, self._get_pair_seed(src_name, target),
                    self._p_for(src_name, target),
                    max(int(src.w), csr._log_rows),
                    max(int(tgt.w), csr._log_cols))
                if r:
                    csr.expand(csr._log_rows, csr._log_cols,
                               torch.cat(r), torch.cat(c), torch.cat(v))
                    grew = True
            if grew:
                # Once: the fibers are non-empty now, so this cannot recur.
                return self.project_into(
                    target, from_stimuli, from_areas,
                    plasticity_enabled=plasticity_enabled,
                    record_activation=record_activation)
            return ProjectionResult(
                winners=tgt.winners.cpu().numpy().astype(np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w)

        # --- Sample new winner candidates via truncated normal ---
        input_sizes = (
            [self._stimuli[s].size for s in from_stimuli]
            + [area_fiber_activity(int(self._areas[a].winners.numel()),
                                   self._areas[a].k, self.norm_init)
               for a in from_areas])
        # Presynaptic POPULATION per fiber, parallel to input_sizes. Used only
        # to price candidates on the incumbent scale -- see
        # `core._pricing.candidate_divisor`. Stimulus fibers use the target's
        # own n, matching the convention in `inverse_indegree`.
        src_pops = (
            [tgt.n for _ in from_stimuli]
            + [self._areas[a].n for a in from_areas])

        if self.readonly or (self._no_recruitment and tgt.w >= tgt.k):
            # No new candidates -> topk selects only among materialized neurons,
            # so the projection never grows the area (deterministic inference).
            # The second arm is `brain.read_only()` / `probe()`: same
            # semantics, scoped to the context manager instead of the
            # engine's lifetime. Gated on w >= k exactly like the numpy
            # engine -- below k there is nothing to select from, and a
            # silently short assembly would be worse than growing.
            potential_new = torch.empty(0, dtype=torch.float32,
                                        device=self._device)
        elif self.dense_drive:
            # Score EVERY unmaterialized neuron, not just k order statistics, so
            # the subsequent topk over n is exact (Lever A). See
            # _sample_dense_candidates and docs/gpu_scale_design.md.
            if self.heterogeneous():
                raise NotImplementedError(
                    "dense_drive prices candidates at the global p; this "
                    "brain has per-fiber densities (add_connectivity). "
                    "Refusing rather than sampling with the wrong statistics.")
            potential_new = self._sample_dense_candidates(
                input_sizes, tgt.n - tgt.w, rng)
        elif self._gpu_sampling and not self.heterogeneous():
            potential_new = self._sample_truncated_normal_gpu(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p, rng)
        else:
            # Heterogeneous brains take the SHARED CPU sampler: it already
            # carries the per-fiber law (Poisson-binomial moment-matched by
            # `_pricing.effective_binomial` -- see NumpySparseEngine.
            # add_connectivity), and candidates are O(k), so there is no law
            # duplicated on the GPU and nothing material lost off it.
            input_ps = ([self._p_for(s, target) for s in from_stimuli]
                        + [self._p_for(a, target) for a in from_areas]
                        ) if self.heterogeneous() else self.p
            old_rng = self._sparse_sim.rng
            self._sparse_sim.rng = rng
            if self._deterministic:
                potential_new_np = self._sparse_sim.sample_new_winner_inputs_legacy(
                    input_sizes, tgt.n, tgt.w, tgt.k, input_ps)
            else:
                potential_new_np = self._sparse_sim.sample_new_winner_inputs(
                    input_sizes, tgt.n, tgt.w, tgt.k, input_ps)
            self._sparse_sim.rng = old_rng
            if hasattr(potential_new_np, 'get'):
                potential_new_np = potential_new_np.get()
            potential_new_np = np.asarray(potential_new_np, dtype=np.float32)
            potential_new = torch.from_numpy(potential_new_np).to(self._device)

        # norm_init: candidates are sampled on the unit-weight scale; bring them
        # onto the normalized scale by dividing by the mean in-degree (n*p), so
        # they compete with the 1/d_j-scaled materialized drive above. Stored
        # weights and the sampler stay unit-scale (see _norm_candidate_divisor).
        if self.norm_init:
            potential_new = potential_new / self._norm_candidate_divisor(
                tgt.n, input_sizes, src_pops)

        if prev_winner_inputs.numel() > 0:
            all_inputs = torch.cat([prev_winner_inputs, potential_new])
        else:
            all_inputs = potential_new

        # --- Snapshot raw prev_winner_inputs before penalties ---
        if record_activation:
            _raw_prev_t = prev_winner_inputs.clone()

        # --- LRI: penalise recently-fired neurons ---
        if (tgt.refractory_period > 0
                and tgt.inhibition_strength > 0
                and len(tgt._refractory_history) > 0):
            pen_indices = []
            pen_values = []
            n_inputs = len(all_inputs)
            for steps_ago_idx, winner_set in enumerate(
                    reversed(list(tgt._refractory_history))):
                steps_ago = steps_ago_idx + 1
                decay = 1.0 - (steps_ago - 1) / tgt.refractory_period
                penalty = tgt.inhibition_strength * decay
                for cidx in winner_set:
                    if cidx < n_inputs:
                        pen_indices.append(cidx)
                        pen_values.append(penalty)
            if pen_indices:
                idx_t = torch.tensor(pen_indices, dtype=torch.long,
                                     device=self._device)
                val_t = torch.tensor(pen_values, dtype=torch.float32,
                                     device=self._device)
                all_inputs.scatter_add_(
                    0, idx_t, -val_t)

        # --- Refracted mode: cumulative bias penalty ---
        if tgt.refracted and tgt._cumulative_bias is not None:
            bias = tgt._cumulative_bias
            end = min(len(bias), len(all_inputs))
            if end > 0:
                all_inputs[:end] -= bias[:end]

        # --- Snapshot full all_inputs before top-k ---
        if record_activation:
            _pre_kwta_snapshot = all_inputs.detach().cpu().numpy().astype(
                np.float32).copy()
            _pre_kwta_total_val = float(all_inputs.sum().item())

        # --- Select winners (policy-aware or default top-k) ---
        policy = tgt.winner_policy or TopKPolicy(k=tgt.k)
        # On-device fast path for the default top-k policy: run torch.topk on
        # the GPU-resident drive vector so it never crosses to host, then bring
        # back only the k selected indices for compact-id bookkeeping. The CPU
        # path below copies the whole W-sized drive vector and runs numpy
        # argpartition every round -- measured 55-159x slower at large W. Custom
        # policies (threshold / e-percent / slotted) and additive input noise
        # keep the CPU path, which owns those semantics.
        if isinstance(policy, TopKPolicy) and tgt.input_noise_std == 0.0:
            k_sel = min(int(policy.k), int(all_inputs.numel()))
            _, sel = torch.topk(all_inputs, k_sel, sorted=True)
            winners_gpu = sel.to(torch.int32)
        else:
            inputs_cpu = all_inputs.detach().cpu().numpy().astype(np.float64)
            if tgt.input_noise_std > 0:
                inputs_cpu = inputs_cpu + rng.normal(
                    0.0, tgt.input_noise_std, size=inputs_cpu.shape,
                )
            winner_indices = self._winner_sel.select_with_policy(
                inputs_cpu, policy)
            winners_gpu = torch.tensor(
                [int(i) for i in winner_indices],
                dtype=torch.int32,
                device=self._device,
            )
        k = int(winners_gpu.numel())

        # --- Process first-time winners ---
        first_mask = winners_gpu.long() >= tgt.w
        first_input_vals = all_inputs[winners_gpu[first_mask].long()]
        if self.norm_init and first_input_vals.numel() > 0:
            # New winners are sampled candidates, so their drive was divided by
            # the candidate divisor above. Connectome expansion splits an INTEGER
            # synapse count across fibers, so un-normalize back to unit scale
            # first (mirror of NumpySparseEngine before _expand_connectomes).
            # Must be the SAME divisor that was applied, which now depends on
            # the fibers, not on tgt.n alone -- otherwise the round trip is
            # asymmetric and the recovered synapse count is wrong.
            first_input_vals = (
                first_input_vals * self._norm_candidate_divisor(
                    tgt.n, input_sizes, src_pops))

        winners_cpu = winners_gpu.cpu().tolist()
        first_inputs_cpu = (first_input_vals.cpu().tolist()
                            if first_input_vals.numel() > 0 else [])

        num_first = 0
        first_winner_inputs_cpu = []
        new_winner_indices = list(winners_cpu)
        first_idx = 0

        for i in range(k):
            if new_winner_indices[i] >= tgt.w:
                first_winner_inputs_cpu.append(
                    int(first_inputs_cpu[first_idx]))
                first_idx += 1
                actual_id = tgt.next_neuron_id()
                tgt.compact_to_neuron_id.append(actual_id)
                new_winner_indices[i] = tgt.w + num_first
                num_first += 1

        new_w = tgt.w + num_first
        remapped_gpu = torch.tensor(
            new_winner_indices, dtype=torch.int32, device=self._device)

        # --- Apply plasticity ---
        if plasticity_enabled and self._plasticity_enabled_global:
            self._apply_plasticity(
                target, from_stimuli, from_areas, remapped_gpu)

        # --- Expand connectomes for new winners ---
        if num_first > 0:
            self._expand_connectomes(
                target, from_stimuli, from_areas,
                input_sizes, new_winner_indices,
                first_winner_inputs_cpu, new_w)

        # --- Commit state ---
        tgt.winners = remapped_gpu
        tgt.w = new_w

        # --- Update LRI refractory history ---
        if tgt.refractory_period > 0:
            tgt._refractory_history.append(
                set(int(i) for i in new_winner_indices))

        # --- Update refracted cumulative bias ---
        # Rule and gating live in `core._homeostasis`; see that module for why
        # the increment is proportional to raw drive and why charging is tied
        # to the same condition as the Hebbian update.
        if (tgt.refracted and tgt.refracted_strength > 0
                and plasticity_enabled and self._plasticity_enabled_global):
            if len(tgt._cumulative_bias) < new_w:
                old = tgt._cumulative_bias
                tgt._cumulative_bias = torch.zeros(
                    new_w, dtype=torch.float32, device=self._device)
                if len(old) > 0:
                    tgt._cumulative_bias[:len(old)] = old
            bias = tgt._cumulative_bias
            widx = torch.as_tensor(
                np.asarray(new_winner_indices, dtype=np.int64),
                device=bias.device)
            widx = widx[widx < len(bias)]
            if widx.numel() > 0:
                bias[widx] += refraction_increment(
                    all_inputs[widx], bias[widx], tgt.refracted_strength)

        total_act = float(all_inputs[new_winner_indices].sum().item())

        result = ProjectionResult(
            winners=np.array(new_winner_indices, dtype=np.uint32),
            num_first_winners=num_first,
            num_ever_fired=new_w,
            total_activation=total_act)
        if record_activation:
            result.pre_kwta_inputs = _pre_kwta_snapshot
            result.pre_kwta_prev_only = _raw_prev_t.cpu().numpy().astype(
                np.float32)
            result.pre_kwta_total = _pre_kwta_total_val
        return result

    # -- Plasticity ---------------------------------------------------------

    def _apply_plasticity(self, target, from_stimuli, from_areas,
                          winners_gpu):
        """Hebbian learning: w *= (1 + beta), clamped at w_max."""
        tgt = self._areas[target]
        winners_long = winners_gpu.long()

        for stim_name in from_stimuli:
            conn = self._stim_conns[stim_name][target]
            beta = tgt.beta_by_source.get(stim_name, tgt.beta)
            if beta == 0:
                continue
            valid = winners_long[winners_long < len(conn.weights)]
            if len(valid) > 0:
                conn.weights[valid] *= (1 + beta)
            if self.w_max is not None:
                conn.weights.clamp_(0, self.w_max)

        for src_name in from_areas:
            dense_conn = self._dense_area_conns.get(src_name, {}).get(target)
            if dense_conn is not None and getattr(
                self._areas[src_name], "explicit_source", False,
            ):
                beta = tgt.beta_by_source.get(src_name, tgt.beta)
                if beta > 0:
                    src = self._areas[src_name]
                    valid_rows = src.winners.long()
                    valid_rows = valid_rows[
                        valid_rows < dense_conn.weights.shape[0]
                    ]
                    neuron_ids = [
                        tgt.compact_to_neuron_id[int(w)]
                        for w in winners_gpu.cpu().tolist()
                        if int(w) < tgt.w
                    ]
                    if len(valid_rows) > 0 and len(neuron_ids) > 0:
                        dense_conn.update_weights(
                            valid_rows.cpu().numpy(),
                            neuron_ids,
                            beta,
                        )
                continue

            csr = self._area_conns[src_name][target]
            beta = tgt.beta_by_source.get(src_name, tgt.beta)
            if beta == 0:
                continue
            src = self._areas[src_name]
            csr.hebbian_update(
                src.winners.long(), winners_long, beta, self.w_max)

        self._normalize_area_columns(target, from_areas, winners_long)

    def _normalize_area_columns(self, target, from_areas, winners):
        """Homeostatic synaptic scaling on area->area fibers (torch port).

        The LAW and its measured failure modes live on
        `NumpySparseEngine._normalize_area_columns`; this mirrors its
        per-update form on CSR storage (`CSRConn.scale_columns`). The three
        semantics that MUST match, each a past defect on the numpy side:

        * setpoint priced at the FIBER's own density
          ([[pricing-law-implemented-twice]] -- the global-p setpoint
          renormalized a p=0.40 organ fiber to 1/8 of its natural mass and
          inverted learning);
        * mass summed over the source's LOGICAL rows only (rows past
          ``src.w`` are unallocated bookkeeping);
        * stimulus fibers excluded (1-D pre-summed: normalizing them drives
          every neuron to the same value and erases the representation).

        Explicit-dense bridges (`_dense_area_conns`) are NOT scaled -- the
        numpy engine scales any 2-D block, so a brain using explicit sources
        under scaling differs across engines; none of the scaling organs use
        them, and this note is the tripwire if one ever does.
        """
        if not scaling_applies(self.synaptic_scaling, target):
            return
        check_area_homeostasis(target, refracted=self._areas[target].refracted,
                               synaptic_scaling=True)
        for src_name in from_areas:
            csr = self._area_conns.get(src_name, {}).get(target)
            if csr is None or csr.nnz == 0:
                continue
            rows = min(int(self._areas[src_name].w), int(csr._nrows))
            if rows <= 0:
                continue
            setpoint = scaling_setpoint(rows, self._p_for(src_name, target))
            csr.scale_columns(winners, setpoint, nrows=rows)

    # -- Connectome expansion -----------------------------------------------

    def _maybe_densify(self, src_name, target):
        """Swap a rebuild-bound CSR fiber for dense storage; return the conn.

        Call at every GROWTH site before touching the fiber. Density picks
        the representation at `add_connectivity` time, but a low-density
        fiber that grows every recruitment step (an area self-fiber during
        training) is just as rebuild-bound once it is large -- see
        DENSIFY_MIN_NNZ. Budgeted against the fiber's FINAL dense footprint
        (n_src x n_tgt), not its current extent, so a fiber that will not
        fit never starts migrating.
        """
        conn = self._area_conns[src_name][target]
        if (isinstance(conn, CSRConn)
                and conn.nnz >= DENSIFY_MIN_NNZ):
            final_bytes = (int(self._areas[src_name].n)
                           * int(self._areas[target].n) * 2)
            if final_bytes <= DENSIFY_MAX_BYTES:
                conn = densify(conn, device=self._device,
                               max_rows=int(self._areas[src_name].n),
                               max_cols=int(self._areas[target].n))
                self._area_conns[src_name][target] = conn
        return conn

    def _hash_grow_parts(self, csr, pair_seed, fiber_p,
                         needed_rows, needed_cols):
        """L-shaped hash init of a CSR fiber's uncovered region, as COO parts.

        The one canonical implementation of the grow-to-extent step (it was
        inline in `_expand_connectomes`; `materialize_area` needs the same
        mechanics, and two copies of an init path is how the numpy engine
        got its self-fiber masking defects). Content-addressed: entries
        depend only on (pair_seed, absolute position, fiber_p), so growing
        in any order yields the same fiber. Updates the logical extents;
        the caller merges the returned parts via ``csr.expand``.
        """
        log_rows = csr._log_rows
        log_cols = csr._log_cols
        coo_r, coo_c, coo_v = [], [], []
        if needed_rows > log_rows or needed_cols > log_cols:
            regions = []
            # Block A: new rows x existing cols
            if needed_rows > log_rows and log_cols > 0:
                regions.append((log_rows, needed_rows, 0, log_cols))
            # Block B: existing rows x new cols
            if needed_cols > log_cols and log_rows > 0:
                regions.append((0, log_rows, log_cols, needed_cols))
            # Block C: new rows x new cols
            if needed_rows > log_rows and needed_cols > log_cols:
                regions.append((log_rows, needed_rows, log_cols, needed_cols))
            for r0, r1, c0, c1 in regions:
                r, c, v = hash_bernoulli_coo(
                    r0, r1, c0, c1, pair_seed, fiber_p, device=self._device)
                if len(r) > 0:
                    coo_r.append(r)
                    coo_c.append(c)
                    coo_v.append(v)
            csr._log_rows = max(log_rows, needed_rows)
            csr._log_cols = max(log_cols, needed_cols)
        return coo_r, coo_c, coo_v

    def materialize_area(self, area: str, storage: str = "csr") -> int:
        """Bring ALL ``n`` of an area's neurons into existence at once.

        The torch port of `NumpySparseEngine.materialize_area` (see its
        docstring for WHY this exists: any protocol that drives an area from
        an arbitrary subset of ``n`` -- assigned blocks, uniform seeds --
        silently reads zeros from neurons lazy materialization has not
        created yet). Init is content-addressed by absolute position, so a
        materialized-all-at-once area has exactly the weights it would have
        had if the same neurons had been recruited one at a time.

        ``storage`` is accepted for interface parity and ignored: CSR is
        this engine's native representation (the numpy engine offers
        dense/CSR because its callers index dense blocks in ways CSRWeights
        refuses; nothing indexes a torch fiber that way).

        Returns the number of neurons newly materialized.
        """
        tgt = self._areas[area]
        prior_w = int(tgt.w)
        n = int(tgt.n)
        if prior_w >= n:
            return 0
        if tgt._lazy_ids:
            raise NotImplementedError(
                f"materialize_area({area!r}): lazy neuron-id mode "
                f"(n > {LAZY_ID_THRESHOLD}) has no full-permutation pool to "
                f"draw identities from.")

        # 1. Compact ids for every remaining neuron, drawn from the same
        #    shuffled pool the incremental path consumes, so identities match.
        if tgt.neuron_id_pool is not None:
            pool = np.asarray(tgt.neuron_id_pool)
            need = n - len(tgt.compact_to_neuron_id)
            ptr = int(tgt.neuron_id_pool_ptr)
            take = pool[ptr:ptr + need]
            tgt.compact_to_neuron_id.extend(int(x) for x in take)
            tgt.neuron_id_pool_ptr = ptr + len(take)
        while len(tgt.compact_to_neuron_id) < n:
            tgt.compact_to_neuron_id.append(len(tgt.compact_to_neuron_id))

        # 2. Stim -> area vectors out to n.
        for stim_name, conns in self._stim_conns.items():
            conn = conns.get(area)
            if conn is None or not conn.sparse or conn.weights is None:
                continue
            old = int(conn.weights.numel())
            if old < n:
                add = hash_stim_counts(
                    self._stimuli[stim_name].size, old, n,
                    self._get_pair_seed(stim_name, area),
                    self._p_for(stim_name, area), device=self._device)
                conn.weights = torch.cat([conn.weights, add])

        # 3. Area fibers: hash-grow every block touching this area to full
        #    extent. IN-fibers gain columns; OUT-fibers gain rows; the self
        #    fiber gains both (covered by the first loop, then skipped).
        def _grow(csr, src_name, tgt_name, needed_rows, needed_cols):
            r, c, v = self._hash_grow_parts(
                csr, self._get_pair_seed(src_name, tgt_name),
                self._p_for(src_name, tgt_name), needed_rows, needed_cols)
            if r:
                csr.expand(csr._log_rows, csr._log_cols,
                           torch.cat(r), torch.cat(c), torch.cat(v))

        for src_name, conns in self._area_conns.items():
            csr = conns.get(area)
            if csr is None:
                continue
            src_rows = (n if src_name == area
                        else max(int(self._areas[src_name].w),
                                 csr._log_rows))
            _grow(csr, src_name, area, src_rows, n)
        for tgt_name, csr in self._area_conns.get(area, {}).items():
            if tgt_name == area:
                continue  # self fiber handled above
            tgt_cols = max(int(self._areas[tgt_name].w), csr._log_cols)
            _grow(csr, area, tgt_name, n, tgt_cols)

        tgt.w = n
        return n - prior_w

    def _expand_connectomes(self, target, from_stimuli, from_areas,
                            input_sizes, winners, first_winner_inputs,
                            new_w):
        """Expand connectivity for first-time winners using hash-based init."""
        tgt = self._areas[target]
        inputs_names = list(from_stimuli) + list(from_areas)

        splits_per_new = self._sparse_sim.compute_input_splits(
            input_sizes, first_winner_inputs)

        prior_w = tgt.w
        new_indices = [int(w) for w in winners if int(w) >= prior_w]

        stim_names = [n for n in inputs_names if n in self._stimuli]
        area_names = [n for n in inputs_names if n in self._areas]

        # --- Expand stim->area 1-D vectors ---
        for stim_name in self._stimuli.keys():
            conn = self._stim_conns[stim_name][target]
            if conn.sparse:
                old = len(conn.weights)
                if new_w > old:
                    add_len = new_w - old
                    if stim_name not in stim_names:
                        pair_seed = self._get_pair_seed(stim_name, target)
                        add = hash_stim_counts(
                            self._stimuli[stim_name].size, old, new_w,
                            pair_seed, self._p_for(stim_name, target),
                            device=self._device)
                    else:
                        add = torch.zeros(add_len, dtype=WEIGHT_DTYPE,
                                          device=self._device)
                    conn.weights = torch.cat([conn.weights, add])

        # Write allocations for firing stimuli
        for idx, win in enumerate(new_indices):
            if win >= new_w:
                continue
            split = (splits_per_new[idx]
                     if idx < len(splits_per_new) else None)
            if split is None:
                continue
            for j, name in enumerate(inputs_names):
                alloc = int(split[j])
                if name in self._stimuli:
                    conn = self._stim_conns[name][target]
                    if conn.sparse and win < len(conn.weights):
                        conn.weights[win] = alloc

        # --- Expand area->area CSR matrices ---
        for src_name in area_names:
            csr = self._maybe_densify(src_name, target)
            src = self._areas[src_name]
            pair_seed = self._get_pair_seed(src_name, target)

            src_w_arr = src.winners.long()
            max_src_idx = (
                (int(src_w_arr.max()) + 1)
                if src_w_arr.numel() > 0 else 0)
            # Never shrink below pregrown CSR extent (CONTEXT w can reset to 0
            # while connectome topology is preserved — matches numpy dense path).
            needed_rows = max(
                max_src_idx,
                new_w if src_name == target else src.w,
                csr._nrows,
            )
            needed_cols = max(new_w, csr._ncols)

            coo_r_parts, coo_c_parts, coo_v_parts = self._hash_grow_parts(
                csr, pair_seed, self._p_for(src_name, target),
                needed_rows, needed_cols)

            # Explicit entries from first-timer allocations
            from_index = inputs_names.index(src_name)
            local_rng = np.random.default_rng(
                self._rng.integers(0, 2**32))
            src_winners_cpu = src.winners.cpu().numpy().astype(np.int64)

            # VECTORISED PER NEW WINNER. The draws stay one-per-winner in the
            # same order (they consume `local_rng`, so any reordering would
            # change the connectome), but the chosen rows are kept as ARRAYS
            # instead of appended element by element: the inner Python loop
            # ran once per synapse -- 1.78M list appends in a 4-presentation
            # build -- and then paid again handing a multi-million-element
            # Python list to `torch.tensor`. Same values, same order, one
            # host->device copy.
            exp_rows_parts, exp_cols_parts = [], []
            for idx, win in enumerate(new_indices):
                alloc = (
                    int(splits_per_new[idx][from_index])
                    if idx < len(splits_per_new) else 0)
                if alloc <= 0 or src.w == 0:
                    continue
                sample_size = min(alloc, len(src.winners))
                if sample_size <= 0:
                    continue
                chosen = local_rng.choice(
                    src_winners_cpu, size=sample_size, replace=False)
                col_idx = win - prior_w
                if col_idx < 0 or col_idx >= needed_cols:
                    continue
                chosen = chosen[(chosen >= 0) & (chosen < needed_rows)]
                if chosen.size == 0:
                    continue
                exp_rows_parts.append(chosen)
                exp_cols_parts.append(
                    np.full(chosen.size, col_idx, dtype=np.int64))

            if exp_rows_parts:
                exp_rows = np.concatenate(exp_rows_parts).astype(
                    np.int32, copy=False)
                exp_cols = np.concatenate(exp_cols_parts).astype(
                    np.int32, copy=False)
                coo_r_parts.append(
                    torch.from_numpy(exp_rows).to(self._device))
                coo_c_parts.append(
                    torch.from_numpy(exp_cols).to(self._device))
                coo_v_parts.append(torch.ones(
                    exp_rows.size, dtype=WEIGHT_DTYPE,
                    device=self._device))

            # Merge into CSR
            if coo_r_parts:
                new_r = torch.cat(coo_r_parts)
                new_c = torch.cat(coo_c_parts)
                new_v = torch.cat(coo_v_parts)
                csr.expand(needed_rows, needed_cols, new_r, new_c, new_v)
            elif needed_rows > csr._nrows or needed_cols > csr._ncols:
                e = torch.empty(0, dtype=torch.int32, device=self._device)
                csr.expand(needed_rows, needed_cols,
                           e, e.clone(),
                           torch.empty(0, dtype=WEIGHT_DTYPE,
                                       device=self._device))

    # -- State accessors ----------------------------------------------------

    def get_winners(self, area: str) -> np.ndarray:
        st = self._areas[area]
        return st.winners.cpu().numpy().astype(np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        st = self._areas[area]
        # torch.tensor(uint32_array, dtype=int32, device=cuda) hits a slow
        # element-wise path -- uint32 is not a native torch dtype, so at large k
        # this dominated the whole projection (measured 32ms/round at k=100k).
        # Route through int64 (torch-native) so from_numpy is zero-copy, then a
        # single fused H2D + cast kernel.
        arr = np.ascontiguousarray(winners, dtype=np.int64)
        st.winners = torch.from_numpy(arr).to(
            self._device, dtype=torch.int32, non_blocking=True)

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
        self._areas[area].winner_policy = policy

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = beta

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
        st = self._areas[area]
        check_area_homeostasis(area, refracted=enabled, synaptic_scaling=self.synaptic_scaling)
        st.refracted = enabled
        st.refracted_strength = strength
        if enabled and st._cumulative_bias.numel() == 0:
            st._cumulative_bias = torch.zeros(
                max(st.w, 0), dtype=torch.float32, device=self._device)

    def clear_refracted_bias(self, area: str) -> None:
        st = self._areas[area]
        st._cumulative_bias = torch.zeros(
            max(st.w, 0), dtype=torch.float32, device=self._device)

    # -- Weight normalization -----------------------------------------------

    def normalize_weights(self, target: str, source: str = None) -> None:
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
