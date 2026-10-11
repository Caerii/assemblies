"""NumpySparseEngine: CPU engine using statistical sparse simulation.

This is the extraction of brain.py's ``_project_into_legacy`` sparse path.
Connectivity is stored as growing 1-D (stim->area) and 2-D (area->area)
numpy arrays.  Statistical sampling (truncated normal, binomial PPF)
generates candidate activations for new neurons.
"""

import os

import numpy as np
from typing import Dict, Tuple
from collections import OrderedDict, defaultdict

# `scipy.sparse` is imported ON FIRST USE via `scipy_sparse()`, not here --
# importing it at module scope charged every process that merely constructs a
# Brain ~0.7s and 429 modules for a code path most runs never reach. See
# `_csr_weights.scipy_sparse` for the measurement.


from ..backend import to_cpu, xp_by_name, xp_name
from .._homeostasis import (HomeostasisConfig, validate_lri_parameters)
from ..engine import (
    ComputeEngine,
    validate_deterministic_allocation,
)
from ..registration import (validate_input_noise, validate_stimulus_registration,
                            validate_area_registration, validate_plasticity_rate)
from ..connectome import Connectome
from ..projection_fidelity import ProjectionFidelity
from ..semantics import (
    ArithmeticMode,
    CandidateDomain,
    ConnectomeMode,
    ModelSemantics,
    NormalizationMode,
    PlasticityRule,
    SampledRecurrencePolicy,
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

from ._growth import GrowthMixin, _self_fiber_deferred_init  # noqa: F401

from ._sparse_switches import (  # noqa: F401  re-exported: the torch engine and _exact read them here
    _PRUNE_MAX_FRACTION, _explicit_src_norm_enabled, _fixed_target_plasticity_enabled,
    _strict_drive_enabled, _warn_fixed_target_enabled,
)
from ._degree_norm import DegreeNormMixin
from ._drive_cache import (  # noqa: F401
    DriveCacheMixin, _csr_storage_available, _CSR_MIN_CELLS,
    _CSR_MAX_DENSITY,
)
from ._state import SparseAreaState, StimulusState
from ._csr_weights import CSRWeights, build_csr_from_blocks, scipy_sparse
from ._virtual_weights import VirtualWeights
from ._seeding import (
    fnv1a_pair_seed,
    hash_area_weights,
    rust_kernels,
    stable_seed,
)


def _env_content_init() -> bool:
    """Whether area->area weights are addressed by (row, col) or by draw order.

    On by default. Set ``ASSEMBLIES_STREAM_INIT=1`` to restore the old
    stream-addressed behaviour -- kept only so the two can be A/B'd on the same
    seed, since this switch moves every seeded weight (not their distribution).
    """
    return os.environ.get("ASSEMBLIES_STREAM_INIT", "").strip().lower() not in (
        "1", "true", "yes", "on",
    )


# -- stim->area vector growth policy ----------------------------------------
#
# Every first-time winner in an area appends one slot to the stim->area weight
# vector of EVERY registered stimulus, not just the firing ones (non-firing
# stimuli get a fresh Binomial(stim_size, p) background weight so they can
# still compete later).  The naive implementation reallocated each vector with
# ``concatenate`` on every growth, which is O(w) per stimulus per step and
# therefore O(w^2) overall.
#
# Measured on one TWO_WORD training at n=3000, k=30, seed=42 (line profile of
# ``_expand_connectomes``): 806 calls, 364,216 vector extensions, of which the
# ``concatenate`` was 27.0% and the per-stimulus ``rng.binomial`` call was
# 30.2% of the function's time -- together with the surrounding dict lookups
# the stim block was ~90% of ``_expand_connectomes``.  The area->area 2-D
# block, by contrast, executed only 4 times in the whole run.
#
# The fast path keeps the sampled VALUES and the RNG consumption order
# bit-identical (see ``_expand_stim_vectors_fast``) and only changes how memory
# is allocated and how many calls the draws are batched into, so it is exactly
# equivalent to the legacy path -- verified by A/B running both to convergence
# and comparing winners, ever-fired counts, every stimulus and area connectome
# array and all pairwise assembly overlaps with ``np.array_equal``.
# Measured: ``_expand_connectomes`` 2.61s -> ~1.2s, whole TWO_WORD training
# ~1.3x faster.  Set ``ASSEMBLIES_STIM_FASTPATH=0`` to fall back.
_STIM_FASTPATH = True


# ``stable_seed`` moved to ._seeding and is re-exported above: it belongs with
# the rest of the seeding policy, and importers outside this module rely on the
# name being here.

# Materialized fraction w/n at which an area stops being treated as sparse and
# its stim vectors are allocated to the full n in one shot, so no first-time
# winner ever reallocates again.  Below it, capacity doubles as before.
#
# Default 0.25.  Justification from the same run: the core lexical areas end at
# w = 1808 (NOUN_CORE) / 1739 (DET_CORE) / 1701 (VERB_CORE) out of n = 3000 --
# 57-60% materialized, reached in ~270 increments of median 4 neurons -- so
# they cross 0.25 early and then never reallocate again.  ROLE_AGENT ends at
# w = 38 (1.3%) and never crosses it, so it keeps the doubling path and a
# 64-element vector rather than a 3000-element one.
#
# Be clear about what this buys: measured time inside ``_expand_connectomes``
# is 1.0-1.4s at EVERY setting of the threshold (None, 0.10, 0.20, 0.25, 0.30,
# 0.50), with the ordering flipping between repeats -- the doubling below it
# already amortizes reallocation, so the threshold is worth nothing extra on
# wall-clock at this scale.  What it does buy is a bound: a high-occupancy area
# stops reallocating entirely instead of paying log2(n) more copies as w
# climbs, which matters as n grows.  0.25 sits in the middle of the flat
# region.  Set to ``None`` to always double.
_DENSE_STIM_THRESHOLD = 0.25


def set_stim_fastpath(enabled: bool) -> None:
    """Enable/disable the stim-vector fast path for engines created after this.

    Exists so an A/B equivalence harness can run the legacy and fast paths in
    one process and compare state bit-for-bit.
    """
    global _STIM_FASTPATH
    _STIM_FASTPATH = bool(enabled)


def set_dense_stim_threshold(threshold) -> None:
    """Set the w/n fraction at which stim vectors go dense (``None`` = never)."""
    global _DENSE_STIM_THRESHOLD
    _DENSE_STIM_THRESHOLD = None if threshold is None else float(threshold)


def _env_stim_fastpath() -> bool:
    raw = os.environ.get("ASSEMBLIES_STIM_FASTPATH", "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return False
    if raw in ("1", "on", "true", "yes"):
        return True
    return _STIM_FASTPATH


def _env_dense_stim_threshold():
    raw = os.environ.get("ASSEMBLIES_DENSE_STIM_THRESHOLD", "").strip().lower()
    if raw in ("", "default"):
        return _DENSE_STIM_THRESHOLD
    if raw in ("none", "off"):
        return None
    try:
        return float(raw)
    except ValueError:
        return _DENSE_STIM_THRESHOLD


from ._sparse_projection import NumpyProjection
from ._sparse_candidates import NumpyCandidates
from ._sparse_plasticity import NumpyPlasticity
from ._sparse_prune import NumpyKwtaPrune
from ._sparse_controls import NumpyAreaControls


class NumpySparseEngine(NumpyProjection, NumpyCandidates, NumpyPlasticity, NumpyKwtaPrune, NumpyAreaControls, GrowthMixin, DegreeNormMixin, DriveCacheMixin,
                        ComputeEngine):
    """CPU engine using statistical sparse simulation.

    Connectivity is stored as growing 1-D (stim->area) and 2-D (area->area)
    numpy arrays.  Statistical sampling (truncated normal, binomial PPF)
    generates candidate activations for new neurons.

    Parameters mirror ``Brain.__init__``.
    """

    supports_norm_init = True
    supports_synaptic_scaling = True
    supports_synaptic_scaling_deferred = True
    supports_input_noise = True
    supports_refraction = True
    supports_fiber_learning_masks = True
    supports_sampled_recurrence_policy = True
    supports_compiled_projection = True
    supports_deterministic_allocation = True
    supports_stim_preallocation = True

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False,
                 projection_fidelity: str = ProjectionFidelity.EXACT.value,
                 synaptic_scaling: "bool | frozenset | set | tuple" = False,
                 synaptic_scaling_deferred: bool = False,
                 norm_init: bool = False,
                 sampled_recurrence_policy: str = "warn"):
        deterministic = validate_deterministic_allocation(type(self), deterministic)
        self._sampled_recurrence_policy = SampledRecurrencePolicy.normalize(
            sampled_recurrence_policy
        )
        self.p = p
        #: (source, target) -> density, for fibers that override `p`.
        #: EMPTY BY DEFAULT, and every path below is byte-for-byte the
        #: pre-existing one while it stays empty -- that is the property
        #: that makes heterogeneous density safe to put on the hot path.
        self._fiber_p: Dict[Tuple[str, str], float] = {}
        self.w_max = w_max
        # Signed-weight kernels remain dormant until candidate sampling and
        # newly materialized edge reconstruction implement the same law.
        self.inhibitory_prob = 0.0
        self.inhibitory_weight = -0.2
        # Homeostatic synaptic scaling on area->area fibers. See
        # _normalize_area_columns for why the setpoint is the initial expected
        # column sum rather than 1, and why stimulus fibers are excluded.
        #
        # True scales EVERY target area (the legacy per-fiber form, with its
        # documented attractor-cancellation flaw); a collection of area names
        # scales ONLY those targets. The scoped form exists for
        # stimulus-anchored feature areas (TENSE/NUMBER), whose recall needs
        # the afferent fiber to be DISCRIMINATIVE, not self-sustaining -- the
        # per-fiber objection does not apply where no attractor is required
        # (task #130).
        homeostasis = HomeostasisConfig(norm_init, synaptic_scaling, synaptic_scaling_deferred)
        self.synaptic_scaling = homeostasis.synaptic_scaling
        # E9 (#138): defer scaling to flush_synaptic_scaling() at phase
        # boundaries -- fast Hebbian inside a slowly renormalized envelope.
        self.synaptic_scaling_deferred = synaptic_scaling_deferred
        self._pending_scaling: dict = {}
        # One-time incoming-weight normalization (Dabagia et al. reference
        # `norm_init`).  See _norm_scale for the lazy-materialization
        # formulation and why it is exactly equivalent.
        self.norm_init = norm_init
        self._deterministic = deterministic
        self._rng = np.random.default_rng(seed)
        # Kept alongside the Generator because content-addressed init needs the
        # seed VALUE, not a cursor into a stream. When no seed was given, one is
        # drawn once here so a fiber's identity is still fixed for this engine.
        self._seed = (int(seed) if seed is not None
                      else int(self._rng.integers(0, 2 ** 32)))
        self._content_init = _env_content_init()
        # Order-statistic candidate draw. Fixes non-idempotence (see
        # SparseSimulationEngine.sample_new_winner_inputs), verified by
        # research/experiments/projection_idempotence.py: 4 non-idempotent
        # configurations -> 0.
        #
        # OPT-IN, NOT DEFAULT, and this is a known-incomplete state. Enabling
        # it costs ~13 real test failures, all on one axis: separation between
        # independent inputs and recruitment rate (separate_near_chance,
        # separate_overlap, pnas_scaling, lexicon_entries_are_distinct,
        # neither_seals_nor_exhausts). Independent stimuli read overlap
        # 0.136 +/- 0.203 against a chance of 0.005 -- and that variance is the
        # tell: the failure is bimodal, which is the signature of the
        # recurrent-pricing defect this fix EXPOSED rather than caused.
        # With a recurrent fiber present, candidates are priced as receiving
        # from stimulus AND recurrence (total_k doubles) while the incumbents'
        # recurrent contribution does not reach the comparison, so every
        # candidate wins or none does.
        #
        # The two are coupled and have to land together; turning this on
        # before fixing recurrent pricing trades a long-standing defect that
        # everything is calibrated around for a fresh one that nothing is.
        self.stable_candidates = (
            os.environ.get("NEURAL_ASSEMBLIES_STABLE_CANDIDATES", "0") != "0"
        )
        # {candidate_draw_key: neurons this key has recruited}. The offset into
        # that key's order-statistic tail. Plain dict of tuples/ints so it
        # survives the pickling this repo does constantly.
        #
        # BOUNDED. An exact-repeat key is only needed while that input is still
        # being repeated; the fiber accumulator below is a correct floor for
        # everything older, so evicting the oldest entries costs nothing but a
        # little precision on a long-dormant input.
        self._key_recruited: "OrderedDict[tuple, int]" = OrderedDict()
        self._key_recruited_max = 8192
        # {fiber signature: (cumulative recruit count, {source area: winners})}.
        # See `_fiber_draw_offset` -- this is what stops a slowly DRIFTING input
        # from being priced as a brand-new one every round.
        self._fiber_draw: Dict[tuple, tuple] = {}
        # Size an area->area block on the round it is FIRST NAMED rather than
        # the round after. Needed for PNAS Fig. 2 B1-B3, where y2's competition
        # depends on y1's recurrent input existing.
        #
        # SEPARATE FLAG, DEFAULT OFF, because it is net-negative today. Bundled
        # with `stable_candidates` it took the flag-on suite from 11 non-CUDA
        # failures to 17, and it broke three association tests that were
        # passing (test_association_increases_overlap_substantially,
        # test_associate_creates_shared_response, test_associate_golden) -- an
        # extra fiber delivering on round one changes which neurons the joint
        # assembly recruits. It moves PNAS A2 from 1.000 to 0.220 against a
        # paper value of ~0.50, so it is directionally right and not yet
        # correct.
        self.eager_fiber_init = (
            os.environ.get("NEURAL_ASSEMBLIES_EAGER_FIBER_INIT", "0") != "0"
        )
        # THE ARRAY MODULE THIS ENGINE USES, captured ONCE here rather than
        # re-read from a process-global on every call.
        #
        # `CupySparseEngine` subclasses this one and works by calling
        # `set_backend("cupy")` before `super().__init__`, so the inherited
        # code builds CuPy arrays -- that is why this is `get_xp()` and not
        # plain `np`. But `set_backend` is never restored, so before this
        # attribute existed, constructing a CuPy engine ANYWHERE in the process
        # retroactively changed what an already-built numpy engine did, and the
        # numpy engine started handing itself CuPy arrays mid-run. It took out
        # 11 tests that pass in isolation, and it only reproduces where CuPy is
        # installed, so CI never saw it.
        #
        # Reading it once at construction makes an engine's backend a property
        # of THAT ENGINE. A later global flip cannot reach backwards.
        # Stored as a NAME, resolved through the `_xp` property. The module
        # object itself is not picklable, and Areas, engines and whole Brains
        # are pickled and deep-copied constantly here -- fork, checkpoint, the
        # disk backbone cache, read-only probes. Storing the module took out
        # 143 tests with "cannot pickle 'module' object".
        self._xp_name = xp_name()
        self._pair_seeds: Dict[tuple, int] = {}
        # Set only inside Brain.read_only(); see the guard in project_into.
        self._no_recruitment = False
        # Opt-in: raise when a projection RECRUITS while plasticity is off.
        # See the raise site in project_into for why this is not the default.
        self._strict_probes = bool(int(
            os.environ.get("NEURAL_ASSEMBLIES_STRICT_PROBES", "0") or 0))
        self._plasticity_enabled_global = True
        self._projection_fidelity = ProjectionFidelity.normalize(projection_fidelity)

        # Internal state
        self._areas: Dict[str, SparseAreaState] = {}
        self._sampled_recurrence_warned = False
        self._stimuli: Dict[str, StimulusState] = {}

        # Connectivity: stim_name -> area_name -> Connectome (1-D weights)
        self._stim_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)
        # Connectivity: src_area -> tgt_area -> Connectome (2-D weights)
        self._area_conns: Dict[str, Dict[str, Connectome]] = defaultdict(dict)

        # stim->area vector growth policy (see module docstring above)
        self._stim_fastpath = _env_stim_fastpath()
        self._dense_stim_threshold = _env_dense_stim_threshold()
        # Bumped whenever a stimulus or area is registered, invalidating the
        # per-target (name, connectome) lists cached in _stim_conns_for.
        self._stim_conn_version = 0
        self._stim_target_cache: Dict[str, tuple] = {}

        # CSR mirrors of dense area->area blocks, for the read hot path. See
        # `_csr_row_sum`. Keyed (src, tgt); ONLY ever populated and consulted
        # while plasticity is off, and cleared by every write path, so a stale
        # entry cannot outlive the weights it mirrors.
        self._csr_drive: Dict[tuple, tuple] = {}

        # Reusable math primitives
        self._sparse_sim = SparseSimulationEngine(self._rng, xp=self._xp)
        self._winner_sel = WinnerSelector(self._rng)

    @property
    def sampled_recurrence_policy(self):
        return getattr(
            self,
            "_sampled_recurrence_policy",
            SampledRecurrencePolicy.WARN,
        )

    def _configure_sampled_recurrence_policy(self, policy) -> None:
        """Set Brain-owned admission policy before runtime state exists."""
        if self._areas:
            raise RuntimeError(
                "sampled recurrence policy cannot change after area registration"
            )
        self._sampled_recurrence_policy = SampledRecurrencePolicy.normalize(policy)

    def describe_model_semantics(self) -> ModelSemantics:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-model-semantics"""
        return ModelSemantics(
            connectome=(
                ConnectomeMode.LAZY_CONTENT_ADDRESSED
                if self._content_init
                else ConnectomeMode.LAZY_STREAM_ADDRESSED
            ),
            candidate_domain=CandidateDomain.MATERIALIZED_PLUS_ORDER_STATISTICS,
            stimulus_drive=StimulusDriveLaw.LAZY_CONDITIONED_AFFERENT_COUNT,
            default_tie_break=TieBreakRule.PARTITION_ORDER,
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

    def _weight_bounds(self, scale: float = 1.0):
        """Clip bounds for Hebbian updates, as (low, high).

        The low bound is 0 only when there are no inhibitory synapses. With
        feedforward inhibition (Hoff et al. Eq. 7) weights may legitimately be
        negative, and clipping at 0 would erase every inhibitory synapse the
        first time plasticity touched it -- silently disabling the mechanism.
        Eq. 3 potentiates inhibitory synapses too (they grow more negative),
        so the negative side gets the mirrored bound.
        """
        if self.w_max is None:
            # UNCLAMPED. `w_max=None` is a supported configuration and the one
            # the theorems assume -- they have no clip, because homeostasis IS
            # their boundedness mechanism (PREREG_theorem_regime.md). Every
            # DENSE caller guards on `w_max is not None` before calling, so the
            # only site that reaches here unclamped is `_new_virtual_fiber`,
            # and `VirtualWeights` already treats `w_hi=None` as "no clip"
            # (it guards the np.clip). Returning a bare `self.w_max * scale`
            # raised TypeError there instead, so the virtual representation was
            # unusable in exactly the regime it is most needed for -- the deep,
            # unclipped runs where dense fibers are largest. `w_lo` is dead
            # whenever `w_hi` is None, so it keeps its clamped meaning.
            return 0.0, None
        hi = self.w_max * scale
        if self.inhibitory_prob <= 0.0:
            return 0.0, hi
        return -abs(self.inhibitory_weight) * self.w_max * scale, hi

    def _p_for(self, source: str, target: str) -> float:
        """This fiber's connection probability; the global `p` unless set."""
        value = self._fiber_p.get((source, target))
        return self.p if value is None else float(value)

    def heterogeneous(self) -> bool:
        """Whether any fiber overrides `p`. Guards every fast path that
        assumes one density, so the homogeneous case never changes."""
        return bool(self._fiber_p)

    def _sample_area_weights(self, shape, rng, p=None):
        """Initial area->area weights, with optional feedforward inhibition.

        Each present synapse (probability ``p``) is excitatory (weight 1) with
        probability ``1 - inhibitory_prob`` or inhibitory (``inhibitory_weight``)
        otherwise. With ``inhibitory_prob == 0`` this is the original binomial
        0/1 connectome, bit-for-bit.
        """
        present = rng.random(shape) < (self.p if p is None else p)
        if self.inhibitory_prob <= 0.0:
            return present.astype(np.float32)
        w = present.astype(np.float32)
        inh = present & (rng.random(shape) < self.inhibitory_prob)
        w[inh] = self.inhibitory_weight
        return w

    def _pair_seed(self, source: str, target: str) -> int:
        """Seed identifying one fiber, cached. Depends only on names + seed."""
        key = (source, target)
        seed = self._pair_seeds.get(key)
        if seed is None:
            seed = fnv1a_pair_seed(self._seed, source, target)
            self._pair_seeds[key] = seed
        return seed

    @property
    def _xp(self):
        """This engine's array module, pinned at construction."""
        return xp_by_name(self._xp_name)

    def _to_xp(self, arr):
        """`backend.to_xp` reads the process-global, which is the leak one
        level down: the engine would hold numpy arrays and be handed a CuPy one
        by a helper. This converts into the engine's OWN module."""
        return self._xp.asarray(arr)

    def _init_area_block(self, source, target, r0, r1, c0, c1):
        """Initial weights for absolute rows [r0,r1) x cols [c0,c1) of a fiber.

        Addressed by position, so which cells a caller happens to ask for --
        and in what order -- cannot change any of their values. That is the
        whole point: growth by rows-then-columns and by columns-then-rows must
        produce the same matrix, or "the same brain" depends on parse order.

        The legacy branch draws from the shared stream instead and is order
        dependent by construction; it exists only for A/B'ing the switch.
        """
        fiber_p = self._p_for(source, target)
        if not self._content_init:
            return self._to_xp(self._sample_area_weights(
                (max(r1 - r0, 0), max(c1 - c0, 0)), self._rng, fiber_p))
        return self._to_xp(hash_area_weights(
            r0, r1, c0, c1, self._pair_seed(source, target), fiber_p,
            self.inhibitory_prob, self.inhibitory_weight,
        ))

    # -- norm_init: one-time incoming-weight normalization -------------------


    def _materialize_self_fiber_csr(self, area: str, conn, n: int) -> None:
        """Build the full ``n x n`` self fiber straight into CSR, chunk by chunk.

        Equivalent to what ``_grow``'s vstack/hstack produces, and produced
        without ever holding the dense form -- which is the point, because the
        transient ``O(n^2)`` allocation is the actual wall a bigger ladder hits
        (1.0 GB at ``n=16,000``) even though the result is 95% zeros.

        The final block is exactly::

            [0:pr, 0:pc]   the EXISTING, already-trained sub-block
            elsewhere      fresh `_init_area_block`, which is addressed by
                           ABSOLUTE position, so chunking cannot change a value

        Preserving the trained corner is load-bearing: everything the area has
        learned so far lives there, and regenerating it from the initialiser
        would silently reset the area to naive while leaving every shape and
        count looking right.
        """
        xp = self._xp
        prev = conn.weights
        prev = prev if getattr(prev, "ndim", 0) == 2 else None
        pr, pc = (prev.shape if prev is not None else (0, 0))
        # CLAMP: physical capacity is amortised and may exceed `n` on a
        # connectome grown before that doubling was bounded (or restored from
        # such a pickle). Everything at or past `n` is padding no consumer
        # reads -- it is sliced off by the logical `w` -- so dropping it here
        # is lossless, where copying it raised a broadcast error mid-way
        # through materialisation and left the area half-built.
        pr, pc = min(int(pr), int(n)), min(int(pc), int(n))
        prev_dense = (np.asarray(to_cpu(prev))
                      if prev is not None and pr and pc else None)

        def block_fn(r0, r1):
            out = np.array(
                to_cpu(self._init_area_block(area, area, r0, r1, 0, n)),
                dtype=np.float32, copy=True)
            if prev_dense is not None and r0 < pr:
                rr = min(r1, pr)
                out[:rr - r0, :pc] = prev_dense[r0:rr, :pc]
            return out

        # The Rust kernel emits CSR triplets directly, so the fresh rows never
        # exist densely even one chunk at a time. Only rows below `pr` carry
        # trained content and need the dense overlay, and `pr` is the area's
        # pre-materialisation `w` -- a few hundred against an `n` of tens of
        # thousands. Restricted to the content-addressed initialiser because
        # the legacy branch draws from the shared RNG stream, which a
        # positionally-addressed kernel cannot reproduce.
        rust = rust_kernels()
        if rust is not None and self._content_init:
            sp = scipy_sparse()
            seed = int(self._pair_seed(area, area)) & 0xFFFFFFFF
            parts = []
            if pr > 0:
                parts.append(sp.csr_matrix(block_fn(0, min(pr, n))))
            if n > pr:
                indptr, indices, data = rust.area_weights_csr_rows(
                    pr, n, n, seed, float(self.p),
                    float(self.inhibitory_prob),
                    float(self.inhibitory_weight),
                )
                parts.append(sp.csr_matrix(
                    (data, indices, indptr.astype(np.int32)),
                    shape=(n - pr, n)))
            conn.weights = CSRWeights(
                sp.vstack(parts, format="csr") if len(parts) > 1 else parts[0])
        else:
            conn.weights = build_csr_from_blocks(n, n, block_fn)
        conn._log_rows, conn._log_cols = n, n
        # Rebuilt rather than patched: norm_init reads these, and CSRWeights
        # answers them natively (`column_nnz`) instead of scanning n^2 cells.
        conn._deg_counts_arr = None
        conn._deg_rows = 0
        conn._deg_dirty = None
        del xp


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
        xp = self._xp
        area = SparseAreaState(name=name, n=n, k=k, beta=beta,
                               refractory_period=refractory_period,
                               inhibition_strength=inhibition_strength,
                               winner_policy=winner_policy,
                               input_noise_std=input_noise_std,
                               backend_name=self._xp_name)
        area.neuron_id_pool = self._rng.permutation(np.arange(n, dtype=np.uint32))
        area.neuron_id_pool_ptr = 0
        self._areas[name] = area
        self._stim_conn_version += 1

        # Initialize stim->area connectomes for every already-registered stimulus
        for stim_name, stim in self._stimuli.items():
            conn = Connectome(stim.size, n, self._p_for(stim_name, name),
                              sparse=True)
            conn.weights = xp.empty(0, dtype=xp.float32)
            self._stim_conns[stim_name][name] = conn
            area.beta_by_source[stim_name] = beta

        # Initialize area->area connectomes (both directions) for every existing area
        for other_name, other in self._areas.items():
            if other_name == name:
                self_conn = Connectome(n, n, self._p_for(name, name),
                                       sparse=True)
                self_conn.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[name][name] = self_conn
            else:
                conn_fwd = Connectome(other.n, n,
                                      self._p_for(other_name, name),
                                      sparse=True)
                conn_fwd.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[other_name][name] = conn_fwd

                conn_rev = Connectome(n, other.n,
                                      self._p_for(name, other_name),
                                      sparse=True)
                conn_rev.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[name][other_name] = conn_rev

                area.beta_by_source[other_name] = beta
                other.beta_by_source[name] = beta

    def add_stimulus(self, name: str, size: int) -> None:
        size = validate_stimulus_registration(name, size, existing=self._stimuli,
                                             reserved=self._areas)
        xp = self._xp
        self._stimuli[name] = StimulusState(name=name, size=size)
        self._stim_conn_version += 1

        # Initialize stim->area for every already-registered area.
        # If the area already has ever-fired neurons (w > 0), create a
        # weight vector of length w with random Bernoulli(p) connections
        # so the stimulus can compete with existing trained connections.
        # Without this, the empty weight vector produces zero input and
        # the projection short-circuits, making online learning impossible.
        for area_name, area in self._areas.items():
            fiber_p = self._p_for(name, area_name)
            conn = Connectome(size, area.n, fiber_p, sparse=True)
            if area.w > 0:
                rng = np.random.default_rng(
                    stable_seed(name, area_name, area.w))
                conn.weights = self._to_xp(
                    (rng.random(area.w) < fiber_p).astype(np.float32)
                    * size  # scale by stimulus size for fair competition
                )
            else:
                conn.weights = xp.empty(0, dtype=xp.float32)
            self._stim_conns[name][area_name] = conn
            area.beta_by_source[name] = area.beta

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        """Set one fiber's connection probability, overriding the global `p`.

        WHY IT IS WORTH THE COMPLEXITY. The regime condition kp >= 3 ln n
        ([[SEQ-REGIME]]) is per-AREA, so an organ that needs a dense local
        regime inside an otherwise sparse brain needs density set per fiber --
        that is [[SEQ-ORGAN-EMBEDS]]. `k` is NOT a substitute even though only
        the product appears in the floor: raising it spends capacity
        ([[AC-CAP]]) and forces `n` up with it.

        WHAT HAD TO CHANGE. The candidate sampler prices unmaterialized
        neurons with ONE pooled draw for the whole projection, which is exact
        only when every fiber shares a `p`. With per-fiber densities the pooled
        count is a Poisson-binomial, moment-matched by
        `_pricing.effective_binomial`, and the divisor's numerator generalizes
        from ``p * sum_f a_f`` to ``sum_f a_f * p_f``.

        HOMOGENEOUS BRAINS ARE UNAFFECTED. `_fiber_p` is empty until this is
        called, and every path checks that before taking a heterogeneous
        branch, so a brain that never calls this is byte-for-byte what it was.

        STRUCTURAL, SO IT MUST PRECEDE TRAFFIC. `p` decides which synapses
        exist. Changing it once a fiber has carried drive would leave
        potentiation sitting on synapses that no longer exist and silently
        rewrite formed assemblies, so this raises instead. Requesting the value
        already in force is not a change and stays a no-op.
        """
        if p is None:
            raise ValueError("connectivity probability p must be provided")
        key = (source, target)
        current_p = self._fiber_p.get(key)
        if current_p is None:
            current_p = self.p
        if float(p) == float(current_p):
            return
        if source not in self._areas and source not in self._stimuli:
            raise KeyError(f"unknown source {source!r}")
        if target not in self._areas:
            raise KeyError(f"unknown target area {target!r}")

        conn = (self._area_conns.get(source, {}).get(target)
                if source in self._areas
                else self._stim_conns.get(source, {}).get(target))
        if conn is not None and getattr(conn, "weights", None) is not None:
            if int(np.size(to_cpu(conn.weights))) > 0:
                raise RuntimeError(
                    f"add_connectivity({source!r}, {target!r}, p={p}) after the "
                    f"fiber has carried traffic. Connectivity is structural: "
                    f"changing it now would leave potentiation on synapses that "
                    f"no longer exist. Set it before the first projection.")
        self._fiber_p[key] = float(p)
        if conn is not None:
            conn.p = float(p)

    def ensure_area_conn(self, src_name: str, target: str) -> bool:
        """Give the ``src -> target`` connectome real columns if it has none.

        Connectome columns are normally allocated as a side effect of the
        target recruiting new neurons. A fiber first used *after* its target
        has already grown therefore stays shaped (0, 0): it delivers zero
        input forever, and because plasticity here is multiplicative
        (``w *= 1 + beta``), zero weights can never grow. Training such a
        pathway silently does nothing.

        This allocates the missing block with the same binomial connectivity
        and deterministic per-pair seed the lazy-init path uses, so the result
        matches what the fiber would have had if it had been used earlier.

        Returns True if a block was allocated.
        """
        self.invalidate_csr_drive()
        conns = getattr(self, "_area_conns", None)
        if not conns or src_name not in conns or target not in conns[src_name]:
            return False
        if src_name not in self._areas or target not in self._areas:
            return False

        conn = conns[src_name][target]
        nr = int(self._areas[src_name].w)
        nc = int(self._areas[target].w)
        if nr <= 0 or nc <= 0:
            return False

        cur = conn.weights
        cr, cc = (0, 0) if cur is None else tuple(getattr(cur, "shape", (0, 0)))
        if cr >= nr and cc >= nc:
            return False

        # Only ever grow. Physical capacity is amortised (it doubles), so the
        # existing block can already be larger than the logical w in either
        # dimension; shrinking it would drop live columns.
        nr, nc = max(nr, cr), max(nc, cc)

        # Content-addressed, so this agrees CELL BY CELL with what
        # _expand_connectomes would have written. Under the old per-shape seed
        # the two paths sampled independently, so a cell's weight depended on
        # which path happened to materialise it first -- order dependence of
        # the same kind, one level up.
        fresh = self._init_area_block(src_name, target, 0, nr, 0, nc)

        # Preserve everything already learned. Overwriting the whole block
        # would discard the accumulated Hebbian weights every time the target
        # recruited a neuron, so a pathway could never accumulate strength
        # across training -- only the newly added rows/columns are novel.
        if cur is not None and cr > 0 and cc > 0:
            fresh[:cr, :cc] = cur[:cr, :cc]
        conn.weights = fresh
        return True

    # -- Projection ---------------------------------------------------------

    def materialize_area(self, area: str, storage: str = "csr") -> int:
        """Bring ALL ``n`` of an area's neurons into existence at once.

        WHY THIS EXISTS.  This engine materializes neurons lazily: only the
        ``w`` neurons that have actually won are given a compact index, and
        every weight block is sized to ``w``.  For feed-forward use that is the
        whole point -- it is what makes ``n = 10^6`` tractable.  But it silently
        breaks any protocol that DRIVES THE AREA FROM AN ARBITRARY SUBSET OF
        ``n``, because the neurons it names mostly do not exist yet and deliver
        nothing.

        The measured case is the reference NEMO coin.  Its
        ``RecurrentArea.reset`` allocates ``recurrent_weights`` as a full dense
        ``n x n`` Bernoulli matrix up front, and its ``flip`` then seeds a
        UNIFORM RANDOM ``k``-subset of all ``n``.  Ported onto lazy
        materialization at ``n=2000`` the area had ``w=357`` -- 17.9% of it
        existed -- so a uniform ``k=50`` seed contained on average
        ``9.2 +/- 2.6`` materialized neurons and **82% of the seed had no
        outgoing recurrent synapse at all**.  Settling was resolving ~9 neurons
        of signal, which is why it walked to noise instead of completing to an
        attractor, and why no ``(beta, rounds, settle)`` cell produced a fair
        coin.

        COST, AND WHY ``storage`` EXISTS.  Dense, the self fiber is ``O(n^2)``
        float32: 16 MB at ``n=2000`` but **1.0 GB at ``n=16,000``**, which is
        what bounded the finite-size ladder. The block is only ``~p`` occupied
        (measured 4.99% at ``p=0.05``), so ``storage="csr"`` (the default)
        stores it as ``CSRWeights`` instead -- **10x less memory**, putting
        ``n ~ 50,000`` inside the same budget -- and builds it CHUNKED, so the
        dense form is never held even transiently. See ``_csr_weights`` for why
        a fixed-pattern representation is correct here.

        ``storage="dense"`` restores the plain ndarray for callers that need to
        index the block in ways ``CSRWeights`` deliberately refuses.

        The new weights come from ``_init_area_block``, which is addressed by
        ABSOLUTE position -- so a materialized-all-at-once area has exactly the
        weights it would have had if the same neurons had been recruited one at
        a time.  Materializing does not change the brain, only when it exists.

        Returns the number of neurons newly materialized.
        """
        self.invalidate_csr_drive()
        xp = self._xp
        tgt = self._areas[area]
        prior_w = int(tgt.w)
        n = int(tgt.n)
        if prior_w >= n:
            return 0

        # 1. Give every remaining neuron a compact index, drawn from the same
        #    shuffled pool the incremental path consumes, so identities match.
        if tgt.neuron_id_pool is not None:
            pool = np.asarray(to_cpu(tgt.neuron_id_pool))
            need = n - len(tgt.compact_to_neuron_id)
            ptr = int(tgt.neuron_id_pool_ptr)
            take = pool[ptr:ptr + need]
            tgt.compact_to_neuron_id.extend(int(x) for x in take)
            tgt.neuron_id_pool_ptr = ptr + len(take)
        while len(tgt.compact_to_neuron_id) < n:
            tgt.compact_to_neuron_id.append(len(tgt.compact_to_neuron_id))

        # 2. Stimulus fibers are 1-D over the target's neurons.
        #
        # PASS AN EMPTY FIRING SET, NOT THE CONNECTED ONES.  Both expansion
        # helpers read `stim_names` as "stimuli FIRING on this step", and they
        # deliberately leave those at ZERO because `_expand_connectomes` fills
        # them afterwards from each new winner's own afferent split
        # (`_grow_stim_vector(..., fill=None)` / the `xp.zeros` branch). Every
        # OTHER stimulus takes the background `binomial(stim_size, p)` draw.
        #
        # This site used to pass every stimulus CONNECTED to the area, so all
        # of them took the firing branch -- and nothing fires during
        # materialization, so nothing ever filled them. Measured at n=1000,
        # k=32, p=0.0991 after `materialize_area('A')`:
        #
        #     stimA->A : shape=(1000,) nonzero=0        <- silent
        #     A->A     : shape=(1000,1000) density=0.0988 (p=0.0991)  <- fine
        #
        # A materialized area therefore received zero stimulus drive, took the
        # "Zero signal -> preserve current assembly" early return, and saved an
        # EMPTY winner set every round: on the merge protocol, 52 snapshots of
        # length 0. It looked correct in the protocol this method was built for
        # -- the NEMO coin seeds a k-subset and drives it RECURRENTLY, with no
        # stimulus -- which is why a half-wired area went unnoticed.
        if any(area in per for per in self._stim_conns.values()):
            if self._stim_fastpath and not self.heterogeneous():
                self._expand_stim_vectors_fast(area, tgt, [], n)
            else:
                self._expand_stim_vectors_legacy(area, [], n)

        # 3. Area fibers, BOTH directions. Incoming blocks gain columns;
        #    outgoing blocks gain rows; the self fiber gains both.
        def _grow(conn, src_name, dst_name, rows, cols):
            if not conn.sparse:
                return
            w = conn.weights
            if isinstance(w, VirtualWeights):
                # Materializing a virtual fiber is a bounds update; the base
                # is lazy and the deviations are already absolute-indexed.
                w.resize(max(int(rows), w.n_rows), max(int(cols), w.n_cols))
                conn._log_rows, conn._log_cols = w.n_rows, w.n_cols
                conn._deg_counts_arr = None
                conn._deg_rows = 0
                conn._deg_dirty = None
                return
            if (self._use_virtual() and getattr(w, "size", 0) == 0
                    and rows > 0 and cols > 0):
                # A fiber born here is born virtual. One with dense CONTENT
                # stays dense -- its cells may carry potentiation the virtual
                # store has no history for.
                conn.weights = self._new_virtual_fiber(
                    src_name, dst_name, rows, cols)
                conn._log_rows, conn._log_cols = int(rows), int(cols)
                conn._deg_counts_arr = None
                conn._deg_rows = 0
                conn._deg_dirty = None
                return
            if getattr(w, "ndim", 0) != 2:
                w = xp.empty((0, 0), dtype=xp.float32)
            pr, pc = w.shape
            if rows > pr:
                add = self._init_area_block(src_name, dst_name, pr, rows, 0, pc)
                w = xp.vstack([w, add]) if pc > 0 else xp.zeros(
                    (rows, 0), dtype=xp.float32)
                pr = rows
            if cols > pc:
                add = self._init_area_block(src_name, dst_name, 0, pr, pc, cols)
                w = xp.hstack([w, add]) if pr > 0 else xp.zeros(
                    (0, cols), dtype=xp.float32)
                pc = cols
            conn.weights = w
            conn._log_rows, conn._log_cols = pr, pc
            # The in-degree cache is indexed by (rows, cols); it is rebuilt
            # from scratch rather than patched, because norm_init reads it.
            conn._deg_counts_arr = None
            conn._deg_rows = 0
            conn._deg_dirty = None

        for src_name, per_dst in self._area_conns.items():
            conn = per_dst.get(area)
            if conn is None:
                continue
            if (src_name == area and storage == "csr" and conn.sparse
                    and _csr_storage_available(self._xp)):
                self._materialize_self_fiber_csr(area, conn, n)
                continue
            src_rows = n if src_name == area else int(self._areas[src_name].w)
            _grow(conn, src_name, area, src_rows, n)
        for dst_name, conn in self._area_conns.get(area, {}).items():
            if dst_name == area or conn is None:
                continue
            _grow(conn, area, dst_name, n, int(self._areas[dst_name].w))

        tgt.w = n
        if tgt.refracted and tgt._cumulative_bias is not None:
            old = tgt._cumulative_bias
            tgt._cumulative_bias = xp.zeros(n, dtype=xp.float32)
            if len(old) > 0:
                tgt._cumulative_bias[:len(old)] = old
        return n - prior_w


    # -- k-WTA bound-and-prune ----------------------------------------------

    #: Opt-in from the environment as well as the attribute, so an experiment
    #: can enable the prune on an organ whose Brain it does not construct
    #: itself. Default OFF: every result in this repository was measured
    #: without it, and it does not preserve the ORDER of exactly-tied winners.
    _KWTA_PRUNE_ENV = "ASSEMBLIES_KWTA_PRUNE"

    def clone(self) -> "NumpySparseEngine":
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-brain-clone

        Copy the state graph, preserving internal aliases without reconstructing
        configuration from defaults. Specialized copies must meet this contract.
        """
        import copy

        return copy.deepcopy(self)

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "numpy_sparse"
