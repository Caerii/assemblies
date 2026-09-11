"""NumpySparseEngine: CPU engine using statistical sparse simulation.

This is the extraction of brain.py's ``_project_into_legacy`` sparse path.
Connectivity is stored as growing 1-D (stim->area) and 2-D (area->area)
numpy arrays.  Statistical sampling (truncated normal, binomial PPF)
generates candidate activations for new neurons.
"""

import os
import zlib

import numpy as np
from ..index_spaces import validated_indices, reserve_initial_neuron_ids
from typing import Dict, List
from collections import OrderedDict, defaultdict

# `scipy.sparse` is imported ON FIRST USE via `scipy_sparse()`, not here --
# importing it at module scope charged every process that merely constructs a
# Brain ~0.7s and 429 modules for a code path most runs never reach. See
# `_csr_weights.scipy_sparse` for the measurement.




from ..backend import to_cpu, xp_by_name, xp_name
from .._pricing import (
    area_fiber_activity,
)
from .._homeostasis import (column_scale, refraction_increment,
                            scaling_applies, scaling_setpoint, HomeostasisConfig, check_area_homeostasis, validate_lri_parameters)
from ..engine import ComputeEngine, ProjectionResult
from ..registration import validate_input_noise, validate_stimulus_registration, validate_area_registration
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
    from ...compute.winner_policies import TopKPolicy
except ImportError:
    from compute.sparse_simulation import SparseSimulationEngine
    from compute.winner_selection import WinnerSelector
    from compute.winner_policies import TopKPolicy

from ._growth import GrowthMixin, _self_fiber_deferred_init  # noqa: F401
from ._kwta_prune import (
    PotentiatedSupport, bound_outside, evaluate_set,
)

#: Above this share of the materialised columns the prune saves
#: nothing and still pays for its own decision, so it declines.
#: Not a tuning knob for accuracy -- the answer is exact at any
#: value; it only decides when the fast path is worth taking.
_PRUNE_MAX_FRACTION = 0.5
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


def _warn_fixed_target_enabled() -> bool:
    """Whether to warn on a plasticity-bearing projection INTO a fixed area.

    Off by default (behaviour-neutral); set ``ASSEMBLIES_WARN_FIXED_TARGET=1``
    to surface the silent-no-op footgun during development. See the check at
    the fixed-assembly short-circuit in ``project_into``.
    """
    return os.environ.get("ASSEMBLIES_WARN_FIXED_TARGET", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _explicit_src_norm_enabled() -> bool:
    """Whether an EXPLICIT source area's drive is norm_init-scaled like any other.

    ON by default, because leaving it off silently defeats ``norm_init`` for
    every fiber whose presynaptic area is explicit.

    THE DEFECT.  ``project_into`` accumulates drive from three kinds of source.
    The stimulus path and the general area path both multiply their contribution
    by ``_norm_scale`` (the per-neuron ``1/d_j``).  The explicit-dense path did
    not.  Meanwhile sampled candidates for unmaterialized neurons are ALWAYS
    divided by ``_norm_candidate_divisor`` (``n * p``).  So under ``norm_init``
    the two populations that top-k chooses between were on scales a factor of
    ``n * p`` apart::

        incumbent (explicit source) ~ Binomial(a, p)              # raw counts
        candidate                   ~ Binomial(a, p) / (n * p)    # normalized

    Measured at the [COLT22] parameters (n=1e3, k=1e2, p=0.1, a=165 active):
    incumbent drive min 23.10 / median 25.30 against a candidate max of 0.43 --
    a factor of ~100, exactly ``n * p``.

    TWO CONSEQUENCES, both silent.

    1. **The target seals.**  Once it has materialized ``k`` neurons, no
       candidate can ever outbid an incumbent again, whatever the input.  The
       area's cap becomes constant, so every readout that compares a cap against
       a stored assembly reads exactly 1.0000 -- the perfect-score signature.
    2. **Fiber weighting is corrupted even with no recruitment.**  An area
       driven by BOTH an explicit and a non-explicit source weights the explicit
       one ``n_pre * p`` times too heavily, because only the other fiber is
       divided by its own in-degree.  Normalized, a fiber contributes about
       ``k / n_pre``; unnormalized it contributes about ``k * p`` regardless of
       ``n_pre``, which erases the reference's deliberate geometry.

    The first assembly is affected too: ``_bootstrap_from_explicit_dense``
    selected the initial cap by RAW in-degree, which is precisely the
    "pre-existing hubs" that ``norm_init`` exists to eliminate.

    Set ``ASSEMBLIES_EXPLICIT_SRC_NORM=0`` to restore the unnormalized
    behaviour, which is needed to reproduce results recorded before this fix.

    THE TORCH ENGINE HAD THE SAME DEFECT and did not inherit this fix, because
    it carried a hand-written copy of the pricing math rather than calling a
    shared one.  Both engines now go through ``core/_pricing.py``; the flag
    above is numpy-only and covers only reproduction of pre-fix runs.
    """
    return os.environ.get(
        "ASSEMBLIES_EXPLICIT_SRC_NORM", "1",
    ).strip().lower() in ("1", "true", "yes", "on")




def _fixed_target_plasticity_enabled() -> bool:
    """Whether a projection INTO a fixed area still potentiates its afferents.

    ON by default, because this is what the reference implementation does and
    the divergence was silently breaking its central idiom.

    [ACREF] (``.reference/dmitropolsky-assemblies/brain.py``) handles a fixed
    target by
    pinning the winners and skipping recruitment::

        if target_area.fixed_assembly:
          target_area._new_winners = target_area.winners
          target_area._new_w = target_area.w
          num_first_winners_processed = 0

    and then FALLS THROUGH to the plasticity section, which sits outside that
    branch, so ``from_area -> fixed_target`` synapses are still multiplied by
    ``(1 + beta)`` onto the frozen winners. Holding an area fixed means "do not
    let the winners move", not "do not learn".

    That is the whole mechanism behind the reference's reciprocal idiom --
    ``parser.py``'s "reciprocal until stable, LEX frozen", and
    ``simulations.fixed_assembly_recip_proj``, which freezes A and runs
    ``{"A": ["B"], "B": ["A", "B"]}`` so that B->A is written against a
    stationary A and can later restore it. We short-circuited before plasticity,
    so the back-fiber was never written and restoration was impossible.

    MEASURED against the reference at its own defaults (n=1e5, k=317, p=0.01,
    beta=0.05): first B->A restores 0.246 of A, rising to 0.344 -- which matches
    the expectation recorded in that function's header comment ("first B->A gets
    only 25% ... restore up to 42%"). At the parameters
    ``tests/test_assembly_calculus.py`` uses it reaches 0.75, so that file's
    ``> 0.6`` assertion was calibrated correctly all along; what it was testing
    was broken.

    Set ``ASSEMBLIES_FIXED_TARGET_PLASTICITY=0`` to restore the short-circuit,
    kept so the two can be A/B'd on one seed. Note that the old behaviour makes
    such a projection a silent no-op, which is what
    ``ASSEMBLIES_WARN_FIXED_TARGET`` exists to surface.
    """
    return os.environ.get(
        "ASSEMBLIES_FIXED_TARGET_PLASTICITY", "").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _strict_drive_enabled() -> bool:
    """Whether to warn when a projection delivers NO drive to its target.

    Off by default (behaviour-neutral); set ``ASSEMBLIES_STRICT_DRIVE=1``, which
    the test suite and the research scripts should do.

    WHY THIS EXISTS. The recurring failure mode in this project is not a wrong
    number, it is a mechanism that silently does not fire, because a projection
    with no drive still returns k winners and looks like it worked. Three
    instances are on record and each cost days:

      * `reset_area_connections` zeroing a connectome, so every candidate had
        equal input and the deterministic index tie-break returned the SAME k
        winners for every item -- bit-identical stored assemblies, retrieval at
        exactly chance, invariant to every parameter.
      * a second source area into an already-grown target whose weight block
        was never sized, so it delivered zero and the target silently kept its
        previous assembly. Whichever source projected first was the only one
        that ever worked.
      * projection INTO a fixed area, whose inputs are discarded (that one has
        its own switch above).

    All three have the same signature in results -- "mechanism X turns out to
    have surprisingly little effect" -- which is indistinguishable from a real
    negative result by looking at the numbers. This makes the engine say so.
    """
    return os.environ.get("ASSEMBLIES_STRICT_DRIVE", "").strip().lower() in (
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


class NumpySparseEngine(GrowthMixin, DegreeNormMixin, DriveCacheMixin,
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

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False,
                 projection_fidelity: str = ProjectionFidelity.EXACT.value,
                 inhibitory_prob: float = 0.0,
                 inhibitory_weight: float = -0.2,
                 synaptic_scaling: "bool | frozenset | set | tuple" = False,
                 synaptic_scaling_deferred: bool = False,
                 norm_init: bool = False,
                 sampled_recurrence_policy: str = "warn"):
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
        # Feedforward inhibition (Hoff et al. 2026, Eq. 7): an area->area
        # synapse is inhibitory with probability inhibitory_prob (p_i), taking
        # weight inhibitory_weight (omega_inh < 0); otherwise excitatory (1).
        # p_i = 0 recovers the original excitatory-only Assembly Calculus.
        self.inhibitory_prob = inhibitory_prob
        self.inhibitory_weight = inhibitory_weight
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
        return self._fiber_p.get((source, target), self.p)

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
        refractory_period, inhibition_strength = validate_lri_parameters(
            refractory_period, inhibition_strength)
        xp = self._xp
        area = SparseAreaState(name=name, n=n, k=k, beta=beta,
                               refractory_period=refractory_period,
                               inhibition_strength=inhibition_strength,
                               winner_policy=winner_policy,
                               input_noise_std=input_noise_std)
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
        key = (source, target)
        if float(p) == float(self._fiber_p.get(key, self.p)):
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

    def _candidate_draw_key(self, target, tgt, from_stimuli, from_areas):
        """Identify a projection by its CONTENT, for the candidate sampler.

        Two projections that should elect the same winners must produce the
        same key, and two that should not must not. So the key carries:

          * the target area and the size of its never-fired pool (`w`), which
            is what the truncated-tail draw is a statement about;
          * which stimuli fired -- by name, since a stimulus is a fixed
            pattern;
          * which areas fired AND the assembly each one fired with, because a
            source area's drive is determined by its current winners, not by
            its name. Keying on the name alone would hand the same candidates
            to two different assemblies projecting along the same fiber.

        Winners are digested with crc32 rather than embedded, to keep the key
        small and hashable; `stable_seed` then crc32s the whole tuple.
        """
        # NOTE: `w` is deliberately NOT in the key. It is the OFFSET into the
        # order-statistic sequence this key names; including it would mint a
        # new sequence on every recruitment and defeat the whole fix.
        parts = [target, int(tgt.n), int(tgt.k)]
        parts.extend(sorted(from_stimuli))
        for a in sorted(from_areas):
            w = np.asarray(to_cpu(self._areas[a].winners), dtype=np.uint32)
            parts.append((a, int(zlib.crc32(np.sort(w).tobytes()))))
        return tuple(parts)

    def _fiber_draw_offset(self, target, tgt, from_stimuli, from_areas,
                           input_sizes):
        """How far a DRIFTING input has already eaten into its own tail.

        `_key_recruited` answers this exactly for an input that repeats
        byte-for-byte and gives 0 for everything else. That binary reading is
        wrong in the middle, and the middle is where the interesting protocols
        live: in `merge_sim` the source assemblies shift by a few neurons per
        round, so every round mints a fresh key, every round is priced as a
        brand-new input, and every round is handed candidates from rank ~0 --
        the extreme top of a pool of n. Measured over 50 rounds of the merge
        protocol, support per assembly in units of k (explicit engine = ground
        truth, which does no candidate sampling at all):

            n     k     explicit    old sampler   keyed, no discount
            1000  32       6.1          10.0          18.4
            2000  45       6.1          14.2          30.9
            4000  63       6.9          12.2          25.9

        The model says why. Drive is |{j in x : synapse j->i}|, so two inputs
        overlapping 90% give drives correlated 0.9 -- the neurons already taken
        are still near the top for the shifted input, and only the 10% that is
        genuinely new can reach past them. Treating that as a fresh draw
        recruits a full k every round forever.

        So the offset is discounted by that correlation. `rho` is the fraction
        of this projection's drive that also drove the previous projection
        along the same fibers (stimuli are fixed patterns and count as fully
        shared), and the offset is

            offset = rho * (neurons this fiber has ever recruited)

        rho = 1 recovers the exact-repeat count; rho = 0 (a genuinely
        independent input on the same fibers) gives rank 0, which is what an
        independent input deserves.

        THE COUNT IS CUMULATIVE, NOT DISCOUNTED PER ROUND.  An earlier version
        carried `eff <- rho * eff + recruited`, treating the correlation
        between round t and round t-d as rho^d. That is wrong whenever any part
        of the drive PERSISTS, and here some always does: a stimulus fires the
        identical pattern every round, so a neuron wired to it is favoured in
        all of them. Traced at n=1e5 k=317, rho sat at exactly 1/3 =
        stim/(stim + A + C) for 25 consecutive rounds with winner stability
        0.000 -- the geometric sum pinned the offset near 155 while `w` ran past
        8000, so candidates kept outbidding settled incumbents and the assembly
        never converged at all. Summing without the decay lets the offset track
        `w`; the transient then terminates and w_A at paper scale falls
        8725 -> 4273.

        THIS IS A DAMPER, NOT A DERIVATION, AND THE ALTERNATIVE WAS MEASURED.
        The offset is a rank, and a rank IS directly computable: the incumbents'
        exact drives are in `prev_winner_inputs`, so one can solve
        j* = |{i : d_i > v_j*}| and keep no state at all. That was implemented
        and rejected on evidence -- it saturates at k_eff for any trained area,
        which SEALS it (`test_materialize_area` caught a weight block with zero
        rows, and association overlap fell to 0.020 against a 0.030 floor),
        while merge at paper scale went the wrong way, 4273 -> 10698. The
        reason is that merge's over-recruitment lives in the TRANSIENT, where
        the incumbents genuinely are weak and a correct offset therefore says
        "recruit". No offset policy can fix that: the offset says where in THIS
        input's tail to start and cannot express that this tail is 90% of last
        round's. Charging recruitment history regardless of incumbent strength
        is exactly the crude thing that does damp it.

        Keyed on the FIBER, not the content, so it is bounded by the number of
        distinct projection shapes rather than growing without limit.

        Returns ``(signature, current source winners, rho, cumulative count)``;
        the caller applies `rho * cumulative` and writes the count back.
        """
        sig = (target, int(tgt.n), int(tgt.k),
               tuple(sorted(from_stimuli)), tuple(sorted(from_areas)))
        cur = {}
        for a in sorted(from_areas):
            w = np.asarray(to_cpu(self._areas[a].winners), dtype=np.uint32)
            cur[a] = frozenset(int(x) for x in w)

        prev = self._fiber_draw.get(sig)
        if prev is None:
            return sig, cur, 0.0, 0.0

        eff, last = prev
        # Weight each fiber by how much drive it carries, so a big stimulus
        # does not get the same say as a k-neuron area. `input_sizes` is
        # positionally parallel to from_stimuli + from_areas.
        sizes = list(input_sizes)
        stims = sorted(from_stimuli)
        shared = total = 0.0
        for i, _s in enumerate(stims):
            sz = float(sizes[i]) if i < len(sizes) else 0.0
            shared += sz          # a stimulus is the same pattern every time
            total += sz
        for j, a in enumerate(sorted(from_areas)):
            idx = len(stims) + j
            sz = float(sizes[idx]) if idx < len(sizes) else 0.0
            if sz <= 0:
                continue          # silent fiber: carries no drive, no say
            prev_w = last.get(a)
            if prev_w:
                shared += sz * len(cur[a] & prev_w) / max(1, len(cur[a]))
            total += sz
        rho = (shared / total) if total > 0 else 0.0
        return sig, cur, rho, float(eff)

    def _select_winner_indices(self, tgt, all_inputs, rng, population_sigma=None):
        """Select winner indices using area policy (default top-k)."""
        from ...compute.winner_policies import TopKPolicy

        inputs = all_inputs
        if getattr(tgt, "input_noise_std", 0.0) > 0:
            noise = rng.normal(0, tgt.input_noise_std, size=len(inputs))
            inputs = inputs + self._to_xp(noise.astype(np.float32))

        policy = getattr(tgt, "winner_policy", None) or TopKPolicy(k=tgt.k)
        if (
            isinstance(policy, TopKPolicy)
            and policy.k == tgt.k
            and getattr(tgt, "input_noise_std", 0.0) == 0.0
        ):
            return self._winner_sel.heapq_select_top_k(inputs, tgt.k).tolist()

        selected = self._winner_sel.select_with_policy(
            inputs, policy, population_sigma=population_sigma)
        return [int(i) for i in to_cpu(selected)]

    def _bootstrap_from_explicit_dense(
        self,
        target: str,
        dense_act,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        rng=None,
        record_activation: bool = False,
    ) -> ProjectionResult:
        """First assembly in a sparse area driven by explicit-source dense input."""
        xp = self._xp
        tgt = self._areas[target]
        if rng is None:
            rng = np.random.default_rng(self._rng.integers(0, 2**32))

        act = dense_act.astype(xp.float32, copy=True)
        for stim in from_stimuli:
            stim_conn = self._stim_conns[stim][target]
            if not stim_conn.sparse and len(stim_conn.weights) > 0:
                act += stim_conn.weights.sum(axis=0).astype(xp.float32, copy=False)

        if tgt.input_noise_std > 0:
            act = act + rng.normal(
                0.0, tgt.input_noise_std, size=act.shape,
            ).astype(xp.float32)

        neuron_ids = self._winner_sel.select_with_policy(
            act, tgt.winner_policy or TopKPolicy(k=tgt.k),
        )
        neuron_ids = [int(i) for i in to_cpu(neuron_ids)]
        compact = list(range(len(neuron_ids)))
        pool = reserve_initial_neuron_ids(tgt.neuron_id_pool, neuron_ids, n=tgt.n)
        tgt.compact_to_neuron_id = list(neuron_ids)
        tgt.neuron_id_pool = pool
        tgt.neuron_id_pool_ptr = len(neuron_ids)

        if plasticity_enabled and self._plasticity_enabled_global:
            for src_name in from_areas:
                if not self.fiber_learning_allowed(src_name, target):
                    continue
                conn = self._area_conns[src_name][target]
                src = self._areas[src_name]
                if not conn.sparse and getattr(src, "explicit_source", False):
                    beta = tgt.beta_by_source.get(src_name, tgt.beta)
                    if beta > 0:
                        valid_rows = xp.asarray(src.winners)
                        valid_rows = valid_rows[valid_rows < conn.weights.shape[0]]
                        if len(valid_rows) > 0 and len(neuron_ids) > 0:
                            conn.update_weights(
                                to_cpu(valid_rows),
                                neuron_ids,
                                beta,
                                w_max=self.w_max,
                            )

        tgt.winners = xp.asarray(compact, dtype=xp.uint32)
        tgt.w = len(compact)
        total_act = float(xp.sum(act[neuron_ids])) if neuron_ids else 0.0

        result = ProjectionResult(
            winners=np.array(compact, dtype=np.uint32),
            num_first_winners=len(compact),
            num_ever_fired=len(compact),
            total_activation=total_act,
        )
        if record_activation:
            result.pre_kwta_inputs = np.array(to_cpu(act), dtype=np.float32, copy=True)
            result.pre_kwta_prev_only = np.zeros(0, dtype=np.float32)
            result.pre_kwta_total = float(xp.sum(act))
            result.pre_kwta_count = int(len(act))
        return result

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



    def project_into(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        record_activation: bool = False,
    ) -> ProjectionResult:
        xp = self._xp
        tgt = self._areas[target]
        self.validate_probe_target(target)
        if (
            target in from_areas
            and not tgt.fixed_assembly
            and not self._no_recruitment
            and tgt.winners.size > 0
            and tgt.w > 0
            and tgt.w < tgt.n
        ):
            policy = getattr(
                self,
                "sampled_recurrence_policy",
                SampledRecurrencePolicy.WARN,
            )
            if policy is SampledRecurrencePolicy.FORBID:
                raise RuntimeError(
                    f"Recurrent projection into {target!r} is forbidden because its "
                    "numpy connectome is still sampled. Materialize the area or use "
                    "a fixed-connectome engine. See "
                    "research/notes/sequence/PREREG_sampler_audit.md."
                )
            if (
                policy is SampledRecurrencePolicy.WARN
                and not getattr(self, "_sampled_recurrence_warned", False)
            ):
                import warnings

                warnings.warn(
                    f"Recurrent projection into {target!r} uses the sampled numpy "
                    "connectome. Sequence-dynamics numbers are void until rerun "
                    "materialized or on a fixed-connectome engine (numpy_exact or "
                    "the hashed substrate). See "
                    "research/notes/sequence/PREREG_sampler_audit.md. Use "
                    "Brain.materialize_area before training if materialized "
                    "semantics are intended, or explicitly select "
                    "sampled_recurrence_policy='acknowledged' for a deliberate "
                    "sampled-engine comparison.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._sampled_recurrence_warned = True
        rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # A learning round may rewrite any block, so no CSR mirror survives it.
        # This is the PRIMARY guarantee that `_csr_row_sum` cannot read stale
        # weights; the per-site invalidations below are belt and braces.
        if plasticity_enabled:
            self._csr_drive.clear()

        # Filter out source areas with no assembly
        from_areas = [
            a for a in from_areas
            if self._areas[a].winners.size > 0
            and (
                self._areas[a].w > 0
                or getattr(self._areas[a], "explicit_source", False)
            )
        ]

        # Fixed assembly — the winners do not move. Whether the AFFERENTS still
        # learn is the question, and we used to answer it differently from the
        # reference: inputs were discarded and no plasticity was applied, so a
        # projection into a fixed area looked like training but wrote nothing.
        # That footgun produced three separate "the mechanism doesn't work"
        # investigations (merge, associate, direct binding) -- and a fourth,
        # since it also made `reciprocal_project` unable to restore its source.
        #
        # The reference potentiates the afferents onto the frozen winners; see
        # `_fixed_target_plasticity_enabled` for the code, the idiom it enables
        # and the numbers. So the default now learns, and only the WINNERS are
        # held. The old short-circuit remains available for A/B.
        if tgt.fixed_assembly:
            learn = (plasticity_enabled and (from_stimuli or from_areas)
                     and _fixed_target_plasticity_enabled())
            if learn:
                # Size any never-used source block FIRST -- a multiplicative
                # `w *= 1 + beta` cannot grow an unmaterialised (0, 0) block,
                # so without this the potentiation would be a no-op of exactly
                # the kind this branch is being fixed for.
                self._init_deferred_area_srcs(target, from_areas, int(tgt.w))
                self._apply_plasticity(
                    target, from_stimuli, from_areas, tgt.winners)
            elif (plasticity_enabled and (from_stimuli or from_areas)
                    and _warn_fixed_target_enabled()):
                import warnings
                warnings.warn(
                    f"projection into FIXED area {target!r} with inputs "
                    f"{list(from_stimuli) + list(from_areas)} is a no-op: the "
                    f"inputs are discarded and no plasticity is applied. Fix "
                    f"the SOURCE, or drive the target with a stimulus instead "
                    f"of fixing it (see ops.merge).",
                    RuntimeWarning, stacklevel=3,
                )
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # No inputs -> keep assembly unchanged
        if len(from_stimuli) == 0 and len(from_areas) == 0:
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # --- Accumulate inputs from previous winners ---
        prev_winner_inputs = xp.zeros(tgt.w, dtype=xp.float32)
        explicit_dense_act = None

        # Stimulus inputs (1-D slice up to w)
        limit = tgt.w
        for stim in from_stimuli:
            stim_conn = self._stim_conns[stim][target]
            stim_w = stim_conn.weights
            end = min(limit, len(stim_w))
            if end > 0:
                nscale = self._norm_scale(
                    stim_conn, tgt.n, self._stimuli[stim].size, end,
                    p=self._p_for(stim, target))
                if nscale is None:
                    prev_winner_inputs[:end] += stim_w[:end]
                else:
                    prev_winner_inputs[:end] += stim_w[:end] * nscale[:end]

        # k-WTA BOUND-AND-PRUNE: which columns actually have to be gathered.
        # Decided from the potentiated support alone, before any gather, so a
        # declined prune costs one cheap pass and never a redundant one.
        _cand_cols = self._prune_evaluate_set(
            target, tgt, from_stimuli, from_areas, limit,
            record_activation=record_activation)

        # Area inputs (2-D, vectorised fancy-index)
        # Track sources whose connectomes need deferred initialisation.
        _deferred_init_srcs = []
        # Area sources that were NAMED but delivered no drive this round --
        # their weight block is not materialised yet. Priced out below.
        _silent_area_srcs: set = set()
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            src = self._areas[src_name]
            if not conn.sparse and getattr(src, "explicit_source", False):
                src_w = xp.asarray(src.winners)
                valid = src_w[src_w < conn.weights.shape[0]]
                if len(valid) == 0:
                    continue
                # norm_init applies to THIS fiber too.  The connectome is dense
                # and full-width, so its columns are NEURON IDs rather than
                # compact indices -- ask for the whole width and index the
                # result by neuron id.  All rows exist, so rows_known is the
                # full row count and `_norm_scale`'s unknown-row term is zero.
                # See `_explicit_src_norm_enabled` for what omitting this did.
                enorm = None
                if _explicit_src_norm_enabled():
                    enorm = self._norm_scale(
                        conn, src.n, conn.weights.shape[0],
                        int(conn.weights.shape[1]),
                        p=self._p_for(src_name, target),
                    )
                if tgt.w == 0:
                    contrib = conn.weights[valid].sum(axis=0)
                    if enorm is not None:
                        contrib = contrib * enorm[:len(contrib)]
                    if explicit_dense_act is None:
                        explicit_dense_act = contrib.astype(xp.float32, copy=True)
                    else:
                        explicit_dense_act += contrib
                    continue
                neuron_ids = xp.asarray(
                    tgt.compact_to_neuron_id, dtype=xp.int64,
                )
                valid_cols = neuron_ids[neuron_ids < conn.weights.shape[1]]
                if len(valid_cols) > 0:
                    contrib = conn.weights[valid][:, valid_cols].sum(axis=0)
                    if enorm is not None:
                        contrib = contrib * enorm[valid_cols]
                    end = min(limit, len(contrib))
                    if end > 0:
                        prev_winner_inputs[:end] += contrib[:end]
                continue
            if conn.weights.shape[1] == 0:
                # EAGER INIT: size the block NOW, so the fiber delivers on the
                # round it is first named rather than the one after.
                #
                # Deferring it is what kept PNAS Fig. 2 B1-B3 out of reach. The
                # paper's ~50% overlap(y1,y2) comes from y1's neurons getting
                # potentiated afferent input PLUS recurrent input from y1,
                # which together are comparable to a fresh candidate's
                # unpotentiated afferent plus the same recurrent term. Deferred,
                # the recurrent half simply is not there on the round that
                # decides y2, so the comparison is not the paper's.
                if (self.eager_fiber_init and not self._no_recruitment
                        and conn.sparse
                        and (src_name != target
                             or _self_fiber_deferred_init())
                        and self._areas[src_name].w > 0 and tgt.w > 0):
                    conn.weights = self._init_area_block(
                        src_name, target, 0, int(self._areas[src_name].w),
                        0, int(tgt.w))
                if conn.weights.shape[1] == 0:
                    # Still empty -- either eager init is off, or there is
                    # genuinely nothing to connect yet (w == 0 on one side).
                    # Mark for deferred init so it works on the NEXT round. See
                    # `_self_fiber_deferred_init` for why SELF fibers are
                    # excluded by default and what that costs.
                    #
                    # THIS ROUND THE FIBER DELIVERS NOTHING, and it must not be
                    # charged into the candidate price either -- see
                    # `_silent_area_srcs` below.
                    if (conn.sparse
                            and (src_name != target
                                 or _self_fiber_deferred_init())
                            and self._areas[src_name].w > 0 and tgt.w > 0):
                        _deferred_init_srcs.append(src_name)
                    _silent_area_srcs.add(src_name)
                    continue
            # STALE COVERAGE (#151 dead fiber): growth is recruitment-gated
            # (`_expand_connectomes` returns early with no first-time winner),
            # so a no-recruitment episode after the source has grown freezes
            # this block forever -- out-of-range rows are dropped from the
            # slice below, zero drive recruits nobody, and no recruitment
            # means no expansion. Mark for the deferred repair (same
            # next-round semantics as the empty-block path above). Self
            # fibers keep the `_self_fiber_deferred_init` gate, and a
            # read_only() probe must not repair -- growth is exactly the
            # channel that contract closes, and a stale fiber read under it
            # honestly reports the trained brain as it is.
            if (conn.sparse and not self._no_recruitment
                    and not isinstance(conn.weights, CSRWeights)
                    and getattr(conn.weights, "ndim", 0) == 2
                    and (src_name != target or _self_fiber_deferred_init())):
                _cov_r = min(int(getattr(conn, "_log_rows",
                                         conn.weights.shape[0])),
                             int(conn.weights.shape[0]))
                _cov_c = min(int(getattr(conn, "_log_cols",
                                         conn.weights.shape[1])),
                             int(conn.weights.shape[1]))
                if int(src.w) > _cov_r or int(tgt.w) > _cov_c:
                    _deferred_init_srcs.append(src_name)
            src_w = xp.asarray(src.winners)
            internal = src_w[src_w < conn.weights.shape[0]]
            if len(internal) > 0 and limit > 0:
                col_end = min(limit, conn.weights.shape[1])
                # A materialised block is ~p occupied; CSR gathers k rows out
                # of it 3-11x faster and bit-identically. Only consulted with
                # plasticity off (see `_csr_row_sum`), so it cannot go stale.
                contrib = None
                if isinstance(conn.weights, VirtualWeights):
                    # The drive kernel regenerates the k winner rows
                    # (~k x n_cols hashed cells) instead of traversing a
                    # stored block; deviations come from per-row dicts.
                    contrib = self._to_xp(conn.weights.row_sum(
                        np.asarray(to_cpu(internal)), col_end))
                elif isinstance(conn.weights, CSRWeights):
                    # Stored sparse: answer natively, no mirror needed, and
                    # safe with plasticity ON because there is nothing cached.
                    contrib = conn.weights.row_sum(internal, col_end)
                elif not plasticity_enabled and _cand_cols is None:
                    contrib = self._csr_row_sum(
                        src_name, target, conn.weights, internal, col_end)
                if contrib is None and _cand_cols is not None:
                    # Gather ONLY the columns whose bound has not already lost.
                    # Values stay bit-identical: taking a column subset never
                    # reorders any column's row sum. Pruned slots keep 0, which
                    # is below their true drive and far below tau, so they
                    # cannot enter the top-k either way -- and leaving them at
                    # 0 rather than -inf keeps every other consumer of this
                    # vector (the zero-signal check, total_activation) honest.
                    sub = _cand_cols[_cand_cols < col_end]
                    if len(sub) > 0:
                        prev_winner_inputs[sub] += conn.weights[
                            xp.ix_(internal, sub)].sum(axis=0)
                    continue
                if contrib is None:
                    contrib = conn.weights[internal, :col_end].sum(axis=0)
                nscale = self._norm_scale(
                    conn, self._areas[src_name].n,
                    self._areas[src_name].w, col_end,
                    p=self._p_for(src_name, target))
                if nscale is not None:
                    contrib = contrib * nscale[:col_end]
                prev_winner_inputs[:col_end] += contrib

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

        # Zero signal -> preserve current assembly
        # Specification: neural_assemblies/ir/VERIFICATION.md#contract-noise-only-observation
        zero_signal = len(prev_winner_inputs) > 0 and not bool(xp.any(prev_winner_inputs))
        if zero_signal and tgt.input_noise_std > 0 and tgt.w < tgt.n:
            raise ValueError('noise-only projection requires a fully materialized population')
        if zero_signal and tgt.input_noise_std == 0:
            # RUN THE DEFERRED INIT BEFORE RETURNING. Without this the
            # mechanism is UNREACHABLE in the one case it exists for: a source
            # area projecting into an already-grown target for the first time.
            # Its connectome block is empty, so it contributes nothing, so the
            # total is zero, so we return here -- before the consumption site
            # at the end of this function. Next round is identical, forever.
            #
            # The observable symptom is not a zero assembly but a STALE one:
            # this branch preserves `tgt.winners`, so every item driven through
            # the dead fiber "stores" whatever the target last held. Measured
            # on a role area fed by LEX_NOUN then LEX_VERB: all 20 verbs
            # returned the 40th noun's assembly, and swapping the training
            # order swapped which category collapsed (40 nouns onto the last
            # verb). Whichever source happens to go first is the only one that
            # ever works, and nothing warns.
            self._init_deferred_area_srcs(
                target, _deferred_init_srcs, int(tgt.w))
            if _strict_drive_enabled() and (from_stimuli or from_areas):
                import warnings
                empty = [s for s in from_areas
                         if getattr(self._area_conns.get(s, {}).get(target),
                                    "weights", None) is None
                         or tuple(getattr(
                             self._area_conns[s][target].weights,
                             "shape", (0, 0)) or (0, 0))[1:2] == (0,)]
                warnings.warn(
                    f"projection into {target!r} from "
                    f"{list(from_stimuli) + list(from_areas)} delivered ZERO "
                    f"drive; {target!r} keeps its previous assembly, so this "
                    f"looks like it worked and stored nothing new"
                    + (f". Empty weight blocks: "
                       f"{', '.join(f'{s}->{target}' for s in empty)}"
                       if empty else "")
                    + ". Common causes: reset_area_connections zeroed the "
                      "connectome (k-WTA then returns the same index "
                      "tie-break winners for every input), or a source is "
                      "projecting into this area for the first time.",
                    RuntimeWarning, stacklevel=3,
                )
            return ProjectionResult(
                winners=np.array(to_cpu(tgt.winners), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
            )

        # --- Compiled topology: top-k on pregrown columns only ---
        # Skips truncated-normal sampling and connectome expansion when
        # bridge pathways were pregrown (freeze + ring).  Used during
        # EmergentParser bridge training; parity-checked via predict_next.
        if self._use_compiled_projection(tgt):
            limit = int(tgt.w)
            inputs_slice = prev_winner_inputs[:limit]
            # Same noise and competition contract as ordinary selection; only
            # the candidate population is restricted by compiled topology.
            new_winner_indices = self._select_winner_indices(tgt, inputs_slice, rng)
            new_winner_indices = xp.asarray(new_winner_indices, dtype=xp.uint32)
            if plasticity_enabled and self._plasticity_enabled_global:
                self._apply_plasticity(
                    target, from_stimuli, from_areas, new_winner_indices,
                )
            tgt.winners = new_winner_indices
            total_act = float(xp.sum(inputs_slice[new_winner_indices]))
            result = ProjectionResult(
                winners=np.array(to_cpu(new_winner_indices), dtype=np.uint32),
                num_first_winners=0,
                num_ever_fired=tgt.w,
                total_activation=total_act,
            )
            if record_activation:
                snap = np.array(to_cpu(inputs_slice), dtype=np.float32, copy=True)
                result.pre_kwta_inputs = snap
                result.pre_kwta_prev_only = snap
                result.pre_kwta_total = float(xp.sum(inputs_slice))
                result.pre_kwta_count = int(len(inputs_slice))
            return result

        # --- Sample new winner candidates via truncated normal ---
        # A source area contributes drive from the neurons that are ACTUALLY
        # firing, so that -- not the area's nominal cap -- is the size the
        # candidate sampler must price Binomial(size, p) against.  The two
        # agree whenever an assembly is complete, which is why the nominal `k`
        # was harmless in ordinary projection; they diverge exactly when a
        # PARTIAL assembly is presented, i.e. during pattern completion.
        # Measured: cueing 25 of 50 neurons, candidates were sampled as if 50
        # were active and came in at 0.060-0.090 normalized drive while the
        # genuine missing assembly members sat at 0.043-0.064, so sampled
        # candidates outbid real completions and completion stalled at 0.427.
        # Materialized non-assembly neurons were correctly below the assembly
        # (max 0.049), confirming the fault was in the sampler, not the
        # dynamics.  Gated on norm_init only to keep default results
        # bit-identical; it is a no-op whenever len(winners) == k.
        # A fiber whose block is not materialised yet contributed ZERO to
        # `prev_winner_inputs` above. Charging it into the candidate price
        # anyway is what made recurrence catastrophic: naming B->B on the round
        # it first appears doubled total_k (100 -> 200), lifting candidates from
        # [17,18] to [30,31], while the incumbents' drive was byte-identical
        # with and without it (min 18.00 max 22.00 mean 18.77 either way). Every
        # candidate then beat every incumbent -- 0 of 100 survived -- and
        # overlap(y1,y2) went 1.000 -> 0.000 with y2 an entirely fresh cohort.
        # PNAS 2020 predicts ~0.50 here.
        # Zeroed, NOT dropped. `input_sizes` is parallel to
        # from_stimuli + from_areas and `_expand_connectomes` indexes the split
        # it produces by that position; shortening the list raised IndexError
        # on the first multi-source projection. A zero entry costs nothing in
        # `total_k = sum(input_sizes)` and allocates no synapses in the split.
        def _priced(a: str) -> int:
            if self.stable_candidates and a in _silent_area_srcs:
                return 0
            return area_fiber_activity(self._areas[a].winners.size,
                                       self._areas[a].k, self.norm_init)

        input_sizes = (
            [self._stimuli[s].size for s in from_stimuli]
            + [_priced(a) for a in from_areas]
        )
        if sum(input_sizes) == 0:
            # EVERY source is silent, so this is a bootstrap round: nothing is
            # materialised yet and deferred init has not run. Pricing at zero
            # would leave `compute_input_splits` with total_k == 0, which
            # returns empty split vectors and makes `_expand_connectomes` raise
            # IndexError on `split[j]`. Fall back to the nominal sizes: with no
            # incumbents to protect there is nothing for the silent-fiber
            # correction to fix, and the area still needs to recruit.
            input_sizes = (
                [self._stimuli[s].size for s in from_stimuli]
                + [area_fiber_activity(self._areas[a].winners.size,
                                       self._areas[a].k, self.norm_init)
                   for a in from_areas]
            )
        # Presynaptic POPULATION per fiber, parallel to input_sizes. Used only
        # to price candidates on the incumbent scale -- see
        # `_norm_candidate_divisor`. Stimulus fibers use the target's own n,
        # matching `_norm_scale`'s convention for them.
        src_pops = (
            [tgt.n for _ in from_stimuli]
            + [self._areas[a].n for a in from_areas]
        )
        # Per-fiber densities, parallel to input_sizes. None while no fiber
        # overrides `p`, which keeps the pooled draw and the divisor on their
        # original scalar code paths -- see `add_connectivity`.
        input_ps = ([self._p_for(s, target) for s in from_stimuli]
                    + [self._p_for(a, target) for a in from_areas]
                    ) if self.heterogeneous() else None

        draw_key = None
        if self._no_recruitment and tgt.w >= tgt.k:
            # A READ-ONLY probe answers "which of the neurons you already have
            # respond best?", so no candidates are offered and the area cannot
            # grow. Recruitment is the last channel by which measuring changes
            # the thing measured: frozen() stops weights changing but not w,
            # and two probe orders that recruit different numbers of neurons
            # are structurally different brains no matter how init is seeded.
            #
            # Gated on w >= k because below it there is nothing to select from,
            # and a silently short assembly would be worse than growing.
            potential_new = np.empty(0, dtype=np.float32)
            old_rng = self._sparse_sim.rng
            draw_key = None          # read-only probe: recruits nothing
        else:
            old_rng = self._sparse_sim.rng
            self._sparse_sim.rng = rng
            # Content key: what this projection IS, so the same projection
            # draws the same candidates. See sample_new_winner_inputs.
            draw_key = (self._candidate_draw_key(target, tgt, from_stimuli,
                                                 from_areas)
                        if self.stable_candidates else None)
            # How far THIS key has already eaten into its own tail. Not the
            # area's `w` -- see _order_statistic_candidates for what that
            # sealed.
            # Two accumulators, and the offset is the larger:
            #   * the exact-repeat count, correct to the neuron for an input
            #     repeated byte-for-byte, and correct THROUGH interleaving
            #     with other inputs, which the fiber term is not;
            #   * the correlation-discounted fiber count, which is what covers
            #     a slowly drifting input -- see _fiber_draw_offset.
            draw_offset = None
            if draw_key is not None:
                fiber_sig, fiber_cur, rho, eff = self._fiber_draw_offset(
                    target, tgt, from_stimuli, from_areas, input_sizes)
                draw_offset = max(self._key_recruited.get(draw_key, 0),
                                  int(round(rho * eff)))
            if self._deterministic:
                potential_new = self._sparse_sim.sample_new_winner_inputs_legacy(
                    input_sizes, tgt.n, tgt.w, tgt.k,
                    self.p if input_ps is None else input_ps, key=draw_key,
                )
            else:
                potential_new = self._sparse_sim.sample_new_winner_inputs(
                    input_sizes, tgt.n, tgt.w, tgt.k,
                    self.p if input_ps is None else input_ps, key=draw_key,
                    offset=draw_offset,
                )
        self._sparse_sim.rng = old_rng

        potential_new = self._to_xp(potential_new)
        # norm_init: bring sampled candidates onto the normalized scale (see
        # _norm_candidate_divisor).  Stored weights and the sampler stay on the
        # unit scale; only the drive comparison is rescaled.
        norm_div = (self._norm_candidate_divisor(tgt, input_sizes, src_pops,
                                                 input_ps)
                    if self.norm_init else None)
        if norm_div is not None:
            potential_new = potential_new / norm_div
        if len(prev_winner_inputs) > 0:
            all_inputs = xp.concatenate([prev_winner_inputs, potential_new])
        else:
            all_inputs = potential_new

        # --- Snapshot raw prev_winner_inputs before penalties ---
        if record_activation:
            _raw_prev = np.array(to_cpu(prev_winner_inputs),
                                 dtype=np.float32, copy=True)

        # --- LRI: penalise recently-fired neurons ---
        if (tgt.refractory_period > 0
                and tgt.inhibition_strength > 0
                and len(tgt._refractory_history) > 0):
            n_inputs = len(all_inputs)
            for steps_ago_idx, winner_set in enumerate(
                    reversed(list(tgt._refractory_history))):
                steps_ago = steps_ago_idx + 1
                decay = 1.0 - (steps_ago - 1) / tgt.refractory_period
                penalty = tgt.inhibition_strength * decay
                for cidx in winner_set:
                    if cidx < n_inputs:
                        all_inputs[cidx] -= penalty

        # --- Refracted mode: cumulative bias penalty ---
        # A MASKED READ ranks the raw drive: the bias is skipped on a read
        # (no plasticity) when the area asks for it. Never on a write.
        masked_read = bool(getattr(tgt, "masked_readout", False)) and not plasticity_enabled
        if tgt.refracted and tgt._cumulative_bias is not None and not masked_read:
            bias = tgt._cumulative_bias
            end = min(len(bias), len(all_inputs))
            if end > 0:
                all_inputs[:end] -= bias[:end]

        # --- Snapshot full all_inputs before top-k ---
        if record_activation:
            _pre_kwta_snapshot = np.array(to_cpu(all_inputs),
                                          dtype=np.float32, copy=True)
            _pre_kwta_total_val = float(xp.sum(all_inputs))
            _pre_kwta_count_val = int(len(all_inputs))

        # --- Select winners (top-k or area policy) ---
        # Analytic sigma of the population input distribution: each source
        # contributes Binomial(active_count, p), so variances add.
        # Each source contributes Binomial(active_count, p), so variances add.
        # This assumes UNIT weights, which is right: the population is the whole
        # area, and all but the |assembly| neurons that have actually fired are
        # still unpotentiated. Working the mixture variance out confirms it --
        # with f_p = w/n on the order of 60/20000, the potentiated group's
        # within- and between-group contributions are both negligible and
        # sigma_mixture ~= sigma_unit.
        #
        # Do NOT try to rescale this by the observed mean of `all_inputs` to
        # account for potentiation: that vector is top-biased on both halves
        # (potentiated incumbents plus SAMPLED TOP ORDER STATISTICS for the
        # unmaterialized neurons), so its mean cannot separate potentiation
        # from selection bias, and the correction overshoots badly.
        #
        # Known limitation, and the reason window="sigma" is not yet the
        # default for trained language areas: once an assembly is potentiated
        # its drive sits many population-sigma above the bulk, so the window
        # h_max - sigma_c*sigma admits only a couple of neurons and selection
        # falls to min_winners. Using the empirical std of `all_inputs`
        # instead swings the other way and admits every candidate. Neither is
        # emergent; a faithful fix needs the window referenced to an
        # inhibitory pool driven by RECENT ACTIVITY rather than by the silent
        # bulk, which is not modelled here yet.
        # Per-fiber densities where they are set: this is a population spread
        # over the SAME per-fiber binomials the sampler prices.
        _sigma_ps = input_ps if input_ps is not None else [self.p] * len(input_sizes)
        pop_sigma = float(np.sqrt(sum(sz * pp * (1.0 - pp)
                                      for sz, pp in zip(input_sizes, _sigma_ps)))) or None
        if pop_sigma is not None and norm_div is not None:
            pop_sigma = pop_sigma / norm_div
        new_winner_indices = self._select_winner_indices(
            tgt, all_inputs, rng, population_sigma=pop_sigma)

        # --- Process first-time winners ---
        num_first = 0
        first_winner_inputs = []
        ring_mode = getattr(tgt, '_ring_mode', False)
        ring_capacity = int(getattr(tgt, '_ring_capacity_cols', 0))
        ring_slot = 0
        for i in range(len(new_winner_indices)):
            if new_winner_indices[i] >= tgt.w:
                if ring_mode and ring_capacity > 0:
                    slot_idx = tgt.w + ring_slot
                    if slot_idx < ring_capacity:
                        new_winner_indices[i] = slot_idx
                        while len(tgt.compact_to_neuron_id) <= slot_idx:
                            tgt.compact_to_neuron_id.append(
                                len(tgt.compact_to_neuron_id),
                            )
                        ring_slot += 1
                        continue
                # Un-normalize: connectome expansion splits an INTEGER synapse
                # count across the input fibers, so it needs the unit-scale
                # drive, not the norm_init-scaled one.
                _fwi = float(all_inputs[new_winner_indices[i]])
                if norm_div is not None:
                    _fwi *= norm_div
                first_winner_inputs.append(int(_fwi))
                if tgt.neuron_id_pool is not None:
                    pid = tgt.neuron_id_pool_ptr
                    if pid >= len(tgt.neuron_id_pool):
                        raise RuntimeError(f"Neuron id pool exhausted for area {tgt.name}")
                    actual_id = int(tgt.neuron_id_pool[pid])
                    tgt.neuron_id_pool_ptr += 1
                else:
                    actual_id = tgt.w + num_first
                tgt.compact_to_neuron_id.append(actual_id)
                new_winner_indices[i] = tgt.w + num_first
                num_first += 1

        if ring_mode and ring_slot > 0:
            new_w = tgt.w + ring_slot
        else:
            new_w = tgt.w + num_first

        # getattr, not attribute access: Brains are pickled to the disk backbone
        # cache and an engine restored from an entry written before this flag
        # existed has no such attribute.
        if (num_first and not plasticity_enabled
                and getattr(self, "_strict_probes", False)):
            # RECRUITMENT WHILE PLASTICITY IS OFF is the signature of a probe
            # written against `frozen()` that meant `read_only()`. frozen()
            # stops weights changing; it does not stop the area GROWING, and
            # growth is what makes a measurement change the measured -- two
            # probe orders that recruit different numbers of neurons are
            # structurally different brains however init is seeded.
            #
            # Off by default and opt-in via NEURAL_ASSEMBLIES_STRICT_PROBES=1,
            # same discipline as _VERIFY_NNZ: it converts "which of these 50
            # frozen() sites is a contaminating probe?" from an argument into a
            # measurement you can run over the whole suite. It is NOT on by
            # default because switching a site to read_only() CHANGES ITS
            # NUMBERS -- suppressing recruitment moves what the probe reads
            # (stored/probe overlap 0.038 -> 0.180 on one protocol) -- so each
            # site is a measured decision, not a mechanical rename.
            raise RuntimeError(
                f"STRICT PROBES: projection into {target!r} recruited "
                f"{num_first} neurons while plasticity was disabled. A read "
                f"that grows the area contaminates what it measures; use "
                f"brain.read_only() rather than brain.frozen() for probes, or "
                f"unset NEURAL_ASSEMBLIES_STRICT_PROBES if this projection is "
                f"meant to build structure without learning.")

        # Advance this input's position in its own tail by what it just took.
        # A repeat of the same input now resumes below the neurons it already
        # holds (idempotent); a novel input still starts near rank 0; a
        # drifting one resumes at the correlation-discounted position.
        # NOT gated on num_first: the fiber entry must be rewritten even when
        # nothing was recruited, because `fiber_cur` is what the NEXT round
        # measures rho against, and leaving it stale prices that round as if
        # this round's drift had not happened.
        if draw_key is not None:
            if num_first > 0:
                self._key_recruited[draw_key] = (
                    self._key_recruited.get(draw_key, 0) + num_first)
                self._key_recruited.move_to_end(draw_key)
                while len(self._key_recruited) > self._key_recruited_max:
                    self._key_recruited.popitem(last=False)
            # Cumulative, and rewritten every round so `fiber_cur` tracks the
            # sources rho is measured against. See _fiber_draw_offset for why
            # this must NOT decay by rho.
            self._fiber_draw[fiber_sig] = (eff + num_first, fiber_cur)

        # --- Apply plasticity ---
        if plasticity_enabled and self._plasticity_enabled_global:
            self._apply_plasticity(target, from_stimuli, from_areas, new_winner_indices)

        # --- Expand connectomes for new winners ---
        if num_first > 0:
            self._expand_connectomes(
                target, from_stimuli, from_areas,
                input_sizes, new_winner_indices,
                first_winner_inputs, new_w,
            )

        # --- Commit state ---
        tgt.winners = xp.asarray(new_winner_indices, dtype=xp.uint32)
        tgt.w = new_w

        # --- Update LRI refractory history ---
        if tgt.refractory_period > 0:
            tgt._refractory_history.append(
                set(int(i) for i in new_winner_indices))

        # --- Update refracted cumulative bias ---
        # Gated on plasticity, and on the SAME condition as the Hebbian update
        # above. The reference charges the bias inside `RefractedArea.update`,
        # so `update=False` stops learning and charging together; ours did not,
        # which meant a no-learn readout kept charging and altered the very
        # trajectory it was meant to observe -- one step of a test sequence
        # changing the next.
        if (tgt.refracted and tgt.refracted_strength > 0
                and plasticity_enabled and self._plasticity_enabled_global):
            if len(tgt._cumulative_bias) < new_w:
                old = tgt._cumulative_bias
                tgt._cumulative_bias = xp.zeros(new_w, dtype=xp.float32)
                if len(old) > 0:
                    tgt._cumulative_bias[:len(old)] = old
            bias = tgt._cumulative_bias
            widx = xp.asarray(new_winner_indices)
            widx = widx[widx < len(bias)]
            if len(widx) > 0:
                bias[widx] += refraction_increment(
                    all_inputs[widx], bias[widx], tgt.refracted_strength)

        total_act = float(xp.sum(all_inputs[new_winner_indices]))

        # --- Deferred connectome initialisation --------------------------------
        # Sources whose connectomes were empty this round get initialised now
        # so they can contribute signal on the NEXT projection round.  Uses a
        # deterministic per-pair seed to avoid disturbing the main RNG.
        self._init_deferred_area_srcs(target, _deferred_init_srcs, new_w)

        result = ProjectionResult(
            winners=np.array(new_winner_indices, dtype=np.uint32),
            num_first_winners=num_first,
            num_ever_fired=new_w,
            total_activation=total_act,
        )
        if record_activation:
            result.pre_kwta_inputs = _pre_kwta_snapshot
            result.pre_kwta_prev_only = _raw_prev
            result.pre_kwta_total = _pre_kwta_total_val
            result.pre_kwta_count = _pre_kwta_count_val
        return result

    # -- Plasticity ---------------------------------------------------------

    def _normalize_area_columns(self, target, from_areas, winners):
        """Homeostatic synaptic scaling on area->area fibers.

        Holds each postsynaptic neuron's TOTAL incoming weight on a fiber at a
        setpoint, so Hebbian learning redistributes a fixed budget across
        presynaptic sources instead of inflating the total. This is what stops
        the rich-get-richer runaway: belonging to an established assembly no
        longer buys extra total drive, only a different share of it.

        Two deliberate departures from the reference implementation
        (reference/nemo_numpy/areas.py:145-149), both forced by this engine's
        sparse representation:

        1. The setpoint is the initial expected column sum (``src.w * p``),
           NOT 1. Neurons that have never fired are not materialized here;
           their input is *sampled* as ~Binomial(active, p), which presumes
           weights of order 1. Normalizing materialized columns to sum to 1
           would leave them at total drive ~1 while fresh candidates sample
           ~k*p, so unmaterialized neurons would win every competition and no
           assembly could ever stabilize. Scaling to the initial expected sum
           keeps materialized and sampled neurons on the same scale, and is
           also the biologically accurate statement of synaptic scaling: a
           homeostatic setpoint, not unity.

        2. Stimulus fibers are excluded. The reference normalizes a 2-D
           (presynaptic x postsynaptic) matrix where only the *active* subset
           of presynaptic neurons delivers drive, so normalization redistributes
           across sources. This engine stores stimulus connectomes as 1-D
           pre-summed input, and a stimulus fires in full every step, so
           normalizing would drive every neuron to the identical value and
           erase the representation entirely. Stimulus weights stay bounded by
           w_max instead.

        Only the columns plasticity just touched are rescaled, so the cost is
        O(k) columns per step rather than the whole matrix.

        STATUS: opt-in and OFF by default, because this per-fiber formulation
        is not yet correct. Measured with recurrence enabled, it reduces the
        runaway substantially (independent-stimulus overlap 0.67 -> 0.22, and
        10-assembly overlap 3.0 against the literature's 4.0), but it breaks
        the two defining properties of an assembly: post-projection stability
        collapses to 0.01 (needs > 0.9) and pattern completion falls to 0.000
        (needs > 0.6).

        The reason is structural, not a tuning issue. Restoring each fiber to
        its own original total exactly cancels the net potentiation that makes
        an assembly self-sustaining -- an attractor requires the assembly's
        internal loop to end up stronger than its surroundings, and a per-fiber
        setpoint removes that gain by construction. The reference avoids this
        by normalizing EVERY fiber to a common scale, so a neuron's recurrent
        share can grow at the expense of its feedforward share.

        The fix is therefore joint normalization across all fibers into a
        neuron (stimulus + every area), holding the neuron's TOTAL drive
        constant while letting the recurrent/feedforward split shift. That
        requires the stimulus fibers to participate in the budget, which in
        turn needs their 1-D pre-summed representation reconciled with the
        2-D per-synapse form the split is defined over.
        """
        # The gate's spelling is interpreted ONCE, in the owner.
        if not scaling_applies(self.synaptic_scaling, target):
            return
        # SLOW HOMEOSTASIS (E9, #138): biological synaptic scaling operates
        # over hours-to-days, segregated from fast Hebbian plasticity --
        # and E8 measured why: per-update renormalization fights repeated
        # writes seed-bistably (variance +/-0.058 -> +/-0.120 under
        # repetition). Deferred mode ACCUMULATES touched columns and
        # normalizes only at flush_synaptic_scaling() (called by trainers
        # at phase boundaries): fast Hebbian inside a slowly renormalized
        # envelope. Default False = per-update, byte-identical.
        # RESEARCH KNOB, default "winners" = unchanged.
        #
        # THE ASYMMETRY. k-WTA selects among CANDIDATES. This rule only ever
        # touches columns that have ALREADY WON, so it cannot influence the
        # selection that produced them -- it arrives one step too late, every
        # step. `norm_init` divides EVERY candidate by its own degree at read
        # time, so a hub never gets its advantage in the first place; here a
        # hub keeps its full raw mass right up to the moment it wins and is cut
        # down only afterwards. Measured (`seq_scaling_merger_forensics`):
        # substrate C's multiply-shared neurons are the highest-degree columns
        # in the area (815 against a population 552) and the earliest recruited
        # (mean compact rank 2.0), yet their drive AFTER training is only
        # ~1.19x the population -- because normalization removed the advantage
        # after it had already been spent.
        #
        # "all" rescales every materialized column each round, normalizing the
        # candidates before the comparison instead of the winners after it.
        # MEASURED: cuts pairwise overlap 0.188 -> 0.142 and quadruples
        # half-cue completion 0.125 -> 0.500, on all three seeds. Real, and
        # only PART of the story -- the sampled (not yet materialized)
        # candidates cannot be rescaled at all, because they do not exist.
        # The full repair is `norm_init` alongside this, which cancels every
        # candidate's degree potentiation-invariantly at read time and takes
        # the overlap to 0.018-0.027, i.e. the chance floor.
        if getattr(self, "synaptic_scaling_scope", "winners") == "all":
            tgt = self._areas.get(target)
            if tgt is not None and int(tgt.w) > 0:
                winners = self._xp.arange(int(tgt.w))

        if getattr(self, "synaptic_scaling_deferred", False):
            pending = self._pending_scaling
            for src_name in from_areas:
                pending.setdefault((src_name, target), set()).update(
                    int(c) for c in winners)
            return
        self._scale_columns_now(target, from_areas, winners)

    # -- k-WTA bound-and-prune ----------------------------------------------

    #: Opt-in from the environment as well as the attribute, so an experiment
    #: can enable the prune on an organ whose Brain it does not construct
    #: itself. Default OFF: every result in this repository was measured
    #: without it, and it does not preserve the ORDER of exactly-tied winners.
    _KWTA_PRUNE_ENV = "ASSEMBLIES_KWTA_PRUNE"

    def _kwta_prune_on(self):
        if getattr(self, "kwta_prune", False):
            return True
        return os.environ.get(self._KWTA_PRUNE_ENV, "") == "1"

    def _support_for(self, src_name, target):
        """Per-fiber index of the cells plasticity has touched. Lazy, so an
        engine that never prunes never pays for one."""
        m = getattr(self, "_pot_support", None)
        if m is None:
            m = self._pot_support = {}
        key = (src_name, target)
        sup = m.get(key)
        if sup is None:
            sup = m[key] = PotentiatedSupport()
        return sup

    def drop_potentiated_support(self):
        """Forget every fiber's index.

        MUST be called by anything that renumbers compact indices -- the index
        is keyed on them, and a stale row->col map points at other neurons'
        columns ([[consolidation-resets-the-index-space]]). Dropping it only
        costs the prune; keeping a wrong one costs the science.
        """
        m = getattr(self, "_pot_support", None)
        if m:
            m.clear()

    def _prune_evaluate_set(self, target, tgt, from_stimuli, from_areas,
                            limit, record_activation=False):
        """Columns that must be gathered exactly, or None to gather all.

        THE DECISION IS MADE BEFORE ANY GATHER, which is what makes this safe
        to wire in without a fallback path. `drive[c] >= stim[c] + corr[c]`
        because the base term is non-negative, so the k-th largest of
        `stim + corr` over the evaluated set is a LOWER bound on the true tau.
        If that already clears `bound_outside`, pruning is provably valid and
        nothing has been computed twice. It declines more often than a
        gather-then-check would, and never guesses.
        """
        if not self._kwta_prune_on() or limit <= 0:
            return None
        # `record_activation` snapshots the FULL drive vector, so a pruned one
        # would hand the caller a partial vector that still looks like a
        # measurement. Checked HERE rather than at the call site so the hit
        # counter is not incremented for a projection that did not prune --
        # a counter that lies makes the guard untestable, which is how a
        # broken guard stays broken.
        if record_activation:
            return None
        k = int(tgt.k)
        if k <= 0:
            return None
        # GUARDS. Each breaks the bound; see `_kwta_prune`'s module docstring.
        if self.norm_init or self.synaptic_scaling:
            return None
        if float(getattr(tgt, "input_noise_std", 0.0) or 0.0) > 0.0:
            return None
        if getattr(tgt, "winner_policy", None) not in (None, "topk"):
            return None

        xp = self._xp
        stim_total = None
        if from_stimuli:
            stim_total = np.zeros(limit, dtype=np.float64)
            for stim in from_stimuli:
                sw = self._stim_conns[stim][target].weights
                end = min(limit, len(sw))
                if end > 0:
                    stim_total[:end] += np.asarray(to_cpu(sw[:end]),
                                                   dtype=np.float64)

        corr = np.zeros(limit, dtype=np.float64)
        touched = []
        total_active = 0
        for src_name in from_areas:
            if not self.fiber_learning_allowed(src_name, target):
                continue
            conn = self._area_conns[src_name][target]
            w = conn.weights
            # Only DENSE blocks carry a maintained index -- the sparse
            # representations answer `row_sum` natively and were never noted.
            if not isinstance(w, xp.ndarray) or getattr(w, "ndim", 0) != 2:
                return None
            src_w = xp.asarray(self._areas[src_name].winners)
            internal = np.asarray(to_cpu(src_w[src_w < w.shape[0]]))
            # COUNT EVERY ACTIVE ROW, not just the ones this block currently
            # covers. `eager_fiber_init` can materialise or widen a block
            # INSIDE the gather loop, after this decision has been taken, and
            # those fresh rows contribute base drive to columns whose bound was
            # computed without them. Counting `src.winners` is an upper bound
            # and therefore always safe; clipping to `w.shape[0]` undercounts
            # and makes the bound too low, which silently drops real winners.
            total_active += int(len(src_w))
            if len(internal) == 0:
                continue
            cols = min(limit, int(w.shape[1]))
            c, t = self._support_for(src_name, target).correction(
                internal, w, cols)
            corr[:cols] += c
            if len(t):
                touched.append(t)
        if total_active == 0:
            return None

        touch = (np.unique(np.concatenate(touched)) if touched
                 else np.empty(0, dtype=np.int64))
        ev = evaluate_set(touch, stim_total, k, limit)
        if len(ev) < k:
            return None

        def _tau(cand):
            lo = corr[cand] + (stim_total[cand]
                               if stim_total is not None else 0.0)
            return float(np.partition(lo, -k)[-k])

        tau_lo = _tau(ev)
        # WIDEN BY THE STIMULUS BEFORE GIVING UP. The base term is bounded by
        # |S| because it is Bernoulli 0/1; the STIMULUS term is not bounded by
        # anything, and on a real organ one symbol drives thousands of columns
        # hard. Evaluating only its top-k therefore leaves `bound_outside`
        # enormous and the prune declines on exactly the workload it was built
        # for -- measured on the Z60 arc: potentiated support 147 columns of
        # 19,999 (0.7%), and it still declined 1,798 times out of 1,798.
        #
        # So take every column whose stimulus alone could still reach tau, in
        # one pass. Widening can only help twice over: more candidates can only
        # raise the k-th largest, and every column moved inside lowers the max
        # left outside.
        if stim_total is not None and tau_lo > total_active:
            need = tau_lo - float(total_active)
            extra = np.nonzero(stim_total[:limit] >= need)[0]
            if len(extra):
                ev = np.union1d(ev, extra.astype(np.int64))
                if len(ev) >= k:
                    tau_lo = _tau(ev)

        # A prune that evaluates most of the area saves nothing and still pays
        # for the decision. Checked AFTER widening, since widening is what
        # decides how big the evaluated set really is.
        if len(ev) >= _PRUNE_MAX_FRACTION * limit or len(ev) < k:
            self._prune_misses = getattr(self, "_prune_misses", 0) + 1
            return None
        if tau_lo > bound_outside(stim_total, ev, limit, total_active):
            self._prune_hits = getattr(self, "_prune_hits", 0) + 1
            return ev.astype(np.int64)
        self._prune_misses = getattr(self, "_prune_misses", 0) + 1
        return None

    def _scale_columns_now(self, target, from_areas, winners):
        check_area_homeostasis(target, refracted=self._areas[target].refracted,
                               synaptic_scaling=True)
        xp = self._xp
        cols = xp.asarray(winners, dtype=xp.int64)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            w = conn.weights
            if w is None or getattr(w, "ndim", 0) != 2 or w.shape[1] == 0:
                continue
            # LAYOUT FOLLOWS THE FIBER'S SHAPE. A scaled fiber pays two strided
            # ops per projection: this function's column gather/scatter over
            # src_rows x k (wants columns contiguous, F-order) and
            # `project_into`'s drive row-gather over k x tgt_cols (wants rows
            # contiguous, C-order). Whichever slice is LARGER should own the
            # contiguity: measured on the organ_p=0.5 pair, the tall 20000x4200
            # arc->state fiber scales at 34 ms in C vs 4 ms in F (scaling
            # dominates, 20000 > 4200), while the wide 4200x20000 state->arc
            # fiber's drive-read was 57% of project_into under blanket F-order
            # (drive dominates, 20000 > 4200 the other way). So: F-order iff
            # rows > cols. `asfortranarray` preserves logical [i,j], so either
            # layout is BYTE-IDENTICAL end-to-end -- proven by identical census
            # and connectome crc at PRES=8 and PRES=24; layout only moves time.
            # Converted lazily and persisted on the connection; growth
            # reallocates C-order (amortised doubling, front-loaded to early
            # recruitment ~log2(n/k) times), and this reconverts on the next
            # touch, so there is no per-step thrash.
            if (isinstance(w, xp.ndarray) and w.shape[0] > w.shape[1]
                    and not w.flags.f_contiguous):
                w = xp.asfortranarray(w)
                conn.weights = w
            valid = cols[cols < w.shape[1]]
            if len(valid) == 0:
                continue
            # Use the LOGICAL source size, not w.shape[0]: physical capacity is
            # amortised (it doubles), so the rows beyond src.w are unallocated
            # padding. Summing or setting the setpoint over them inflates it
            # and silently under-normalizes.
            rows = min(int(self._areas[src_name].w), int(w.shape[0]))
            if rows <= 0:
                continue
            sub = w[:rows, valid]
            sums = sub.sum(axis=0)
            # THE FIBER'S p, NOT THE BRAIN'S -- third member of the defect
            # class ([[pricing-law-implemented-twice]]; _norm_scale 79fba4f,
            # the stimulus w_max clamp). The setpoint is the initial expected
            # column sum OF THIS FIBER; pricing it at the global p renormalized
            # every trained column of a p=0.40 organ fiber inside a p=0.05
            # brain to 1/8 of its natural mass, while untouched columns kept
            # full mass -- inverting learning exactly like substrate B did.
            # Found by the substrate-C smoke run (every transition soft).
            setpoint = scaling_setpoint(rows, self._p_for(src_name, target))
            # RESEARCH KNOB, default "population" = the line above, unchanged.
            # "degree" restores neuron j to the mass IT started with rather
            # than to the mass an average neuron started with.
            #
            # MEASURED AND REFUTED -- kept because a refuted arm is evidence,
            # and because the idea is the obvious one to re-try. It is not a
            # near-miss, it is the worst arm ever measured on this substrate:
            # `seq_scaling_merger_forensics` (n=2000 k=50 p=0.5 T=8 M=8, in
            # regime) gives pairwise overlap 0.667 = 26.7x chance against the
            # population setpoint's 0.188, and breaks retrieval from the FULL
            # cue (rank-1 0.625, where every other arm scores 1.000).
            #
            # The reason is that it is not a normalization at all. Restoring a
            # column to its own initial mass is a no-op on the structure and
            # cancels only the potentiation, so the raw in-degree competition
            # comes back undamped and the hubs win everything -- the collapse
            # `norm_init` exists to prevent ([[recurrence-needs-norm-init]]),
            # and the per-column form of the per-fiber failure this method's
            # own docstring already records.
            if getattr(self, "synaptic_scaling_setpoint", "population") == "degree":
                deg = xp.count_nonzero(sub, axis=0).astype(sums.dtype)
                setpoint = xp.where(deg > 0, deg, 1.0)
            scale = column_scale(sums, setpoint, xp=xp)
            w[:rows, valid] = sub * scale

    def flush_synaptic_scaling(self) -> int:
        """Apply deferred homeostatic scaling to every touched column.

        The slow half of the fast/slow separation (see
        _normalize_area_columns). Setpoints use the areas' CURRENT logical
        sizes -- slow homeostasis regulates the state as it stands at the
        boundary, which is the semantics the timescale argument wants.
        Returns the number of (src, tgt) fibers scaled.
        """
        pending = getattr(self, "_pending_scaling", None)
        if not pending:
            return 0
        n = 0
        for (src_name, target), cols in list(pending.items()):
            if not self.fiber_learning_allowed(src_name, target):
                continue
            self._scale_columns_now(target, [src_name], sorted(cols))
            del pending[(src_name, target)]
            n += 1
        return n

    def _apply_plasticity(self, target, from_stimuli, from_areas, winners):
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-fiber-learning

        Hebbian learning and triggered scaling on permitted fibers only.
        """
        if not self._plasticity_enabled_global:
            return
        from_stimuli = [s for s in from_stimuli if self.fiber_learning_allowed(s, target)]
        from_areas = [s for s in from_areas if self.fiber_learning_allowed(s, target)]
        xp = self._xp
        tgt = self._areas[target]
        winners_arr = xp.asarray(winners, dtype=xp.int64)

        # Stimulus -> area (1-D weights)
        for stim_name in from_stimuli:
            conn = self._stim_conns[stim_name][target]
            beta = tgt.beta_by_source.get(stim_name, tgt.beta)
            if beta == 0:
                continue
            valid = winners_arr[winners_arr < len(conn.weights)]
            if len(valid) > 0:
                conn.weights[valid] *= (1 + beta)
            if self.w_max is not None:
                # w_max means "multiples of the initial weight". Area->area
                # weights start at 1, so the raw cap is correct there. A
                # stimulus connectome instead stores PRE-SUMMED input, which
                # starts at about stim_size * p -- with the defaults that is
                # ~20 for a cap of 20, so the very first update clipped every
                # winner to exactly w_max and pinned it there. Plasticity then
                # had no effect at any beta: assembly recovery measured a flat
                # 0.30 for beta = 0.001, 0.01 and 0.1 alike. Scaling the cap
                # by the initial magnitude restores the intended semantics.
                stim = self._stimuli.get(stim_name)
                # THIS FIBER's density, not the global one. A stimulus
                # connectome stores PRE-SUMMED input starting near
                # ``size * p``, so the clamp has to be scaled by the same p the
                # weights were DRAWN at. Using the global p clamped a dense
                # fiber inside a sparse brain at its sparse ceiling: drawn at
                # p=0.4 the weights start near 28 but were clipped at
                # w_max * 70 * 0.05 = 70, so 15 presentations of Hebbian growth
                # (1.1^15 = 4.18, reaching ~117) saturated instead. A saturated
                # conjunct cannot discriminate, and the mod-3 FSM fell from
                # 10/10 correct trajectories to 1/10.
                scale = max(1.0, float(getattr(stim, "size", 1))
                            * self._p_for(stim_name, target))
                lo, hi = self._weight_bounds(scale)
                xp.clip(conn.weights, lo, hi, out=conn.weights)

        # Area -> area (2-D weights)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            beta = tgt.beta_by_source.get(src_name, tgt.beta)
            if beta == 0:
                continue
            src = self._areas[src_name]
            src_w = xp.asarray(src.winners)
            if not conn.sparse and getattr(src, "explicit_source", False):
                valid_rows = src_w[src_w < conn.weights.shape[0]]
                post_ids = [
                    tgt.compact_to_neuron_id[int(c)]
                    for c in winners
                    if int(c) < len(tgt.compact_to_neuron_id)
                ]
                if len(valid_rows) > 0 and len(post_ids) > 0:
                    conn.update_weights(
                        to_cpu(valid_rows), post_ids, beta, w_max=self.w_max)
                continue
            if isinstance(conn.weights, VirtualWeights):
                vw = conn.weights
                valid_rows = src_w[src_w < vw.n_rows]
                valid_cols = winners_arr[winners_arr < vw.n_cols]
                if len(valid_rows) > 0 and len(valid_cols) > 0:
                    try:
                        vw.bump(np.asarray(to_cpu(valid_rows)),
                                np.asarray(to_cpu(valid_cols)), float(beta))
                    except ValueError:
                        # Beta changed mid-life: one exponent store cannot
                        # carry two growth factors, so DENSIFY -- correct
                        # under any beta history -- and apply this event on
                        # the dense block below.
                        conn.weights = self._to_xp(vw.todense())
                        conn._deg_counts_arr = None
                        conn._deg_rows = 0
                        ix = xp.ix_(valid_rows, valid_cols)
                        conn.weights[ix] *= (1 + beta)
                        if self.w_max is not None:
                            sub = conn.weights[ix]
                            _lo, _hi = self._weight_bounds()
                            xp.clip(sub, _lo, _hi, out=sub)
                            conn.weights[ix] = sub
                continue
            if conn.weights.ndim == 2:
                valid_rows = src_w[src_w < conn.weights.shape[0]]
                valid_cols = winners_arr[winners_arr < conn.weights.shape[1]]
                if len(valid_rows) > 0 and len(valid_cols) > 0:
                    ix = xp.ix_(valid_rows, valid_cols)
                    conn.weights[ix] *= (1 + beta)
                    # Record the touched support for the k-WTA prune. This is
                    # the ONLY place a dense fiber learns which cells are
                    # potentiated -- the block stores their values but not
                    # their index, and recovering it later would cost exactly
                    # the O(k*n) scan the prune exists to avoid. O(k^2) here
                    # against O(k*n) there.
                    if self._kwta_prune_on():
                        self._support_for(src_name, target).note(
                            np.asarray(to_cpu(valid_rows)),
                            np.asarray(to_cpu(valid_cols)))
                    if self.w_max is not None:
                        sub = conn.weights[ix]
                        _lo, _hi = self._weight_bounds()
                        xp.clip(sub, _lo, _hi, out=sub)
                        conn.weights[ix] = sub
            else:
                valid = winners_arr[winners_arr < len(conn.weights)]
                if len(valid) > 0:
                    conn.weights[valid] *= (1 + beta)
                if self.w_max is not None:
                    _lo, _hi = self._weight_bounds()
                    xp.clip(conn.weights, _lo, _hi, out=conn.weights)

        # Homeostatic scaling closes the loop on the update just applied.
        # (Runs BEFORE connectome expansion, so a first-time winner's
        # freshly-expanded column can sit above the setpoint until its next
        # update -- pinned in test_scoped_synaptic_scaling.py.)
        self._normalize_area_columns(target, from_areas, winners)

    # -- Connectome expansion for new winners --------------------------------









    def get_winners(self, area: str) -> np.ndarray:
        st = self._areas[area]
        return np.array(to_cpu(st.winners), dtype=np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-winner-inputs"""
        xp = self._xp
        st = self._areas[area]
        st.winners = validated_indices(winners, upper=st.n, label=f"{area} winners",
                                       xp=xp, unique=True)

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].w

    def get_neuron_id_mapping(self, area: str) -> list:
        """Return the compact_to_neuron_id list for stable winner IDs."""
        return self._areas[area].compact_to_neuron_id

    # -- Materialization (see ComputeEngine.fiber_extent for the rationale) --

    def materialized_count(self, area: str):
        st = self._areas.get(area)
        return None if st is None else int(st.w)

    def fiber_extent(self, source: str, target: str):
        """Logical column watermark of ``source -> target``.

        Returns ``None`` when the fiber is dense (every column exists, so the
        watermark is vacuous) or absent, and the integer watermark when it is
        lazily materialized. The PHYSICAL shape is deliberately not returned:
        growth doubles capacity, so it over-runs the logical content and
        columns past the watermark are allocated-but-uninitialised zeros.
        """
        conn = self._area_conns.get(source, {}).get(target)
        if conn is None or not getattr(conn, "sparse", False):
            return None
        w = getattr(conn, "weights", None)
        if w is None or getattr(w, "ndim", 0) != 2:
            return None
        return int(min(getattr(conn, "_log_cols", w.shape[1]), w.shape[1]))

    # -- Projection fidelity ------------------------------------------------

    def set_projection_fidelity(self, fidelity: str) -> None:
        self._projection_fidelity = ProjectionFidelity.normalize(fidelity)

    def get_projection_fidelity(self) -> str:
        return self._projection_fidelity.value

    def preallocate_stim_targets(self, target: str, min_columns: int) -> None:
        """Extend all stim→*target* vectors to at least *min_columns* (zeros)."""
        if min_columns <= 0 or target not in self._areas:
            return
        xp = self._xp
        n = self._areas[target].n
        for stim_name, tgt_map in self._stim_conns.items():
            conn = tgt_map.get(target)
            if conn is None or not conn.sparse:
                continue
            old = len(conn.weights)
            if min_columns > old:
                if self._stim_fastpath:
                    self._grow_stim_vector(conn, n, old, min_columns, None)
                else:
                    add = xp.zeros(min_columns - old, dtype=xp.float32)
                    conn.weights = (
                        xp.concatenate([conn.weights, add])
                        if old > 0 else add
                    )

    def _use_compiled_projection(self, tgt) -> bool:
        """True when compiled/fuzzy top-k on pregrown columns should run."""
        if not getattr(tgt, "_freeze_connectome_growth", False):
            return False
        if getattr(tgt, "_force_exact_projection", False):
            return False
        if tgt.w < tgt.k:
            return False
        if getattr(tgt, "_plasticity_only_mode", False):
            return True
        return self._projection_fidelity == ProjectionFidelity.COMPILED

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
        if st.winners is None or (hasattr(st.winners, '__len__') and len(st.winners) == 0):
            raise ValueError(f"Area {area} has no winners to fix.")
        st.fixed_assembly = True

    def unfix_assembly(self, area: str) -> None:
        self._areas[area].fixed_assembly = False

    def is_fixed(self, area: str) -> bool:
        return self._areas[area].fixed_assembly

    # -- Connection reset ---------------------------------------------------

    def reset_area_connections(self, area: str) -> None:
        """Reset area->area connections involving *area* to initial state."""
        self.invalidate_csr_drive()
        xp = self._xp
        for src_name in list(self._area_conns.keys()):
            if area not in self._area_conns[src_name]:
                continue
            conn = self._area_conns[src_name][area]
            if conn.sparse:
                conn.weights = xp.empty((0, 0), dtype=xp.float32)
                if hasattr(conn, '_log_rows'):
                    del conn._log_rows
                if hasattr(conn, '_log_cols'):
                    del conn._log_cols
            else:
                rows, cols = conn.weights.shape
                conn.weights = xp.asarray(
                    (self._rng.random((rows, cols)) < self.p
                     ).astype(np.float32),
                )

    # -- LRI control --------------------------------------------------------

    def clear_refractory(self, area: str) -> None:
        """Clear refractory history for an area."""
        self._areas[area]._refractory_history.clear()

    def set_lri(self, area: str, refractory_period: int,
                inhibition_strength: float) -> None:
        """Update LRI parameters for an area at runtime."""
        from collections import deque
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
        """Enable or disable refracted mode for an area."""
        st = self._areas[area]
        check_area_homeostasis(area, refracted=enabled, synaptic_scaling=self.synaptic_scaling)
        st.refracted = enabled
        st.refracted_strength = strength
        if enabled and len(st._cumulative_bias) == 0:
            xp = self._xp
            st._cumulative_bias = xp.zeros(max(st.w, 0), dtype=xp.float32)

    def clear_refracted_bias(self, area: str) -> None:
        """Reset accumulated refracted bias to zero."""
        xp = self._xp
        st = self._areas[area]
        st._cumulative_bias = xp.zeros(max(st.w, 0), dtype=xp.float32)

    # -- Weight normalization -----------------------------------------------

    def normalize_weights(self, target: str, source: str = None) -> None:
        """Column-normalize weights into *target* so each neuron sums to 1.0."""
        check_area_homeostasis(target, refracted=self._areas[target].refracted,
                               synaptic_scaling=True)
        self.invalidate_csr_drive()
        xp = self._xp
        eps = 1e-8

        def _norm_conn(conn):
            w = conn.weights
            if w.ndim == 2 and w.size > 0:
                col_sums = w.sum(axis=0, keepdims=True)
                col_sums = xp.maximum(col_sums, eps)
                conn.weights = w / col_sums
            elif w.ndim == 1 and w.size > 0:
                total = float(xp.sum(w))
                if total > eps:
                    conn.weights = w / total

        if source is not None:
            if source in self._stim_conns and target in self._stim_conns[source]:
                _norm_conn(self._stim_conns[source][target])
            if source in self._area_conns and target in self._area_conns[source]:
                _norm_conn(self._area_conns[source][target])
            return

        for stim_name in self._stim_conns:
            if target in self._stim_conns[stim_name]:
                _norm_conn(self._stim_conns[stim_name][target])
        for src_name in self._area_conns:
            if target in self._area_conns[src_name]:
                _norm_conn(self._area_conns[src_name][target])

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
