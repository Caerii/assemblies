"""NumpySparseEngine: CPU engine using statistical sparse simulation.

This is the extraction of brain.py's ``_project_into_legacy`` sparse path.
Connectivity is stored as growing 1-D (stim->area) and 2-D (area->area)
numpy arrays.  Statistical sampling (truncated normal, binomial PPF)
generates candidate activations for new neurons.
"""

import os
import zlib

import numpy as np
from typing import Dict, List
from collections import defaultdict

from ..backend import get_xp, to_cpu, to_xp
from ..engine import ComputeEngine, ProjectionResult
from ..connectome import Connectome
from ..projection_fidelity import ProjectionFidelity

try:
    from ...compute.sparse_simulation import SparseSimulationEngine
    from ...compute.winner_selection import WinnerSelector
    from ...compute.winner_policies import TopKPolicy
except ImportError:
    from compute.sparse_simulation import SparseSimulationEngine
    from compute.winner_selection import WinnerSelector
    from compute.winner_policies import TopKPolicy

from ._state import SparseAreaState, StimulusState


def _warn_fixed_target_enabled() -> bool:
    """Whether to warn on a plasticity-bearing projection INTO a fixed area.

    Off by default (behaviour-neutral); set ``ASSEMBLIES_WARN_FIXED_TARGET=1``
    to surface the silent-no-op footgun during development. See the check at
    the fixed-assembly short-circuit in ``project_into``.
    """
    return os.environ.get("ASSEMBLIES_WARN_FIXED_TARGET", "").strip().lower() in (
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


def stable_seed(*parts) -> int:
    """A 32-bit seed from `parts` that is identical in every process.

    MUST be used instead of ``hash(...)`` for anything that seeds an RNG.
    Python randomizes ``hash()`` of str/bytes per process (PEP 456), so
    ``hash((src, tgt, nr, nc))`` is stable WITHIN a run and different across
    runs. Three lazy-connectome sites here seeded ``default_rng`` that way and
    were commented "deterministic per-pair seed" -- they were not, and the
    result was that `Brain(seed=42)` trained different weights from one process
    to the next. Measured on the Geschwind lesion study: ~1/3 of PYTHONHASHSEED
    values changed the reported accuracy (1.00 vs 0.50), and one crashed.
    """
    return zlib.crc32(repr(parts).encode("utf-8")) & 0xFFFFFFFF

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


class NumpySparseEngine(ComputeEngine):
    """CPU engine using statistical sparse simulation.

    Connectivity is stored as growing 1-D (stim->area) and 2-D (area->area)
    numpy arrays.  Statistical sampling (truncated normal, binomial PPF)
    generates candidate activations for new neurons.

    Parameters mirror ``Brain.__init__``.
    """

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 deterministic: bool = False,
                 projection_fidelity: str = ProjectionFidelity.EXACT.value,
                 inhibitory_prob: float = 0.0,
                 inhibitory_weight: float = -0.2,
                 synaptic_scaling: bool = False,
                 norm_init: bool = False):
        self.p = p
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
        self.synaptic_scaling = synaptic_scaling
        # One-time incoming-weight normalization (Dabagia et al. reference
        # `norm_init`).  See _norm_scale for the lazy-materialization
        # formulation and why it is exactly equivalent.
        self.norm_init = norm_init
        self._deterministic = deterministic
        self._rng = np.random.default_rng(seed)
        self._plasticity_enabled_global = True
        self._projection_fidelity = ProjectionFidelity.normalize(projection_fidelity)

        # Internal state
        self._areas: Dict[str, SparseAreaState] = {}
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

        # Reusable math primitives
        self._sparse_sim = SparseSimulationEngine(self._rng)
        self._winner_sel = WinnerSelector(self._rng)

    def _weight_bounds(self, scale: float = 1.0):
        """Clip bounds for Hebbian updates, as (low, high).

        The low bound is 0 only when there are no inhibitory synapses. With
        feedforward inhibition (Hoff et al. Eq. 7) weights may legitimately be
        negative, and clipping at 0 would erase every inhibitory synapse the
        first time plasticity touched it -- silently disabling the mechanism.
        Eq. 3 potentiates inhibitory synapses too (they grow more negative),
        so the negative side gets the mirrored bound.
        """
        hi = self.w_max * scale
        if self.inhibitory_prob <= 0.0:
            return 0.0, hi
        return -abs(self.inhibitory_weight) * self.w_max * scale, hi

    def _sample_area_weights(self, shape, rng):
        """Initial area->area weights, with optional feedforward inhibition.

        Each present synapse (probability ``p``) is excitatory (weight 1) with
        probability ``1 - inhibitory_prob`` or inhibitory (``inhibitory_weight``)
        otherwise. With ``inhibitory_prob == 0`` this is the original binomial
        0/1 connectome, bit-for-bit.
        """
        present = rng.random(shape) < self.p
        if self.inhibitory_prob <= 0.0:
            return present.astype(np.float32)
        w = present.astype(np.float32)
        inh = present & (rng.random(shape) < self.inhibitory_prob)
        w[inh] = self.inhibitory_weight
        return w

    # -- norm_init: one-time incoming-weight normalization -------------------

    def _norm_scale(self, conn, n_pre: int, rows_known: int, needed: int):
        """Per-postsynaptic-neuron read-time scale ``1/d_j`` for one fiber.

        Reproduces the reference implementation's ``norm_init``
        (``.reference/mdabagia-nemo/brain.py``: ``FFArea.normalize`` /
        ``RecurrentArea.normalize``, which are called ONLY from ``reset()`` --
        a ONE-TIME initialization, not ongoing homeostasis).  There, each
        fiber's weight matrix is divided by its own column sums exactly once at
        init, when every present weight is 1.  The column sum is therefore the
        postsynaptic neuron's IN-DEGREE ``d_j``, and normalization is precisely
        "initialize every incoming synapse of neuron j to ``1/d_j``".

        WHY A READ-TIME SCALE RATHER THAN SCALED WEIGHTS.  This engine
        materializes neurons lazily, so neuron j's full incoming column does
        not exist when it would need to be divided.  But plasticity here is
        purely MULTIPLICATIVE (``w *= 1 + beta``), so

            (w_0 / d_j) * prod_t (1 + beta_t) == (w_0 * prod_t (1 + beta_t)) / d_j

        i.e. dividing a neuron's *summed drive* by a per-neuron constant is
        algebraically identical to having initialized its incoming weights at
        ``1/d_j``.  Storage therefore stays on the unit scale -- which also
        preserves ``w_max``'s "multiples of the initial weight" semantics and
        keeps the lazily *sampled* candidate drive commensurable -- and the
        division happens at read time.

        HOW ``d_j`` IS OBTAINED.  Sampling ``d_j ~ Binomial(n_pre, p)``
        independently of the neuron's realized wiring was tried first and does
        nothing (measured: winner in-degree z 1.66 -> 1.53, overlap
        0.940 -> 0.916).  It cannot work: an independent draw does not cancel
        the neuron's ACTUAL degree advantage, it only adds noise.  The
        reference divides by the neuron's OWN column sum, so the divisor must
        track realized wiring.  Here that is

            d_j = (present synapses over the rows that exist so far)
                  + p * (rows that do not exist yet)

        which is an unbiased running estimate of the full-population in-degree:
        each row that later materializes contributes a present synapse with
        probability p, exactly the rate the second term assumes, so ``d_j``
        stays centred on ``n_pre * p`` while tracking each neuron's own
        realized excess.  Present synapses are COUNTED, not summed, so the
        divisor is potentiation-invariant, matching the reference's
        take-it-once-at-init semantics.

        For STIMULUS fibers ``n_pre`` is the TARGET area's ``n``, not the
        stimulus size.  The reference's inputs are areas of size n with only a
        cap of k neurons active, so every fiber delivers drive of order k/n.
        This engine's stimuli instead fire in full and are stored pre-summed,
        so dividing by their own in-degree would set every neuron's stimulus
        drive to exactly 1.0 -- a constant, which erases the stimulus
        representation entirely (constants do not affect top-k).  Treating a
        stimulus of size s as the active cap of an implicit input population of
        size n restores the reference's geometry: stimulus drive ~ s/n,
        recurrent drive ~ k/n, so the fibers compete on equal terms and the
        stimulus keeps its per-neuron identity.

        Returns ``None`` when norm_init is off or there is nothing to scale.
        """
        if not self.norm_init:
            return None
        xp = get_xp()
        w = conn.weights
        if w is None or getattr(w, "size", 0) == 0:
            return None

        if getattr(w, "ndim", 1) == 2:
            rows = int(min(rows_known, w.shape[0]))
            cols = int(min(needed, w.shape[1]))
            if cols <= 0:
                return None
            key = (rows, cols)
            if getattr(conn, "_norm_deg_key", None) != key:
                conn._norm_deg = xp.asarray(
                    (w[:rows, :cols] != 0).sum(axis=0), dtype=xp.float32)
                conn._norm_deg_key = key
            deg = conn._norm_deg[:cols]
            unknown = max(int(n_pre) - rows, 0)
        else:
            cols = int(min(needed, len(w)))
            if cols <= 0:
                return None
            # 1-D stimulus fiber: the stored value IS the observed in-degree
            # from the stimulus population -- but only until plasticity scales
            # it, so snapshot each column the first time it is seen (a fresh
            # column is always read before it is ever potentiated).
            base = getattr(conn, "_norm_deg_base", None)
            have = 0 if base is None else len(base)
            if have < cols:
                add = xp.asarray(w[have:cols], dtype=xp.float32)
                base = add if base is None or have == 0 else xp.concatenate(
                    [base, add])
                conn._norm_deg_base = base
            deg = base[:cols]
            unknown = max(int(n_pre) - int(rows_known), 0)

        d = deg + xp.float32(unknown * self.p)
        return 1.0 / xp.maximum(d, xp.float32(1.0))

    def _norm_candidate_divisor(self, tgt) -> float:
        """Scale factor applied to sampled (unmaterialized) candidate drive.

        Candidates are sampled on the unit-weight scale as top order statistics
        of ``Binomial(sum(input_sizes), p)``.  Under norm_init a materialized
        neuron's drive is that same count divided by its in-degree, whose mean
        is ``n * p``.  Dividing candidates by ``n * p`` puts both populations
        on one scale.  Candidate in-degree *variance* is deliberately not
        modelled: an unmaterialized neuron has no persistent in-degree yet (it
        is drawn only when the neuron materializes), which is exactly the
        "no pre-existing hubs" property norm_init is here to enforce.
        """
        return max(float(tgt.n) * self.p, 1e-12)

    # -- Registration -------------------------------------------------------

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0,
                 winner_policy=None,
                 input_noise_std: float = 0.0) -> None:
        xp = get_xp()
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
            conn = Connectome(stim.size, n, self.p, sparse=True)
            conn.weights = xp.empty(0, dtype=xp.float32)
            self._stim_conns[stim_name][name] = conn
            area.beta_by_source[stim_name] = beta

        # Initialize area->area connectomes (both directions) for every existing area
        for other_name, other in self._areas.items():
            if other_name == name:
                self_conn = Connectome(n, n, self.p, sparse=True)
                self_conn.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[name][name] = self_conn
            else:
                conn_fwd = Connectome(other.n, n, self.p, sparse=True)
                conn_fwd.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[other_name][name] = conn_fwd

                conn_rev = Connectome(n, other.n, self.p, sparse=True)
                conn_rev.weights = xp.empty((0, 0), dtype=xp.float32)
                self._area_conns[name][other_name] = conn_rev

                area.beta_by_source[other_name] = beta
                other.beta_by_source[name] = beta

    def add_stimulus(self, name: str, size: int) -> None:
        xp = get_xp()
        self._stimuli[name] = StimulusState(name=name, size=size)
        self._stim_conn_version += 1

        # Initialize stim->area for every already-registered area.
        # If the area already has ever-fired neurons (w > 0), create a
        # weight vector of length w with random Bernoulli(p) connections
        # so the stimulus can compete with existing trained connections.
        # Without this, the empty weight vector produces zero input and
        # the projection short-circuits, making online learning impossible.
        for area_name, area in self._areas.items():
            conn = Connectome(size, area.n, self.p, sparse=True)
            if area.w > 0:
                rng = np.random.default_rng(
                    stable_seed(name, area_name, area.w))
                conn.weights = to_xp(
                    (rng.random(area.w) < self.p).astype(np.float32)
                    * size  # scale by stimulus size for fair competition
                )
            else:
                conn.weights = xp.empty(0, dtype=xp.float32)
            self._stim_conns[name][area_name] = conn
            area.beta_by_source[name] = area.beta

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        pass

    def _select_winner_indices(self, tgt, all_inputs, rng, population_sigma=None):
        """Select winner indices using area policy (default top-k)."""
        from ...compute.winner_policies import TopKPolicy

        xp = get_xp()
        inputs = all_inputs
        if getattr(tgt, "input_noise_std", 0.0) > 0:
            noise = rng.normal(0, tgt.input_noise_std, size=len(inputs))
            inputs = inputs + to_xp(noise.astype(np.float32))

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
        xp = get_xp()
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
        tgt.compact_to_neuron_id = list(neuron_ids)
        if tgt.neuron_id_pool is not None:
            tgt.neuron_id_pool_ptr = len(neuron_ids)

        if plasticity_enabled and self._plasticity_enabled_global:
            for src_name in from_areas:
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

        lazy_seed = stable_seed(src_name, target, nr, nc)
        lazy_rng = np.random.default_rng(lazy_seed)
        fresh = to_xp(self._sample_area_weights((nr, nc), lazy_rng))

        # Preserve everything already learned. Overwriting the whole block
        # would discard the accumulated Hebbian weights every time the target
        # recruited a neuron, so a pathway could never accumulate strength
        # across training -- only the newly added rows/columns are novel.
        if cur is not None and cr > 0 and cc > 0:
            fresh[:cr, :cc] = cur[:cr, :cc]
        conn.weights = fresh
        return True

    # -- Projection ---------------------------------------------------------

    def project_into(
        self,
        target: str,
        from_stimuli: List[str],
        from_areas: List[str],
        plasticity_enabled: bool = True,
        record_activation: bool = False,
    ) -> ProjectionResult:
        xp = get_xp()
        tgt = self._areas[target]
        rng = np.random.default_rng(self._rng.integers(0, 2**32))

        # Filter out source areas with no assembly
        from_areas = [
            a for a in from_areas
            if self._areas[a].winners.size > 0
            and (
                self._areas[a].w > 0
                or getattr(self._areas[a], "explicit_source", False)
            )
        ]

        # Fixed assembly — short-circuit. The inputs are SILENTLY DISCARDED and
        # no plasticity is applied: a fixed area holds its assembly and ignores
        # drive. That is correct for holding a SOURCE steady, but projecting
        # WITH inputs INTO a fixed area is almost always a caller bug -- it
        # looks like training but writes nothing. This exact footgun produced
        # three separate "the mechanism doesn't work" investigations (merge,
        # associate, direct binding). The guard below makes it loud when opted
        # in (ASSEMBLIES_WARN_FIXED_TARGET=1); default behaviour is unchanged.
        if tgt.fixed_assembly:
            if (plasticity_enabled and (from_stimuli or from_areas)
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
                    stim_conn, tgt.n, self._stimuli[stim].size, end)
                if nscale is None:
                    prev_winner_inputs[:end] += stim_w[:end]
                else:
                    prev_winner_inputs[:end] += stim_w[:end] * nscale[:end]

        # Area inputs (2-D, vectorised fancy-index)
        # Track sources whose connectomes need deferred initialisation.
        _deferred_init_srcs = []
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            src = self._areas[src_name]
            if not conn.sparse and getattr(src, "explicit_source", False):
                src_w = xp.asarray(src.winners)
                valid = src_w[src_w < conn.weights.shape[0]]
                if len(valid) == 0:
                    continue
                if tgt.w == 0:
                    contrib = conn.weights[valid].sum(axis=0)
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
                    end = min(limit, len(contrib))
                    if end > 0:
                        prev_winner_inputs[:end] += contrib[:end]
                continue
            if conn.weights.shape[1] == 0:
                # Mark cross-area connections for deferred init so they
                # are available on the NEXT projection round.
                if (conn.sparse and src_name != target
                        and self._areas[src_name].w > 0 and tgt.w > 0):
                    _deferred_init_srcs.append(src_name)
                continue
            src_w = xp.asarray(src.winners)
            internal = src_w[src_w < conn.weights.shape[0]]
            if len(internal) > 0 and limit > 0:
                col_end = min(limit, conn.weights.shape[1])
                contrib = conn.weights[internal, :col_end].sum(axis=0)
                nscale = self._norm_scale(
                    conn, self._areas[src_name].n,
                    self._areas[src_name].w, col_end)
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
        if len(prev_winner_inputs) > 0 and float(xp.sum(prev_winner_inputs)) == 0.0:
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
            new_winner_indices = self._winner_sel.heapq_select_top_k(
                inputs_slice, tgt.k,
            )
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
        input_sizes = (
            [self._stimuli[s].size for s in from_stimuli]
            + [(int(self._areas[a].winners.size) if self.norm_init
                else self._areas[a].k) for a in from_areas]
        )

        old_rng = self._sparse_sim.rng
        self._sparse_sim.rng = rng
        if self._deterministic:
            potential_new = self._sparse_sim.sample_new_winner_inputs_legacy(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p,
            )
        else:
            potential_new = self._sparse_sim.sample_new_winner_inputs(
                input_sizes, tgt.n, tgt.w, tgt.k, self.p,
            )
        self._sparse_sim.rng = old_rng

        potential_new = to_xp(potential_new)
        # norm_init: bring sampled candidates onto the normalized scale (see
        # _norm_candidate_divisor).  Stored weights and the sampler stay on the
        # unit scale; only the drive comparison is rescaled.
        norm_div = self._norm_candidate_divisor(tgt) if self.norm_init else None
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
        if tgt.refracted and tgt._cumulative_bias is not None:
            bias = tgt._cumulative_bias
            end = min(len(bias), len(all_inputs))
            if end > 0:
                all_inputs[:end] -= bias[:end]

        # --- Snapshot full all_inputs before top-k ---
        if record_activation:
            _pre_kwta_snapshot = np.array(to_cpu(all_inputs),
                                          dtype=np.float32, copy=True)
            _pre_kwta_total_val = float(xp.sum(all_inputs))

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
        pop_sigma = float(np.sqrt(sum(sz * self.p * (1.0 - self.p)
                                      for sz in input_sizes))) or None
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
        if tgt.refracted and tgt.refracted_strength > 0:
            if len(tgt._cumulative_bias) < new_w:
                old = tgt._cumulative_bias
                tgt._cumulative_bias = xp.zeros(new_w, dtype=xp.float32)
                if len(old) > 0:
                    tgt._cumulative_bias[:len(old)] = old
            for cidx in new_winner_indices:
                if cidx < len(tgt._cumulative_bias):
                    tgt._cumulative_bias[cidx] += tgt.refracted_strength

        total_act = float(xp.sum(all_inputs[new_winner_indices]))

        # --- Deferred connectome initialisation --------------------------------
        # Sources whose connectomes were empty this round get initialised now
        # so they can contribute signal on the NEXT projection round.  Uses a
        # deterministic per-pair seed to avoid disturbing the main RNG.
        if _deferred_init_srcs:
            for src_name in _deferred_init_srcs:
                conn = self._area_conns[src_name][target]
                if conn.weights.shape[1] > 0:
                    continue  # already sized by expand_connectomes
                src = self._areas[src_name]
                nr, nc = src.w, new_w
                if nr > 0 and nc > 0:
                    lazy_seed = stable_seed(src_name, target, nr, nc)
                    lazy_rng = np.random.default_rng(lazy_seed)
                    conn.weights = to_xp(
                        self._sample_area_weights((nr, nc), lazy_rng)
                    )

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
        if not self.synaptic_scaling:
            return
        xp = get_xp()
        cols = xp.asarray(winners, dtype=xp.int64)
        for src_name in from_areas:
            conn = self._area_conns[src_name][target]
            w = conn.weights
            if w is None or getattr(w, "ndim", 0) != 2 or w.shape[1] == 0:
                continue
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
            setpoint = max(float(rows) * self.p, 1e-12)
            # Guard the denominator itself; xp.where evaluates both branches,
            # so dividing first would still emit divide-by-zero on empty cols.
            safe = xp.where(xp.abs(sums) > 1e-12, sums, 1.0)
            scale = xp.where(xp.abs(sums) > 1e-12, setpoint / safe, 1.0)
            w[:rows, valid] = sub * scale

    def _apply_plasticity(self, target, from_stimuli, from_areas, winners):
        """Hebbian learning: w *= (1 + beta), clamped at w_max."""
        xp = get_xp()
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
                scale = max(1.0, float(getattr(stim, "size", 1)) * self.p)
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
            if conn.weights.ndim == 2:
                valid_rows = src_w[src_w < conn.weights.shape[0]]
                valid_cols = winners_arr[winners_arr < conn.weights.shape[1]]
                if len(valid_rows) > 0 and len(valid_cols) > 0:
                    ix = xp.ix_(valid_rows, valid_cols)
                    conn.weights[ix] *= (1 + beta)
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

    # -- Connectome expansion for new winners --------------------------------

        # Homeostatic scaling closes the loop on the update just applied.
        self._normalize_area_columns(target, from_areas, winners)

    def _expand_connectomes(self, target, from_stimuli, from_areas,
                            input_sizes, winners, first_winner_inputs, new_w):
        """Expand connectivity for first-time winners.

        Uses amortised buffer growth for 2-D area->area matrices: physical
        capacity doubles when exceeded, avoiding repeated vstack/hstack
        reallocation on every step.
        """
        xp = get_xp()
        tgt = self._areas[target]
        inputs_names = list(from_stimuli) + list(from_areas)

        prior_w = tgt.w
        new_indices = [int(w) for w in winners if int(w) >= prior_w]
        if not new_indices:
            return

        splits_per_new = self._sparse_sim.compute_input_splits(
            input_sizes, first_winner_inputs,
        )

        if getattr(tgt, '_freeze_connectome_growth', False):
            area_names = [name for name in inputs_names if name in self._areas]
            can_freeze = True
            for src_name in area_names:
                conn = self._area_conns[src_name][target]
                if not conn.sparse or conn.weights.ndim != 2:
                    continue
                phys_rows, phys_cols = conn.weights.shape
                src = self._areas[src_name]
                src_w_arr = xp.asarray(src.winners)
                max_src_idx = (
                    int(xp.max(src_w_arr)) + 1 if src_w_arr.size > 0 else 0
                )
                needed_rows = max(
                    max_src_idx,
                    new_w if src_name == target else src.w,
                )
                needed_cols = new_w
                if needed_rows > phys_rows or needed_cols > phys_cols:
                    can_freeze = False
                    break
            if can_freeze:
                for src_name in area_names:
                    conn = self._area_conns[src_name][target]
                    if not conn.sparse or conn.weights.ndim != 2:
                        continue
                    _, phys_cols = conn.weights.shape
                    self._write_area_expansion_edges(
                        src_name, target, conn, inputs_names,
                        new_indices, splits_per_new, prior_w, phys_cols,
                    )
                return

        # --- Expand stim->area 1-D vectors ---
        stim_names = [name for name in inputs_names if name in self._stimuli]
        area_names = [name for name in inputs_names if name in self._areas]

        if new_w > prior_w:
            if self._stim_fastpath:
                self._expand_stim_vectors_fast(target, tgt, stim_names, new_w)
            else:
                self._expand_stim_vectors_legacy(target, stim_names, new_w)

        # Write allocations for firing stimuli
        for idx, win in enumerate(new_indices):
            if win >= new_w:
                continue
            split = splits_per_new[idx] if idx < len(splits_per_new) else None
            if split is None:
                continue
            for j, name in enumerate(inputs_names):
                alloc = int(split[j])
                if name in self._stimuli:
                    conn = self._stim_conns[name][target]
                    if conn.sparse and win < len(conn.weights):
                        conn.weights[win] = alloc

        # --- Expand area->area 2-D matrices ---
        for src_name in area_names:
            conn = self._area_conns[src_name][target]
            if not conn.sparse:
                continue
            src = self._areas[src_name]
            if conn.weights.ndim != 2:
                conn.weights = xp.empty((0, 0), dtype=xp.float32)

            phys_rows, phys_cols = conn.weights.shape
            src_w_arr = xp.asarray(src.winners)
            max_src_idx = (int(xp.max(src_w_arr)) + 1) if src_w_arr.size > 0 else 0
            needed_rows = max(max_src_idx, new_w if src_name == target else src.w)
            needed_cols = new_w

            if self._deterministic:
                if needed_rows > phys_rows:
                    nr = needed_rows - phys_rows
                    new_rows = to_xp(
                        self._sample_area_weights((nr, phys_cols), self._rng)
                    )
                    conn.weights = xp.vstack([conn.weights, new_rows]) if phys_cols > 0 else xp.zeros((needed_rows, 0), dtype=xp.float32)
                    phys_rows = needed_rows
                if needed_cols > phys_cols:
                    nc = needed_cols - phys_cols
                    new_cols = to_xp(
                        self._sample_area_weights((phys_rows, nc), self._rng)
                    )
                    conn.weights = xp.hstack([conn.weights, new_cols]) if phys_rows > 0 else xp.zeros((0, needed_cols), dtype=xp.float32)
                    phys_cols = needed_cols
            else:
                log_rows = getattr(conn, '_log_rows', phys_rows)
                log_cols = getattr(conn, '_log_cols', phys_cols)
                if log_rows > phys_rows or log_cols > phys_cols:
                    log_rows = min(log_rows, phys_rows)
                    log_cols = min(log_cols, phys_cols)

                new_pr, new_pc = phys_rows, phys_cols
                need_realloc = False
                if needed_rows > phys_rows:
                    new_pr = max(needed_rows, phys_rows * 2, 2 * src.k)
                    need_realloc = True
                if needed_cols > phys_cols:
                    new_pc = max(needed_cols, phys_cols * 2, 2 * tgt.k)
                    need_realloc = True

                if need_realloc:
                    buf = xp.zeros((new_pr, new_pc), dtype=xp.float32)
                    if phys_rows > 0 and phys_cols > 0:
                        buf[:phys_rows, :phys_cols] = conn.weights
                    conn.weights = buf
                    phys_rows, phys_cols = new_pr, new_pc

                nr = needed_rows - log_rows
                nc = needed_cols - log_cols
                if nr > 0 and log_cols > 0:
                    conn.weights[log_rows:needed_rows, :log_cols] = to_xp(
                        self._sample_area_weights((nr, log_cols), self._rng)
                    )
                if nc > 0 and log_rows > 0:
                    conn.weights[:log_rows, log_cols:needed_cols] = to_xp(
                        self._sample_area_weights((log_rows, nc), self._rng)
                    )
                if nr > 0 and nc > 0:
                    conn.weights[log_rows:needed_rows, log_cols:needed_cols] = to_xp(
                        self._sample_area_weights((nr, nc), self._rng)
                    )

                conn._log_rows = max(getattr(conn, '_log_rows', 0), needed_rows)
                conn._log_cols = max(getattr(conn, '_log_cols', 0), needed_cols)

            # -- Write specific allocations for first-time winners --
            from_index = inputs_names.index(src_name)
            local_rng = np.random.default_rng(self._rng.integers(0, 2**32))
            src_winners_cpu = np.asarray(
                to_cpu(src.winners) if hasattr(src.winners, 'get') else src.winners
            )
            for idx, win in enumerate(new_indices):
                alloc = int(splits_per_new[idx][from_index]) if idx < len(splits_per_new) else 0
                if alloc <= 0 or src.w == 0:
                    continue
                sample_size = min(alloc, len(src.winners))
                if sample_size <= 0:
                    continue
                chosen = local_rng.choice(src_winners_cpu, size=sample_size, replace=False)
                col_idx = self._expansion_col(int(win), prior_w)
                if 0 <= col_idx < phys_cols:
                    conn.weights[chosen, col_idx] = 1.0

    # -- stim->area vector growth -------------------------------------------

    def _stim_conns_for(self, target: str):
        """Cached ``[(stim_name, connectome)]`` for *target*, in registration order.

        Same sequence the legacy path gets from iterating ``_stim_conns.items()``
        and dropping targets with no connectome, so the order in which
        stimuli are offered to the growth loop -- and therefore the order in
        which they draw from ``self._rng`` -- is unchanged.
        """
        cached = self._stim_target_cache.get(target)
        if cached is not None and cached[0] == self._stim_conn_version:
            return cached[1], cached[2]
        entries = []
        for stim_name, tgt_map in self._stim_conns.items():
            conn = tgt_map.get(target)
            if conn is not None:
                entries.append((stim_name, conn))
        by_name = {name: conn for name, conn in entries}
        self._stim_target_cache[target] = (self._stim_conn_version, entries, by_name)
        return entries, by_name

    def _stim_capacity(self, needed: int, n: int, cur_cap: int) -> int:
        """Physical capacity for a stim vector that must hold *needed* slots.

        Once the area is materialized past ``_dense_stim_threshold`` it is
        clearly not sparse, so allocate straight to ``n`` and never reallocate
        again. Areas that stay below the threshold keep the doubling growth.
        Capacity never affects sampled values -- only how much room exists.
        """
        thr = self._dense_stim_threshold
        if thr is not None and n > 0 and needed >= thr * n:
            return n
        cap = max(needed, cur_cap * 2, 16)
        return min(cap, n) if n > 0 else cap

    def _grow_stim_vector(self, conn, n: int, old: int, new_len: int, fill) -> None:
        """Extend one stim->area vector to *new_len*, writing *fill* in the tail.

        ``conn.weights`` is kept as a view of exactly ``new_len`` elements over
        an over-allocated buffer, so every reader still sees a vector whose
        length is the area's ever-fired count -- identical to the ``concatenate``
        the legacy path did, minus the O(w) copy on every step.
        """
        xp = get_xp()
        cur = conn.weights
        buf = getattr(conn, "_cap_buf", None)
        # getattr(cur, "base", None) is not buf catches a weights array that was
        # replaced wholesale (normalize_weights, unpickling, clone) and so is no
        # longer backed by our buffer.
        if buf is None or getattr(cur, "base", None) is not buf or new_len > buf.shape[0]:
            cap = self._stim_capacity(
                new_len, n, 0 if buf is None else int(buf.shape[0]),
            )
            buf = xp.zeros(cap, dtype=xp.float32)
            if old > 0:
                buf[:old] = cur[:old]
            conn._cap_buf = buf
        if fill is None:
            buf[old:new_len] = 0.0
        else:
            buf[old:new_len] = to_xp(fill)
        conn.weights = buf[:new_len]

    def _expand_stim_vectors_legacy(self, target, stim_names, new_w) -> None:
        """Original per-step ``concatenate`` growth. Kept as the A/B reference."""
        xp = get_xp()
        # dict.fromkeys, not set(): the loop below consumes ``self._rng`` once
        # per stimulus, so ITERATION ORDER DECIDES WHICH SLICE OF THE SEEDED
        # STREAM EACH STIMULUS GETS. These are str keys, and set-of-str order
        # varies with PYTHONHASHSEED, so the same seed produced different
        # stimulus weights in every process (same names, same shapes, different
        # values). Insertion order here is deterministic.
        stim_to_extend = dict.fromkeys(stim_names)
        for stim_name, tgt_map in self._stim_conns.items():
            conn = tgt_map.get(target)
            if conn is not None and conn.sparse and len(conn.weights) < new_w:
                stim_to_extend[stim_name] = None
        for stim_name in stim_to_extend:
            conn = self._stim_conns[stim_name][target]
            if conn.sparse:
                old = len(conn.weights)
                if new_w > old:
                    add_len = new_w - old
                    if stim_name not in stim_names:
                        stim_size = self._stimuli[stim_name].size
                        add = to_xp(self._rng.binomial(
                            stim_size, self.p, size=add_len).astype(np.float32))
                    else:
                        add = xp.zeros(add_len, dtype=xp.float32)
                    conn.weights = xp.concatenate([conn.weights, add])

    def _expand_stim_vectors_fast(self, target, tgt, stim_names, new_w) -> None:
        """Amortized-capacity growth with batched background sampling.

        Bit-identical to ``_expand_stim_vectors_legacy``:

        * the same ordered ``dict`` is built from the same insertion sequence,
          so it is iterated in the same order. This was a ``set`` and the claim
          was only true within one process: str hashing is randomized per
          process, so the RNG slice each stimulus received changed from run to
          run. Both paths must keep using ``dict.fromkeys`` or they diverge
          from each other as well as from themselves;
        * ``self._rng`` is consumed by the same stimuli in the same order;
        * consecutive draws are merged into one call only when the scalar
          ``(stim_size, add_len)`` match, and ``Generator.binomial`` with
          scalar parameters fills element-by-element, so
          ``binomial(s, p, a) ++ binomial(s, p, a) == binomial(s, p, 2a)``;
        * non-firing stimuli that draw nothing never split a run because they
          consume no randomness.

        Only the allocation changes: capacity is over-allocated (doubling, or
        straight to ``n`` past ``_dense_stim_threshold``) instead of a fresh
        ``concatenate`` per stimulus per step.
        """
        entries, by_name = self._stim_conns_for(target)
        stim_to_extend = dict.fromkeys(stim_names)
        for stim_name, conn in entries:
            if conn.sparse and len(conn.weights) < new_w:
                stim_to_extend[stim_name] = None

        # Pass 1 -- resolve, in insertion order, what each stimulus needs.
        plan = []  # (conn, old, add_len, stim_size or None when no rng draw)
        for stim_name in stim_to_extend:
            conn = by_name.get(stim_name)
            if conn is None:
                conn = self._stim_conns[stim_name][target]
            if not conn.sparse:
                continue
            old = len(conn.weights)
            if new_w <= old:
                continue
            size = (None if stim_name in stim_names
                    else self._stimuli[stim_name].size)
            plan.append((conn, old, new_w - old, size))
        if not plan:
            return

        # Pass 2 -- batch maximal runs of identical (stim_size, add_len) draws.
        draws = {}
        rng_idx = [i for i, item in enumerate(plan) if item[3] is not None]
        i = 0
        while i < len(rng_idx):
            size, add_len = plan[rng_idx[i]][3], plan[rng_idx[i]][2]
            j = i + 1
            while (j < len(rng_idx)
                   and plan[rng_idx[j]][3] == size
                   and plan[rng_idx[j]][2] == add_len):
                j += 1
            count = j - i
            if count == 1:
                draws[rng_idx[i]] = self._rng.binomial(
                    size, self.p, size=add_len).astype(np.float32)
            else:
                block = self._rng.binomial(
                    size, self.p, size=add_len * count).astype(np.float32)
                for t in range(count):
                    draws[rng_idx[i + t]] = block[t * add_len:(t + 1) * add_len]
            i = j

        # Pass 3 -- write.
        n = tgt.n
        for idx, (conn, old, _add_len, size) in enumerate(plan):
            self._grow_stim_vector(
                conn, n, old, new_w, draws[idx] if size is not None else None,
            )

    def _write_area_expansion_edges(
        self,
        src_name: str,
        target: str,
        conn,
        inputs_names: list,
        new_indices: list,
        splits_per_new: list,
        prior_w: int,
        phys_cols: int,
    ) -> None:
        """Write Hebbian edge allocations without matrix reallocation."""
        src = self._areas[src_name]
        from_index = inputs_names.index(src_name)
        local_rng = np.random.default_rng(self._rng.integers(0, 2**32))
        src_winners_cpu = np.asarray(
            to_cpu(src.winners) if hasattr(src.winners, 'get') else src.winners
        )
        for idx, win in enumerate(new_indices):
            alloc = int(splits_per_new[idx][from_index]) if idx < len(splits_per_new) else 0
            if alloc <= 0 or src.w == 0:
                continue
            sample_size = min(alloc, len(src.winners))
            if sample_size <= 0:
                continue
            chosen = local_rng.choice(src_winners_cpu, size=sample_size, replace=False)
            col_idx = self._expansion_col(int(win), prior_w)
            if 0 <= col_idx < phys_cols:
                conn.weights[chosen, col_idx] = 1.0

    def _expansion_col(self, win: int, prior_w: int) -> int:
        """Column to write a first-time winner's sampled afferents into.

        This SHOULD simply be ``win``: ``win`` is the new neuron's compact
        index, and the allocation drawn from ``compute_input_splits`` describes
        that neuron's own incoming synapses.  The legacy expression
        ``win - prior_w`` instead writes them into columns ``0, 1, 2, ...`` --
        the neurons materialized in the very first projection.  It is only
        correct on the first projection into an area, where ``prior_w == 0``.

        MEASURED CONSEQUENCE.  From the second projection onward every recruit
        dumps its afferents onto the oldest columns, which are exactly the
        neurons most likely to be in the surviving assembly.  Assembly-internal
        connection density inflates to 0.189 against p=0.05 (the dense/explicit
        engine and the reference implementation both give ~0.09), and the
        stored assembly ends up receiving 3.5x the recurrent drive of the rest
        of the area from an UNRELATED assembly (dense engine and reference:
        ~1.0x).  That turns every stored assembly into a hair-trigger attractor
        which captures any independent stimulus, and it is the dominant reason
        self-recurrence collapsed in this engine.  Corrected, independent
        stimuli hold chance overlap at 15 rounds (see Brain.project_rounds).

        WHY IT IS GATED ON ``norm_init`` RATHER THAN JUST FIXED.  The write
        affects EVERY sparse projection, not only recurrent ones, so
        correcting it unconditionally would shift every existing result in the
        repository.  It is therefore scoped to the opt-in norm_init path.  This
        is a defect worth fixing globally on its own, with its own regression
        sweep -- it is not specific to norm_init.
        """
        return win if self.norm_init else win - prior_w

    # -- State accessors ----------------------------------------------------

    def get_winners(self, area: str) -> np.ndarray:
        st = self._areas[area]
        return np.array(to_cpu(st.winners), dtype=np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        xp = get_xp()
        st = self._areas[area]
        st.winners = xp.asarray(winners, dtype=xp.uint32)

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].w

    def get_neuron_id_mapping(self, area: str) -> list:
        """Return the compact_to_neuron_id list for stable winner IDs."""
        return self._areas[area].compact_to_neuron_id

    # -- Projection fidelity ------------------------------------------------

    def set_projection_fidelity(self, fidelity: str) -> None:
        self._projection_fidelity = ProjectionFidelity.normalize(fidelity)

    def get_projection_fidelity(self) -> str:
        return self._projection_fidelity.value

    def preallocate_stim_targets(self, target: str, min_columns: int) -> None:
        """Extend all stim→*target* vectors to at least *min_columns* (zeros)."""
        if min_columns <= 0 or target not in self._areas:
            return
        xp = get_xp()
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
        xp = get_xp()
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
        st.refracted = enabled
        st.refracted_strength = strength
        if enabled and len(st._cumulative_bias) == 0:
            xp = get_xp()
            st._cumulative_bias = xp.zeros(max(st.w, 0), dtype=xp.float32)

    def clear_refracted_bias(self, area: str) -> None:
        """Reset accumulated refracted bias to zero."""
        xp = get_xp()
        st = self._areas[area]
        st._cumulative_bias = xp.zeros(max(st.w, 0), dtype=xp.float32)

    # -- Weight normalization -----------------------------------------------

    def normalize_weights(self, target: str, source: str = None) -> None:
        """Column-normalize weights into *target* so each neuron sums to 1.0."""
        xp = get_xp()
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
        """Structural clone for sweep forks — copies numpy state, not Python graph walk."""
        import copy
        from collections import deque

        new = NumpySparseEngine(
            self.p,
            seed=0,
            w_max=self.w_max,
            deterministic=self._deterministic,
            projection_fidelity=self._projection_fidelity.value,
            inhibitory_prob=self.inhibitory_prob,
            inhibitory_weight=self.inhibitory_weight,
            synaptic_scaling=self.synaptic_scaling,
            norm_init=self.norm_init,
        )
        new._rng = copy.deepcopy(self._rng)
        new._plasticity_enabled_global = self._plasticity_enabled_global
        new._stim_fastpath = self._stim_fastpath
        new._dense_stim_threshold = self._dense_stim_threshold

        for name, stim in self._stimuli.items():
            new.add_stimulus(name, stim.size)
        for name, area in self._areas.items():
            new.add_area(
                name,
                area.n,
                area.k,
                area.beta,
                refractory_period=area.refractory_period,
                inhibition_strength=area.inhibition_strength,
                winner_policy=area.winner_policy,
                input_noise_std=area.input_noise_std,
            )

        def _copy_conn(src_conn: Connectome, dst_conn: Connectome) -> None:
            dst_conn.source_size = src_conn.source_size
            dst_conn.target_size = src_conn.target_size
            dst_conn.sparse = src_conn.sparse
            if src_conn.weights is not None and getattr(src_conn.weights, "size", 0) > 0:
                dst_conn.weights = src_conn.weights.copy()
            else:
                dst_conn.weights = src_conn.weights
            # norm_init per-neuron scales are part of the network's identity,
            # not derived state -- a clone must inherit them.
            base = getattr(src_conn, "_norm_deg_base", None)
            if base is not None:
                dst_conn._norm_deg_base = base.copy()

        for stim_name, area_map in self._stim_conns.items():
            for area_name, conn in area_map.items():
                _copy_conn(conn, new._stim_conns[stim_name][area_name])

        for src_name, tgt_map in self._area_conns.items():
            for tgt_name, conn in tgt_map.items():
                _copy_conn(conn, new._area_conns[src_name][tgt_name])

        xp = get_xp()
        for name, src in self._areas.items():
            dst = new._areas[name]
            dst.w = src.w
            dst.winners = src.winners.copy()
            dst.compact_to_neuron_id = list(src.compact_to_neuron_id)
            if src.neuron_id_pool is not None:
                dst.neuron_id_pool = src.neuron_id_pool.copy()
            dst.neuron_id_pool_ptr = src.neuron_id_pool_ptr
            dst.fixed_assembly = src.fixed_assembly
            dst.beta_by_source = dict(src.beta_by_source)
            dst.refractory_period = src.refractory_period
            dst.inhibition_strength = src.inhibition_strength
            dst.refracted = src.refracted
            dst.refracted_strength = src.refracted_strength
            dst.explicit_source = src.explicit_source
            dst.winner_policy = src.winner_policy
            dst.input_noise_std = src.input_noise_std
            if src._refractory_history is not None:
                dst._refractory_history = deque(
                    (set(h) for h in src._refractory_history),
                    maxlen=src._refractory_history.maxlen,
                )
            if src._cumulative_bias is not None and len(src._cumulative_bias) > 0:
                dst._cumulative_bias = src._cumulative_bias.copy()
            else:
                dst._cumulative_bias = xp.zeros(max(dst.w, 0), dtype=xp.float32)

        return new

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "numpy_sparse"
