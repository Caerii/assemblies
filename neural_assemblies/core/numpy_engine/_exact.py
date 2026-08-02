"""NumpyExactEngine: compute the drive, never store the substrate (task #85).

WHAT THIS IS FOR. `numpy_sparse` invents a drive for neurons that have not
fired (`sample_new_winner_inputs`), and that approximation is the entire
accuracy gap. Measured (`research/notes/graded_similarity_and_sampler_load.md`):
the exact substrate maps similar inputs to similar assemblies -- chance overlap
for disjoint inputs, rising monotonically to 1.0 -- while the sampler flattens
that to **0.906 for fully disjoint inputs** at low area load. Every graded
readout in this repo is scored on the property the sampler destroys.

`explicit` gets it right and cannot scale: n^2 floats per fiber is 400 MB at
n=1e4. But it is MEMORY, not arithmetic, that makes it unusable --

    w_ij(t) = f(i, j, seed) * (1 + beta)^{c_ij}

`f` is already content-addressed by ABSOLUTE (row, col) via `hash_area_weights`,
so it is recomputable and never needs storing, and `c_ij` is nonzero only for
pairs that co-fired AND have a synapse (measured: 9.2% of co-firing pairs,
0.44 MB against 16 MB dense at n=2000). So the drive decomposes into a dense
recomputable term plus a sparse learned one.

Benchmarked: `hash_area_weights` runs at 1.15-1.5 G cells/s with the Rust
kernel, so exact drive at n=1e4, k=200 costs ~3.4 ms/round and the one-time
exact-in-degree pass costs 0.09 s.

WHAT IS DIFFERENT FROM THE OTHER ENGINES.

* **No recruitment, no compact index space.** Every neuron is addressable from
  t=0, which is what the model says: G(n,p) is fixed at t=0 and firing never
  creates a synapse. `winners` are neuron ids, `w == n` always, and
  [[two-index-spaces-compact-vs-neuron-id]] cannot arise here.
* **norm_init is EXACT, not estimated.** `inverse_indegree` charges rows that do
  not exist yet at the ambient rate `p * (n_pre - rows_known)`. Here every row
  exists, so `rows_known == n_pre`, the correction term is zero, and `d_j` is
  the neuron's true in-degree. This engine removes an approximation rather than
  reproducing one -- expect numbers to differ from `numpy_sparse`, and that is
  the point.
* **Draw order cannot matter.** Everything is a function of (row, col, seed),
  so door 5 (#81) and door 6 are gone by construction rather than by patch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ..backend import to_cpu
from ..engine import ComputeEngine, ProjectionResult
from .._pricing import inverse_indegree
from ._seeding import fnv1a_pair_seed, hash_area_weights, hash_stim_counts
from ._state import StimulusState


class _Potentiation:
    """Sparse `(1+beta)^c` for one area->area fiber, keyed by (row, col).

    Only pairs that co-fired AND carry a synapse are ever stored -- `w *= 1+b`
    leaves a zero at zero -- so this tracks what was LEARNED, not n^2.

    STORED LOW-RANK, and that is the whole trick. Every update is a full outer
    product of (source winners) x (target winners), so

        c_ij = |{t : i in S_t and j in T_t}|

    is a sum of rank-one terms. Keeping the DISTINCT (S, T) pairs with
    multiplicities costs O(events * k) instead of O(pairs), and they repeat
    heavily -- study I presents ~431 distinct bigrams over thousands of rounds.

    MEASURED at n=1e4, k=200, 1293 rounds over 431 distinct bigrams, counts
    verified identical to the naive store (max abs diff 0):

        store            bump      read      total
        dict-of-dicts   17.19 ms  34.30 ms  51.49 ms/round
        low-rank         0.07 ms  26.22 ms  26.29 ms/round   <- 1.96x

    Writing becomes essentially free. A per-row sorted-numpy variant was also
    tried and is SLOWER than the dict (24.80 ms bump): each row grows to ~4500
    columns and every insertion re-sorts it. Vectorising the wrong structure
    does not help; changing the structure does.
    """

    __slots__ = ("_sets", "_ids", "_events", "_by_src", "_members")

    def __init__(self) -> None:
        self._sets: List[np.ndarray] = []          # id -> sorted index array
        self._ids: Dict[bytes, int] = {}           # interning
        self._events: Dict[tuple, int] = {}        # (sid, tid) -> multiplicity
        self._by_src: Dict[int, List[int]] = defaultdict(list)
        self._members: Dict[int, set] = defaultdict(set)   # row -> {sid}

    def __len__(self) -> int:
        """Distinct potentiated (row, col) pairs this represents."""
        return sum(self._sets[s].size * self._sets[t].size
                   for (s, t) in self._events)

    def _intern(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr, dtype=np.int64)
        key = arr.tobytes()
        sid = self._ids.get(key)
        if sid is None:
            sid = len(self._sets)
            self._ids[key] = sid
            self._sets.append(arr)
        return sid

    def bump(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Record one potentiation event over the outer product rows x cols."""
        sid = self._intern(rows)
        tid = self._intern(cols)
        key = (sid, tid)
        if key not in self._events:
            self._events[key] = 0
            self._by_src[sid].append(tid)
            for r in self._sets[sid]:
                self._members[int(r)].add(sid)
        self._events[key] += 1

    def intersects(self, rows: np.ndarray) -> bool:
        """Does anything stored here touch these rows? Cheap enough to ask
        before deciding whether the caller needs a materialised block."""
        for r in rows:
            if self._members.get(int(r)):
                return True
        return False

    def apply_to(self, block: np.ndarray, rows: np.ndarray,
                 beta: float) -> bool:
        """Multiply `block` in place by `(1+beta)^c`. True if anything applied.

        NO DENSE COUNT MATRIX IS BUILT. Exponents ADD, so

            (1+beta)^(c1 + c2) == (1+beta)^c1 * (1+beta)^c2

        which means each stored outer product can be applied independently to
        its own sub-block. Materialising `(len(rows), n)` counts and calling
        `np.power` on it costs 2e6 pow evaluations at n=1e4, k=200 where only
        ~2% of entries are nonzero -- measured at 32 ms/round, worse than the
        naive store it replaced. Touching only the sub-blocks costs one small
        in-place multiply per event.

        The per-event factor comes from a cached table indexed by
        multiplicity: the exponents are small integers, so this is a gather
        rather than a transcendental.
        """
        rows = np.asarray(rows, dtype=np.int64)
        pos_of = {int(r): i for i, r in enumerate(rows)}
        touched: set = set()
        for r in rows:
            got = self._members.get(int(r))
            if got:
                touched |= got
        if not touched:
            return False
        applied = False
        for sid in touched:
            hit = np.array([pos_of[int(m)] for m in self._sets[sid]
                            if int(m) in pos_of], dtype=np.int64)
            if hit.size == 0:
                continue
            for tid in self._by_src[sid]:
                mult = self._events[(sid, tid)]
                block[np.ix_(hit, self._sets[tid])] *= (1.0 + beta) ** mult
                applied = True
        return applied


class ExactAreaState:
    """Per-area state. No `compact_to_neuron_id`: the index IS the neuron id."""

    __slots__ = ("name", "n", "k", "beta", "winners", "fixed_assembly",
                 "beta_by_source", "ever_fired", "w")

    def __init__(self, name: str, n: int, k: int, beta: float) -> None:
        self.name, self.n, self.k, self.beta = name, n, k, beta
        self.winners = np.empty(0, dtype=np.uint32)
        self.fixed_assembly = False
        self.beta_by_source: Dict[str, float] = {}
        self.ever_fired = np.zeros(n, dtype=bool)
        # `w` is num-ever-fired here, and it is n from the start because every
        # neuron exists. Kept only so consumers reading `w` do not crash.
        self.w = n

    @property
    def num_ever_fired(self) -> int:
        return int(self.ever_fired.sum())


class NumpyExactEngine(ComputeEngine):
    """Exact drive for every neuron, with no weight storage."""

    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 norm_init: bool = True, inhibitory_prob: float = 0.0,
                 inhibitory_weight: float = -1.0, **_ignored) -> None:
        self.p = float(p)
        self.seed = int(seed)
        self.w_max = w_max
        self.norm_init = bool(norm_init)
        self.inhibitory_prob = float(inhibitory_prob)
        self.inhibitory_weight = float(inhibitory_weight)
        self._plasticity_enabled_global = True

        self._areas: Dict[str, ExactAreaState] = {}
        self._stimuli: Dict[str, StimulusState] = {}
        # stim -> area : per-target-neuron afferent COUNT (this is d_j too)
        self._stim_base: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        # stim -> area : per-target-neuron potentiation exponent
        self._stim_pot: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        # (src, tgt) -> sparse potentiation
        self._area_pot: Dict[tuple, _Potentiation] = {}
        # (src, tgt) -> exact 1/d_j, computed once and cached
        self._norm_cache: Dict[tuple, np.ndarray] = {}
        self._stim_norm_cache: Dict[tuple, np.ndarray] = {}

    # -- wiring -------------------------------------------------------------

    def _pair_seed(self, source: str, target: str) -> int:
        return fnv1a_pair_seed(self.seed, source, target)

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0, inhibition_strength: float = 0.0,
                 slot_count: int = 0) -> None:
        area = ExactAreaState(name, n, k, beta)
        self._areas[name] = area
        for stim_name, stim in self._stimuli.items():
            self._wire_stim(stim_name, stim.size, name)
            area.beta_by_source[stim_name] = beta
        for other_name, other in self._areas.items():
            area.beta_by_source.setdefault(other_name, beta)
            other.beta_by_source.setdefault(name, other.beta)

    def add_stimulus(self, name: str, size: int) -> None:
        self._stimuli[name] = StimulusState(name=name, size=size)
        for area_name, area in self._areas.items():
            self._wire_stim(name, size, area_name)
            area.beta_by_source[name] = area.beta

    def _wire_stim(self, stim_name: str, size: int, area_name: str) -> None:
        n = self._areas[area_name].n
        base = np.asarray(
            hash_stim_counts(size, 0, n, self._pair_seed(stim_name, area_name),
                             self.p),
            dtype=np.float64)
        self._stim_base[stim_name][area_name] = base
        self._stim_pot[stim_name][area_name] = np.zeros(n, dtype=np.float64)

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        pass

    # -- the substrate, recomputed ------------------------------------------

    def _fiber_rows(self, source: str, target: str, rows: np.ndarray,
                    n_cols: int) -> np.ndarray:
        """Initial weights of `rows` x [0, n_cols) for one area fiber.

        Addressed by ABSOLUTE (row, col), so which rows a caller asks for --
        and in what order -- cannot change any value.

        Kept in the kernel's own float32: converting each row to float64 on the
        way in costs `k * n` casts per round for no precision that survives the
        k-WTA comparison.
        """
        ps = self._pair_seed(source, target)
        out = np.empty((len(rows), n_cols), dtype=np.float32)
        for i, r in enumerate(rows):
            r = int(r)
            out[i] = hash_area_weights(
                r, r + 1, 0, n_cols, ps, self.p,
                self.inhibitory_prob, self.inhibitory_weight).reshape(-1)
        return out

    def _fiber_sum(self, source: str, target: str, rows: np.ndarray,
                   n_cols: int) -> np.ndarray:
        """Column sums of `rows` x [0, n_cols) WITHOUT materialising the block.

        The un-potentiated drive only ever needs the sum, so the (k, n) block
        need not exist: at n=1e4, k=200 that is a 8 MB allocation per round
        bought for nothing. Used whenever the fiber carries no potentiation
        (every readout, and every fiber before it has learned anything).
        """
        ps = self._pair_seed(source, target)
        acc = np.zeros(n_cols, dtype=np.float64)
        for r in rows:
            r = int(r)
            acc += hash_area_weights(
                r, r + 1, 0, n_cols, ps, self.p,
                self.inhibitory_prob, self.inhibitory_weight).reshape(-1)
        return acc

    def _area_norm(self, source: str, target: str) -> Optional[np.ndarray]:
        """Exact `1/d_j` for an area fiber. No unknown-row correction needed.

        `d_j` is the TRUE in-degree over all `n_pre` rows, because every row
        exists. The sparse engine has to estimate this term; here it is
        measured, so `inverse_indegree` is called with `rows_known == n_pre`.
        Cached: this is the one O(n_pre * n) pass in the engine (0.09 s at
        n=1e4).
        """
        if not self.norm_init:
            return None
        key = (source, target)
        cached = self._norm_cache.get(key)
        if cached is not None:
            return cached
        n_pre = self._areas[source].n
        n_post = self._areas[target].n
        deg = np.zeros(n_post, dtype=np.float64)
        ps = self._pair_seed(source, target)
        chunk = max(1, min(n_pre, max(1, 4_000_000 // max(n_post, 1))))
        for r0 in range(0, n_pre, chunk):
            r1 = min(r0 + chunk, n_pre)
            blk = np.asarray(
                hash_area_weights(r0, r1, 0, n_post, ps, self.p,
                                  self.inhibitory_prob, self.inhibitory_weight),
                dtype=np.float64)
            deg += (blk != 0).sum(axis=0)
        scale = inverse_indegree(deg, n_pre, n_pre, self.p, xp=np)
        self._norm_cache[key] = scale
        return scale

    def _stim_norm(self, stim: str, target: str) -> Optional[np.ndarray]:
        """Exact `1/d_j` for a stimulus fiber; the base count IS `d_j`."""
        if not self.norm_init:
            return None
        key = (stim, target)
        cached = self._stim_norm_cache.get(key)
        if cached is not None:
            return cached
        size = self._stimuli[stim].size
        deg = self._stim_base[stim][target]
        scale = inverse_indegree(deg, size, size, self.p, xp=np)
        self._stim_norm_cache[key] = scale
        return scale

    # -- projection ---------------------------------------------------------

    def project_into(self, target: str, from_stimuli: List[str],
                     from_areas: List[str], plasticity_enabled: bool = True,
                     record_activation: bool = False,
                     external_drive: Optional[np.ndarray] = None
                     ) -> ProjectionResult:
        tgt = self._areas[target]
        from_areas = [a for a in from_areas if self._areas[a].winners.size > 0]

        if tgt.fixed_assembly:
            return ProjectionResult(
                winners=np.array(tgt.winners, dtype=np.uint32),
                num_first_winners=0, num_ever_fired=tgt.num_ever_fired)

        n = tgt.n
        drive = np.zeros(n, dtype=np.float64)

        for stim in from_stimuli:
            base = self._stim_base[stim][target]
            pot = self._stim_pot[stim][target]
            beta = tgt.beta_by_source.get(stim, tgt.beta)
            w = base * self._clamped(np.power(1.0 + beta, pot))
            scale = self._stim_norm(stim, target)
            drive += w if scale is None else w * scale

        for src_name in from_areas:
            src = self._areas[src_name]
            rows = np.asarray(src.winners, dtype=np.int64)
            beta = tgt.beta_by_source.get(src_name, tgt.beta)
            pot = self._area_pot.get((src_name, target))
            if pot is None or beta == 0 or not pot.intersects(rows):
                # Nothing learned on this fiber reaches these rows, so the sum
                # is all that is needed and the block never has to exist.
                summed = self._fiber_sum(src_name, target, rows, n)
            else:
                block = self._fiber_rows(src_name, target, rows, n)
                if pot.apply_to(block, rows, beta) and self.w_max is not None:
                    # `w_max` is a ceiling in MULTIPLES of the initial weight,
                    # and storage is on the unit scale, so clamping the weight
                    # IS clamping the multiplier. Applied once over the block
                    # rather than per event, because the events compose.
                    np.minimum(block, np.float32(self.w_max), out=block)
                summed = block.sum(axis=0, dtype=np.float64)
            scale = self._area_norm(src_name, target)
            drive += summed if scale is None else summed * scale

        if external_drive is not None and len(external_drive) == n:
            drive += np.asarray(external_drive, dtype=np.float64)

        winners = self._select(drive, tgt.k)

        if plasticity_enabled and self._plasticity_enabled_global:
            for stim in from_stimuli:
                if tgt.beta_by_source.get(stim, tgt.beta) != 0:
                    self._stim_pot[stim][target][winners] += 1.0
            for src_name in from_areas:
                if tgt.beta_by_source.get(src_name, tgt.beta) == 0:
                    continue
                key = (src_name, target)
                store = self._area_pot.get(key)
                if store is None:
                    store = self._area_pot[key] = _Potentiation()
                store.bump(np.asarray(self._areas[src_name].winners,
                                      dtype=np.int64), winners)

        tgt.winners = np.asarray(winners, dtype=np.uint32)
        tgt.ever_fired[winners] = True

        return ProjectionResult(
            winners=np.array(tgt.winners, dtype=np.uint32),
            num_first_winners=0,
            num_ever_fired=tgt.num_ever_fired,
            total_activation=float(drive[winners].sum()))

    def _clamped(self, mult: np.ndarray) -> np.ndarray:
        """`w_max` is a ceiling in MULTIPLES of the initial weight.

        Storage stays on the unit scale (that is what makes the norm_init
        read-time scale legal), so the clamp applies to the multiplier.
        """
        if self.w_max is None:
            return mult
        return np.minimum(mult, float(self.w_max))

    @staticmethod
    def _select(drive: np.ndarray, k: int) -> np.ndarray:
        """Top-k by drive, ties broken by LOWEST INDEX.

        Chosen deliberately rather than inherited: a tie-break decides
        borderline results, and at low `k*p` a large fraction of every assembly
        sits inside a tied band (measured 62% at k*p=1, 12% at 15.9). `argsort`
        with a stable kind makes the rule "highest drive, then lowest index",
        which is the same rule the explicit engine's selector applies.
        """
        k = int(min(k, drive.size))
        order = np.argsort(-drive, kind="stable")[:k]
        return np.sort(order).astype(np.int64)

    # -- accessors ----------------------------------------------------------

    def get_winners(self, area: str) -> np.ndarray:
        return np.array(to_cpu(self._areas[area].winners), dtype=np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        st = self._areas[area]
        st.winners = np.asarray(winners, dtype=np.uint32)
        if st.winners.size:
            st.ever_fired[np.asarray(st.winners, dtype=np.int64)] = True

    def get_num_ever_fired(self, area: str) -> int:
        return self._areas[area].num_ever_fired

    def set_beta(self, target: str, source: str, beta: float) -> None:
        self._areas[target].beta_by_source[source] = beta

    def get_beta(self, target: str, source: str) -> float:
        tgt = self._areas[target]
        return tgt.beta_by_source.get(source, tgt.beta)

    def fix_assembly(self, area: str) -> None:
        st = self._areas[area]
        if st.winners is None or len(st.winners) == 0:
            raise ValueError(f"Area {area} has no winners to fix.")
        st.fixed_assembly = True

    def unfix_assembly(self, area: str) -> None:
        self._areas[area].fixed_assembly = False

    def is_fixed(self, area: str) -> bool:
        return self._areas[area].fixed_assembly

    def reset_area_connections(self, area: str) -> None:
        """Drop everything LEARNED into `area`; the substrate is untouched.

        There is no connectome to re-randomise here -- initial weights are a
        function of (row, col, seed). Resetting therefore means forgetting
        potentiation, which is what the operation is actually for.
        """
        for key in [k for k in self._area_pot if k[1] == area]:
            del self._area_pot[key]
        for stim in self._stim_pot:
            if area in self._stim_pot[stim]:
                self._stim_pot[stim][area][:] = 0.0

    def materialize_area(self, area: str, storage: str = "csr") -> int:
        """No-op: every neuron already exists. Present so arbiter code works."""
        return self._areas[area].n

    @property
    def name(self) -> str:
        return "numpy_exact"
