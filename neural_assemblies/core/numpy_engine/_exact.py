"""NumpyExactEngine: compute the drive, never store the substrate (task #85).

WHAT THIS IS FOR. `numpy_sparse` invents a drive for neurons that have not
fired (`sample_new_winner_inputs`), and that approximation is the entire
accuracy gap. Measured (`research/notes/substrate/graded_similarity_and_sampler_load.md`):
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

from collections import OrderedDict, defaultdict
from typing import Dict, List, Mapping, Optional

import numpy as np
from ..index_spaces import validated_indices

from ..backend import to_cpu
from ..engine import ComputeEngine, ProjectionResult
from ..activity import ActivityState
from .._pricing import inverse_indegree
from ._seeding import (fnv1a_pair_seed, hash_area_cells, hash_area_indegree,
                       hash_area_rows, hash_stim_counts)
from ._state import StimulusState


def _fixed_target_learns() -> bool:
    """Whether a projection INTO a fixed area still potentiates its afferents.

    ONE definition, owned by `_sparse`, read through here. Restating the policy
    would let the two engines drift apart exactly as they twice drifted apart
    on the k-WTA pricing law ([[pricing-law-implemented-twice]]) -- and this
    engine has already diverged on it once by not implementing it at all.

    Imported lazily so `_exact` does not take `_sparse`'s import cost just to
    read an environment variable; import time is measured and defended here
    (#74, #78). No cycle: `_sparse` does not import `_exact`.
    """
    from ._sparse import _fixed_target_plasticity_enabled
    return _fixed_target_plasticity_enabled()


def _reject_unsupported(where: str, supported_defaults: Mapping[str, object],
                        given: Mapping[str, object]) -> None:
    """Raise if a caller asked for a mechanism this engine does not implement.

    `Brain` forwards a common kwarg set to every engine, so an engine that
    implements a subset has two options: swallow the rest in `**kwargs`, or
    say so. Swallowing produces the failure this repo keeps rediscovering --
    a mechanism that is wired, configured, and never runs, whose signature is
    "X seems to have little effect". See [[silent-no-op-dead-fibers]].

    Passing the DEFAULT is not a request, so it is accepted silently; passing
    anything else raises. Unknown kwargs are accepted and ignored, because the
    ABC may grow parameters this engine has no opinion on -- those still show
    up as an explicit signature mismatch rather than as wrong numbers.
    """
    asked = [f"{key}={given[key]!r}"
             for key, default in supported_defaults.items()
             if key in given and given[key] != default]
    if asked:
        raise NotImplementedError(
            f"{where} does not implement: {', '.join(asked)}. "
            f"Use numpy_sparse for these, or leave them at their defaults "
            f"({', '.join(f'{k}={v!r}' for k, v in supported_defaults.items())})."
        )


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

    __slots__ = ("_sets", "_ids", "_events", "_by_src", "_members",
                 "_touch_memo")

    def __init__(self) -> None:
        self._sets: List[np.ndarray] = []          # id -> sorted index array
        self._ids: Dict[bytes, int] = {}           # interning
        self._events: Dict[tuple, int] = {}        # (sid, tid) -> multiplicity
        self._by_src: Dict[int, List[int]] = defaultdict(list)
        self._members: Dict[int, set] = defaultdict(set)   # row -> {sid}
        self._touch_memo: Dict[bytes, tuple] = {}          # assembly -> sids

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

    @staticmethod
    def _axis_index(idx: np.ndarray, extent: int):
        """`slice(None)` when `idx` selects the whole axis in order, else `idx`.

        `np.ix_` fancy indexing COPIES: `block[ix] *= f` gathers the sub-block,
        multiplies, and scatters it back, and the clamp repeats that -- four
        non-contiguous passes over 300k elements, measured at 12.2 ms. When the
        sub-block IS the event's own rows x cols -- the common case, since the
        columns were chosen as exactly the touched ones -- the indices are the
        identity and a plain in-place multiply works.
        """
        if idx.size != extent:
            return idx
        if idx[0] != 0 or idx[-1] != extent - 1:
            return idx
        return slice(None) if np.array_equal(
            idx, np.arange(extent, dtype=idx.dtype)) else idx

    @staticmethod
    def _positions(rows_sorted: np.ndarray, members: np.ndarray) -> np.ndarray:
        """Positions of `members` within `rows`, dropping absentees.

        Both arrays are sorted, so this is a searchsorted. The dict-and-list
        spelling -- `{int(r): i for ...}` then `[pos[int(m)] for m in ...]` --
        costs ~1600 interpreter operations per event at k=548 and measured
        6-12 ms per round against 0.5-1.4 ms for the gather it feeds.
        """
        if rows_sorted.size == 0 or members.size == 0:
            return np.empty(0, dtype=np.int64)
        # `rows` is NOT guaranteed sorted -- `set_winners` accepts any order,
        # and L3 drives from deliberately unsorted patterns. searchsorted on an
        # unsorted array returns nonsense silently, so sort when needed and map
        # the positions back through the permutation.
        if rows_sorted.size > 1 and not np.all(rows_sorted[:-1] <= rows_sorted[1:]):
            order = np.argsort(rows_sorted, kind="stable")
            srt = rows_sorted[order]
            idx = np.searchsorted(srt, members)
            np.clip(idx, 0, srt.size - 1, out=idx)
            return order[idx[srt[idx] == members]]
        idx = np.searchsorted(rows_sorted, members)
        np.clip(idx, 0, rows_sorted.size - 1, out=idx)
        return idx[rows_sorted[idx] == members]

    def intersects(self, rows: np.ndarray) -> bool:
        """Does anything stored here touch these rows? Cheap enough to ask
        before deciding whether the caller needs a materialised block."""
        for r in rows:
            if self._members.get(int(r)):
                return True
        return False

    def _touched(self, rows: np.ndarray):
        """Sorted sids whose source set meets `rows`, memoised per assembly.

        Assemblies repeat, so this is computed once per distinct assembly
        rather than once per round; the memo is invalidated whenever a new
        source set is interned, which is the only thing that can change it.
        """
        key = rows.tobytes()
        hit = self._touch_memo.get(key)
        if hit is not None and hit[0] == len(self._sets):
            return hit[1]
        touched = set()
        for r in rows:
            got = self._members.get(int(r))
            if got:
                touched |= got
        out = sorted(touched)
        self._touch_memo[key] = (len(self._sets), out)
        return out

    def touched_cols(self, rows: np.ndarray) -> Optional[np.ndarray]:
        """The columns any stored outer product reaches from these rows.

        The learned factor is confined to these, so the drive's correction term
        is too -- everything else is exactly the initial weight and is already
        accounted for by the fused sum. Returns None when nothing applies.
        """
        touched = self._touched(rows)
        if not touched:
            return None
        cols: set = set()
        for sid in touched:
            for tid in self._by_src[sid]:
                cols.update(self._sets[tid].tolist())
        if not cols:
            return None
        return np.fromiter(sorted(cols), dtype=np.int64, count=len(cols))

    def apply_to(self, block: np.ndarray, rows: np.ndarray, beta: float,
                 w_max: Optional[float] = None,
                 col_index: Optional[np.ndarray] = None) -> bool:
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
        touched = self._touched(rows)
        if not touched:
            return False
        applied = False
        regions = []
        cap = np.float32(w_max) if w_max is not None else None
        # CLAMP AFTER EVERY EVENT when the multiplier is >= 1, which is the
        # only way to keep the running product finite. This engine reached
        # `overflow encountered in multiply` on a real parser run: a cell
        # potentiated by enough events sends `f * (1+beta)^sum` past float32's
        # 3.4e38 BEFORE the deferred clamp ever sees it, and inf then poisons
        # the column sum.
        #
        # The comment this replaces said clamping between events would "cap a
        # product still being built". That is true for beta < 0 and FALSE for
        # beta >= 0, because for any multiplier m >= 1 and cap c > 0
        #
        #     min(min(x, c) * m, c) == min(x * m, c)
        #
        #   x <= c:  left is min(x*m, c), same as right.
        #   x >  c:  left is min(c*m, c) = c since m >= 1; right is c too
        #            because x*m > x > c.
        #
        # so clamping early is not an approximation, it is the same number. The
        # deferred pass is kept for beta < 0, where the identity does not hold
        # and no overflow is possible either.
        clamp_now = cap is not None and beta >= 0.0

        def _clamp(blk, ix, c):
            if isinstance(ix[0], slice) and isinstance(ix[1], slice):
                np.minimum(blk, c, out=blk)      # a real in-place view
            else:
                # Assignment, NOT `out=blk[ix]`: fancy indexing returns a COPY,
                # so an in-place write there lands in a temporary and is
                # silently discarded.
                blk[ix] = np.minimum(blk[ix], c)
        # SORTED, not raw set order. Floating-point multiplication is not
        # associative, so the order events are applied in decides the last ulp
        # of a cell touched by more than one. Set iteration happens to be
        # deterministic for int keys (hash(i) == i), but that is an
        # undocumented CPython internal and depends on the table's growth
        # history; sorting makes the guarantee explicit and costs nothing at
        # these sizes.
        for sid in touched:                      # already sorted; see below
            hit = self._positions(rows, self._sets[sid])
            if hit.size == 0:
                continue
            for tid in self._by_src[sid]:
                mult = self._events[(sid, tid)]
                cols = self._sets[tid]
                if col_index is not None:
                    # `block` spans only the touched columns, so translate.
                    # BOTH arrays are sorted, so this is a searchsorted, not a
                    # dict lookup per column: the generator version ran 200
                    # Python lookups per event per round and was 0.83 ms, the
                    # single largest cost in the projection.
                    cols = np.searchsorted(col_index, cols)
                factor = (1.0 + beta) ** mult
                r = self._axis_index(hit, block.shape[0])
                c = self._axis_index(cols, block.shape[1])
                if isinstance(r, slice) and isinstance(c, slice):
                    block *= factor                       # contiguous, in place
                    ix = (r, c)
                elif isinstance(r, slice):
                    block[:, c] *= factor                 # one fancy axis
                    ix = (r, c)
                else:
                    ix = np.ix_(hit, cols)
                    block[ix] *= factor
                if clamp_now:
                    _clamp(block, ix, cap)
                regions.append(ix)
                applied = True

        # `w_max` can only ever bind where something was potentiated -- an
        # untouched cell still holds its initial weight, which is 1 (or the
        # inhibitory value) and cannot exceed the cap. So clamp the regions,
        # not the block: the whole-block pass is 2e6 elements at n=1e4, k=200.
        # Only reached for beta < 0; see `clamp_now` above for why beta >= 0
        # clamps in the loop instead.
        if applied and cap is not None and not clamp_now:
            for ix in regions:
                _clamp(block, ix, cap)
        return applied


class ExactAreaState(ActivityState):
    """Per-area state. No `compact_to_neuron_id`: the index IS the neuron id."""
    _activity_fields = ("winners", "w", "ever_fired", "fixed_assembly", "explicit_source")

    __slots__ = ("name", "n", "k", "beta", "winners", "fixed_assembly",
                 "beta_by_source", "ever_fired", "w", "explicit_source",
                 "compact_to_neuron_id", "winner_policy")

    def __init__(self, name: str, n: int, k: int, beta: float) -> None:
        self.name, self.n, self.k, self.beta = name, n, k, beta
        self.winners = np.empty(0, dtype=np.uint32)
        self.fixed_assembly = False
        # `Brain._project_impl` sets this on every source area. On the sparse
        # engine it means "these winners are neuron IDs, not compact indices"
        # -- a real distinction there ([[two-index-spaces-compact-vs-neuron-id]]).
        # Here the two spaces COINCIDE, so it is inert by construction rather
        # than ignored: there is no id remapping for it to control.
        self.explicit_source = False
        # PERMANENTLY EMPTY, and that is the correct value rather than a stub.
        # An empty mapping means "compact index IS the neuron id", which every
        # reader already handles (`Brain._project_impl` falls through to
        # `result.winners`), and here it is true by construction: this engine
        # never renumbers, because every neuron exists from the start. The
        # attribute exists at all because the parser's reset paths ASSIGN to it
        # (`parser_mixins/incremental.py`), and `__slots__` turned that into an
        # AttributeError that stopped the parser reading assemblies at all.
        self.compact_to_neuron_id: list = []
        # None means plain k-WTA. Anything else goes through
        # `compute.winner_selection.select_with_policy` -- see `add_area`.
        self.winner_policy = None
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

    #: Precision of the drive vector and the cache. NOT merely a speed knob:
    #: k-WTA is a COMPARISON, so precision decides how many neurons land in the
    #: tied band at the k-th boundary -- and a tie is resolved by index
    #: convention, not by the model. Lower precision therefore hands more of
    #: each assembly to the tie-break. See `research/notes/exact_drive_precision.md`
    #: for the measured tie-inflation and assembly divergence per dtype.
    #:
    #: float32 is the default because it is EXACT for the un-normalised drive
    #: (a count below 2^24) and is what `numpy_sparse` accumulates in, so it
    #: matches the arbiter rather than diverging from it.
    DEFAULT_DTYPE = np.float32

    #: Constructor kwargs `Brain` forwards to every engine that this one has no
    #: implementation for. Listed rather than swallowed by `**kwargs`, so that
    #: asking for one is an ERROR and not a mechanism that silently never runs
    #: -- the repo's dominant defect class, see [[silent-no-op-dead-fibers]].
    #: Each maps to the only value that means "not requested".
    supports_fiber_learning_masks = True

    _UNSUPPORTED_INIT = {
        "synaptic_scaling": False,
        "deterministic": False,   # this engine has no RNG stream to stabilise
    }

    # `norm_init` DEFAULTS TO FALSE, matching `numpy_sparse`, and the default
    # is load-bearing rather than a taste call: `Brain` forwards this kwarg
    # ONLY when it is True (`brain.py`, "so other engines' constructors are
    # unaffected" -- `numpy_explicit` does not accept it), so OMISSION IS HOW
    # `Brain` SAYS FALSE. An engine that defaults to True therefore silently
    # ignores `Brain(norm_init=False)` and runs the production substrate while
    # a literature reproduction believes it pinned the un-normalised one
    # ([[norm-init-substrate-vs-reference]]). This engine did exactly that
    # until it was measured. Pinned for every registered engine by
    # `test_engine_norm_init_contract`.
    def __init__(self, p: float, seed: int = 0, w_max: float = 20.0,
                 norm_init: bool = False, inhibitory_prob: float = 0.0,
                 inhibitory_weight: float = -1.0, dtype=None,
                 **kwargs) -> None:
        _reject_unsupported("NumpyExactEngine()", self._UNSUPPORTED_INIT, kwargs)
        self.p = float(p)
        self.seed = int(seed)
        self.w_max = w_max
        from .._homeostasis import HomeostasisConfig
        self.norm_init = HomeostasisConfig(norm_init=norm_init).norm_init
        self.dtype = np.dtype(dtype) if dtype is not None else self.DEFAULT_DTYPE
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
        # (src, tgt, assembly) -> base drive. O(assemblies * n), LRU-bounded.
        self._drive_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._drive_bytes = 0
        # (src, tgt) -> per-fiber connection probability; see add_connectivity.
        self._fiber_p: Dict[tuple, float] = {}
        # Counts projections so `add_connectivity` can refuse to change the
        # substrate under weights that are already written.
        self._projections = 0

    # -- wiring -------------------------------------------------------------

    def _pair_seed(self, source: str, target: str) -> int:
        return fnv1a_pair_seed(self.seed, source, target)

    #: Per-area mechanisms `Brain.add_area` forwards that this engine has no
    #: implementation for, with the value that means "not requested". Same
    #: rationale as `_UNSUPPORTED_INIT`: a k-WTA modifier that is configured
    #: and never applied changes the answer without changing the log.
    _UNSUPPORTED_AREA = {
        "refractory_period": 0,
        "inhibition_strength": 0.0,
        "input_noise_std": 0.0,
        "slot_count": 0,
    }

    def add_area(self, name: str, n: int, k: int, beta: float,
                 refractory_period: int = 0,
                 inhibition_strength: float = 0.0,
                 winner_policy=None, **kwargs) -> None:
        """`winner_policy` selects the competition rule; None is plain k-WTA.

        SUPPORTED HERE, and it matters where it is supported. E%-WTA (Hoff et
        al. 2026) makes assembly SIZE emergent -- the firing set is
        `{j : h_j in [(1-eps) h_max, h_max]}` rather than a fixed k. Running
        that on `numpy_sparse` means running an emergent-size rule on top of
        drive the candidate sampler INVENTED for neurons that have not fired,
        so the size it discovers is partly a property of the sampler. This
        engine rejected policies outright, which left every E%-WTA and
        literature-parity result in the repository sampler-side by default.

        One property worth stating: `select_with_policy` takes an optional
        `population_sigma` for `EPercentPolicy(window="sigma")`, because a
        caller whose feature vector is not the whole population has to estimate
        the spread. Here the drive vector IS the whole population, so the
        estimate is unnecessary and the sigma window is exact by construction.

        `input_noise_std` remains unsupported and is not an oversight: it needs
        an RNG stream, and this engine deliberately has none -- that is what
        makes it reproducible by content-addressing rather than by seeding.
        """
        _reject_unsupported(
            f"NumpyExactEngine.add_area({name!r})", self._UNSUPPORTED_AREA,
            dict(kwargs, refractory_period=refractory_period,
                 inhibition_strength=inhibition_strength))
        area = ExactAreaState(name, n, k, beta)
        area.winner_policy = winner_policy
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
                             self._p_of(stim_name, area_name)),
            dtype=np.float64)
        self._stim_base[stim_name][area_name] = base
        self._stim_pot[stim_name][area_name] = np.zeros(n, dtype=np.float64)

    def add_connectivity(self, source: str, target: str, p: float) -> None:
        """Set this fiber's connection probability, overriding the global `p`.

        WHY THIS EXISTS.  Mitropolsky & Papadimitriou (2025) do not give every
        fiber the same density: "four of these 2m + 6 fibers ... have increased
        parameters beta AND p, making them stronger conduits of synaptic
        input". Those four are what makes the noun/verb split emerge without a
        label -- LEX1 is wired more densely to VISUAL, LEX2 to MOTOR, and the
        class of a word falls out of which lexical area can hold a stable
        assembly for it. Per-fiber beta already exists (`set_beta`); per-fiber
        `p` did not, so that architecture was unbuildable here.

        WHY IT IS THIS METHOD AND NOT A NEW ONE.  `add_connectivity(source,
        target, p)` was already on the engine interface, already documented in
        `engine.py` with a worked example -- and was `pass` in all three
        engines. Every caller that believed it had set a per-fiber density
        silently got the global one. Implementing the existing name is the fix;
        adding `set_fiber_p` beside it would have left the trap in place.

        STRUCTURAL, SO IT MUST PRECEDE TRAFFIC.  `p` selects which synapses
        exist. Changing it after a fiber has carried drive would leave
        potentiation sitting on synapses that no longer exist and silently
        rewrite already-formed assemblies, so this raises instead. Stimulus
        fibers are re-wired here because `_wire_stim` baked the old `p` into
        `_stim_base` at `add_stimulus` time.
        """
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}")
        if self._projections and p != self._p_of(source, target):
            raise RuntimeError(
                f"add_connectivity({source!r}, {target!r}, p={p}) after "
                f"{self._projections} projections. Connectivity is structural: "
                f"changing p decides which synapses EXIST, and potentiation "
                f"already written would be left on synapses that no longer do. "
                f"Set connectivity before any projection.")
        self._fiber_p[(source, target)] = p
        self._norm_cache.pop((source, target), None)
        self._stim_norm_cache.pop((source, target), None)
        if source in self._stimuli and target in self._areas:
            self._wire_stim(source, self._stimuli[source].size, target)

    def _p_of(self, source: str, target: str) -> float:
        """This fiber's connection probability, defaulting to the global `p`."""
        return self._fiber_p.get((source, target), self.p)

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
        return hash_area_rows(
            rows, n_cols, self._pair_seed(source, target),
            self._p_of(source, target),
            self.inhibitory_prob, self.inhibitory_weight)

    def _fiber_cells(self, source: str, target: str, rows: np.ndarray,
                     cols: np.ndarray) -> np.ndarray:
        """Initial weights on an arbitrary rows x cols GATHER.

        The potentiated part of a fiber sits on scattered columns, so a
        contiguous range would span nearly all of `n` and there would be no
        saving. This is the only block the plastic path materialises.
        """
        return hash_area_cells(
            rows, cols, self._pair_seed(source, target),
            self._p_of(source, target),
            self.inhibitory_prob, self.inhibitory_weight)

    def _fiber_sum(self, source: str, target: str, rows: np.ndarray,
                   n_cols: int) -> np.ndarray:
        """Column sums of `rows` x [0, n_cols) WITHOUT materialising the block.

        The un-potentiated drive only ever needs the sum, so the (k, n) block
        need not exist: at n=1e4, k=200 that is an 8 MB allocation per round
        bought for nothing. Used whenever the fiber carries no potentiation
        (every readout, and every fiber before it has learned anything).

        The Rust kernel fuses the sum into the hash loop and parallelises over
        COLUMNS, so no accumulator is shared between threads and the answer
        does not depend on how many cores ran it.

        Where the fiber's edge list is cached, the sum is a scatter-add over
        `k * n * p` entries instead of `k * n` hash evaluations -- 14.6x at
        n=1e4, p=0.05, and 52x at p=0.01. Both paths give the same numbers;
        `_fiber_csr` decides only on whether the cache is affordable.
        """
        return self._cached_fiber_sum(source, target, rows, n_cols)

    #: Budget for the base-drive cache. See `_cached_fiber_sum` for why this is
    #: the cache that scales; it is bounded so a protocol whose assemblies never
    #: repeat degrades to recomputation instead of exhausting memory.
    DRIVE_CACHE_BUDGET_BYTES = 512 * 1024 * 1024

    def _cached_fiber_sum(self, source: str, target: str, rows: np.ndarray,
                          n_cols: int) -> np.ndarray:
        """Base drive for one assembly through one fiber, memoised.

        CACHE THE DRIVE, NOT THE GRAPH. An edge list is `n_pre * n_post * p`
        entries -- the same asymptote as the dense matrix it replaces, so it
        only postpones the memory wall. The base drive of an ASSEMBLY is one
        `n`-vector, and there are only as many of them as there are distinct
        assemblies, so this is O(A * n): linear in n rather than quadratic.

        Measured per fiber at k*p = 10, k ~ sqrt(n), 50 assemblies:

            n        dense n^2   full CSR    hot-row edges   THIS
            1e4         400 MB      40 MB           20 MB     1 MB
            1e5          40 GB    1.27 GB          200 MB    10 MB
            1e6           4 TB      40 GB            2 GB   100 MB
            1e7         400 TB    1.27 PB           20 GB     1 GB

        The edge/drive ratio is exactly `2*k*p`, independent of n.

        It is safe to memoise forever because G(n,p) is drawn at t=0 and never
        changes, so an assembly's BASE drive is immutable; everything learned is
        applied as a separate correction by the caller.

        Hit rate is high for the reason the low-rank potentiation store works --
        assemblies repeat. Measured on a study-I protocol: 3,168 presentations
        over 50 distinct source assemblies, 63.4x reuse, **98.4% hits**.
        """
        arr = np.ascontiguousarray(rows, dtype=np.int64)
        key = (source, target, arr.tobytes())
        hit = self._drive_cache.get(key)
        if hit is not None:
            self._drive_cache.move_to_end(key)
            return hit.copy()

        val = np.asarray(hash_area_rows(
            arr, n_cols, self._pair_seed(source, target),
            self._p_of(source, target),
            self.inhibitory_prob, self.inhibitory_weight, want_sum=True),
            dtype=np.float64)
        # NORM FOLDED IN BEFORE STORING. `1/d_j` is immutable per fiber, so a
        # cache hit then needs no multiply at all -- one fewer O(n) pass, and
        # at n=3e5 that pass cost 1.2 ms.
        scale = self._area_norm(source, target)
        if scale is not None:
            val = val * scale
        store = val.astype(self.dtype)
        self._drive_cache[key] = store
        self._drive_bytes += store.nbytes
        while (self._drive_bytes > self.DRIVE_CACHE_BUDGET_BYTES
               and self._drive_cache):
            _, evicted = self._drive_cache.popitem(last=False)
            self._drive_bytes -= evicted.nbytes
        return store.copy()

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
        fp = self._p_of(source, target)
        deg = np.asarray(
            hash_area_indegree(n_pre, n_post, self._pair_seed(source, target),
                               fp),
            dtype=np.float64)
        scale = inverse_indegree(deg, n_pre, n_pre, fp, xp=np)
        self._norm_cache[key] = scale
        return scale

    def _stim_norm(self, stim: str, target: str) -> Optional[np.ndarray]:
        """`1/d_j` for a stimulus fiber, over the IMPLICIT input population.

        THE OBVIOUS READING IS DEGENERATE, and it was what this engine did
        first. A stimulus fires in full and is stored pre-summed, so the stored
        base count IS the neuron's in-degree from the stimulus -- and dividing
        it by itself gives exactly 1.0 at every neuron. A constant cannot be
        ranked, so k-WTA falls through to the index tie-break and EVERY
        stimulus elects the same k neurons. Measured, before the fix: overlap
        0.89 between assemblies built from unrelated stimuli, identical whether
        norm_init was on or off and whether recurrence was on or off -- three
        arms agreeing to four decimals, which is what a degenerate arm looks
        like ([[fake-perfect-probe-signatures]]).

        The reference's inputs are AREAS of size n with only k neurons active,
        so every fiber delivers drive of order k/n. A stimulus of size s is the
        same object: the active cap of an implicit input population of size
        `n_post`. Charging the other `n_post - s` rows at the ambient rate `p`
        restores that geometry, so stimulus drive ~ s/n and recurrent drive ~
        k/n compete on equal terms. `numpy_sparse._norm_scale` documents this
        and passes `n_pre=tgt.n, rows_known=stim.size`; this line is the same
        law, which is why the two engines can be compared at all.
        """
        if not self.norm_init:
            return None
        key = (stim, target)
        cached = self._stim_norm_cache.get(key)
        if cached is not None:
            return cached
        size = self._stimuli[stim].size
        deg = self._stim_base[stim][target]
        scale = inverse_indegree(deg, self._areas[target].n, size,
                                 self._p_of(stim, target), xp=np)
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
        # Marks the substrate as in use; `add_connectivity` refuses after this.
        self._projections += 1

        if tgt.fixed_assembly:
            # A FIXED TARGET STILL LEARNS. Only the winners are pinned.
            #
            # This engine used to return here having applied no plasticity at
            # all, which is the divergence `numpy_sparse` fixed and documented
            # at `_fixed_target_plasticity_enabled` -- the reference pins the
            # winners and skips RECRUITMENT, then potentiates the afferents
            # onto those frozen winners. Skipping the potentiation silently
            # breaks every protocol whose whole point is to write INTO a held
            # assembly: `reciprocal_project`, `associate`, and the paper's
            # PHON -> LEX -> PHON round trip, which needs LEX -> PHON trained
            # while PHON is held at the word ([[fixing-an-area-must-still-learn]]).
            #
            # Measured before the fix, on the acquisition harness: trained
            # round-trip recall 0.00-0.20 against an UNTRAINED control of
            # 0.06-0.18 -- i.e. training bought exactly nothing, silently.
            #
            # The gate is imported from `_sparse` rather than restated, so the
            # two engines cannot drift apart on the policy the way they twice
            # drifted apart on the pricing law ([[pricing-law-implemented-twice]]).
            learn = (plasticity_enabled and self._plasticity_enabled_global
                     and (from_stimuli or from_areas)
                     and _fixed_target_learns())
            if learn:
                self._apply_plasticity(
                    target, from_stimuli, from_areas,
                    np.asarray(tgt.winners, dtype=np.int64))
            return ProjectionResult(
                winners=np.array(tgt.winners, dtype=np.uint32),
                num_first_winners=0, num_ever_fired=tgt.num_ever_fired)

        n = tgt.n
        # FLOAT32 END TO END. The drive is a small count (Binomial(k, p)),
        # which float32 represents exactly, and `numpy_sparse` accumulates in
        # float32 too -- so this matches the arbiter rather than diverging from
        # it. Measured at n=3e5: the f64 round spent 5.0 ms in `argpartition`
        # against 1.8 ms in f32, plus 0.9 ms upcasting.
        drive = np.zeros(n, dtype=self.dtype)

        for stim in from_stimuli:
            base = self._stim_base[stim][target]
            pot = self._stim_pot[stim][target]
            beta = tgt.beta_by_source.get(stim, tgt.beta)
            w = base * self._clamped(np.power(1.0 + beta, pot))
            scale = self._stim_norm(stim, target)
            drive += np.asarray(w if scale is None else w * scale,
                                dtype=np.float32)

        for src_name in from_areas:
            src = self._areas[src_name]
            rows = np.asarray(src.winners, dtype=np.int64)
            beta = tgt.beta_by_source.get(src_name, tgt.beta)
            pot = self._area_pot.get((src_name, target))
            # THE BULK IS ALWAYS THE FUSED SUM. Learning only ever perturbs the
            # columns the stored outer products reach, so the drive is
            #
            #     d = sum_i f(i, j)            <- fused, nothing materialised
            #       + sum_i (w_ij - f(i, j))   <- confined to touched columns
            #
            # and the (k, n) block never has to exist even on the plastic path.
            # Already normalised: `_cached_fiber_sum` folds the per-neuron
            # 1/d_j in before storing, so a cache hit needs no multiply at all.
            summed = self._fiber_sum(src_name, target, rows, n)
            cols = (None if (pot is None or beta == 0)
                    else pot.touched_cols(rows))
            if cols is not None:
                sub = self._fiber_cells(src_name, target, rows, cols)
                # Reduce in float32, not float64. The block is float32, so a
                # float64 accumulator forces an upcast of every element on both
                # the before and after sums -- 1.1 ms of the round at n=3e5,
                # its single largest item. Summing k terms each bounded by
                # w_max carries a relative error of about sqrt(k)*eps ~ 1e-6,
                # far below the spacing the k-WTA boundary resolves.
                before = sub.sum(axis=0, dtype=np.float32)
                # `w_max` is a ceiling in MULTIPLES of the initial weight, and
                # storage is on the unit scale, so clamping the weight IS
                # clamping the multiplier. Capped per potentiated region.
                pot.apply_to(sub, rows, beta, self.w_max, cols)
                corr = sub.sum(axis=0, dtype=np.float32) - before
                # `summed` is already normalised; the correction is raw, so it
                # takes the same per-neuron scale before being added in.
                scale = self._area_norm(src_name, target)
                if scale is not None:
                    corr = corr * scale[cols]
                summed[cols] += corr.astype(summed.dtype)
            drive += summed

        if external_drive is not None and len(external_drive) == n:
            drive += np.asarray(external_drive, dtype=self.dtype)

        winners = self._select_winners(drive, tgt)

        if plasticity_enabled and self._plasticity_enabled_global:
            self._apply_plasticity(target, from_stimuli, from_areas, winners)

        tgt.winners = np.asarray(winners, dtype=np.uint32)
        tgt.ever_fired[winners] = True

        result = ProjectionResult(
            winners=np.array(tgt.winners, dtype=np.uint32),
            num_first_winners=0,
            num_ever_fired=tgt.num_ever_fired,
            total_activation=float(drive[winners].sum()))
        if record_activation:
            # The ERP components read PRE-k-WTA energy, because post-k-WTA the
            # winner set is renormalised and the sign of the effect flips under
            # norm_init -- see [[erp-prekwta-not-postkwta]]. `pre_kwta` here is
            # the SUMMED AFFERENT DRIVE, the same quantity `numpy_sparse` snaps
            # from `all_inputs`, and on this engine it is exact rather than
            # part-sampled.
            #
            # `pre_kwta_prev_only` is left EMPTY, not zero-filled: this engine
            # has no separate previous-winner input path to split out, and a
            # zero vector of length n would read as "measured, and it was zero".
            result.pre_kwta_inputs = np.array(drive, dtype=np.float32, copy=True)
            result.pre_kwta_prev_only = np.zeros(0, dtype=np.float32)
            result.pre_kwta_total = float(drive.sum())
            result.pre_kwta_count = int(drive.size)
        return result

    def _apply_plasticity(self, target: str, from_stimuli: List[str],
                          from_areas: List[str], winners: np.ndarray) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-fiber-learning

        Potentiate permitted afferents of `winners` in `target`.

        Extracted so the ordinary path and the FIXED-TARGET path share one
        implementation. They previously did not share anything -- the fixed
        path applied no plasticity at all -- and the cost of writing this twice
        is on record in this repository under
        [[pricing-law-implemented-twice]].

        `winners` are indices into `target`; on this engine they are neuron ids
        and compact indices at once, so no remapping is needed here (see the
        module docstring).
        """
        tgt = self._areas[target]
        for stim in from_stimuli:
            if (tgt.beta_by_source.get(stim, tgt.beta) != 0
                    and self.fiber_learning_allowed(stim, target)):
                self._stim_pot[stim][target][winners] += 1.0
        for src_name in from_areas:
            if (tgt.beta_by_source.get(src_name, tgt.beta) == 0
                    or not self.fiber_learning_allowed(src_name, target)):
                continue
            key = (src_name, target)
            store = self._area_pot.get(key)
            if store is None:
                store = self._area_pot[key] = _Potentiation()
            store.bump(np.asarray(self._areas[src_name].winners,
                                  dtype=np.int64),
                       np.asarray(winners, dtype=np.int64))

    def _clamped(self, mult: np.ndarray) -> np.ndarray:
        """`w_max` is a ceiling in MULTIPLES of the initial weight.

        Storage stays on the unit scale (that is what makes the norm_init
        read-time scale legal), so the clamp applies to the multiplier.
        """
        if self.w_max is None:
            return mult
        return np.minimum(mult, float(self.w_max))

    #: Oversample factor for the probabilistic pivot, and the sampling stride.
    #: 4x leaves the candidate set ~4k, small enough to refine cheaply and
    #: loose enough that the pivot essentially never overshoots.
    PIVOT_OVERSAMPLE = 4
    PIVOT_STRIDE = 64

    @staticmethod
    def _exact_topk(drive: np.ndarray, k: int,
                    cand: Optional[np.ndarray] = None) -> np.ndarray:
        """Top-k by drive, ties broken by LOWEST INDEX.

        `cand`, when given, must be an ASCENDING index array that provably
        contains the top k -- the tie-break relies on that ordering.
        """
        vals = drive if cand is None else drive[cand]
        thresh = np.partition(vals, vals.size - k)[vals.size - k]
        if cand is None:
            above = np.flatnonzero(drive > thresh)
            tied = None
        else:
            above = cand[vals > thresh]
        if above.size >= k:
            return above[:k].astype(np.int64)
        tied = (np.flatnonzero(drive == thresh) if cand is None
                else cand[vals == thresh])
        return np.sort(np.concatenate(
            [above, tied[:k - above.size]])).astype(np.int64)

    def _select_winners(self, drive: np.ndarray, tgt) -> np.ndarray:
        """k-WTA unless the area carries a `winner_policy`.

        The plain path is kept separate because it is the hot one and has its
        own probabilistic-pivot optimisation; policies go through the shared
        `compute.winner_selection` implementation so the two engines cannot
        disagree about what a policy MEANS ([[pricing-law-implemented-twice]]).
        """
        policy = getattr(tgt, "winner_policy", None)
        if policy is None:
            return self._select(drive, tgt.k)
        from ...compute.winner_policies import TopKPolicy
        if isinstance(policy, TopKPolicy) and policy.k == tgt.k:
            return self._select(drive, tgt.k)
        from ...compute.winner_selection import WinnerSelector
        # `WinnerSelector` takes an RNG for the policies that need one; none of
        # the paths reached here do, and this engine deliberately holds no RNG
        # stream -- content-addressing is what makes it reproducible. A fresh
        # default_rng is passed rather than None so a future policy that does
        # sample fails loudly at ITS call site instead of on an AttributeError.
        selected = WinnerSelector(np.random.default_rng(0)).select_with_policy(
            drive, policy)
        return np.asarray(to_cpu(selected), dtype=np.int64)

    def _select(self, drive: np.ndarray, k: int) -> np.ndarray:
        """Top-k with a PROBABILISTIC PIVOT and an EXACT result.

        Guess a threshold from a cheap sample; if the set above it holds at
        least k elements then the true top-k is PROVABLY inside it (anything
        excluded is below the pivot, hence below the k-th largest of a set that
        already has k members above the pivot). So the guess only decides how
        much work the exact refinement does -- never what comes out. A pivot
        that overshoots is caught by the size check and costs one fallback.

        Measured at n=3e5, k=548, identical winners in every case:

            argpartition on indices        4.879 ms
            partition on values            2.182 ms
            histogram select               5.541 ms   <- SLOWER, more passes
            sampled pivot + verify         0.167 ms   <- 13x

        The candidate set is ~2,283 for k=548, so the refinement sorts 2e3
        elements instead of 3e5. Fallback rate over 200 assemblies: 0/200.

        The sample is a fixed STRIDE, not a random draw: it needs no RNG, so it
        cannot perturb reproducibility, and correctness does not depend on the
        sample being unbiased -- only the amount of work does.
        """
        n = drive.size
        k = int(min(k, n))
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        if k >= n:
            return np.arange(n, dtype=np.int64)
        stride = self.PIVOT_STRIDE
        # Only worth a pivot when the array is much larger than both the sample
        # and k; otherwise the full partition is already cheap.
        if n >= 8192 and k * stride < n:
            sample = drive[::stride]
            m = sample.size
            want = min(m - 1, max(1, int(self.PIVOT_OVERSAMPLE * k * m / n)))
            pivot = np.partition(sample, m - want)[m - want]
            cand = np.flatnonzero(drive > pivot)
            if cand.size >= k:
                return self._exact_topk(drive, k, cand)
        return self._exact_topk(drive, k)

    # -- accessors ----------------------------------------------------------

    def get_winners(self, area: str) -> np.ndarray:
        return np.array(to_cpu(self._areas[area].winners), dtype=np.uint32)

    def set_winners(self, area: str, winners: np.ndarray) -> None:
        """Specification: neural_assemblies/ir/VERIFICATION.md#contract-winner-inputs"""
        st = self._areas[area]
        st.winners = validated_indices(winners, upper=st.n, label=f"{area} winners",
                                       xp=np, unique=True)
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
