"""Which wired mechanisms in this repo NEVER fire?

The recurring failure in this project is not a wrong number, it is a mechanism
that silently does not run. A projection with no drive still returns k winners;
a guarded branch whose guard is never true still ships, is still documented, and
is still cited in prose as "the model does X". The signature in results is
"mechanism X turns out to have surprisingly little effect", which is
indistinguishable from a real negative by inspection of the numbers alone.

One instance is already established: `Brain._apply_mutual_inhibition` fires only
when >= 2 areas of a declared group are targets of the SAME `project()` call,
and across a full suite run ZERO calls satisfy that. Every claim phrased "the
paper's inter-area inhibition does X" is therefore a claim about a variant.

Measured here on the 2026-07-30 slice, that stands with one refinement worth
stating precisely, because "dormant" and "unreachable" are not the same claim.
Of 142,844 `project()` calls made on a brain that had declared a group, ONE
co-targeted -- and it was `test_mechanisms_fire.py`, a test written expressly to
force the mechanism. `EmergentParser`, which declares the ROLE group, still
never co-targets (its guard test remains XFAIL). But `NemoParser(competitive=
True)` DOES: its project map is DERIVED from `InhibitionState` rather than named
by the caller, so ROLE_AGENT and ROLE_PATIENT land in one call and the paper's
inhibition runs (3 co-targets in a single 3-word parse). Both tests that reach
it are marked `slow`, so the dev slice never sees it. The mechanism is reachable
by exactly one route, and nothing in the default suite takes it.

That figure was found reactively, by one investigation. `fiber_census`, the
`ASSEMBLIES_STRICT_DRIVE` guard and `pricing_exposure` all exist but have only
ever been pointed at a suspect after the fact. This turns them around: run the
whole suite and ask, of every mechanism, whether its firing condition was ever
met by anything.

HOW.  Same trick as `research/experiments/pricing_exposure_sweep.py`: the
interesting constructions live inside pytest cases, not behind importable entry
points, so rather than reverse-engineering entry points this installs tracers on
`Brain` / the engines and lets the TEST SUITE drive them.

FOUR THINGS ARE MEASURED
------------------------
1.  DEAD FIBERS INTO LIVE AREAS (`diagnostics.FiberState.silently_ignored`).
    Measured at the point of USE, not only as an end-state census: for every
    `project_into`, each named source area whose weight block is still 0-column
    while the target already has neurons delivered exactly zero from that
    source. Note that a fiber being dead on its FIRST use is normal -- the
    deferred-init path exists to fix precisely that on the next round -- so the
    reported statistic is dead-uses / total-uses per fiber, and a fiber dead on
    EVERY use is the finding. An end-of-test whole-brain census is run too, so
    fibers materialized but never driven are also visible.

2.  ZERO-DELIVERED-DRIVE PROJECTIONS. `ProjectionResult.total_activation` is the
    summed input to the winners. Zero means the k-WTA decided on nothing. The
    engine has three early returns that produce this and they are NOT equally
    interesting, so they are classified rather than pooled:
      * `no inputs`   -- called with neither stimuli nor source areas. Internal,
                         harmless, and mostly an artifact of the batch config.
      * `FIXED target`-- the target is fixed, so inputs are discarded. A no-op
                         by design, but a no-op the caller usually did not mean.
      * `cold target` -- target had w == 0 before the call, i.e. this is its
                         first materialization; drive is genuinely undefined.
      * `ZERO DRIVE`  -- had inputs, not fixed, target already grown, and the
                         delivered total was exactly 0.0. This is the dangerous
                         one; it is the case `ASSEMBLIES_STRICT_DRIVE` warns on.

3.  MUTUAL INHIBITION, specifically. Counts (a) `project()` calls, (b) calls on
    a brain that has at least one group declared, (c) calls where >= 2 areas of
    one group are co-targeted -- the paper's actual firing condition -- and
    (d) calls where the mechanism actually silenced an area. This confirms or
    refutes the standing "zero" on the CURRENT code.

4.  GATE CENSUS. A curated list of other wired mechanisms, each wrapped so that
    both its CALL count and its FIRE count are recorded -- fire meaning the
    guard inside it was satisfied. Called-but-never-fired is the shape being
    hunted; never-called is weaker evidence (the slice may just not exercise it)
    and is reported separately rather than pooled with it.

MEASURED POWER (`--validate`).  A detector whose true-positive case has never
been constructed is a detector of unknown sensitivity, and this one is aimed at
a failure that by definition leaves no trace in the numbers. So the tool ships
with its own known positive: the reciprocal protocol on the TORCH engine, where
the `B->A` weight block is never materialised (`CSRConn` with rows=cols=nnz=0),
the back-projection therefore delivers `total_activation == 0.0`, the engine's
"Zero signal -- preserve current assembly" branch returns the incumbent winners
and restoration reads EXACTLY 1.0000. `--validate` runs it and asserts that the
at-use detector fires on `B->A` and does NOT fire on the live `A->B` beside it,
which is the half that makes the check meaningful. The same protocol on the
numpy engine is the negative control (block 92x93, nnz 446, activation 330.4,
restoration 0.6000), so the finding is engine-specific rather than protocol-
specific.

That known positive also measured the tool this sweep was built to turn around.
`diagnostics.fiber_census` reads `conn.weights`; the torch engine's `CSRConn`
has no such attribute, so EVERY torch fiber reads shape (0, 0) and is reported
`silently_ignored`. On the validation brain it flags 4 of 4 fibers, including
the demonstrably live `A->B` (nnz 1314). On the torch engine that function
cannot distinguish the true positive from a healthy fiber -- which is why the
dimension reader here is `_conn_dims` and not `getattr(conn, "weights")`.

WHAT THIS CANNOT SEE.  `Brain`, the two numpy engines and (when importable) the
torch engine are instrumented; the CuPy and C++ engines are not. The default
slice additionally EXCLUDES the GPU test files, so torch coverage in a default
run comes only from tests that reach the torch engine incidentally -- run
`--validate`, or name a torch file explicitly, to exercise that path. A
mechanism that fires once in a test this slice does not run reads as "never
called" here; that is why never-called and called-but-never-fired are kept
apart.

Usage::

    uv run python research/experiments/dormant_mechanism_sweep.py --validate
    uv run python research/experiments/dormant_mechanism_sweep.py
    uv run python research/experiments/dormant_mechanism_sweep.py -k inhibition
    uv run python research/experiments/dormant_mechanism_sweep.py \
        neural_assemblies/tests/test_emergent_parser.py

The default slice is ~770 tests and is dominated by a handful of very heavy
ones, so in practice run it SHARDED -- split the file list N ways, give each
shard ``--sweep-json <path>``, and sum the JSONs (every statistic here is a
count, so the merge is exact). Measured: 8 shards over 8 processes finished in
6 min wall clock against >2 h serial. See `dump_json`.
"""
import os
import sys
import weakref
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# The slice. `-m "not slow"` over the test tree, minus every file that touches
# torch/CUDA. That exclusion costs this sweep almost nothing and buys a lot:
# only `Brain` and the two NUMPY engines are instrumented, so a torch-only test
# contributes no traced projection at all -- while on a shared card it can
# stall for minutes per test. (Measured mid-run: 9 compute processes on the one
# RTX 3080, and the slice crawled to ~1 test / 30 s inside the batched-torch
# files.) `test_cuda_kernels` is separately stale -- task #40, 9 of 15 fail.
_GPU_FILES = [
    "test_backend.py", "test_batched_next_token.py", "test_batched_projection.py",
    "test_batched_trainer.py", "test_cross_engine_projection.py",
    "test_cuda_kernels.py", "test_docs_examples_smoke.py", "test_engine_parity.py",
    "test_engine_pricing.py", "test_epwta_gpu.py", "test_image_activation.py",
    "test_literature_parity.py", "test_stim_init_engine_parity.py",
    "test_torch_parity.py", "test_training_perf.py",
    # Not a GPU file: it asserts determinism ACROSS PROCESSES, so the work
    # happens in subprocesses and the tracer -- which lives in this one --
    # would record none of it while paying the full runtime.
    "test_cross_process_determinism.py",
]
DEFAULT_ARGS = (
    ["neural_assemblies/tests/", "-m", "not slow"]
    + [f"--ignore=neural_assemblies/tests/{f}" for f in _GPU_FILES]
)

# Measured slice, 2026-07-30: 83 of the 99 test files, 764 passed / 1 skipped /
# 107 deselected / 3 xfailed / 1 xpassed, 275,834 `Brain.project()` calls,
# 631,925 engine-level projections. 8 shards over 8 processes, 5m39s wall clock
# for the longest.


# ---------------------------------------------------------------------------
# Weight-block dimensions, across engine representations
# ---------------------------------------------------------------------------

def _conn_dims(conn):
    """(rows, cols, live) for one area->area weight block, any engine.

    NOT `getattr(conn, "weights", None).shape`. That is what
    `diagnostics.fiber_census` does, and it is why that function is blind on the
    torch engine: `CSRConn` stores `_nrows` / `_ncols` / `_col` / `_val` and has
    no `weights` attribute at all, so the shape read falls back to (0, 0) and
    every torch fiber -- live ones included -- is reported dead. Measured on the
    reciprocal known positive: 4 of 4 fibers flagged, one of which carried
    nnz=1314 and drove the projection that materialised it.

    `live` is deliberately not just "cols > 0": a CSR block can carry the right
    dimensions with no stored entries, and a numpy block can be allocated and
    then zeroed by `reset_area_connections`. Rows are returned separately
    because row-side death is invisible to a column-based census (see the
    ROW-dead accounting in the report).
    """
    if conn is None:
        return 0, 0, False
    if hasattr(conn, "_nrows"):                     # torch CSRConn
        return int(conn._nrows), int(conn._ncols), bool(conn.nnz)
    w = getattr(conn, "weights", None)
    if w is None:
        return 0, 0, False
    shape = tuple(getattr(w, "shape", (0, 0)) or (0, 0))
    rows, cols = (shape + (0, 0))[:2]
    return int(rows), int(cols), bool(rows and cols)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    def __init__(self):
        self.current_test = None

        # -- topology
        self.areas = defaultdict(dict)          # bid -> {name: (n, k, explicit)}
        self.mi_groups = defaultdict(list)      # bid -> [[area, ...]]
        self.brains = []                        # [(bid, weakref)]

        # -- project() accounting
        self.project_calls = 0
        self.impl_calls = 0
        self.mi_declared_calls = 0              # call on a brain with a group
        self.mi_cotarget_calls = 0              # >=2 group areas co-targeted
        self.mi_fired_calls = 0                 # actually silenced someone
        self.mi_cotarget_examples = []
        self.mi_cotarget_by_file = Counter()
        self.mi_declared_by_file = Counter()

        # -- per project_into accounting
        self.into_calls = 0
        self.drive_class = Counter()            # class -> count
        self.zero_drive = []                    # (test, target, srcs) samples
        self.zero_drive_sites = Counter()       # (test, target, srcs) -> count

        # -- fiber use, keyed (bid, src, dst)
        self.fiber_uses = Counter()
        self.fiber_dead_uses = Counter()        # no columns  (fiber_census sees)
        self.fiber_rowdead_uses = Counter()     # no rows     (it does NOT)
        self.fiber_test = {}                    # (bid,src,dst) -> test id

        # -- gate census
        self.gate_calls = Counter()
        self.gate_fires = Counter()
        self.gate_tests = defaultdict(set)

        # -- end-of-test whole-brain census
        self.census_fibers = 0
        self.census_dead_into_live = 0          # dead block, target has neurons
        self.census_dead_undriven = 0           # ... and nothing ever drove it
        self.census_dead_driven = Counter()     # ... but something DID drive it

    # -- brain registry ----------------------------------------------------

    def note_brain(self, b):
        bid = id(b)
        if bid not in self.areas:
            self.areas[bid] = {}
            try:
                self.brains.append((bid, weakref.ref(b)))
            except TypeError:                                # noqa: BLE001
                pass
        return bid

    def note_area(self, b, name, n, k, explicit):
        self.areas[self.note_brain(b)][name] = (int(n), int(k), bool(explicit))

    # -- gate census -------------------------------------------------------

    def gate(self, label, fired):
        self.gate_calls[label] += 1
        if fired:
            self.gate_fires[label] += 1
        if self.current_test:
            self.gate_tests[label].add(self.current_test)


TRACER = Tracer()


# ---------------------------------------------------------------------------
# Gate probes
# ---------------------------------------------------------------------------
#
# Each entry wraps one function so that CALLS and FIRES are counted separately.
# `fired` is a predicate over (self, args, kwargs, result); returning True means
# the guarded body actually ran. Keep the predicates cheap -- these sit on hot
# paths and the point is to run a whole suite under them.

def _wrap_gate(owner, attr, label, fired):
    orig = getattr(owner, attr, None)
    if orig is None:
        return False

    def wrapper(self, *a, **kw):
        out = orig(self, *a, **kw)
        try:
            TRACER.gate(label, bool(fired(self, a, kw, out)))
        except Exception:                                    # noqa: BLE001
            TRACER.gate(label + " [probe-error]", False)
        return out

    setattr(owner, attr, wrapper)
    return True


def _wrap_gate_fn(module, attr, label, fired):
    """Same, for a module-level function (no `self`)."""
    orig = getattr(module, attr, None)
    if orig is None:
        return False

    def wrapper(*a, **kw):
        out = orig(*a, **kw)
        try:
            TRACER.gate(label, bool(fired(None, a, kw, out)))
        except Exception:                                    # noqa: BLE001
            TRACER.gate(label + " [probe-error]", False)
        return out

    setattr(module, attr, wrapper)
    return True


ALWAYS = lambda s, a, kw, out: True                          # noqa: E731
TRUTHY = lambda s, a, kw, out: bool(out)                     # noqa: E731


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install():
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.core.numpy_engine._sparse import NumpySparseEngine
    from neural_assemblies.core.numpy_engine._explicit import NumpyExplicitEngine
    from neural_assemblies.core import inhibition as inh_mod

    # -- topology ----------------------------------------------------------
    orig_add_area = Brain.add_area

    def add_area(self, name, n, k, *a, **kw):
        out = orig_add_area(self, name, n, k, *a, **kw)
        TRACER.note_area(self, name, n, k, bool(kw.get("explicit", False)))
        return out

    Brain.add_area = add_area

    orig_add_mi = Brain.add_mutual_inhibition

    def add_mutual_inhibition(self, area_names):
        TRACER.mi_groups[TRACER.note_brain(self)].append(list(area_names))
        TRACER.gate("Brain.add_mutual_inhibition (group declared)", True)
        return orig_add_mi(self, area_names)

    Brain.add_mutual_inhibition = add_mutual_inhibition

    # -- project() ---------------------------------------------------------
    orig_project = Brain.project

    def project(self, *a, **kw):
        TRACER.project_calls += 1
        return orig_project(self, *a, **kw)

    Brain.project = project

    orig_impl = Brain._project_impl

    def _project_impl(self, areas_by_stim, dst_areas_by_src_area,
                      verbose=0, external_drive=None):
        TRACER.impl_calls += 1
        bid = TRACER.note_brain(self)

        targets = set()
        for areas in (areas_by_stim or {}).values():
            targets.update(areas)
        for dsts in (dst_areas_by_src_area or {}).values():
            targets.update(dsts)

        # THE PAPER'S FIRING CONDITION, evaluated here rather than inside
        # _apply_mutual_inhibition, because that method is only reached when a
        # group exists at all -- and the question is how often the co-target
        # condition holds, not how often the method is entered.
        groups = getattr(self, "_mutual_inhibition_groups", None) or []
        if groups:
            TRACER.mi_declared_calls += 1
            f = (TRACER.current_test or "?").split("::")[0]
            TRACER.mi_declared_by_file[f] += 1
            for g in groups:
                hit = [x for x in g if x in targets]
                if len(hit) >= 2:
                    TRACER.mi_cotarget_calls += 1
                    TRACER.mi_cotarget_by_file[f] += 1
                    if len(TRACER.mi_cotarget_examples) < 12:
                        TRACER.mi_cotarget_examples.append(
                            (TRACER.current_test, sorted(hit)))
                    break

        TRACER.gate("Brain.project (external_drive supplied)",
                    bool(external_drive))
        return orig_impl(self, areas_by_stim, dst_areas_by_src_area,
                         verbose, external_drive)

    Brain._project_impl = _project_impl

    orig_apply_mi = Brain._apply_mutual_inhibition

    def _apply_mutual_inhibition(self, activation_scores):
        fired = any(
            len([n for n in g if n in activation_scores]) > 1
            for g in (self._mutual_inhibition_groups or [])
        )
        if fired:
            TRACER.mi_fired_calls += 1
        TRACER.gate("Brain._apply_mutual_inhibition (>=2 co-targeted)", fired)
        return orig_apply_mi(self, activation_scores)

    Brain._apply_mutual_inhibition = _apply_mutual_inhibition

    # -- engine project_into ----------------------------------------------
    def _make_into(engine_cls, tag):
        orig_into = engine_cls.project_into

        def project_into(self, target, from_stimuli=(), from_areas=(),
                         plasticity_enabled=True, **kw):
            TRACER.into_calls += 1
            from_stimuli = list(from_stimuli or ())
            from_areas = list(from_areas or ())

            st = getattr(self, "_areas", {}).get(target)
            w_before = int(getattr(st, "w", 0) or 0)
            fixed = bool(getattr(st, "fixed_assembly", False))

            # DEAD-AT-USE. A source whose weight block into this target still
            # has zero columns while the target already has neurons delivers
            # exactly nothing, yet the k-WTA still returns k winners.
            if w_before > 0:
                conns = getattr(self, "_area_conns", {})
                for s in from_areas:
                    conn = conns.get(s, {}).get(target)
                    rows, cols, live = _conn_dims(conn)
                    key = (id(self), s, target)
                    TRACER.fiber_uses[key] += 1
                    TRACER.fiber_test.setdefault(key, TRACER.current_test)
                    if not live:
                        TRACER.fiber_dead_uses[key] += 1
                        continue
                    # ROW-SIDE DEATH, which `fiber_census` cannot see: it
                    # judges a fiber on columns and nnz, but the engine indexes
                    # rows by the SOURCE's compact winner indices and drops any
                    # that exceed the block's row count
                    # (`internal = src_w[src_w < conn.weights.shape[0]]`). A
                    # fiber whose target has grown but whose source has
                    # recruited past the block's rows therefore delivers
                    # exactly zero while looking perfectly healthy in a census.
                    src_st = getattr(self, "_areas", {}).get(s)
                    win = getattr(src_st, "winners", None)
                    if win is not None and len(win):
                        # torch states hold cuda tensors; .min() on those is a
                        # device scalar, so go through the tensor's own reduce
                        # rather than numpy, which would force a host copy of
                        # every winner array on the hot path.
                        lo = int(win.min()) if hasattr(win, "min") else \
                            int(np.min(np.asarray(win)))
                        if lo >= rows:
                            TRACER.fiber_rowdead_uses[key] += 1

            out = orig_into(self, target, from_stimuli, from_areas,
                            plasticity_enabled, **kw)

            total = float(getattr(out, "total_activation", 0.0) or 0.0)
            if not (from_stimuli or from_areas):
                cls = "no inputs"
            elif fixed:
                cls = "FIXED target (inputs discarded)"
            elif w_before == 0:
                cls = "cold target (first materialization)"
            elif total == 0.0:
                cls = "ZERO DRIVE"
            else:
                cls = "live"
            TRACER.drive_class[cls] += 1
            if cls == "ZERO DRIVE":
                site = (TRACER.current_test, tag, target,
                        tuple(from_stimuli), tuple(from_areas))
                TRACER.zero_drive_sites[site] += 1
            return out

        engine_cls.project_into = project_into

    _make_into(NumpySparseEngine, "sparse")
    _make_into(NumpyExplicitEngine, "explicit")

    # The torch engine is where the validated known positive lives, so it is
    # instrumented whenever it imports. Guarded rather than assumed: the module
    # imports torch at module scope and a CPU-only checkout must still be able
    # to run the numpy half of this sweep.
    try:
        from neural_assemblies.core.torch_engine._engine import TorchSparseEngine
    except Exception:                                        # noqa: BLE001
        pass
    else:
        _make_into(TorchSparseEngine, "torch")

    # -- gate census -------------------------------------------------------
    _wrap_gate(Brain, "inhibit_areas", "Brain.inhibit_areas",
               lambda s, a, kw, out: bool(a and a[0]))
    _wrap_gate(Brain, "remove_mutual_inhibition",
               "Brain.remove_mutual_inhibition", ALWAYS)
    _wrap_gate(Brain, "set_lri", "Brain.set_lri (inhibition_strength > 0)",
               lambda s, a, kw, out: float(
                   kw.get("inhibition_strength", a[2] if len(a) > 2 else 0.0)
               ) > 0.0)
    _wrap_gate(Brain, "set_refracted", "Brain.set_refracted (enabled)",
               lambda s, a, kw, out: bool(
                   kw.get("enabled", a[1] if len(a) > 1 else False)))
    _wrap_gate(Brain, "normalize_weights", "Brain.normalize_weights", ALWAYS)
    _wrap_gate(Brain, "project_rounds", "Brain.project_rounds", ALWAYS)

    _wrap_gate(NumpySparseEngine, "ensure_area_conn",
               "engine.ensure_area_conn (materialized a dead fiber)", TRUTHY)
    _wrap_gate(NumpySparseEngine, "_use_compiled_projection",
               "engine._use_compiled_projection (compiled path taken)", TRUTHY)
    _wrap_gate(NumpySparseEngine, "_bootstrap_from_explicit_dense",
               "engine._bootstrap_from_explicit_dense", ALWAYS)
    _wrap_gate(NumpySparseEngine, "_init_deferred_area_srcs",
               "engine._init_deferred_area_srcs (had srcs to init)",
               lambda s, a, kw, out: bool(a[1] if len(a) > 1 else
                                          kw.get("src_names")))
    _wrap_gate(NumpySparseEngine, "reset_area_connections",
               "engine.reset_area_connections", ALWAYS)
    _wrap_gate(NumpySparseEngine, "_sample_area_weights",
               "engine feedforward inhibition (inhibitory_prob > 0)",
               lambda s, a, kw, out: float(
                   getattr(s, "inhibitory_prob", 0.0)) > 0.0)
    _wrap_gate(NumpySparseEngine, "_norm_scale",
               "engine._norm_scale (norm_init actually scaled)",
               lambda s, a, kw, out: out is not None)

    _wrap_gate(inh_mod.InhibitionState, "project_map",
               "InhibitionState.project_map", ALWAYS)
    _wrap_gate(inh_mod.InhibitionState, "check_war_of_fibers",
               "InhibitionState.check_war_of_fibers (raised)",
               lambda s, a, kw, out: False)   # returns only when it did NOT
    _wrap_gate_fn(inh_mod, "prepare_targets", "inhibition.prepare_targets",
                  ALWAYS)
    _wrap_gate_fn(inh_mod, "apply_rule", "inhibition.apply_rule", ALWAYS)


# ---------------------------------------------------------------------------
# End-of-test whole-brain census
# ---------------------------------------------------------------------------
#
# `diagnostics.fiber_census` densifies each weight block to compute a
# potentiation ratio. That is fine for one forensic call and unaffordable across
# a suite, so this is the shape/nnz half of the same judgement -- exactly the
# inputs `FiberState.silently_ignored` needs -- with no copies.

def census(test_id):
    for bid, ref in TRACER.brains:
        b = ref()
        if b is None:
            continue
        try:
            areas = list(b.areas)
        except Exception:                                    # noqa: BLE001
            continue
        seen_engines = []
        for name in areas:
            try:
                eng = b._engine_for(b.areas[name])
            except Exception:                                # noqa: BLE001
                continue
            if any(eng is e for e in seen_engines):
                continue
            seen_engines.append(eng)
        for eng in seen_engines:
            conns = getattr(eng, "_area_conns", {})
            for src, per_dst in conns.items():
                for dst, conn in per_dst.items():
                    if conn is None or dst not in b.areas:
                        continue
                    _rows, _cols, live = _conn_dims(conn)
                    st = getattr(eng, "_areas", {}).get(dst)
                    dst_w = int(getattr(st, "w", 0) or 0)
                    TRACER.census_fibers += 1
                    # COLUMN COUNT ONLY, deliberately. `fiber_census` also
                    # counts nonzeros, which means touching every byte of every
                    # weight block -- affordable once, ruinous at every test
                    # teardown of a whole suite (measured: it turned a ~25 min
                    # slice into a >2 h one). Blocks in this engine are
                    # allocated with drawn weights, so a 0-column block is the
                    # deadness that actually occurs; the nonzero term would only
                    # add all-zero-but-allocated blocks, which
                    # `reset_area_connections` produces and which the at-use
                    # ZERO-DRIVE classification above already catches.
                    if not ((not live) and dst_w > 0):
                        continue
                    TRACER.census_dead_into_live += 1
                    # SPLIT ON WHETHER ANYTHING EVER DROVE IT. `fiber_census`'s
                    # `silently_ignored` cannot make this distinction and so
                    # over-reports enormously: a parser declares dozens of areas
                    # and the engine materializes a connectome entry for pairs
                    # no rule ever opens. Only a fiber that was DRIVEN and is
                    # still dead was a projection that silently did nothing.
                    if TRACER.fiber_uses.get((id(eng), src, dst), 0):
                        TRACER.census_dead_driven[(test_id, src, dst)] += 1
                    else:
                        TRACER.census_dead_undriven += 1


class Plugin:
    def pytest_runtest_setup(self, item):
        TRACER.current_test = item.nodeid

    def pytest_runtest_teardown(self, item):
        census(item.nodeid)
        TRACER.brains = []          # drop refs; each test gets fresh brains
        TRACER.current_test = None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report():
    W = 78
    print("\n" + "=" * W)
    print("DORMANT MECHANISM SWEEP -- what never fired")
    print("=" * W)

    # -- 3. mutual inhibition (reported first: it is the standing claim) ----
    print("\n[3] MUTUAL INHIBITION -- the paper's inter-area WTA")
    print(f"    Brain.project() calls                     {TRACER.project_calls:>8}")
    print(f"    projections into the engine               {TRACER.into_calls:>8}")
    print(f"    calls on a brain with a group declared    {TRACER.mi_declared_calls:>8}")
    print(f"    calls co-targeting >=2 areas of a group   {TRACER.mi_cotarget_calls:>8}"
          "   <-- the firing condition")
    print(f"    calls where an area was actually silenced {TRACER.mi_fired_calls:>8}")
    if TRACER.mi_cotarget_calls == 0:
        print("    VERDICT: DORMANT. The mechanism never ran on this slice.")
    else:
        print("    VERDICT: reached. Per FILE -- a test written expressly to "
              "force\n    the mechanism is not evidence that production uses "
              "it:")
        for f, n in TRACER.mi_declared_by_file.most_common():
            print(f"      {TRACER.mi_cotarget_by_file.get(f, 0):>6} / {n:<6} "
                  f"{f}")
        print("    examples:")
        for t, hit in TRACER.mi_cotarget_examples:
            print(f"      {t}  ->  {hit}")

    # -- 2. zero delivered drive ------------------------------------------
    print("\n[2] DELIVERED DRIVE, per engine-level projection")
    tot = sum(TRACER.drive_class.values()) or 1
    for cls, n in TRACER.drive_class.most_common():
        print(f"    {cls:<38} {n:>8}  {n / tot:6.2%}")
    zd = TRACER.drive_class.get("ZERO DRIVE", 0)
    if zd:
        print(f"\n    {len(TRACER.zero_drive_sites)} distinct ZERO-DRIVE sites "
              f"(target already grown, had inputs, delivered 0.0):")
        for (test, tag, target, stims, srcs), n in \
                TRACER.zero_drive_sites.most_common(25):
            print(f"      x{n:<5} [{tag}] {target} <- stim{list(stims)} "
                  f"area{list(srcs)}")
            print(f"             {test}")
    else:
        print("\n    No ZERO-DRIVE projections: every projection into an "
              "already-grown, unfixed target delivered nonzero drive.")

    # -- 1. dead fibers into live areas ------------------------------------
    print("\n[1] DEAD FIBERS INTO LIVE AREAS (silently_ignored)")
    always_dead, sometimes = [], []
    for key, uses in TRACER.fiber_uses.items():
        dead = TRACER.fiber_dead_uses.get(key, 0)
        if dead == 0:
            continue
        (always_dead if dead == uses else sometimes).append((key, dead, uses))
    print(f"    fibers driven while target live           "
          f"{len(TRACER.fiber_uses):>8}")
    print(f"    fibers DEAD on every use                  "
          f"{len(always_dead):>8}   <-- silently ignored")
    print(f"    fibers dead on some uses (deferred init)  "
          f"{len(sometimes):>8}")
    rowdead = sorted(TRACER.fiber_rowdead_uses.items(), key=lambda x: -x[1])
    print(f"    fibers ROW-dead on >=1 use (census-blind)  "
          f"{len(rowdead):>8}   <-- every source winner past the block's rows")
    for (bid, src, dst), n in rowdead[:15]:
        print(f"      {src:>18} -> {dst:<18} row-dead {n}/"
              f"{TRACER.fiber_uses.get((bid, src, dst), 0)} uses"
              f"   {TRACER.fiber_test.get((bid, src, dst))}")
    for (bid, src, dst), dead, uses in sorted(
            always_dead, key=lambda x: -x[1])[:25]:
        print(f"      {src:>18} -> {dst:<18} dead {dead}/{uses} uses")
        print(f"          first seen in {TRACER.fiber_test.get((bid, src, dst))}")
    # Dead on SOME uses is the deferred-init path doing its job, but a fiber
    # dead for many rounds before waking up lost those rounds silently, so the
    # worst offenders are worth a look.
    worst = sorted(sometimes, key=lambda x: -x[1])[:10]
    if worst:
        print("    worst deferred-init lags (dead uses before waking):")
        for (bid, src, dst), dead, uses in worst:
            print(f"      {src:>18} -> {dst:<18} dead {dead}/{uses} uses"
                  f"   {TRACER.fiber_test.get((bid, src, dst))}")

    print(f"\n    end-of-test whole-brain census over "
          f"{TRACER.census_fibers} fiber-observations:")
    print(f"      dead block into a LIVE area (raw silently_ignored) "
          f"{TRACER.census_dead_into_live:>7}")
    print(f"        ...of which NOTHING ever drove (topology only)   "
          f"{TRACER.census_dead_undriven:>7}")
    print(f"        ...of which something DID drive (real)           "
          f"{sum(TRACER.census_dead_driven.values()):>7}")
    for (test, src, dst), n in TRACER.census_dead_driven.most_common(20):
        print(f"      {src:>18} -> {dst:<18}  {test}")

    # -- 4. gate census ----------------------------------------------------
    print("\n[4] GATE CENSUS -- called vs actually fired")
    labels = sorted(set(TRACER.gate_calls) | set(TRACER.gate_fires))
    never, partial, always = [], [], []
    for lab in labels:
        c, f = TRACER.gate_calls.get(lab, 0), TRACER.gate_fires.get(lab, 0)
        (never if f == 0 else (always if f == c else partial)).append((lab, c, f))
    print("    CALLED BUT NEVER FIRED (the dormant shape):")
    if never:
        for lab, c, f in never:
            print(f"      {lab:<58} {c:>7} calls, 0 fires")
    else:
        print("      (none)")
    print("    fired sometimes:")
    for lab, c, f in partial:
        print(f"      {lab:<58} {f:>7}/{c} ({f / c:.1%})")
    print("    fired every call:")
    for lab, c, f in always:
        print(f"      {lab:<58} {c:>7}")

    # Mechanisms whose probe was installed but whose function was never
    # reached at all. Weaker evidence than called-but-never-fired -- the slice
    # may simply not exercise them -- so kept separate on purpose.
    print("\n    NEVER CALLED on this slice (weaker evidence -- coverage, not "
          "dormancy):")
    for lab in EXPECTED_LABELS:
        if lab not in TRACER.gate_calls:
            print(f"      {lab}")


EXPECTED_LABELS = [
    "Brain.add_mutual_inhibition (group declared)",
    "Brain._apply_mutual_inhibition (>=2 co-targeted)",
    "Brain.inhibit_areas",
    "Brain.remove_mutual_inhibition",
    "Brain.set_lri (inhibition_strength > 0)",
    "Brain.set_refracted (enabled)",
    "Brain.normalize_weights",
    "Brain.project_rounds",
    "Brain.project (external_drive supplied)",
    "engine.ensure_area_conn (materialized a dead fiber)",
    "engine._use_compiled_projection (compiled path taken)",
    "engine._bootstrap_from_explicit_dense",
    "engine._init_deferred_area_srcs (had srcs to init)",
    "engine.reset_area_connections",
    "engine feedforward inhibition (inhibitory_prob > 0)",
    "engine._norm_scale (norm_init actually scaled)",
    "InhibitionState.project_map",
    "InhibitionState.check_war_of_fibers (raised)",
    "inhibition.prepare_targets",
    "inhibition.apply_rule",
]


# ---------------------------------------------------------------------------
# Known positive
# ---------------------------------------------------------------------------

def validate():
    """Does the at-use detector fire on a construction KNOWN to be dead?

    The failure this sweep hunts leaves no trace in the numbers by definition,
    so "we ran it and found N things" says nothing about the N+1th. This runs
    the one case established by hand -- torch engine, reciprocal protocol,
    `B->A` never materialised -- through the SAME instrumented code path the
    suite run uses, and asserts both halves:

      * the dead `B->A` fiber is counted dead on every use, and
      * the live `A->B` fiber beside it is NOT.

    The second is the half that makes the first mean anything: a detector that
    calls everything dead would pass the first check alone, and that is exactly
    the state `diagnostics.fiber_census` is in on this engine.

    The numpy engine runs the identical protocol as a negative control, so a
    failure here separates "the detector broke" from "the engine was fixed".
    """
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus.ops import (
        project, reciprocal_project, _snap)

    N, K, BETA, P, ROUNDS = 1000, 50, 0.1, 0.05, 10
    ok = True
    for engine, expect_dead in (("torch_sparse", True), ("numpy_sparse", False)):
        print("=" * 78)
        print(f"KNOWN POSITIVE -- reciprocal protocol on {engine}")
        try:
            b = Brain(p=P, seed=1, engine=engine, norm_init=False)
        except Exception as e:                               # noqa: BLE001
            print(f"  engine unavailable, skipped: {e}")
            continue
        b.add_stimulus("stim", K)
        b.add_area("A", N, K, BETA)
        b.add_area("B", N, K, BETA)
        base = TRACER.fiber_dead_uses.copy(), TRACER.fiber_uses.copy()

        original = project(b, "stim", "A", rounds=ROUNDS)
        reciprocal_project(b, "A", "B", rounds=ROUNDS)
        b.project({}, {"B": ["A"]})
        act = float(b.last_activation_scores.get("A", float("nan")))
        for _ in range(ROUNDS - 1):
            b.project({}, {"B": ["A"], "A": ["A"]})
        restored = original.overlap(_snap(b, "A"))

        eng = b._engine_for(b.areas["A"])
        conns = getattr(eng, "_area_conns", {})
        print(f"  B->A block   {_conn_dims(conns.get('B', {}).get('A'))}"
              f"   (rows, cols, live)")
        print(f"  A->B block   {_conn_dims(conns.get('A', {}).get('B'))}")
        print(f"  first back-projection delivered total_activation = {act}")
        print(f"  restoration overlap = {restored:.4f}"
              + ("   <-- EXACTLY 1.0, i.e. preserved, not restored"
                 if restored == 1.0 else ""))

        def delta(src, dst):
            key = (id(eng), src, dst)
            return (TRACER.fiber_dead_uses[key] - base[0][key],
                    TRACER.fiber_uses[key] - base[1][key])
        ba_dead, ba_uses = delta("B", "A")
        ab_dead, ab_uses = delta("A", "B")
        print(f"  detector: B->A dead {ba_dead}/{ba_uses} uses,"
              f"  A->B dead {ab_dead}/{ab_uses} uses")

        if expect_dead:
            hit = ba_uses > 0 and ba_dead == ba_uses
            miss = ab_uses > 0 and ab_dead == ab_uses
            print(f"  [{'PASS' if hit else 'FAIL'}] fires on the known-dead "
                  f"fiber")
            print(f"  [{'PASS' if not miss else 'FAIL'}] does NOT fire on the "
                  f"live fiber beside it")
            ok = ok and hit and not miss
        else:
            clean = ba_dead < ba_uses
            print(f"  [{'PASS' if clean else 'FAIL'}] negative control: the "
                  f"same protocol on numpy carries drive")
            ok = ok and clean
    print("=" * 78)
    print(f"VALIDATION {'PASSED' if ok else 'FAILED'} -- the detector's "
          f"true-positive case is {'constructed and caught' if ok else 'NOT caught'}")
    return 0 if ok else 1


def dump_json(path):
    """Machine-readable counters, so SHARDS can be summed exactly.

    The slice is gated by a handful of very heavy tests (measured: ~2 min each
    through `test_checkpoint_fork`), so the practical way to run it is several
    processes over disjoint files. Every statistic here is a count or a per-key
    count, i.e. additive across shards -- which is why the aggregation is exact
    rather than an eyeballed merge of printed reports. `pytest-xdist` cannot do
    this job: its workers are separate processes and the tracer state lives in
    whichever process ran the test.
    """
    import json

    def ckeys(counter):
        return {" | ".join(str(x) for x in k) if isinstance(k, tuple) else str(k): v
                for k, v in counter.items()}

    with open(path, "w") as fh:
        json.dump({
            "project_calls": TRACER.project_calls,
            "impl_calls": TRACER.impl_calls,
            "into_calls": TRACER.into_calls,
            "mi_declared_calls": TRACER.mi_declared_calls,
            "mi_cotarget_calls": TRACER.mi_cotarget_calls,
            "mi_fired_calls": TRACER.mi_fired_calls,
            "mi_declared_by_file": ckeys(TRACER.mi_declared_by_file),
            "mi_cotarget_by_file": ckeys(TRACER.mi_cotarget_by_file),
            "mi_cotarget_examples": [[t, h]
                                     for t, h in TRACER.mi_cotarget_examples],
            "drive_class": ckeys(TRACER.drive_class),
            "zero_drive_sites": ckeys(TRACER.zero_drive_sites),
            "gate_calls": ckeys(TRACER.gate_calls),
            "gate_fires": ckeys(TRACER.gate_fires),
            "fiber_always_dead": {
                f"{s} -> {d}": [dead, TRACER.fiber_uses[(b, s, d)]]
                for (b, s, d), dead in TRACER.fiber_dead_uses.items()
                if dead == TRACER.fiber_uses[(b, s, d)]},
            "fiber_sometimes_dead": {
                f"{s} -> {d}": [dead, TRACER.fiber_uses[(b, s, d)]]
                for (b, s, d), dead in TRACER.fiber_dead_uses.items()
                if dead != TRACER.fiber_uses[(b, s, d)]},
            "fiber_rowdead": {
                f"{s} -> {d}": [n, TRACER.fiber_uses[(b, s, d)]]
                for (b, s, d), n in TRACER.fiber_rowdead_uses.items()},
            "fiber_uses_total": len(TRACER.fiber_uses),
            "census_fibers": TRACER.census_fibers,
            "census_dead_into_live": TRACER.census_dead_into_live,
            "census_dead_undriven": TRACER.census_dead_undriven,
            "census_dead_driven": ckeys(TRACER.census_dead_driven),
        }, fh, indent=1)


def main():
    install()
    import pytest

    argv = list(sys.argv[1:])
    if "--validate" in argv:
        return validate()
    json_path = None
    if "--sweep-json" in argv:
        i = argv.index("--sweep-json")
        json_path = argv[i + 1]
        del argv[i:i + 2]

    args = (argv or DEFAULT_ARGS) + ["-q", "-p", "no:randomly"]
    code = pytest.main(args, plugins=[Plugin()])
    report()
    if json_path:
        dump_json(json_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
