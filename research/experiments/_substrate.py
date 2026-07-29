"""Shared substrate harness: the correct thing, made the easy thing.

WHY THIS EXISTS
---------------
On 2026-07-28 three separate "findings" about the Assembly Calculus dissolved
into three measurement defects, each of which had been made repeatedly, by
someone who had read the code, across four experiment files:

  1. READING THE WRONG ARRAY. ``brain.areas[X].winners`` holds COMPACT engine
     indices; ``Assembly.winners`` (what every ``ops.*`` returns) holds STABLE
     NEURON IDS. Measured n=2000 k=45: area.winners max 1395 (w=1398), snap max
     1988 (n=2000), overlap between them 0.0. They are DISJOINT, so comparing
     across them returns EXACTLY CHANCE -- silently, with no error, looking
     precisely like a negative result.
  2. NOT ACTUALLY PROJECTING RECURRENTLY. ``ops.project`` defaults to a path
     that drops ``target -> target``, so the "assembly" was never one.
  3. TRAINING OUTSIDE THE WINDOW. Below it a lone assembly dissolves; above it
     a shared target collapses onto whatever was stored first.

Each was available to anyone reading the source. That they were made anyway is
a fact about the API surface, not about care. So this module does not document
the sharp edges -- it removes them. Read via ``read()``, probe via ``probe()``,
build via ``build()``, and none of the three is expressible.

THE ONE RULE. Nothing in an experiment should ever touch
``brain.areas[X].winners``. If you find yourself reaching for it, you want
``read()``.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import statistics
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

#: Minimum trials before a rate may be REPORTED. This project has already
#: retracted a headline that read 0.375 at 24 trials and 0.104 at 96; 24 cannot
#: separate 0.125 from 0.375. Trials = items x seeds.
MIN_TRIALS = 96

#: 95% two-sided t multipliers for small seed counts.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262}


# --------------------------------------------------------------------------
# Readout -- the single sanctioned door between the two index spaces
# --------------------------------------------------------------------------

def read(brain, area: str) -> np.ndarray:
    """Current assembly in *area* as STABLE NEURON IDS.

    The only sanctioned readout. ``brain.areas[area].winners`` is a different
    coordinate system (compact engine indices, renumbered as the area recruits)
    and comparing the two returns chance -- see the module docstring.
    """
    from neural_assemblies.assembly_calculus.ops import _snap
    return np.asarray(_snap(brain, area).winners, dtype=np.int64)


def similarity(a, b) -> float:
    """Overlap between two things ``read()`` returned. Order-insensitive."""
    from neural_assemblies.assembly_calculus.assembly import overlap
    return float(overlap(np.asarray(a, dtype=np.int64),
                         np.asarray(b, dtype=np.int64)))


def rank1(live, table: Dict) -> object:
    """Key of the stored assembly in *table* best matching *live*."""
    return max((similarity(live, asm), key) for key, asm in table.items())[1]


def spread(assemblies: Iterable) -> float:
    """Mean pairwise overlap -- the distinctness of a set of assemblies."""
    pairs = list(itertools.combinations(list(assemblies), 2))
    return statistics.mean(similarity(x, y) for x, y in pairs) if pairs else 0.0


# --------------------------------------------------------------------------
# Probing -- measuring must not change the thing measured
# --------------------------------------------------------------------------

@contextlib.contextmanager
def probe(brain):
    """Read-only context for any measurement.

    Wraps ``brain.read_only()``, which blocks weight change AND recruitment and
    restores winners on exit. ``frozen()`` is NOT equivalent and must not be
    substituted: it stops weights changing but not ``w``, and two probe orders
    that recruit differently are structurally different brains. That is the
    contamination that inverted the P600 sign.

    KNOWN LIMIT (#41): this does not roll back connectome MATERIALIZATION -- a
    fiber first used inside a probe stays materialized afterwards.
    """
    with brain.read_only():
        yield


# --------------------------------------------------------------------------
# Building -- the protocol, opted in
# --------------------------------------------------------------------------

def build(brain, stimulus: str, area: str, rounds: int) -> np.ndarray:
    """Form an assembly in *area* from *stimulus*, WITH self-recurrence.

    Passes ``recurrent=True``, which ``ops.project`` does not default to; see
    its docstring for why the wrong default is deliberate. Returns neuron IDs.
    """
    from neural_assemblies.assembly_calculus.ops import project
    project(brain, stimulus, area, rounds=rounds, recurrent=True)
    return read(brain, area)


@contextlib.contextmanager
def pinned(brain, area: str, assembly):
    """Hold *assembly* (NEURON IDS) active in *area* for the duration.

    This is the other direction of the index-space trap, and it is the one that
    bit `merge_chain_primitive.py`: writing neuron IDs straight into
    ``area.winners`` puts them in a COMPACT-index slot, so the area is pinned to
    whatever those integers happen to mean as positions -- a different assembly
    entirely, silently. The conversion goes through ``ops._compact_index``,
    which the codebase already centralises for exactly this reason.

    Needed because every constituent above level 1 is composed and has no
    stimulus of its own, so it cannot be driven -- only held. NOTE that
    ``ops.merge`` documents the cost: a projection INTO a fixed area is
    short-circuited before plasticity, so a pinned parent gets no
    back-projection written (measured 0.000 vs 3.47x driven). That is the
    structural question depth has to answer, and it is why this helper exists
    rather than being avoided.
    """
    from neural_assemblies.assembly_calculus.ops import _compact_index

    area_obj = brain.areas[area]
    engine = brain._engine_for(area_obj)
    inv = _compact_index(engine, area)
    ids = [int(x) for x in np.asarray(assembly).ravel()]
    idx = [inv[i] for i in ids if i in inv] if inv else ids
    saved = np.asarray(area_obj.winners).copy()
    area_obj.winners = np.asarray(idx, dtype=np.int64)
    area_obj.fix_assembly()
    try:
        yield
    finally:
        area_obj.unfix_assembly()
        area_obj.winners = saved


def cue(brain, stimulus: str, area: str, rounds: int) -> np.ndarray:
    """Re-present *stimulus* and read what *area* settles on. Changes nothing.

    Use the SAME ``rounds`` as the build. A cue that does not itself clear the
    stability threshold presents a different assembly than the one whose
    synapses were written downstream, so the probe would answer the wrong
    question however well the operation under test worked.
    """
    with probe(brain):
        return build(brain, stimulus, area, rounds)


# --------------------------------------------------------------------------
# The training window
# --------------------------------------------------------------------------

def potentiation(beta: float, rounds: int) -> float:
    """``(1+beta)^rounds`` -- the quantity that governs both window walls."""
    return (1.0 + beta) ** rounds


def measure_window(n: int, k: int, beta: float = 0.1, p: float = 0.05,
                   seeds: Sequence[int] = (42, 7, 123),
                   candidates: Sequence[int] = (6, 8, 10, 14, 20),
                   steps: int = 8) -> Tuple[int, Dict[int, float]]:
    """MEASURE the lower wall at this regime. Returns (rounds, {rounds: overlap}).

    Deliberately measured rather than tabulated. The threshold is not a
    constant: it rises with n, because the competitor is the POPULATION MAXIMUM
    and a bigger population has a bigger maximum. Measured at
    ``(1+beta)^T = 2.6``, overlap was 0.630 / 0.150 / 0.007 for n = 2000 / 1e4 /
    5e4. A hardcoded number would be right for one regime and quietly wrong
    everywhere else, which is the failure mode this whole module exists to stop.

    The returned ``rounds`` is the smallest candidate whose assembly still
    overlaps itself above 0.9 after ``steps`` autonomous self-projections.
    """
    from neural_assemblies.core.brain import Brain

    curve = {}
    for rounds in candidates:
        vals = []
        for seed in seeds:
            brain = Brain(p=p, seed=seed)
            brain.add_area("W", n, k, beta=beta)
            brain.add_stimulus("w", k)
            trained = build(brain, "w", "W", rounds)
            for _ in range(steps):
                brain.project({}, {"W": ["W"]})
            vals.append(similarity(read(brain, "W"), trained))
        curve[rounds] = statistics.mean(vals)
    ok = [r for r in candidates if curve[r] > 0.9]
    return (ok[0] if ok else max(candidates, key=lambda r: curve[r])), curve


# --------------------------------------------------------------------------
# Running seeds in parallel
# --------------------------------------------------------------------------

def _seed_worker(payload):
    import importlib
    mod_name, fn_name, args, seed = payload
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return fn(*args, seed)


def parallel_seeds(fn, seeds: Sequence[int], *args, workers: int = 0):
    """Run ``fn(*args, seed)`` for each seed in a separate process.

    Seeds are INDEPENDENT trials -- each builds its own ``Brain(seed=...)`` and
    shares no state -- so this is exactly equivalent to the serial loop, not an
    approximation. The engine is single-threaded CPU-bound numpy (profiled: 79%
    of a trial is inside ``project``), so wall-clock scales close to linearly
    with cores until the seed count runs out.

    PYTHONHASHSEED IS PINNED, and must be. Multiprocessing on Windows spawns
    fresh interpreters, and this project has already been bitten once by
    ``hash()``-derived RNG seeds making ``Brain(seed=)`` differ ACROSS processes
    while staying stable within one -- a failure no single-process test can
    catch. Without this line every worker would draw a different hash seed and
    the "same" seed would mean something different in each.

    ``fn`` must be a module-level function taking ``seed`` LAST, since spawn
    pickles by qualified name.

    Falls back to the serial loop for a single seed or worker, which keeps
    debugging (and tracebacks) usable.
    """
    import concurrent.futures as _cf
    import multiprocessing as _mp

    seeds = list(seeds)
    n_workers = workers or min(len(seeds), max(1, (os.cpu_count() or 2) - 1))
    if n_workers <= 1 or len(seeds) <= 1:
        return [fn(*args, s) for s in seeds]

    os.environ["PYTHONHASHSEED"] = "0"
    payloads = [(fn.__module__, fn.__name__, args, s) for s in seeds]
    ctx = _mp.get_context("spawn")
    with _cf.ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=ctx) as pool:
        return list(pool.map(_seed_worker, payloads))


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and 95% half-width across seeds; half-width nan for n < 2."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n < 2:
        return values[0], float("nan")
    t = _T95.get(n, 1.96)
    return statistics.mean(values), t * statistics.stdev(values) / (n ** 0.5)


def report_rate(label: str, hits: int, trials: int, chance: float) -> str:
    """Format a rate, and SAY SO when it is under-powered rather than not."""
    rate = hits / trials if trials else float("nan")
    warn = "" if trials >= MIN_TRIALS else (
        f"  [UNDER-POWERED: {trials} trials < {MIN_TRIALS}; do not quote]")
    return f"  {label:<34}{rate:>8.4f}  ({hits}/{trials}, chance {chance:.4f}){warn}"


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

def main() -> None:
    """Demonstrate that each removed error class is really removed."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from neural_assemblies.core.brain import Brain

    n, k, beta = 2000, 45, 0.1
    print(f"\n  SELF-CHECK  n={n} k={k} beta={beta}, chance = {k / n:.4f}")

    print("\n  1. THE TWO INDEX SPACES")
    brain = Brain(p=0.05, seed=42)
    brain.add_area("A", n, k, beta=beta)
    brain.add_stimulus("s", k)
    trained = build(brain, "s", "A", 12)
    raw = np.asarray(brain.areas["A"].winners, dtype=np.int64)
    print(f"     read()           max {trained.max():>5}  (neuron IDs, n={n})")
    print(f"     areas[A].winners max {raw.max():>5}  (compact, w={brain.areas['A'].w})")
    print(f"     overlap between them: {similarity(raw, trained):.4f}"
          f"   vs chance {k / n:.4f}")
    print("     <- AT CHANCE, not zero. That is the trap: a cross-space")
    print("        comparison does not look broken, it looks like a NEGATIVE")
    print("        RESULT, and it is indistinguishable from one by inspection.")

    print("\n  2. RECURRENCE IS OPTED IN")
    from neural_assemblies.assembly_calculus.ops import project
    b2 = Brain(p=0.05, seed=42)
    b2.add_area("A", n, k, beta=beta)
    b2.add_stimulus("s", k)
    project(b2, "s", "A", rounds=12)                 # default path
    off = read(b2, "A")
    b3 = Brain(p=0.05, seed=42)
    b3.add_area("A", n, k, beta=beta)
    b3.add_stimulus("s", k)
    on = build(b3, "s", "A", 12)                     # recurrent=True
    print(f"     default vs recurrent assemblies overlap: {similarity(off, on):.4f}"
          f"   <- different operations, as intended")

    print("\n  3. THE WINDOW IS MEASURED, NOT ASSUMED")
    rounds, curve = measure_window(n, k, beta=beta)
    for r, v in curve.items():
        mark = "  <- chosen" if r == rounds else ""
        print(f"     rounds={r:<3} (1+b)^T={potentiation(beta, r):>6.1f}"
              f"   self-overlap after 8 autonomous steps {v:.3f}{mark}")

    print("\n  4. PROBES DO NOT MUTATE")
    before_w = brain.areas["A"].w
    again = cue(brain, "s", "A", 12)
    print(f"     w {before_w} -> {brain.areas['A'].w}, "
          f"cue matches trained at {similarity(again, trained):.4f}")

    print("\n  5. UNDER-POWERED RATES ARE LABELLED")
    print(report_rate("8 items x 3 seeds", 24, 24, 0.125))
    print(report_rate("16 items x 6 seeds", 90, 96, 0.0625))
    print()


if __name__ == "__main__":
    main()
