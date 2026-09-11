"""The complexity ladder: push the whole corrected stack until something breaks.

WHAT IS BEING SCALED
--------------------
Every fix from the 2026-07-28/29 collapse arc, composed into one stack, and then
driven up until it fails:

    lexicon    feed-forward build (no `area -> area`). No measured ceiling:
               1.0000 to M=256 at n=1000, 12.8x oversubscription.
    merge      ops.merge(parent_self=False, target_self=False,
               back_project=False) at T=2. 512 constituents at n=4000, acc
               1.0000, margin 6.67x, ceiling not found.
    depth      one SHARED area per level. Flat at 1.0000 to depth 3,
               margin 4.58x.

Each was measured alone or two-deep. This runs them TOGETHER and further: more
constituents, more depth, larger n. The point is to find the wall, because every
capacity number in this project so far is a lower bound produced by a sweep that
ran out before the substrate did.

THE THREE AXES, and why they are swept together
------------------------------------------------
They are not independent, and that is the whole reason to do this.
`depth_corrected_substrate.py` found that composition AMPLIFIES overlap: level-1
assemblies sharing 16% of their neurons produce level-3 assemblies sharing 56%.
So the M that works at depth 1 need not work at depth 5, and the merge rounds
that maximise depth-1 fidelity (T=10) actively destroy depth 3. A per-axis
sweep would report three ceilings that do not exist jointly.

WHAT IS REPORTED, AND WHY MARGIN LEADS
---------------------------------------
`margin` (best stored match / second best) is the continuous quantity; accuracy
is a step function of it and only reports that the gap has ALREADY closed. At
M=256/n=1000 accuracy read 0.9544 while margin was 1.47x -- healthy-looking and
one step from failure. Ceilings are located by where margin approaches ~1.1,
not by where accuracy breaks.

`spr_L` (mean pairwise overlap at level L, against the k/n floor) is the leading
indicator for the level ABOVE it, per the amplification law.

OPERATIONALLY
-------------
Results are flushed after every cell, so an interrupted run still yields
everything completed. Cells are ordered cheapest-first so the most informative
region is covered even if the run is cut short. A cell that raises is recorded
as FAILED and the ladder continues -- one bad configuration must not cost the
rest of the night.

PRE-REGISTERED
--------------
N1 Depth 3 at 1.0000 reproduces at n=4000 with M=256, i.e. the depth result is
   not specific to the small regime it was found in.
N2 There is a joint ceiling: some (n, M, depth) fails while its (n, M, depth-1)
   holds. Depth costs capacity, and the exchange rate is what this measures.
N3 Margin at the deepest passing level falls with depth at fixed M, because
   overlap amplifies. If margin is FLAT in depth, amplification has been
   defeated by the T=2 operating point and depth is free -- a stronger result
   than N2 and the one worth chasing next.
N4 The ceiling scales with n at every depth, so capacity is crowding rather
   than a fixed structural limit.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import traceback

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import (  # noqa: E402
    assert_machine_idle, check_distinct, parallel_seeds, probe, read,
    similarity, spread,
)

#: beta is an ENV KNOB because depth_beta_rescue.py found depth peaks at an
#: INTERIOR beta: 0.10 starves the chain (fiber under-potentiated, assemblies
#: distinct but margin decaying) while 0.20 reaches depth 5-6. The first pass of
#: this ladder ran at 0.10 and its depth failures at large n show exactly the
#: starvation signature -- spr_D at the floor with margin collapsed -- so they
#: are operating-point artifacts, not capacity ceilings. Re-run at 0.20 to
#: separate the two.
K_, P_ = 50, 0.05
BETA = float(os.environ.get("LADDER_BETA", "0.10"))
BUILD_ROUNDS, MERGE_ROUNDS = 6, 2
LEAF = "A"
GATED = dict(parent_self=False, target_self=False, back_project=False)
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.environ.get("LADDER_OUT", "complexity_ladder_results.txt"))

#: (n, M, depth). Cheapest first so a short night still covers the useful part.
LADDER = [
    (1000, 64, 3), (1000, 128, 3), (1000, 256, 3),
    (4000, 64, 3), (4000, 256, 3), (4000, 512, 3),
    (1000, 64, 5), (1000, 128, 5), (4000, 256, 5), (4000, 512, 5),
    (4000, 256, 7), (4000, 512, 7),
    (10000, 512, 3), (10000, 1024, 3), (10000, 512, 5), (10000, 1024, 5),
    (10000, 1024, 7), (10000, 2048, 3), (10000, 2048, 5),
]


# Restrict the sweep to named cells, e.g. LADDER_ONLY="4000:512:3".
# Added to reproduce the #49 divergence on the ONE cell that showed it, without
# spending an hour walking the whole ladder to reach it. The worker function
# must stay in this module for that reproduction to be faithful -- the fault
# needs the duplicated module to be the one holding the experiment's globals,
# which a wrapper that delegates to an imported ladder does NOT provide.
_ONLY = os.environ.get("LADDER_ONLY", "").strip()
if _ONLY:
    _want = {tuple(int(x) for x in c.split(":")) for c in _ONLY.split(",")}
    LADDER = [c for c in LADDER if c in _want]


def seeds_for(m):
    return (42, 7, 123) if m <= 256 else (42, 7)


def build_ff(brain, stim, area, rounds):
    for _ in range(rounds):
        brain.project({stim: [area]}, {})
    return read(brain, area)


def partner(L):
    return f"P{L}"


def level(L):
    return f"C{L}"


def trial(n, m_items, depth, seed):
    from neural_assemblies.assembly_calculus.ops import merge
    from _substrate import pinned
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P_, seed=seed)
    brain.add_area(LEAF, n, K_, beta=BETA)
    for L in range(1, depth + 1):
        brain.add_area(partner(L), n, K_, beta=BETA)
        brain.add_area(level(L), n, K_, beta=BETA)
    for m in range(m_items):
        brain.add_stimulus(f"a{m}", K_)
        for L in range(1, depth + 1):
            brain.add_stimulus(f"p{L}_{m}", K_)
    for m in range(m_items):
        build_ff(brain, f"a{m}", LEAF, BUILD_ROUNDS)
        for L in range(1, depth + 1):
            build_ff(brain, f"p{L}_{m}", partner(L), BUILD_ROUNDS)

    stored = {L: {} for L in range(1, depth + 1)}
    for m in range(m_items):
        merge(brain, LEAF, partner(1), level(1), stim_a=f"a{m}",
              stim_b=f"p1_{m}", rounds=MERGE_ROUNDS, **GATED)
        stored[1][m] = read(brain, level(1))
    for L in range(2, depth + 1):
        for m in range(m_items):
            with pinned(brain, level(L - 1), stored[L - 1][m]):
                merge(brain, level(L - 1), partner(L), level(L),
                      stim_b=f"p{L}_{m}", rounds=MERGE_ROUNDS,
                      unstimulated_source_mode="require-fixed", **GATED)
                stored[L][m] = read(brain, level(L))

    # A collapsed level makes every number below it read at chance for a
    # reason unrelated to the hypothesis. Catch it here rather than in a
    # results table -- a cell that reads spr 0.9999 has not measured depth.
    degenerate = []
    for L in range(1, depth + 1):
        _s, note = check_distinct(stored[L].values(), n, K_)
        if note:
            degenerate.append(f"L{L}:{note}")

    full = {L: 0 for L in range(1, depth + 1)}
    margins = {L: [] for L in range(1, depth + 1)}
    for m in range(m_items):
        with probe(brain):
            build_ff(brain, f"a{m}", LEAF, BUILD_ROUNDS)
            src = LEAF
            for L in range(1, depth + 1):
                brain.project({}, {src: [level(L)]})
                live = read(brain, level(L))
                sims = sorted(((similarity(live, a), j)
                               for j, a in stored[L].items()), reverse=True)
                full[L] += sims[0][1] == m
                if len(sims) > 1 and sims[1][0] > 0:
                    margins[L].append(sims[0][0] / sims[1][0])
                src = level(L)

    _ = degenerate
    return (full, m_items,
            {L: (statistics.mean(v) if v else float("nan"))
             for L, v in margins.items()},
            {L: spread(stored[L].values()) for L in stored})


#: Wall-clock ceiling per cell. The first run of this ladder spent over an hour
#: on single n=10000/M=2048 cells with NO output during them, because cost grows
#: as n*M*depth and the tail cells are ~100x the head cells. A sweep that cannot
#: say what it is doing, or how long it will take, is not usable overnight.
CELL_BUDGET_S = float(os.environ.get("LADDER_CELL_BUDGET_S", "900"))


def work_units(n, m_items, depth):
    """Rough cost model: merges and builds both scale as M*depth, and the
    sparse ops scale with n. Only RATIOS matter -- it is calibrated from the
    first completed cell, so the absolute constant is irrelevant."""
    return (n / 1000.0) * m_items * depth


def emit(line, fh):
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    assert_machine_idle()
    with open(OUT, "w", encoding="utf-8") as fh:
        emit(f"  COMPLEXITY LADDER  k={K_} beta={BETA} p={P_}, feed-forward "
             f"lexicon, gated merge T={MERGE_ROUNDS}", fh)
        emit(f"  one SHARED area per level; margin = best/second-best stored "
             f"match; floor = k/n", fh)
        emit(f"  full_D = accuracy at the DEEPEST level; a cell passes when "
             f"full_D > 0.90\n", fh)
        emit(f"  {'n':>7}{'M':>6}{'D':>3}{'load':>7}{'chance':>9}"
             f"{'full_1':>9}{'full_D':>9}{'marg_1':>8}{'marg_D':>8}"
             f"{'spr_1':>8}{'spr_D':>8}{'trials':>8}  verdict", fh)

        rate = None          # seconds per work unit, calibrated as we go
        for n, m_items, depth in LADDER:
            units = work_units(n, m_items, depth)
            if rate is not None:
                predicted = rate * units
                if predicted > CELL_BUDGET_S:
                    # LOG the skip. A silent cap reads as "covered everything".
                    emit(f"  {n:>7}{m_items:>6}{depth:>3}{'':>41}  SKIPPED: "
                         f"~{predicted / 60:.0f} min > budget "
                         f"{CELL_BUDGET_S / 60:.0f} min", fh)
                    continue
                emit(f"    .. starting n={n} M={m_items} D={depth}, est "
                     f"{predicted / 60:.1f} min", fh)
            t0 = time.monotonic()
            try:
                seeds = seeds_for(m_items)
                # Seeds are independent trials, so this is exactly equivalent
                # to the serial loop -- verified identical, and 1.57x on a
                # 56s cell (spawn overhead amortises on the long cells, which
                # are the ones that hurt).
                res = parallel_seeds(trial, seeds, n, m_items, depth)
                tot = sum(x[1] for x in res)
                f1 = sum(x[0][1] for x in res) / tot
                fd = sum(x[0][depth] for x in res) / tot
                m1 = statistics.mean(x[2][1] for x in res)
                md = statistics.mean(x[2][depth] for x in res)
                s1 = statistics.mean(x[3][1] for x in res)
                sdv = statistics.mean(x[3][depth] for x in res)
                verdict = "PASS" if fd > 0.90 else (
                    "MARGINAL" if fd > 0.50 else "FAIL")
                elapsed = time.monotonic() - t0
                rate = elapsed / units if units else rate
                emit(f"  {n:>7}{m_items:>6}{depth:>3}{m_items * K_ / n:>7.1f}"
                     f"{1 / m_items:>9.5f}{f1:>9.4f}{fd:>9.4f}{m1:>8.2f}"
                     f"{md:>8.2f}{s1:>8.4f}{sdv:>8.4f}{tot:>8}  {verdict}"
                     f"   [{elapsed:.0f}s]", fh)
            except Exception as exc:  # keep the ladder going
                emit(f"  {n:>7}{m_items:>6}{depth:>3}"
                     f"{'':>41}  FAILED: {type(exc).__name__}: {exc}", fh)
                traceback.print_exc()

        emit("\n  DONE. Read the margin columns, not the accuracies: a cell "
             "can read 0.95 with", fh)
        emit("  a margin of 1.47x, which is one step from failing. The ceiling "
             "is where", fh)
        emit("  margin approaches ~1.1, and spr_1 is the leading indicator "
             "for the level above.", fh)


if __name__ == "__main__":
    main()
