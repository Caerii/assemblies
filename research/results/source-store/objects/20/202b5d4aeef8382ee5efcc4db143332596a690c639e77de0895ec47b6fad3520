"""Does splitting a lexicon across many areas buy compute, capacity, or both?

WHERE THE QUESTION COMES FROM
-----------------------------
Profiling the capacity sweep showed the cost is not the drive -- that is O(n)
per round and 16% of the time -- but `apply_to`, which REPLAYS every stored
outer product on every projection. At M items in one area that is 232 events
per projection at M=128, so the protocol is O(M^2) and no per-round constant
touches it.

Splitting M items across A areas of M/A items each should therefore cost
O(M^2 / A): the same 12M projections, each replaying A times fewer events. That
is a clean linear prediction and it is worth checking, because if it holds then
"more areas" is the only lever that changes the asymptote.

THE CATCH, WHICH IS THE INTERESTING PART
----------------------------------------
`ceiling_n_scaling_on_exact_drive.md` measured the recurrence ceiling growing
SUPER-linearly in n: 16 -> 64 for n = 1000 -> 2000, four-fold per doubling.
If capacity really goes like n^a with a > 1, then splitting a FIXED neuron
budget into A areas gives total capacity

    A * (n/A)^a  =  n^a / A^(a-1)

which DECREASES in A. Many small areas would then be a compute win and a
capacity loss, out of the same neurons. That trade is the thing to measure, and
it is not obvious which side wins at the sizes anyone actually uses.

So two arms, and they answer different questions:

  SPLIT   fixed TOTAL neurons: A areas of n_total/A.  "I have a fixed brain,
          should I partition it?"
  GROW    fixed AREA size:     A areas of n_0 each.   "I can add neurons,
          should I add areas or make one area bigger?"

ROUTING IS ASSUMED GIVEN, AND THAT IS NOT FREE
----------------------------------------------
With A areas each item lives in a known area, so retrieval only ranks against
the M/A items sharing it -- chance rises from 1/M to A/M. That is a REAL
benefit of modularity and also makes the accuracy columns non-comparable across
A. Both chance levels are printed on every row, and the honest cross-A metric
is CAPACITY PER NEURON, which does not move with the chance level.

Where the routing signal comes from is out of scope here. In the parser it is
the category areas; in this file it is handed over for free, which is an upper
bound on what modularity can buy, not an estimate of it.

PRE-REGISTERED
--------------
P1 Wall-clock falls roughly like 1/A in both arms. This is the arithmetic of
   the profile, so failure means the model of the cost is wrong.
P2 SPLIT: capacity per neuron FALLS as A rises, because the ceiling is
   super-linear in n. Predicted from the n-scaling measurement, on one clean
   doubling, so this is the weaker of the two.
P3 GROW: capacity per neuron is roughly FLAT in A -- A areas of n_0 hold about
   A times what one holds. Anything else means areas interact through something
   this protocol does not model (they share no fibers here).
"""

from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import probe, rank1, read, similarity, spread  # noqa: E402

K_, P_, BETA, ROUNDS = 50, 0.05, 0.10, 6
N_TOTAL = 4000          # SPLIT arm: this is divided among the areas
N_AREA = 1000           # GROW  arm: every area is this big
M = 128                 # items, divided among the areas in both arms
A_SWEEP = (1, 2, 4, 8)
SEEDS = (42, 7, 123)
ENGINE = "numpy_exact"


def trial(n_area, n_areas, m_total, seed):
    """m_total items spread over n_areas areas of n_area neurons each."""
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.core.brain import Brain

    per = m_total // n_areas
    b = Brain(p=P_, seed=seed, norm_init=True, engine=ENGINE)
    areas = [f"L{a}" for a in range(n_areas)]
    for a in areas:
        b.add_area(a, n_area, K_, beta=BETA)
    for a_i, a in enumerate(areas):
        for j in range(per):
            b.add_stimulus(f"w{a_i}_{j}", K_)

    t0 = time.perf_counter()
    stored = {}
    for a_i, a in enumerate(areas):
        for j in range(per):
            project(b, f"w{a_i}_{j}", a, rounds=ROUNDS, recurrent=True)
            stored[(a_i, j)] = read(b, a)

    hits, idents, spreads = 0, [], []
    for a_i, a in enumerate(areas):
        local = {key: v for key, v in stored.items() if key[0] == a_i}
        for j in range(per):
            with probe(b):
                project(b, f"w{a_i}_{j}", a, rounds=ROUNDS, recurrent=True)
                live = read(b, a)
            hits += rank1(live, local) == (a_i, j)
            idents.append(similarity(live, stored[(a_i, j)]))
        spreads.append(spread(local.values()))
    secs = time.perf_counter() - t0

    return (hits / (per * n_areas), statistics.mean(idents),
            statistics.mean(spreads), secs, per)


def run(n_area, n_areas, m_total):
    res = [trial(n_area, n_areas, m_total, s) for s in SEEDS]
    return tuple(statistics.mean(x[i] for x in res) for i in range(4)) \
        + (res[0][4],)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  Many areas: compute win, capacity loss, or both?")
    print(f"  k={K_} beta={BETA} p={P_} buildT={ROUNDS}, M={M} items total, "
          f"{len(SEEDS)} seeds, engine {ENGINE}")
    print(f"  routing is GIVEN, so chance = A/M and rises with A -- read "
          f"capacity/neuron across rows, not acc\n")

    out = {}
    for arm, label in (("split", f"SPLIT: fixed {N_TOTAL} neurons total"),
                       ("grow", f"GROW:  fixed {N_AREA} neurons per area")):
        print(f"  --- {label} ---")
        print(f"  {'A':>3} {'n/area':>7} {'M/area':>7} {'neurons':>8} "
              f"{'chance':>7} {'acc':>8} {'spread':>8} {'floor':>7} "
              f"{'secs':>8} {'speedup':>8}")
        base = None
        for a in A_SWEEP:
            n_area = (N_TOTAL // a) if arm == "split" else N_AREA
            acc, ident, spr, secs, per = run(n_area, a, M)
            base = base or secs
            out[(arm, a)] = (acc, spr, secs, n_area * a, per)
            print(f"  {a:>3} {n_area:>7} {per:>7} {n_area * a:>8} "
                  f"{a / M:>7.4f} {acc:>8.4f} {spr:>8.4f} {K_ / n_area:>7.4f} "
                  f"{secs:>8.1f} {base / secs:>7.1f}x")
        print()

    print("  READING\n")
    for arm in ("split", "grow"):
        secs = [out[(arm, a)][2] for a in A_SWEEP]
        ideal = [secs[0] / a for a in A_SWEEP]
        print(f"    {arm:>5} wall-clock  " + "  ".join(
            f"A={a}:{s:.0f}s" for a, s in zip(A_SWEEP, secs)))
        print(f"    {'':>5} if exactly 1/A " + "  ".join(
            f"{i:.0f}s" for i in ideal))
    p1 = all(out[(arm, A_SWEEP[-1])][2] < 0.6 * out[(arm, 1)][2]
             for arm in ("split", "grow"))
    print(f"\n    P1 wall-clock falls ~1/A:            {str(p1):>5}")

    # capacity per neuron: items RETAINED (acc-weighted) per neuron
    def cap(arm, a):
        acc, _, _, neurons, per = out[(arm, a)]
        return acc * per * a / neurons * 1000.0     # items per 1000 neurons

    for arm in ("split", "grow"):
        caps = [cap(arm, a) for a in A_SWEEP]
        print(f"    {arm:>5} items retained / 1000 neurons  " + "  ".join(
            f"A={a}:{c:.1f}" for a, c in zip(A_SWEEP, caps)))
    p2 = cap("split", A_SWEEP[-1]) < cap("split", 1) * 0.9
    p3 = abs(cap("grow", A_SWEEP[-1]) - cap("grow", 1)) < 0.25 * cap("grow", 1)
    print(f"\n    P2 SPLIT loses capacity/neuron:      {str(p2):>5}")
    print(f"    P3 GROW  holds capacity/neuron flat: {str(p3):>5}")

    print()
    if p1 and p2:
        print("    MANY AREAS ARE A COMPUTE WIN PAID FOR IN CAPACITY.")
        print("    Out of a FIXED neuron budget, partitioning buys back the")
        print("    quadratic -- the same items cost ~1/A -- and gives up")
        print("    capacity per neuron, because one big area holds more than")
        print("    the sum of its parts. Which side wins is a choice about what")
        print("    is scarce, not a fact about the model.")
    elif p1:
        print("    MANY AREAS ARE A FREE COMPUTE WIN AT THIS SIZE. The capacity")
        print("    loss predicted from super-linear n-scaling did NOT show up,")
        print("    which is evidence against that scaling being steep here --")
        print("    and worth chasing, because the two measurements disagree.")
    else:
        print("    P1 FAILED. The cost model taken from the profile is wrong:")
        print("    replaying stored events is not what makes this quadratic.")
        print("    Nothing else in this file can be read until that is fixed.")


if __name__ == "__main__":
    main()
