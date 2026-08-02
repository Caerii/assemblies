"""#90: measure the capacity exponent, instead of reading it off a doubling grid.

WHAT WAS WRONG WITH THE PREVIOUS ANSWER
---------------------------------------
`task90_ceiling_n_scaling.py` reports the largest M on a FACTOR-2 grid whose
accuracy still clears 0.90. That is a grid point, not a crossing: the true
ceiling lies anywhere in `[M, 2M)`. Propagating that through a ratio,

    n=1000  M* in [16, 32)
    n=2000  M* in [64, 128)
    ratio       in (2, 8)      -- an exponent anywhere in (1, 3)

so "four-fold per doubling, roughly n^2" was a point estimate the data could not
carry. Super-linearity IS established (the ratio exceeds 2 strictly, since
M*(1000) < 32 and M*(2000) >= 64), and that already refutes both the coverage
law (M_max ~ n) and alpha* (1.15 n/k). The EXPONENT was not established.

TWO FIXES, AND ONLY THE SECOND IS SUFFICIENT
--------------------------------------------
1. Interpolate the crossing instead of taking a grid point --
   `_substrate.ceiling_from_curve`, which uses every point and works in
   log2(M) because these sweeps are geometric.

2. Put points INSIDE the transition. Fix 1 alone is not enough and the harness
   says so: run on the existing coarse data, two of the three n values report
   `[CLIFF: no interior point]`, because accuracy goes 1.0000 -> 0.0208 in one
   step and interpolating across that is drawing a line over a hole. Only
   n=1000 had an interior point (0.7188 at M=32) and it yields M* = 20.5.

So this file re-sweeps M at x1.25 steps INSIDE each already-known bracket,
which is where the information is, and re-estimates. `.supported` on each
`Ceiling` says whether the interpolation is doing real work; anything still
flagged CLIFF is reported as a bracket and not as a number.

PRE-REGISTERED
--------------
E1 The fine grid produces `supported` crossings (>=1 interior point) at every
   n. If the transition is genuinely a step -- all-or-nothing between two
   adjacent M at x1.25 -- then no grid resolves it and the ceiling is not a
   continuous quantity, which would itself be the finding.

E2 The fitted exponent lands in (1, 3), consistent with the bracket derived
   from the coarse grid. If it lands outside, the two estimators disagree and
   neither should be quoted.

E3 The exponent is > 1 with the bracket uncertainty propagated, i.e.
   super-linearity survives the more careful measurement. This is the claim
   that actually matters; the specific exponent is secondary.

WHY THE BAR IS HIGH HERE
------------------------
An `n^1.49` capacity claim in this repo was withdrawn as a fixed-absolute-gain
artifact ([[critical-load-alpha-star]]). Naming an exponent from three points
is how that happened. This file reports a RANGE from the bracket extremes
alongside the least-squares slope, and treats disagreement between them as a
reason not to quote either.
"""

from __future__ import annotations

import math
import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import ceiling_from_curve  # noqa: E402
from lexicon_capacity_law import BETA, K_, P_, PARENT_ROUNDS, run  # noqa: E402

ENGINE = "numpy_exact"

#: (n, coarse bracket lo, coarse bracket hi) from task90_n_scaling.log, plus
#: the n=4000 bracket which task90_ceiling_extend_n4000.py is resolving. The
#: x1.25 steps are generated inside each bracket; the endpoints are already
#: measured and are re-run so every row comes from one code path.
BRACKETS = ((1000, 16, 32), (2000, 64, 128), (4000, 256, 512))


def fine_grid(lo, hi, step=1.25):
    ms, m = [], float(lo)
    while m < hi:
        v = int(round(m))
        if v not in ms:
            ms.append(v)
        m *= step
    ms.append(hi)
    return ms


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  #90 -- capacity exponent on a x1.25 grid inside each bracket")
    print(f"  k={K_} beta={BETA} p={P_} buildT={PARENT_ROUNDS}, engine {ENGINE}")
    print(f"  crossing interpolated in log2(M); brackets from the coarse sweep\n")

    ceilings = {}
    for n, lo, hi in BRACKETS:
        grid = fine_grid(lo, hi)
        print(f"  --- n={n},  bracket [{lo}, {hi}),  grid {grid} ---")
        print(f"  {'M':>6} {'cover':>7} {'acc':>8} {'ident':>8} {'spread':>8}"
              f" {'secs':>8}")
        pts = []
        for m_words in grid:
            t = time.perf_counter()
            acc, ident, spr, _ = run(n, m_words, "rec", ENGINE)
            pts.append((m_words, acc))
            print(f"  {m_words:>6} {m_words * K_ / n:>7.2f} {acc:>8.4f} "
                  f"{ident:>8.4f} {spr:>8.4f} {time.perf_counter() - t:>8.1f}",
                  flush=True)
        c = ceiling_from_curve(pts)
        ceilings[n] = c
        print(f"      -> {c}\n", flush=True)

    print("  READING\n")
    ns = [n for n, _, _ in BRACKETS]
    cs = [ceilings[n] for n in ns]
    for n, c in zip(ns, cs):
        print(f"    n={n:>5}  {c}")

    e1 = all(c.supported for c in cs)
    print(f"\n    E1 every crossing has an interior point: {str(e1):>5}")
    if not e1:
        print("       Transitions with no interior point at x1.25 are STEPS at")
        print("       this resolution. No finer grid fixes that; report the")
        print("       bracket and stop calling it an exponent.")

    def slope(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

    lx = [math.log2(n) for n in ns]
    a_fit = slope(lx, [math.log2(c.m_star) for c in cs])
    # bracket extremes: steepest = lowest first, highest last; and vice versa
    a_hi = slope(lx, [math.log2(c.lo if i == 0 else (c.hi or c.lo))
                      for i, c in enumerate(cs)])
    a_lo = slope(lx, [math.log2((c.hi or c.lo) if i == 0 else c.lo)
                      for i, c in enumerate(cs)])
    print(f"\n    exponent a in M_max ~ n^a")
    print(f"      least squares on interpolated M*   a = {a_fit:.2f}")
    print(f"      from the coarse bracket extremes   a in "
          f"[{min(a_lo, a_hi):.2f}, {max(a_lo, a_hi):.2f}]")

    e2 = 1.0 < a_fit < 3.0
    e3 = min(a_lo, a_hi) > 1.0
    print(f"\n    E2 fit lands inside the coarse bracket: {str(e2):>5}")
    print(f"    E3 super-linear survives the bracket:   {str(e3):>5}")

    print()
    if e1 and e2 and e3:
        print(f"    THE EXPONENT IS MEASURED, NOT READ OFF A GRID: a ~ {a_fit:.2f}.")
        print(f"    Three points and one estimator, so quote it with the range")
        print(f"    and not alone. What is robust either way: a > 1, so the")
        print(f"    coverage law (a = 1) and alpha* are both refuted, and one")
        print(f"    big area holds MORE than the sum of its parts -- which is")
        print(f"    the same fact the many-areas split arm measured from the")
        print(f"    other side.")
    elif e3:
        print(f"    SUPER-LINEARITY HOLDS, THE EXPONENT DOES NOT RESOLVE.")
        print(f"    a > 1 survives every bracket, so the coverage law and alpha*")
        print(f"    stay refuted. The point estimate {a_fit:.2f} disagrees with")
        print(f"    the bracket range or rests on an unsupported crossing --")
        print(f"    report the range, not the number.")
    else:
        print(f"    SUPER-LINEARITY DOES NOT SURVIVE THE CAREFUL MEASUREMENT.")
        print(f"    That overturns this session's reading of the n-scaling")
        print(f"    sweep, and the coverage law is back on the table. Check the")
        print(f"    coarse and fine sweeps agree at the shared M before")
        print(f"    believing either.")


if __name__ == "__main__":
    main()
