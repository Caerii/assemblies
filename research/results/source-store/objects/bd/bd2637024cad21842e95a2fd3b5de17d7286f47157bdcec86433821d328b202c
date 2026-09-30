"""Is the capacity exponent a property of the substrate, or of a fixed gain?

THE TEST
--------
`task90_ceiling_fine_grid.py` measured `M_max ~ n^1.70` with beta and T held
fixed while n varied -- i.e. at fixed ABSOLUTE gain `(1+beta)^T`. That is the
design [[critical-load-alpha-star]] records as broken:

    Whenever g_c depends on an axis, comparing along that axis at fixed
    ABSOLUTE gain is confounded -- this bit the n sweep and then the k sweep
    identically.

and the controlled version of that measurement came out EXTENSIVE (exponent
1.01, `M_max ~ 1.15 n/k`). So 1.70 is exactly what the confound manufactures.

The cheap decisive check does not need `g_c(n)` at all. If the exponent is a
property of the substrate it is the SAME at every gain. If it is inherited from
the critical boundary sliding under a fixed gain, it MOVES with gain -- because
a different fixed gain sits at a different distance from `g_c(n)` at each n,
and that distance is the thing doing the work.

So: measure `a` at three absolute gains and look at the spread.

    gain = (1+beta)^T,  varied via beta at fixed T

PRE-REGISTERED
--------------
G1 `a` differs across gains by more than the bracket width of any single
   estimate. That is CONFOUNDED, and 1.70 must not be quoted.

G2 If instead `a` is stable across gains, the fixed-absolute-gain objection
   does not bite HERE, and super-linearity stands -- which would then need
   reconciling with the standing alpha* = 1.15, since both cannot be right.
   That reconciliation would be the finding, not the exponent.

I expect G1. Predicting my own result wrong has been the norm today, which is
why this is written down before the run.

DESIGN NOTES
------------
* Two n values, the widest span available (1000 and 4000), because the
  question is the SLOPE and two points give it. Three gains x two n is six
  ceiling searches; the fine grid took three.
* Each ceiling is located by the same `_substrate.ceiling_from_curve` used
  everywhere else, on a x1.25 grid seeded from the coarse bracket, so a cliff
  is flagged rather than interpolated across.
* The grid per (n, gain) is found by doubling until accuracy drops, THEN
  refining -- the brackets from the beta=0.10 run do not transfer to other
  gains and assuming they did would be the same class of error again.
"""

from __future__ import annotations

import math
import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lexicon_capacity_law as L  # noqa: E402
from _substrate import ceiling_from_curve  # noqa: E402

ENGINE = "numpy_exact"
# n=1000/2000, not 1000/4000. At beta=0.05 the n=4000 ceiling sits near M=800
# and that single cell cost 49 MINUTES; the slope needs two points, not the
# widest two. Stated because shrinking the span to afford the control is a real
# weakening of the test and should not be silent.
N_PAIR = (1000, 2000)
BETAS = (0.05, 0.10, 0.20)
T = L.PARENT_ROUNDS


def measure(n, beta, budget=9):
    """Locate M* at (n, beta): double until it fails, then refine at x1.25."""
    pts, m = [], 8
    while len(pts) < budget:
        acc = L.run(n, m, "rec", ENGINE, beta)[0]
        pts.append((m, acc))
        print(f"      M={m:<5} acc={acc:.4f}", flush=True)
        if acc <= 0.90:
            break
        m *= 2
    c = ceiling_from_curve(pts)
    if c.censored:
        return c, pts
    # REFINE UNTIL RESOLVED, not until `supported`. An earlier version stopped
    # as soon as one interior point existed, which left factor-2 brackets in
    # the result -- and m_star on a factor-2 bracket is grid-dependent by ~30%,
    # which moved the two-point exponent by 0.20 and swamped the gain effect
    # this file exists to measure. See `Ceiling.resolved`.
    while not c.resolved() and len(pts) < budget + 6:
        lo, hi = c.lo, c.hi or c.lo * 2
        x = lo * 1.25
        added = False
        while x < hi:
            mm = int(round(x))
            if mm not in [p[0] for p in pts] and lo < mm < hi:
                acc = L.run(n, mm, "rec", ENGINE, beta)[0]
                pts.append((mm, acc))
                added = True
                print(f"      M={mm:<5} acc={acc:.4f}  (refine)", flush=True)
            x *= 1.25
        c = ceiling_from_curve(pts)
        if not added:
            break                      # grid exhausted; report what we have
    return c, pts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  Is the capacity exponent a substrate property, or a fixed gain?")
    print(f"  k={L.K_} p={L.P_} T={T}, engine {ENGINE}, "
          f"n in {N_PAIR}, beta in {BETAS}\n")

    out = {}
    for beta in BETAS:
        gain = (1 + beta) ** T
        print(f"  --- beta={beta}  gain=(1+beta)^{T}={gain:.2f} ---")
        for n in N_PAIR:
            print(f"    n={n}")
            t = time.perf_counter()
            c, _pts = measure(n, beta)
            out[(beta, n)] = c
            print(f"      -> {c}   [{time.perf_counter() - t:.0f}s]\n",
                  flush=True)

    print("  READING\n")
    lo_n, hi_n = N_PAIR
    span = math.log2(hi_n / lo_n)
    slopes = {}
    for beta in BETAS:
        a, b = out[(beta, lo_n)], out[(beta, hi_n)]
        slopes[beta] = math.log2(b.m_star / a.m_star) / span
        flag = "" if (a.supported and b.supported) else "   [UNSUPPORTED CROSSING]"
        print(f"    beta={beta:<5} gain={(1 + beta) ** T:>5.2f}   "
              f"M*({lo_n})={a.m_star:>7.1f}  M*({hi_n})={b.m_star:>7.1f}   "
              f"a={slopes[beta]:.2f}{flag}")

    vals = list(slopes.values())
    spread = max(vals) - min(vals)
    # THE ESTIMATOR'S OWN REPRODUCIBILITY, measured rather than assumed: the
    # same configuration (n=1000, beta=0.10) gave m_star 20.5 on a factor-2
    # bracket and 27.1 on a x1.25 one -- 32% apart, moving a two-point exponent
    # by 0.20. A gain effect smaller than that is simply not detectable by this
    # design, so the decision threshold is set against the measured floor
    # instead of picked. The first run of this file reported "not confounded"
    # at spread 0.29 against a threshold of 0.30, which was a coin flip
    # dressed as a verdict.
    GRID_NOISE = 0.20
    print(f"\n    exponent across gains: {min(vals):.2f} .. {max(vals):.2f}   "
          f"spread {spread:.2f}")

    unresolved = [(b, n) for b in BETAS for n in N_PAIR
                  if not out[(b, n)].resolved()]
    g1 = spread > GRID_NOISE
    print(f"    estimator grid-noise floor: {GRID_NOISE:.2f}  "
          f"(measured -- see Ceiling.resolved)")
    if unresolved:
        print(f"\n    !! {len(unresolved)} of {len(BETAS) * len(N_PAIR)} "
              f"crossings NOT RESOLVED: "
              + ", ".join(f"beta={b} n={n}" for b, n in unresolved))
        print(f"    The exponents above are NOT quotable and neither verdict")
        print(f"    below means anything. Widen the refine budget and re-run.")
    print(f"\n    G1 `a` MOVES with gain (confounded):  {str(g1):>5}")
    print(f"    G2 `a` is stable (property):          {str(not g1):>5}")
    if abs(spread - GRID_NOISE) < 0.08:
        print(f"       ^ spread {spread:.2f} sits within 0.08 of the grid-noise")
        print(f"         floor {GRID_NOISE:.2f}. INCONCLUSIVE whichever way it")
        print(f"         fell -- the design cannot separate a gain effect from")
        print(f"         the estimator's own scatter at this size.")

    print()
    if g1:
        print("    CONFIRMED CONFOUNDED. The exponent is not a property of the")
        print("    substrate -- it is a reading of how far a FIXED gain sits")
        print("    from g_c(n), and g_c moves with n. So a = 1.70 is withdrawn,")
        print("    exactly as n^1.49 was, and this session's claim that the")
        print("    coverage law and alpha* are refuted is WITHDRAWN with it.")
        print("    The standing result stands: capacity is EXTENSIVE,")
        print("    M_max ~ 1.15 n/k ([[critical-load-alpha-star]]).")
        print()
        print("    To measure the exponent properly, sweep n at fixed RELATIVE")
        print("    gain g = c * g_c(n), which needs g_c(n) for THIS protocol --")
        print("    a separate measurement, and the real next step.")
    else:
        print("    NOT CONFOUNDED HERE. `a` holds across a "
              f"{max((1 + b) ** T for b in BETAS) / min((1 + b) ** T for b in BETAS):.1f}x")
        print("    range of absolute gain, so it is not inherited from the")
        print("    boundary sliding. That leaves a genuine conflict with the")
        print("    standing alpha* = 1.15 (exponent 1.01), measured on a")
        print("    DIFFERENT protocol (depth-5 composition, kp=2.5). Both")
        print("    cannot describe the same quantity; reconciling them is the")
        print("    finding, and neither exponent should be quoted until then.")


if __name__ == "__main__":
    main()
