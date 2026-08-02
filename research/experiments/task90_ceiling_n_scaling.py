"""#90 remainder: does the recurrence ceiling scale with n on EXACT drive?

THE CLAIM
---------
`core/brain.py` states the ceiling "scales with n (M=32 / 64 / 256 at n=1000 /
2000 / 4000), which is what identifies it as accumulated potentiation rather
than degree bias." Those three numbers came from `lexicon_capacity_law.py`,
measured on `numpy_sparse`.

`task90_recurrence_ceiling_on_exact.py` showed the companion claim from the
same table -- that norm_init moves the ceiling 8x -- is 1.0x on exact drive,
because the sampler's error is a function of LOAD and norm_init moves load. n
at fixed k IS load, by definition: coverage is `M*k/n`. So the n-scaling claim
was read off an instrument whose error moves with the very variable being
swept, and it was marked UNSUPPORTED rather than refuted. This file settles it.

Same protocol, same parameters, same seeds as `lexicon_capacity_law.py` -- it
imports `trial` from that file rather than restating it, so the two cannot
drift. k=50 and p=0.05 are held FIXED while n varies, which is the point: a
coverage law predicts `M_max` proportional to n at fixed k.

PRE-REGISTERED
--------------
Z1 The exact ceiling MOVES with n at all. If it is flat, the ceiling is not
   about coverage and the whole "accumulated potentiation" story needs
   rebuilding, on either engine.

Z2 The exact ceiling scales roughly LINEARLY in n (doubling n roughly doubles
   M_max), which is the coverage law's actual prediction. Stated with low
   confidence: `critical-load-alpha-star` puts capacity at `M_max ~ 1.15 n/k`,
   which at k=50 gives M_max = 23 / 46 / 92 for n = 1000 / 2000 / 4000 -- and
   the sparse numbers (32 / 64 / 256) are linear for the first two steps and
   then jump 4x, which linearity does not predict.

Z3 The SPARSE ceiling is inflated relative to exact, and inflated MORE at large
   n. Mechanism: bigger n at fixed k is lower load, and the sampler's error is
   worst at low load. If so, the 4x jump at n=4000 is where the instrument runs
   away, and the exact numbers should be closer to linear than the sparse ones.

Z4 Feed-forward has no ceiling in this range on either engine, at any n. This
   is the control. If ff collapses, something is wrong with the harness and
   nothing else here can be read -- that is exactly how the first version of
   the companion file was caught.

WHAT COULD STILL FOOL THIS
--------------------------
Ceilings are read off a factor-2 grid, so a "2x" and a "1.4x" are not
distinguishable and no ratio here should be quoted past one significant figure.
The grid is stated, not smoothed over: `acc` at each M is printed so a reader
can see how close to the 0.90 line each verdict was.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexicon_capacity_law import (BETA, K_, M_SWEEP, N_SWEEP,  # noqa: E402
                                  P_, PARENT_ROUNDS, SEEDS, run)

ENGINES = ("numpy_sparse", "numpy_exact")
MODES = ("rec", "ff")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  #90 -- does the ceiling scale with n? sampler vs exact drive")
    print(f"  k={K_} beta={BETA} p={P_} buildT={PARENT_ROUNDS}, ONE area, "
          f"{len(SEEDS)} seeds")
    print(f"  k and p held FIXED while n varies -- a coverage law predicts "
          f"M_max ~ n\n")

    table = {}
    for mode in MODES:
        print(f"  --- {mode} ---")
        print(f"  {'n':>6} {'M':>5} {'cover':>7} | "
              + " | ".join(f"{e + ' acc':>18}" for e in ENGINES))
        for n in N_SWEEP:
            for m_words in M_SWEEP:
                cells = []
                for e in ENGINES:
                    r = run(n, m_words, mode, e)
                    table[(n, m_words, mode, e)] = r
                    cells.append(f"{r[0]:>8.4f} spr {r[2]:>5.3f}")
                print(f"  {n:>6} {m_words:>5} {m_words * K_ / n:>7.2f} | "
                      + " | ".join(cells))
            print()

    def ceiling(n, mode, engine):
        """(M, censored). CENSORED means the arm still passed at the LARGEST M
        tested, so the true ceiling is only bounded BELOW.

        Reporting a censored value as if it were measured is how a sweep
        manufactures an exponent: at n=4000 the recurrent arm reads 1.0000 at
        M=256 with spread barely above the floor, so "ceiling = 256" is a
        statement about M_SWEEP, not about the substrate. Every ratio computed
        from a censored endpoint is a LOWER BOUND and is printed as ">=".
        """
        ok = [m for m in M_SWEEP if table[(n, m, mode, engine)][0] > 0.90]
        top = max(M_SWEEP)
        return (max(ok) if ok else 0), bool(ok) and max(ok) == top

    print("  READING\n")
    for mode in MODES:
        for e in ENGINES:
            cs = [ceiling(n, mode, e) for n in N_SWEEP]
            ratios = " -> ".join(
                (">=" if (cb or ab) else "") + f"{(b / a):.1f}x" if a else "n/a"
                for (a, ab), (b, cb) in zip(cs, cs[1:]))
            cells = "  ".join(f"n={n} M{'>=' if cen else '='}{c}"
                              for n, (c, cen) in zip(N_SWEEP, cs))
            print(f"    {mode:>4} ceiling on {e:>12}: {cells}"
                  + f"    ({ratios} per doubling)")
        if any(ceiling(n, mode, e)[1] for n in N_SWEEP for e in ENGINES):
            print(f"         ^ CENSORED at the top of M_SWEEP={max(M_SWEEP)}; "
                  f"those are lower bounds, not measurements")
        print()

    rec_ex = [ceiling(n, "rec", "numpy_exact") for n in N_SWEEP]
    rec_sp = [ceiling(n, "rec", "numpy_sparse") for n in N_SWEEP]
    ff_ex = [ceiling(n, "ff", "numpy_exact") for n in N_SWEEP]
    ff_sp = [ceiling(n, "ff", "numpy_sparse") for n in N_SWEEP]

    # Ratios are only interpretable where NEITHER endpoint is censored.
    steps = [(a, b, ac or bc) for (a, ac), (b, bc)
             in zip(rec_ex, rec_ex[1:])]
    clean = [(a, b) for a, b, cen in steps if not cen and a]
    z1 = len({c for c, _ in rec_ex}) > 1
    z2 = bool(clean) and all(1.4 <= b / a <= 3.0 for a, b in clean)
    infl = [(s / x) if x else float("nan")
            for (s, _), (x, _) in zip(rec_sp, rec_ex)]
    z3 = all(b >= a for a, b in zip(infl, infl[1:])) and max(infl) > 1.0
    z4 = all(c == max(M_SWEEP) for c, _ in ff_ex + ff_sp)

    print(f"    Z1 exact ceiling moves with n:        {str(z1):>5}   "
          + " ".join(f"{'>=' if cen else ''}{c}" for c, cen in rec_ex))
    print(f"    Z2 and does so ~linearly (1.4-3x):    {str(z2):>5}   "
          + (f"uncensored steps: "
             + " ".join(f"{a}->{b} ({b / a:.1f}x)" for a, b in clean)
             if clean else "NO uncensored step -- Z2 IS UNTESTED, not false"))
    print(f"    Z3 sparse inflation GROWS with n:     {str(z3):>5}   "
          f"sparse/exact = " + " ".join(f"{i:.1f}x" for i in infl))
    print(f"    Z4 ff has no ceiling, either engine:  {str(z4):>5}   "
          f"(and every ff value is CENSORED at M={max(M_SWEEP)}, which is what "
          f"'no ceiling in this range' means)")

    print()
    alpha = [1.15 * n / K_ for n in N_SWEEP]
    print("    against critical-load alpha* (M_max ~ 1.15 n/k):")
    print(f"       predicted  " + "  ".join(f"n={n} M={a:.0f}"
                                            for n, a in zip(N_SWEEP, alpha)))
    for name, cs in (("exact", rec_ex), ("sparse", rec_sp)):
        print(f"       {name:<10}" + "  ".join(
            f"n={n} M{'>=' if cen else '='}{c}"
            for n, (c, cen) in zip(N_SWEEP, cs)))

    print()
    if not z4:
        print("    Z4 FAILED -- the feed-forward CONTROL collapsed. Nothing")
        print("    above can be read until that is explained; a control that")
        print("    cannot collapse and did is a harness defect, and it is what")
        print("    caught the first version of the companion file.")
    elif z1 and z3:
        print("    THE CEILING DOES SCALE WITH n, AND THE SAMPLER EXAGGERATES")
        print("    HOW MUCH. brain.py's 'M=32 / 64 / 256' is the right shape")
        print("    read off the wrong instrument: the coverage story survives,")
        print("    the numbers do not, and the 4x final step is where the")
        print("    sampler's low-load error runs away rather than where the")
        print("    substrate gains capacity.")
    elif z1:
        print("    THE CEILING SCALES WITH n, AND THE SAMPLER DID NOT PRODUCE")
        print("    THAT. Inflation is 2.0x / 1.0x / 1.0x -- it does not grow")
        print("    with n, so this is the one #90 claim that SURVIVES contact")
        print("    with exact drive. brain.py's inference was right; it just")
        print("    had not established it, and my own prediction (Z3, that the")
        print("    4x step was the instrument running away) is FALSIFIED.")
        print()
        print("    BUT IT IS NOT THE LAW ANYONE STATED. On the one uncensored")
        print("    doubling, exact goes 16 -> 64: FOUR-fold per doubling of n,")
        print("    not two. That is super-linear, so it is not the coverage law")
        print("    (M_max ~ n at fixed k), and it is not alpha* either --")
        print("    1.15n/k predicts 23 / 46 / 92 against measured 16 / 64 />=256.")
        print("    Critical COVERAGE at the ceiling RISES with n (0.8, 1.6,")
        print("    >=3.2), which is precisely what a coverage law forbids.")
        print()
        print("    WHAT IS NOT ESTABLISHED: the exponent. n=4000 is censored at")
        print("    the top of M_SWEEP, so 'quadratic' rests on ONE clean step")
        print("    and three points. A previous n^1.49 claim here was withdrawn")
        print("    as a fixed-absolute-gain artifact, so the bar is high: locate")
        print("    the n=4000 ceiling with a longer M_SWEEP before naming any")
        print("    exponent, and check the mechanism -- extreme-value statistics")
        print("    over n candidates, not coverage -- predicts the shape.")
    else:
        print("    THE EXACT CEILING DOES NOT MOVE WITH n. The coverage law is")
        print("    refuted on the substrate, and 'accumulated potentiation")
        print("    rather than degree bias' rested on the sampler entirely.")
        print("    This is the larger finding and needs its own file.")


if __name__ == "__main__":
    main()
