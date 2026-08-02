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
        ok = [m for m in M_SWEEP if table[(n, m, mode, engine)][0] > 0.90]
        return max(ok) if ok else 0

    print("  READING\n")
    for mode in MODES:
        for e in ENGINES:
            cs = [ceiling(n, mode, e) for n in N_SWEEP]
            ratios = " -> ".join(
                f"{(b / a):.1f}x" if a else "n/a"
                for a, b in zip(cs, cs[1:]))
            print(f"    {mode:>4} ceiling on {e:>12}: "
                  + "  ".join(f"n={n} M={c}" for n, c in zip(N_SWEEP, cs))
                  + f"    ({ratios} per doubling)")
        print()

    rec_ex = [ceiling(n, "rec", "numpy_exact") for n in N_SWEEP]
    rec_sp = [ceiling(n, "rec", "numpy_sparse") for n in N_SWEEP]
    ff_ex = [ceiling(n, "ff", "numpy_exact") for n in N_SWEEP]
    ff_sp = [ceiling(n, "ff", "numpy_sparse") for n in N_SWEEP]

    z1 = len(set(rec_ex)) > 1
    z2 = all(1.4 <= (b / a) <= 3.0 for a, b in zip(rec_ex, rec_ex[1:]) if a)
    # inflation = sparse / exact, and Z3 says it GROWS with n
    infl = [(s / x) if x else float("nan") for s, x in zip(rec_sp, rec_ex)]
    z3 = all(b >= a for a, b in zip(infl, infl[1:])) and max(infl) > 1.0
    z4 = all(c == max(M_SWEEP) for c in ff_ex + ff_sp)

    print(f"    Z1 exact ceiling moves with n:        {str(z1):>5}   {rec_ex}")
    print(f"    Z2 and does so ~linearly (1.4-3x):    {str(z2):>5}")
    print(f"    Z3 sparse inflation GROWS with n:     {str(z3):>5}   "
          f"sparse/exact = " + " ".join(f"{i:.1f}x" for i in infl))
    print(f"    Z4 ff has no ceiling, either engine:  {str(z4):>5}   "
          f"exact {ff_ex}  sparse {ff_sp}")

    print()
    alpha = [1.15 * n / K_ for n in N_SWEEP]
    print("    against critical-load alpha* (M_max ~ 1.15 n/k):")
    print(f"       predicted  " + "  ".join(f"n={n} M={a:.0f}"
                                            for n, a in zip(N_SWEEP, alpha)))
    print(f"       exact      " + "  ".join(f"n={n} M={c}"
                                            for n, c in zip(N_SWEEP, rec_ex)))
    print(f"       sparse     " + "  ".join(f"n={n} M={c}"
                                            for n, c in zip(N_SWEEP, rec_sp)))

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
        print("    The ceiling scales with n on exact drive, and the sampler's")
        print("    inflation does NOT grow with n -- so the scaling is real and")
        print("    the instrument is not what produced it. brain.py's inference")
        print("    was right for a reason it had not established.")
    else:
        print("    THE EXACT CEILING DOES NOT MOVE WITH n. The coverage law is")
        print("    refuted on the substrate, and 'accumulated potentiation")
        print("    rather than degree bias' rested on the sampler entirely.")
        print("    This is the larger finding and needs its own file.")


if __name__ == "__main__":
    main()
