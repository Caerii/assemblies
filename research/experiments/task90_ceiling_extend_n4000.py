"""#90: locate the n=4000 recurrence ceiling, which the main sweep only bounded.

`task90_ceiling_n_scaling.py` stops at M=256, and at n=4000 BOTH engines still
read acc 1.0000 there with spread barely above the floor. So that row is
CENSORED: the ceiling is >=256, and every exponent computed from it is a lower
bound. This extends M upward at n=4000 only, which is the cheapest way to turn
the bound into a measurement.

WHY IT MATTERS. The one uncensored doubling gives 16 -> 64, four-fold per
doubling of n. If that held to n=4000 the ceiling would be ~256 and the law
would be roughly M_max ~ n^2 -- super-linear, which contradicts BOTH the
coverage law (M_max ~ n at fixed k) and alpha* (1.15 n/k = 92 here). A previous
n^1.49 claim in this repo was withdrawn as a fixed-absolute-gain artifact, so
naming an exponent from three points and one clean step would be repeating that
mistake. This file exists to get a second clean step.

Cost grows about quadratically in M, so M=512 at n=4000 is roughly an hour of
the budget and M=1024 several. 512 is run first because it is decisive if the
ceiling is where the two-step extrapolation puts it.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexicon_capacity_law import BETA, K_, P_, PARENT_ROUNDS, run  # noqa: E402

N = 4000
M_EXTEND = (256, 512)
ENGINES = ("numpy_sparse", "numpy_exact")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  #90 -- locating the n={N} recurrence ceiling (main sweep "
          f"censored it at M=256)")
    print(f"  k={K_} beta={BETA} p={P_} buildT={PARENT_ROUNDS}, "
          f"chance floor {K_ / N:.4f}\n")
    print(f"  {'M':>6} {'cover':>7} {'engine':>13} {'acc':>9} {'ident':>8}"
          f" {'spread':>8} {'secs':>7}")

    ceiling = {e: 0 for e in ENGINES}
    censored = {e: True for e in ENGINES}
    for m_words in M_EXTEND:
        for e in ENGINES:
            t = time.perf_counter()
            acc, ident, spr, _ = run(N, m_words, "rec", e)
            print(f"  {m_words:>6} {m_words * K_ / N:>7.2f} {e:>13} "
                  f"{acc:>9.4f} {ident:>8.4f} {spr:>8.4f} "
                  f"{time.perf_counter() - t:>7.1f}")
            if acc > 0.90:
                ceiling[e] = m_words
            else:
                censored[e] = False
        print()

    print("  READING\n")
    for e in ENGINES:
        mark = ">=" if censored[e] else "="
        print(f"    n={N} rec ceiling on {e:>13}: M{mark}{ceiling[e]}")
    print()
    ex, sp = ceiling["numpy_exact"], ceiling["numpy_sparse"]
    if censored["numpy_exact"]:
        print(f"    STILL CENSORED at M={max(M_EXTEND)}. The ceiling at n={N}")
        print(f"    is above the extended sweep too, so the exponent remains")
        print(f"    unmeasured and no law should be named from these points.")
        print(f"    Next step is M=1024, which costs ~4x this run.")
    else:
        step = ex / 64.0     # n=2000 exact ceiling was 64, uncensored
        print(f"    Second clean step: n=2000 M=64 -> n={N} M={ex}, "
              f"{step:.1f}x per doubling.")
        print(f"    First step was 16 -> 64, 4.0x. Two steps at ~{step:.1f}x and")
        print(f"    4.0x put the law near M_max ~ n^2, NOT M_max ~ n.")
        print()
        print(f"    Coverage at the ceiling: {ex * K_ / N:.2f} here against 0.8")
        print(f"    at n=1000 -- rising with n, which a coverage law forbids.")
        print(f"    alpha* (1.15 n/k) predicts {1.15 * N / K_:.0f}; measured {ex}.")
        print()
        print(f"    STILL ONLY THREE POINTS. An n^1.49 claim here was withdrawn")
        print(f"    once as an artifact; two clean steps are enough to say")
        print(f"    'super-linear, and not the stated law', and NOT enough to")
        print(f"    fit an exponent. The mechanism to test next is extreme-value")
        print(f"    statistics over n candidates -- an assembly survives its own")
        print(f"    recurrence while (1+beta)^T beats the max of n draws -- which")
        print(f"    predicts a ceiling growing faster than n and has nothing to")
        print(f"    do with coverage.")


if __name__ == "__main__":
    main()
