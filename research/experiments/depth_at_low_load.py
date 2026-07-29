"""Does depth fail, or does depth fail AT LOAD? (task #46)

The overnight ladder invites a wrong reading. Its deep cells fail, so it looks
like there is a depth ceiling -- but every one of them is also at high load,
because the cost model spent its budget on large M. The single cheapest and
most informative cell, LOW LOAD AND DEEP (n=4000, M=64, D=5), was never run.

The ladder's own beta=0.20 numbers already argue against a depth ceiling:

    n=4000 M=64  D=3   margin 8.05 at the deepest level   PASS
    n=4000 M=256 D=3   margin 1.78                        MARGINAL
    n=4000 M=512 D=3   margin 1.02                        (level 1 fine: 6.34)

Depth 3 is not degrading with depth there -- it is degrading with M at fixed
depth. So the hypothesis under test is that depth is cheap when the substrate
is not crowded, and that the ladder's deep failures are load failures wearing a
depth costume.

This sweeps depth against beta at FIXED low load, which is the one cut that
separates the two. Memory of this project says the two pressures oppose:
capacity wants low beta (less overlap), depth wants high beta (a fiber must be
potentiated enough to transmit). If that is right, the depth ceiling should
MOVE with beta rather than sit at a fixed level.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402

N = int(os.environ.get("DLL_N", "4000"))
M = int(os.environ.get("DLL_M", "64"))
DEPTHS = [int(x) for x in os.environ.get("DLL_DEPTHS", "3,4,5,6").split(",")]
BETAS = [float(x) for x in os.environ.get("DLL_BETAS", "0.10,0.20,0.35").split(",")]
SEEDS = [int(x) for x in os.environ.get("DLL_SEEDS", "42,43").split(",")]


def run_cell(beta, depth):
    """Mean over seeds of the DEEPEST level's margin, spread and distinctness."""
    ladder.BETA = beta          # trial reads the module-level constant
    margs, sprs, dists = [], [], []
    for s in SEEDS:
        distinct, total, margins, spreads = ladder.trial(N, M, depth, s)
        margs.append(margins[depth])
        sprs.append(spreads[depth])
        dists.append(distinct[depth] / total if total else float("nan"))
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    return mean(margs), mean(sprs), mean(dists)


if __name__ == "__main__":
    load = M * 50 / N   # k=50 in the ladder
    print(f"\n  DEPTH AT LOW LOAD   n={N} M={M} load={load:.1f} "
          f"seeds={len(SEEDS)}")
    print("  margin/spread/distinct are at the DEEPEST level; "
          f"floor={50 / N:.4f}\n")
    print(f"  {'beta':>6} {'depth':>6} {'marg_D':>8} {'spr_D':>8} "
          f"{'distinct':>9}  {'':>4}")

    for beta in BETAS:
        for depth in DEPTHS:
            t0 = time.time()
            try:
                marg, spr, dist = run_cell(beta, depth)
            except Exception as exc:  # noqa: BLE001
                print(f"  {beta:>6.2f} {depth:>6} "
                      f"      FAILED: {type(exc).__name__}: {exc}")
                continue
            # A margin at 1.00 with everything still distinct is signal loss;
            # a collapsed spread is crowding. They need different fixes, so
            # never report one as the other.
            if dist < 0.9:
                verdict = "CROWDED"
            elif marg < 1.10:
                verdict = "SIGNAL-LOSS"
            elif marg < 2.0:
                verdict = "WEAK"
            else:
                verdict = "OK"
            print(f"  {beta:>6.2f} {depth:>6} {marg:>8.2f} {spr:>8.4f} "
                  f"{dist:>9.3f}  {verdict:<12} [{time.time() - t0:.0f}s]")
        print()
