"""Measure M_max HEAD-ON, and sweep k. (task #46)

WHY. The capacity claim so far is an INFERENCE, not a measurement. g_c was
measured against gain, and capacity was then read off backwards by inverting
two exchange rates:

    doubling M at fixed n costs 0.339 of gain
    doubling n at fixed M buys 0.539 of gain
    => M_max ~ n^1.6   at fixed gain

Superlinear capacity is a strong claim -- it says capacity is NOT extensive,
that a 2x larger area holds ~3x more composable items -- and it deserves a
direct test rather than an extrapolation from four cells fitted with a model
already known to be imperfect (the g_c increments grow where a logarithm
requires them constant).

So: FIX THE GAIN, PUSH M UNTIL RETRIEVAL BREAKS, and read M_max off the
accuracy curve. That is the quantity the claim is about, measured as itself.

SECOND ARM: k. Every sweep so far held k=50 while varying n, so nothing here
speaks to capacity per ASSEMBLY size, which is the more natural reading of "how
many memories fit". k is swept two ways, because the two are different
questions and the project has already been caught conflating them:

  * fixed p     -- k*p rises with k, so this measures the combined effect
  * fixed k*p   -- p scaled as 1/k, isolating assembly size from afferent count

k*p was shown to flip the SIGN of beta's effect, so leaving it uncontrolled
would make an afferent-count effect look like an assembly-size effect.
"""

from __future__ import annotations

import csv
import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.environ.get("CAP_OUT", "capacity_direct.csv"))
DEPTH = int(os.environ.get("CAP_DEPTH", "5"))
T = 2
GAIN = float(os.environ.get("CAP_GAIN", "1.69"))   # inside the working regime
SEEDS = [int(x) for x in os.environ.get("CAP_SEEDS", "42,43").split(",")]
MODE = os.environ.get("CAP_MODE", "n")             # "n" or "k"

#: Geometric ladders, sized per n so the crossing is bracketed without paying
#: for cells far above it.
N_LADDER = {
    2000: [16, 32, 64, 128],
    4000: [32, 64, 128, 256],
    8000: [64, 128, 256, 512],
}
K_LADDER = [int(x) for x in os.environ.get("CAP_KS", "25,50,100").split(",")]


def run(n, m_items, k, p):
    ladder.BETA = GAIN ** (1.0 / T) - 1.0
    ladder.MERGE_ROUNDS = T
    ladder.K_ = k
    ladder.P_ = p
    accs, margs = [], []
    for s in SEEDS:
        full, total, margins, _spreads = ladder.trial(n, m_items, DEPTH, s)
        accs.append(full[DEPTH] / total)
        margs.append(margins[DEPTH])
    return sum(accs) / len(accs), sum(margs) / len(margs)


def crossing(ms, accs, level=0.5):
    """M where accuracy falls through `level`, interpolated in log M."""
    import math
    for i in range(len(ms) - 1):
        if accs[i] >= level > accs[i + 1]:
            t = (accs[i] - level) / max(accs[i] - accs[i + 1], 1e-12)
            return math.exp(math.log(ms[i])
                            + t * (math.log(ms[i + 1]) - math.log(ms[i])))
    return float("nan")


def main():
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["mode", "n", "k", "p", "kp", "M", "gain", "depth",
                    "acc", "margin"])
        fh.flush()

    print(f"\n  DIRECT CAPACITY   gain={GAIN} (beta={GAIN ** (1 / T) - 1:.3f}) "
          f"depth={DEPTH} seeds={len(SEEDS)}  mode={MODE}\n")

    if MODE == "n":
        results = []
        for n, ms in N_LADDER.items():
            print(f"  n={n}  k=50 p=0.05")
            accs = []
            for m in ms:
                t0 = time.time()
                acc, marg = run(n, m, 50, 0.05)
                accs.append(acc)
                w.writerow([MODE, n, 50, 0.05, 2.5, m, GAIN, DEPTH,
                            f"{acc:.6f}", f"{marg:.6f}"])
                fh.flush()
                print(f"    M={m:<5} acc={acc:.3f} marg={marg:6.2f} "
                      f"[{time.time() - t0:.0f}s]")
            mmax = crossing(ms, accs)
            results.append((n, mmax))
            print(f"    -> M_max ~ {mmax:.0f}\n")

        print("  --- capacity scaling ---")
        import math
        got = [(n, m) for n, m in results if m == m]
        for i in range(len(got) - 1):
            (n0, m0), (n1, m1) = got[i], got[i + 1]
            expo = math.log(m1 / m0) / math.log(n1 / n0)
            print(f"    n {n0}->{n1}:  M_max {m0:.0f}->{m1:.0f}   "
                  f"exponent {expo:.2f}")
        print("    (inferred claim was ~1.6; 1.0 would mean capacity is "
              "extensive)")

    else:
        for k in K_LADDER:
            for label, p in (("fixed p", 0.05), ("fixed kp", 2.5 / k)):
                n = 4000
                ms = [16, 32, 64, 128, 256]
                print(f"  k={k}  {label}  p={p:.4f}  kp={k * p:.2f}")
                accs = []
                for m in ms:
                    t0 = time.time()
                    acc, marg = run(n, m, k, p)
                    accs.append(acc)
                    w.writerow([MODE, n, k, f"{p:.5f}", f"{k * p:.3f}", m,
                                GAIN, DEPTH, f"{acc:.6f}", f"{marg:.6f}"])
                    fh.flush()
                    print(f"    M={m:<5} acc={acc:.3f} marg={marg:6.2f} "
                          f"[{time.time() - t0:.0f}s]")
                print(f"    -> M_max ~ {crossing(ms, accs):.0f}\n")
    fh.close()


if __name__ == "__main__":
    main()
