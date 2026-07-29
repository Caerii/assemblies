"""Fine scan of the gain-driven transition, as a critical point. (task #46)

WHAT IS ALREADY KNOWN. Composition collapses at a boundary set by the GAIN

    g = (1 + beta)^T

and not by beta or T separately: beta ranged 0.18-0.45 across the measured
boundary while g stayed near 1.9. Overlap jumps from the floor to near-total
across a narrow band in g while the control varies smoothly.

WHAT THIS ADDS. That was a coarse grid at one system size, which is enough to
locate a cliff and not enough to call it a critical point. A transition earns
that name by how it behaves with SYSTEM SIZE: at a genuine continuous
transition the curves for different n cross at a single g_c and collapse onto
one function of (g - g_c) * n^(1/nu). A mere threshold does not do this.

So this scans g finely at several n, holding the two things that would
otherwise confound the comparison fixed:

  * LOAD alpha = M*k/n, the fraction of the substrate spoken for. Holding M
    fixed while growing n would change the load and the size together, and the
    load is itself a control parameter.
  * AFFERENT COUNT k*p, which sets whether beta helps at all. Held by scaling
    p with 1/k when k is scaled.

ORDER PARAMETER. Raw overlap has a size-dependent floor q0 = k/n, so it cannot
be compared across n. The scan records the NORMALISED overlap

    Q = (q - q0) / (1 - q0)

which is 0 for independent assemblies at any n and 1 for total collapse.

OUTPUT is tidy CSV, one row per (n, gain, seed, level), written incrementally
so partial runs are still plottable. Nothing is aggregated here -- means and
intervals belong in the analysis, not in the measurement.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import complexity_ladder_overnight as ladder  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.environ.get("CPS_OUT", "critical_point_scan.csv"))

#: System sizes. Load is held fixed by scaling M with n.
NS = [int(x) for x in os.environ.get("CPS_NS", "2000,4000,8000").split(",")]
ALPHA = float(os.environ.get("CPS_ALPHA", "0.8"))   # = M*k/n
DEPTH = int(os.environ.get("CPS_DEPTH", "5"))
T = int(os.environ.get("CPS_T", "2"))
SEEDS = [int(x) for x in os.environ.get("CPS_SEEDS", "42,43,44").split(",")]

#: Gains bracketing the coarse boundary near 1.9, denser where it matters.
GAINS = [float(x) for x in os.environ.get(
    "CPS_GAINS",
    "1.50,1.65,1.75,1.82,1.88,1.92,1.96,2.00,2.06,2.15,2.30").split(",")]


def beta_for(gain, t):
    """Invert g = (1+beta)^T. The scan is over g; beta is the knob it needs."""
    return gain ** (1.0 / t) - 1.0


def main():
    k = ladder.K_
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["n", "k", "p", "M", "alpha", "T", "beta", "gain",
                    "depth", "level", "seed", "acc", "margin", "spread",
                    "floor", "Q"])
        fh.flush()

    print(f"\n  CRITICAL POINT SCAN   alpha={ALPHA} depth={DEPTH} T={T} "
          f"k={k} p={ladder.P_}")
    print(f"  {len(NS)} sizes x {len(GAINS)} gains x {len(SEEDS)} seeds = "
          f"{len(NS) * len(GAINS) * len(SEEDS)} trials -> {OUT}\n")

    for n in NS:
        m_items = max(2, int(round(ALPHA * n / k)))
        floor = k / n
        print(f"  n={n}  M={m_items}  floor={floor:.5f}")
        for gain in GAINS:
            beta = beta_for(gain, T)
            ladder.BETA = beta
            ladder.MERGE_ROUNDS = T
            t0 = time.time()
            accs, Qs = [], []
            for seed in SEEDS:
                full, total, margins, spreads = ladder.trial(
                    n, m_items, DEPTH, seed)
                for L in range(1, DEPTH + 1):
                    q = spreads[L]
                    Q = (q - floor) / (1.0 - floor)
                    w.writerow([n, k, ladder.P_, m_items, ALPHA, T,
                                f"{beta:.6f}", f"{gain:.4f}", DEPTH, L, seed,
                                f"{full[L] / total:.6f}",
                                f"{margins[L]:.6f}", f"{q:.6f}",
                                f"{floor:.6f}", f"{Q:.6f}"])
                    if L == DEPTH:
                        accs.append(full[L] / total)
                        Qs.append(Q)
            fh.flush()
            mean = lambda v: sum(v) / len(v)  # noqa: E731
            print(f"    g={gain:.2f} (beta={beta:.3f})  acc={mean(accs):.3f} "
                  f"Q={mean(Qs):.4f}   [{time.time() - t0:.0f}s]")
        print()
    fh.close()
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
