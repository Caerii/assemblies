"""Finite-size scaling: locate g_c(n) and test for data collapse. (task #46)

A cliff becomes a critical point only if it behaves like one under change of
system size. This does the quantitative version of that check, on the tidy CSV
from critical_point_scan.py.

THREE QUESTIONS, IN ORDER.

1. WHERE is the boundary, per size? Located by the retrieval half-crossing,
   refined by a logistic fit in g so the answer does not depend on which grid
   points happen to straddle 1/2.

2. HOW does it move? Three candidate forms are fitted and compared by residual:

     constant     g_c = a                    -- a substrate constant
     log          g_c = a + b*ln(n)
     extreme      g_c = a + b*sqrt(2 ln(n/k))  -- the shape implied by k-WTA
                                                 selecting the top k of n

   The third is the one the theory sketch predicts: winners are the upper order
   statistics of a candidate pool of size n, and the gap between the mean and
   the k-th largest of n draws grows like sqrt(2 ln(n/k)). If that form fits
   best, the mechanism is competition against the FRESH POOL. This cannot be
   settled by fit quality alone with few sizes, so the fit is reported as
   evidence, not proof.

3. DOES IT COLLAPSE? Under the scaling hypothesis
   Q(g,n) = F((g - g_c(n)) * n^(1/nu)), curves at different n fall on one
   function. nu is scanned to minimise the spread between size curves on a
   common grid. A genuine continuous transition collapses; a threshold that
   merely drifts does not.

REPORTED HONESTLY: with three sizes, exponents are indicative at best. The
purpose is to distinguish "collapses well under some nu" from "does not
collapse at all", which three sizes can do.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSVS = [x for x in os.environ.get(
    "FSS_CSVS", "critical_point_scan.csv").split(",") if x]


def load(paths):
    rows = []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(HERE, p)
        if not os.path.exists(full):
            continue
        with open(full, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    "n": int(r["n"]), "k": int(r["k"]), "M": int(r["M"]),
                    "level": int(r["level"]), "depth": int(r["depth"]),
                    "gain": float(r["gain"]), "acc": float(r["acc"]),
                    "Q": float(r["Q"]), "alpha": float(r["alpha"]),
                })
    return rows


def curves(rows):
    """(n, M) -> (gains, mean acc, mean Q) at the deepest level."""
    acc = defaultdict(lambda: defaultdict(list))
    q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["level"] != r["depth"]:
            continue
        acc[(r["n"], r["M"])][r["gain"]].append(r["acc"])
        q[(r["n"], r["M"])][r["gain"]].append(r["Q"])
    out = {}
    for key in acc:
        gs = np.array(sorted(acc[key]))
        out[key] = (gs,
                    np.array([np.mean(acc[key][g]) for g in gs]),
                    np.array([np.mean(q[key][g]) for g in gs]))
    return out


def logistic_gc(gs, r):
    """g_c from a logistic fit R(g) = 1/(1+exp((g-gc)/w)), least squares."""
    best, bestres = float("nan"), float("inf")
    lo, hi = float(gs.min()), float(gs.max())
    for gc in np.linspace(lo, hi, 400):
        for w in (0.01, 0.02, 0.04, 0.08, 0.15):
            pred = 1.0 / (1.0 + np.exp((gs - gc) / w))
            res = float(np.sum((pred - r) ** 2))
            if res < bestres:
                bestres, best = res, gc
    return best, bestres


def fit_forms(ns, gcs, k):
    ns = np.asarray(ns, float)
    gcs = np.asarray(gcs, float)
    out = {}
    out["constant"] = (float(np.mean(gcs)),
                       float(np.sum((gcs - np.mean(gcs)) ** 2)))
    for name, x in (("log", np.log(ns)),
                    ("extreme", np.sqrt(2.0 * np.log(ns / k)))):
        if len(ns) >= 2:
            A = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(A, gcs, rcond=None)
            res = float(np.sum((A @ coef - gcs) ** 2))
            out[name] = (tuple(float(c) for c in coef), res)
    return out


def collapse_quality(cur, gc_of, nu):
    """Spread between size curves after rescaling; lower is a better collapse."""
    pts = []
    for (n, M), (gs, _r, q) in cur.items():
        gc = gc_of.get((n, M))
        if gc is None or gc != gc:
            continue
        pts.append((np.asarray((gs - gc) * n ** (1.0 / nu)), np.asarray(q), n))
    if len(pts) < 2:
        return float("nan")
    lo = max(float(x.min()) for x, _y, _n in pts)
    hi = min(float(x.max()) for x, _y, _n in pts)
    if not (hi > lo):
        return float("nan")
    grid = np.linspace(lo, hi, 40)
    interp = [np.interp(grid, x, y) for x, y, _n in pts]
    return float(np.mean(np.var(np.vstack(interp), axis=0)))


def main():
    rows = load(CSVS)
    if not rows:
        sys.exit(f"no data; looked for {CSVS} in {HERE}")
    cur = curves(rows)
    k = rows[0]["k"]

    print(f"\n  FINITE-SIZE SCALING   k={k}   {len(cur)} (n, M) cells\n")
    print(f"  {'n':>7} {'M':>6} {'alpha':>7} {'g_c':>8} {'resid':>9}")
    gc_of = {}
    for (n, M) in sorted(cur):
        gs, r, _q = cur[(n, M)]
        if r.max() < 0.5 or r.min() > 0.5:
            print(f"  {n:>7} {M:>6} {M * k / n:>7.2f}   "
                  f"(no half-crossing in range)")
            continue
        gc, res = logistic_gc(gs, r)
        gc_of[(n, M)] = gc
        print(f"  {n:>7} {M:>6} {M * k / n:>7.2f} {gc:>8.3f} {res:>9.4f}")

    # Only cells at MATCHED load may be compared as a size series -- load is
    # itself a control parameter, so mixing loads would confound the fit.
    by_alpha = defaultdict(list)
    for (n, M), gc in gc_of.items():
        by_alpha[round(M * k / n, 3)].append((n, gc))

    for alpha, series in sorted(by_alpha.items()):
        if len(series) < 2:
            continue
        series.sort()
        ns = [n for n, _ in series]
        gcs = [g for _, g in series]
        print(f"\n  --- size series at alpha={alpha} : n={ns} ---")
        fits = fit_forms(ns, gcs, k)
        for name in ("constant", "log", "extreme"):
            if name in fits:
                coef, res = fits[name]
                print(f"    {name:<9} resid {res:.5f}   coef {coef}")
        if len(ns) >= 3:
            best = min(((collapse_quality(
                {kk: vv for kk, vv in cur.items()
                 if round(kk[1] * k / kk[0], 3) == alpha}, gc_of, nu), nu)
                for nu in np.linspace(0.5, 6.0, 45)),
                key=lambda t: (t[0] if t[0] == t[0] else 1e9))
            print(f"    best collapse: nu={best[1]:.2f}  spread={best[0]:.5f}")
        else:
            print("    (collapse needs >= 3 sizes at this load)")

    # The confound cells: same M at different n, and same n at different M.
    print("\n  --- separating n from M ---")
    for (n, M), gc in sorted(gc_of.items()):
        print(f"    n={n:<6} M={M:<4} alpha={M * k / n:<5.2f} g_c={gc:.3f}")
    print("\n  If g_c tracks n at fixed M, the competition is against the "
          "FRESH POOL.\n  If it tracks M at fixed n, it is against OTHER "
          "STORED assemblies.")


if __name__ == "__main__":
    main()
