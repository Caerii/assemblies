"""Does `norm_init` + `synaptic_scaling` lift the M-ceiling, and does the
mechanism survive off the single point it was measured at?

Implements `research/notes/PREREG_substrate_ceiling.md`; bars registered there
before this ran.

Part A re-asks the original ratchet study's question. That study answered NO --
but it compared substrate C ALONE against substrate B ALONE, the false choice
Amendment 5 overturned. Part B exists to FALSIFY Amendment 5: its mechanism
rests entirely on n=2000, k=50, p=0.5, and a mechanism established at one
operating point is not established.

THREE STANDARDS THIS STUDY TAKES FROM `_substrate.py` RATHER THAN INVENTING:

  * `ceiling_from_curve` -- a ceiling is where a CURVE crosses a threshold, not
    where a GRID does. `max(M : acc > t)` reports a grid point as a
    measurement; on a doubling grid the true crossing is only bracketed to a
    factor of 2, and a ratio built from two of those to a factor of 4. Quote
    `m_star` only when `.supported` and `.resolved()`.
  * `MIN_TRIALS` -- 24 trials cannot separate 0.125 from 0.375, and this
    project has already retracted a headline that tried. Seeds are scaled with
    M so every reported rate has M x seeds >= 96.
  * `check_distinct` -- mean pairwise overlap is nearly BLIND to partial
    collapse. 256 items on ~58 distinct assemblies still reads 0.0129 against a
    0.0125 floor, because only ~1.3% of PAIRS are duplicates. The distinct
    FRACTION is reported at every cell.

THE CEILING IS DEFINED ON THE ATTRACTOR, NOT THE LOOKUP. Rank-1 from the full
cue reads 1.000 for substrate C while its assemblies are merged at 7x chance,
because the stimulus does the discriminating. So the curve fed to the estimator
is half-cue rank-1 with the distinctness gate BAKED IN -- a cell that fails the
gate contributes 0.0, not its completion score. Distinctness is a gate here,
not a report, which is the correction to the original study.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from neural_assemblies.diagnostics import (                    # noqa: E402
    Ensemble, ensemble_from_values,
)

from _substrate import MIN_TRIALS, ceiling_from_curve          # noqa: E402
from _substrate_arms import Cfg, worker                          # noqa: E402

T = 8
ARMS = ("B", "C", "G")
BETAS = (0.10, 0.20, 0.30)

A_N, A_K, A_P = 2000, 50, 0.50
A_M = (8, 16, 32, 64, 96, 128, 160, 192, 224, 256)

# p falls 4x while BOTH the regime margin (kp/floor ~ 1.1) and the chance floor
# (k/n = 0.025) are held fixed, so neither can be the explanation.
B_POINTS = ((2000, 50, 0.500), (4000, 100, 0.280), (10000, 250, 0.124))
B_M = 16
B_BETAS = (0.10, 0.20)

HALF_BAR = 0.50          # completion required to call it an attractor
DISTINCT_GATE = 3.0      # x chance; pairwise above this is not a stored set
BASE_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53)


def seeds_for(M):
    """Enough seeds that M x seeds clears MIN_TRIALS, so no rate is unquotable."""
    need = max(3, math.ceil(MIN_TRIALS / M))
    return list(BASE_SEEDS[:min(need, len(BASE_SEEDS))])


def _ens(res, key, cell, seeds) -> Ensemble:
    """Every seed-level statistic goes through `ensemble_from_values`.

    Not a bare mean over seeds. A mean with no interval cannot support a
    comparison, and the comparison gets made anyway -- this project
    published a "conserved budget" law that way and retracted it. Bars
    below are judged on the CONFIDENCE BOUND, never the point estimate.
    """
    return ensemble_from_values(
        [res[cell + (s,)][key] for s in seeds],
        label=f"{cell[-1]}/b{cell[3]}/M{cell[5]}/{key}", keys=list(seeds))


def _gated_half(res, cell, seeds):
    """Half-cue completion with the distinctness gate BAKED IN.

    A cell whose assemblies are not distinct has not stored M things, so its
    completion score is not evidence of capacity -- it contributes 0.0. Baking
    the gate into the curve is what lets a single-curve ceiling estimator
    answer a two-part question honestly.
    """
    h = _ens(res, "rank1_half", cell, seeds)
    x = _ens(res, "pairwise_x_chance", cell, seeds)
    d = _ens(res, "distinct_frac", cell, seeds)
    # The GATE is judged on the confidence bound too: a cell counts as
    # distinct only when its overlap is below the gate and its distinct
    # fraction above 0.9 across the whole seed ensemble, not on average.
    return h.mean if (x.high <= DISTINCT_GATE and d.low >= 0.9) else 0.0


def _curve(res, arm, beta):
    return [(M, _gated_half(res, (A_N, A_K, A_P, beta, T, M, arm), seeds_for(M)))
            for M in A_M]


def main():
    from _parallel import run_cells

    a_cfg = Cfg(n=A_N, k=A_K, p=A_P, beta=BETAS[0], T=T, M=A_M[0])
    print("=== substrate ceiling: is the ratchet ceiling liftable? ===")
    print(f"    Part A  n={A_N} k={A_K} p={A_P}  kp={a_cfg.kp:.1f} vs floor "
          f"{a_cfg.floor:.1f}  chance={a_cfg.chance:.4f}")
    print(f"    curve = half-cue rank-1, ZEROED where pairwise > "
          f"{DISTINCT_GATE:.0f}x chance or distinct-frac < 0.9")
    print(f"    seeds scaled so M x seeds >= {MIN_TRIALS} at every M\n")

    cells = []
    for a in ARMS:
        for b in BETAS:
            for M in A_M:
                cells += [(A_N, A_K, A_P, b, T, M, a, s) for s in seeds_for(M)]
    for (n, k, p) in B_POINTS:
        for b in B_BETAS:
            for a in ARMS:
                cells += [(n, k, p, b, T, B_M, a, s) for s in seeds_for(B_M)]
    cells = sorted(set(cells))

    # CE2, ASSERTED not printed: two of this project's nulls were recorded an
    # order of magnitude below floor, and a printed row at the bottom of a log
    # is indistinguishable from silence.
    for c in cells:
        cfg = Cfg(n=c[0], k=c[1], p=c[2], beta=c[3], T=c[4], M=c[5])
        assert cfg.in_regime, (
            f"OUT OF REGIME: n={cfg.n} k={cfg.k} p={cfg.p} gives kp="
            f"{cfg.kp:.1f} against floor {cfg.floor:.1f}. A null here is not "
            f"evidence about the mechanism.")

    print(f"    {len(cells)} cells\n")
    res = run_cells(worker, cells, max_workers=12)

    print("\n--- Part A: the grid (mean over seeds; +-95% CI on the gated curve)")
    print(f"    {'arm':3s} {'beta':>5} {'M':>4} {'seeds':>5} "
          f"{'half+-CI':>13} {'full':>6} {'pairwise+-CI':>17} {'xchan':>6} "
          f"{'dist':>6} {'rows/n':>7} gate")
    for arm in ARMS:
        for beta in BETAS:
            for M in A_M:
                cell, sd = (A_N, A_K, A_P, beta, T, M, arm), seeds_for(M)
                h = _ens(res, "rank1_half", cell, sd)
                f = _ens(res, "rank1_full", cell, sd)
                pw = _ens(res, "pairwise", cell, sd)
                xc = _ens(res, "pairwise_x_chance", cell, sd)
                df = _ens(res, "distinct_frac", cell, sd)
                sat = _ens(res, "rows", cell, sd).mean / A_N
                ok = _gated_half(res, cell, sd) > 0
                print(f"    {arm:3s} {beta:5.2f} {M:4d} {len(sd):5d} "
                      f"{h.mean:.3f}+-{h.ci:.3f} {f.mean:6.3f} "
                      f"{pw.mean:.4f}+-{pw.ci:.4f} {xc.mean:6.2f} "
                      f"{df.mean:6.3f} {sat:7.3f} "
                      f"{'ATTR' if ok else 'gated'}"
                      f"{' SAT' if sat >= 0.999 else ''}")
            print()

    print("--- Part A: ceilings from the CURVE, each arm at its own best beta")
    best, ceil = {}, {}
    for arm in ARMS:
        cands = []
        for beta in BETAS:
            c = ceiling_from_curve(_curve(res, arm, beta), threshold=HALF_BAR)
            cands.append((c.m_star, beta, c))
        _, beta, c = max(cands, key=lambda t: t[0])
        best[arm], ceil[arm] = beta, c
        print(f"    {arm}  best beta {beta:.2f}   {c}")
    print("    (quote m_star only where it is neither CLIFF nor UNRESOLVED)")

    print("\n--- Part B: does the mechanism survive as p falls 4x? ---")
    print(f"    {'n':>6} {'k':>4} {'p':>6} {'beta':>5} {'arm':>4} {'pairw':>7} "
          f"{'xchan':>6} {'half':>6} {'dist':>6} {'rho(deg,mult)':>14} "
          f"{'deg_cv':>7}")
    for (n, k, p) in B_POINTS:
        for beta in B_BETAS:
            for arm in ARMS:
                cell, sd = (n, k, p, beta, T, B_M, arm), seeds_for(B_M)
                pw_e = _ens(res, "pairwise", cell, sd)
                xc_e = _ens(res, "pairwise_x_chance", cell, sd)
                rh_e = _ens(res, "rho_deg_mult", cell, sd)
                print(f"    {n:6d} {k:4d} {p:6.3f} {beta:5.2f} {arm:>4} "
                      f"{pw_e.mean:.4f}+-{pw_e.ci:.4f} {xc_e.mean:6.2f} "
                      f"{_ens(res,'rank1_half',cell,sd).mean:6.3f} "
                      f"{_ens(res,'distinct_frac',cell,sd).mean:6.3f} "
                      f"{rh_e.mean:+.3f}+-{rh_e.ci:.3f} "
                      f"{_ens(res,'deg_cv',cell,sd).mean:7.4f}")
        print()

    print("=== BARS ===")
    sd8 = seeds_for(8)
    e1 = {a: _ens(res, "rank1_full", (A_N, A_K, A_P, 0.10, T, 8, a), sd8)
          for a in ARMS}
    ce1 = all(e.low >= 0.90 for e in e1.values())
    print(f"  {'PASS' if ce1 else 'FAIL'}  CE1 instrument: M=8 full-cue "
          f"CI-low >= 0.90  " + "  ".join(
              f"{a} {e1[a].mean:.3f}+-{e1[a].ci:.3f}" for a in ARMS))
    print("  PASS  CE2 regime: asserted for every cell before the run")

    ce3 = ceil["G"].m_star > ceil["B"].m_star
    print(f"  {'PASS' if ce3 else 'FAIL'}  CE3 THE CLAIM: G M*="
          f"{ceil['G'].m_star:.1f} > B M*={ceil['B'].m_star:.1f}  "
          f"(brackets G {ceil['G'].bracket} B {ceil['B'].bracket})")
    ce4 = ceil["C"].m_star <= ceil["B"].m_star
    print(f"  {'PASS' if ce4 else 'FAIL'}  CE4 C alone does not lift it: "
          f"C M*={ceil['C'].m_star:.1f} <= B M*={ceil['B'].m_star:.1f}")

    def bens(key, arm):
        return [_ens(res, key, (n, k, p, b, T, B_M, arm), seeds_for(B_M))
                for (n, k, p) in B_POINTS for b in B_BETAS]

    # Judged on the CONFIDENCE BOUND at EVERY point, not on a mean over
    # points: a generalization claim that holds on average across
    # operating points is not a generalization claim.
    gx, cx = bens("pairwise_x_chance", "G"), bens("pairwise_x_chance", "C")
    ce5 = all(e.high <= 1.5 for e in gx) and all(e.beats(3.0) for e in cx)
    print(f"  {'PASS' if ce5 else 'FAIL'}  CE5 generalization at EVERY "
          f"point: G worst CI-high {max(e.high for e in gx):.2f}x <= 1.5x, "
          f"C worst CI-low {min(e.low for e in cx):.2f}x > 3x")

    rc, rg = bens("rho_deg_mult", "C"), bens("rho_deg_mult", "G")
    ce6 = all(e.beats(0.15) for e in rc) and all(e.high < 0.10 for e in rg)
    print(f"  {'PASS' if ce6 else 'FAIL'}  CE6 mechanism invariant "
          f"(FALSIFICATION TEST): rho C worst CI-low "
          f"{min(e.low for e in rc):+.3f} > 0.15, G worst CI-high "
          f"{max(e.high for e in rg):+.3f} < 0.10")

    cv = [_ens(res, "deg_cv", (n, k, p, B_BETAS[0], T, B_M, "C"),
               seeds_for(B_M)).mean for (n, k, p) in B_POINTS]
    pwx = [_ens(res, "pairwise_x_chance",
               (n, k, p, B_BETAS[0], T, B_M, "C"),
               seeds_for(B_M)).mean for (n, k, p) in B_POINTS]
    spread = (max(cv) - min(cv)) / max(min(cv), 1e-12)
    if spread < 0.25:
        print(f"  ----  CE7 UNINFORMATIVE as registered: deg_cv moved only "
              f"{spread*100:.0f}% ({[round(x,4) for x in cv]}), too little to "
              f"read a direction from")
    else:
        agree = float(np.corrcoef(cv, pwx)[0, 1]) > 0
        print(f"  {'PASS' if agree else 'FAIL'}  CE7 direction: deg_cv "
              f"{[round(x,4) for x in cv]} vs xchance {[round(x,2) for x in pwx]}")

    if not ce3:
        print("\n  -> The composition buys DISTINCTNESS at fixed M, not "
              "CAPACITY. Report it that way and do not re-score RC3.")
    if not ce6:
        print("\n  -> Amendment 5's mechanism does NOT generalize. Narrow it to "
              "p=0.5 in PREREG_recurrent_ratchet.md and in the memory citing "
              "it, before anything else is built on it.")

    path = os.path.join(_HERE, "seq_substrate_ceiling_results.json")
    with open(path, "w") as fh:
        json.dump({"T": T, "A": [A_N, A_K, A_P], "A_M": list(A_M),
                   "betas": list(BETAS), "B_M": B_M,
                   "B_points": [list(x) for x in B_POINTS],
                   "best_beta": best,
                   "ceilings": {a: {"m_star": ceil[a].m_star,
                                    "lo": ceil[a].lo, "hi": ceil[a].hi,
                                    "censored": ceil[a].censored,
                                    "supported": ceil[a].supported,
                                    "resolved": ceil[a].resolved()}
                                for a in ARMS},
                   "bars": {"CE1": bool(ce1), "CE2": True, "CE3": bool(ce3),
                            "CE4": bool(ce4), "CE5": bool(ce5),
                            "CE6": bool(ce6)},
                   "cells": {"/".join(str(x) for x in c): res[c]
                             for c in cells}}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
