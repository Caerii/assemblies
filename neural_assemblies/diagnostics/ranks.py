"""Rank statistics in numpy alone (scipy.stats stays off the import path):
rankdata, Spearman and partial Spearman correlation.

Part of neural_assemblies.diagnostics."""
from __future__ import annotations


import numpy as np



# --------------------------------------------------------------------------
# Rank statistics (numpy-only; scipy.stats stays off the import path --
# the perf A2 lesson). Promoted from research/experiments/overlap_ceiling.py
# after six experiments imported them from there (#149 literate pass).
# --------------------------------------------------------------------------

def rankdata(a) -> np.ndarray:
    """Average-rank transform with ties (scipy.stats.rankdata semantics)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation; NaN when either input is constant.

    A NaN here is an OUTCOME, not noise (the undefinedness lesson):
    report the degenerate seeds explicitly, never filter them.
    """
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z) -> float:
    """Spearman(x, y) with z partialled out (least squares on ranks)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([np.ones_like(rz), rz]).T

    def resid(v):
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta

    ex, ey = resid(rx), resid(ry)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])
