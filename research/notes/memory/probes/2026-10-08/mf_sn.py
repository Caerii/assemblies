"""Reduced-model saddle-node load per cell: the L at which max_o F(o) - o crosses 0."""
import math, sys
import numpy as np
from mf2 import F

CELLS = [(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 120, 0.5),
         (8000, 120, 0.5), (8000, 120, 0.25), (16000, 120, 0.5), (4000, 30, 0.5),
         (6000, 90, 0.4), (12000, 80, 0.5)]
OS = [0.5, 0.6, 0.7, 0.8, 0.9]

def margin(n, k, p, L):
    beta = math.sqrt((1 - p) * math.log(n) / (p * k))
    rng = np.random.default_rng(7)
    return max(F(o, n, k, p, beta, L, trials=12, rng=rng)[0] - o for o in OS)

for n, k, p in CELLS:
    unit = n * n * p / (k * math.log(n))
    lo, hi = 0.02 * unit, 0.6 * unit
    if margin(n, k, p, hi) > 0:
        print(f"({n}, {k}, {p}) no crossing below rho 0.6", flush=True); continue
    for _ in range(9):
        mid = math.sqrt(lo * hi)
        if margin(n, k, p, mid) > 0:
            lo = mid
        else:
            hi = mid
    L = math.sqrt(lo * hi)
    print(f"({n}, {k}, {p}) kp={k * p:.0f} 3ln n={3 * math.log(n):.1f}: model saddle-node L={L:.0f} rho={L / unit:.3f}", flush=True)
