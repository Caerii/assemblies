"""Exploratory: rho_50 against tau/(n/k) at small, cheap cells. Seeds 980-999 (20 brains)."""
import os
import math, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
from research.experiments import memory_load_drift as md, memory_load_law as ml, memory_threshold_law as tl
seeds = list(range(980, 1000))
def rho50(n, k, p, tau):
    spec = {"n": n, "k": k, "p": p, "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n), "tau": tau}
    rows, zeros = {}, 0
    for j in range(0, 30):
        r = 0.05 * 2 ** (j / 12)
        L = int(round(r * ml.unit(n, k, p)))
        st = md.reliability(spec, L, seeds, "cuda")
        full = sum(s >= L - 1 for s in st) / len(st)
        rows[str(L)] = {"rho": L / ml.unit(n, k, p), "full": full}
        zeros = zeros + 1 if full == 0 else 0
        if zeros >= 2: break
    return ml.crossing(rows, 0.5), ml.crossing(rows, 0.9)
for n, k, p in ((4000, 400, 0.1), (8000, 400, 0.1), (4000, 200, 0.15), (2000, 60, 0.5)):
    for f in (0.25, 0.5, 1, 2, 4):
        tau = max(1, round(f * n / k))
        r5, r9 = rho50(n, k, p, tau)
        print(f"({n}, {k}, {p}) n/k={n/k:.0f} tau={tau} tau/(n/k)={f}: rho_50 {r5 and round(r5,4)} rho_90 {r9 and round(r9,4)}", flush=True)
