"""Exploratory (not registered): small n/k, where ln n and ln(n/k) forms part. Seeds 980-989; tau = n/k / 2."""
import os
import math, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
from research.experiments import memory_load_drift as md, memory_load_law as ml, memory_threshold_law as tl, memory_load_tau as mt
seeds = list(range(980, 990))
for n, k, p in [(int(a), int(b), float(c)) for a, b, c in (s.split(",") for s in sys.argv[1:])]:
    spec = {"n": n, "k": k, "p": p, "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n), "tau": max(1, mt.rule(n, k))}
    zeros = 0
    for j in range(2, 30):
        r = 0.06 * 2 ** (j / 8)
        L = int(round(r * ml.unit(n, k, p))); t0 = time.time()
        st = md.reliability(spec, L, seeds, "cuda")
        full = sum(s >= L - 1 for s in st) / len(st)
        print(f"({n}, {k}, {p}) tau={spec['tau']} L={L} rho_lnn={r:.3f} rho_lnnk={r*math.log(n/k)/math.log(n):.4f}: full {full:.1f} min {min(st):.0f} [{time.time()-t0:.0f}s]", flush=True)
        zeros = zeros + 1 if full == 0 else 0
        if zeros >= 2: break
