"""Exploratory (not registered): is A38's low cliff tied to n/k or to n? Seeds 980-989."""
import os
import math, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
from research.experiments import memory_load_drift as md, memory_load_law as ml, memory_threshold_law as tl
seeds = list(range(980, 990))
for n, k, p in [(int(a), int(b), float(c)) for a, b, c in (s.split(",") for s in sys.argv[1:])]:
    spec = {"n": n, "k": k, "p": p, "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n)}
    zeros = 0
    for r in ml.LADDER[1:]:
        L = int(round(r * ml.unit(n, k, p))); t0 = time.time()
        st = md.reliability(spec, L, seeds, "cuda")
        full = sum(s >= L - 1 for s in st) / len(st)
        print(f"({n}, {k}, {p}) n/k={n/k:.0f} kp/3lnn={k*p/3/math.log(n):.2f} L={L} rho={r:.3f}: full {full:.1f} "
              f"min {min(st):.0f} [{time.time()-t0:.0f}s]", flush=True)
        zeros = zeros + 1 if full == 0 else 0
        if zeros >= 2: break
