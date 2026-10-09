"""Exploratory (not registered): many short sequences vs one, survey cell, seeds 980-989."""
import os
import sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
from research.experiments import memory_load_many as mm, memory_load_law as ml
seeds = list(range(980, 990))
n, k, p = 4000, 60, 0.5
for spec in mm.plan([(n, k, p)]):
    for r in (0.101, 0.131, 0.156):
        L = int(round(r * ml.unit(n, k, p)))
        if spec["arm"] != "single": L = max(1, round(L / spec["arm"])) * spec["arm"]
        t0 = time.time(); fr = mm.whole(spec, L, seeds, "cuda")
        print(f"tau={spec['tau']} arm={spec['arm']} L={L} rho={L/ml.unit(n,k,p):.3f}: whole {sum(fr)/len(fr):.3f} "
              f"per-brain {[round(f,2) for f in fr]} [{time.time()-t0:.0f}s]", flush=True)
