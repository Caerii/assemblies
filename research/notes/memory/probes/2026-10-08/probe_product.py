"""Exploratory (not registered): the dose-response of transition repetition, after Amendment 44.
U x R fixed, U traded against R: U R = 50 at (U, b) = (10, 2), (20, 8), (40, 32); U R = 36 at (12, 4), (24, 16), (48, 64);
R alone predicts (10,2) worst, U alone (40,32) worst, the product all alike. Word-level reliability as in Amendment 44. Cell (12000, 80, 0.45) (Amendment 44's, judged),
tau = 75, rho = 0.05. Seeds 980-989."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
from research.experiments import memory_reuse_grammar as rg, memory_threshold_law as tl, memory_load_drift as md

seeds = list(range(980, 990))
n, k, p, tau = 12000, 80, 0.45, 75
for u, b in ((10, 2), (20, 8), (40, 32), (12, 4), (24, 16), (48, 64)):
    spec = {"n": n, "k": k, "p": p, "tau": tau, "uses": u, "b": b, "rho": 0.05,
            "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n)}
    m = rg.measure(spec, seeds, "cuda")
    mean = lambda v: sum(x for x in v if x is not None) / max(1, sum(x is not None for x in v))
    print(f"U={u} b={b or 'V'} V={m['V']} R={mean(m['repeats']):.2f}: word {mean(m['word']):.3f} "
          f"(brains {min(m['word']):.2f}-{max(m['word']):.2f}), same-word {mean(m['same']):.3f}", flush=True)
