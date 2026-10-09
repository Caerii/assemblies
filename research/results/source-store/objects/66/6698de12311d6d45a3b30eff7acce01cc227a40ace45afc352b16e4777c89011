"""Exploratory (not registered): the dose-response of transition repetition, after Amendment 44.
the two edges: i.i.d. words at U = 10, 20, 40, 60, 80 (recurrence), and U = 10 at b = 4, 3 (repetition, R ~ 2.5, 3.3);
(b = 5 and 2 at U = 10 are in probe_dose2 and probe_product). Word-level reliability as in Amendment 44. Cell (12000, 80, 0.45) (Amendment 44's, judged),
tau = 75, rho = 0.05. Seeds 980-989."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
from research.experiments import memory_reuse_grammar as rg, memory_threshold_law as tl, memory_load_drift as md

seeds = list(range(980, 990))
n, k, p, tau = 12000, 80, 0.45, 75
for u, b in ((10, None), (20, None), (40, None), (60, None), (80, None), (10, 4), (10, 3)):
    spec = {"n": n, "k": k, "p": p, "tau": tau, "uses": u, "b": b, "rho": 0.05,
            "beta": round(tl.theta(n, k, p), 5), "batch": md.batch_size(n)}
    m = rg.measure(spec, seeds, "cuda")
    mean = lambda v: sum(x for x in v if x is not None) / max(1, sum(x is not None for x in v))
    print(f"U={u} b={b or 'V'} V={m['V']} R={mean(m['repeats']):.2f}: word {mean(m['word']):.3f} "
          f"(brains {min(m['word']):.2f}-{max(m['word']):.2f}), same-word {mean(m['same']):.3f}", flush=True)
