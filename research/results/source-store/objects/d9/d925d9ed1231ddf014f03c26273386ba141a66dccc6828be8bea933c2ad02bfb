"""Exploratory (not registered): a sleep gate with NO reference brains -- a developmental SET POINT.
Amendments 51-54 calibrate sleep's contrast gate on 20 separate healthy reference brains, which no
brain has. Healthy dreams do not settle (contrast ~1.35-1.40 at every cell so far), so a brain's own
dreams BEFORE it learns anything might give the same baseline: each brain fixes its gate once, at
birth, at 1.02 x the largest contrast its own empty network's dreams reach (300 dreams).
Questions: (1) how do empty-network contrasts compare with a healthy store's? (2) is a per-brain
set-point gate safe on a healthy store and does it repair a reused one, against Amendment 54's
median-of-reference-brains rule? (10000, 75, 0.48), tau 67, subjects 980-999, references 960-979."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_lifecycle as lc
from research.experiments import memory_reuse_grammar as rg
from research.experiments import memory_sleep as sl
from research.experiments import memory_threshold_law as tl
from research.experiments import memory_write_separation as ws

dev = "cuda"
seeds = list(range(980, 1000))
ref_seeds = list(range(960, 980))
n, k, p, tau = 10000, 75, 0.48, 67
B = len(seeds)
spec = {"n": n, "k": k, "p": p, "tau": tau, "rho": rg.RHO, "beta": round(tl.theta(n, k, p), 5)}


def contrasts(mem, brains):
    cal = []
    g0 = torch.Generator(device=dev).manual_seed(sl.CAL_SEED)
    for _ in range(sl.CALIBRATION):
        sl.dream(mem, g0, None, dev, cal)
    return torch.cat(cal).float().view(-1, brains)        # [samples, brain]


def describe(name, c):
    m = c.max(0).values
    print(f"  {name:28s} mean {float(c.mean()):.3f}  p99 {float(torch.quantile(c.reshape(-1), 0.99)):.3f}  "
          f"brain maxima {float(m.min()):.3f}-{float(m.max()):.3f} (median {float(m.median()):.3f})", flush=True)
    return m


print("contrasts of 300 dreams", flush=True)
empty = ws.build(n, k, p, tau, seeds, dev)
setpoint = describe("empty networks (subjects)", contrasts(empty, B)) * sl.MARGIN
del empty
ref = sl.build_store(spec, 10, ref_seeds, dev)
ref_max = describe("healthy U=10 (references)", contrasts(ref["mem"], B))
median_thr = float(ref_max.median()) * sl.MARGIN
del ref
h = sl.build_store(spec, 10, seeds, dev)
own = describe("healthy U=10 (subjects)", contrasts(h["mem"], B))
print(f"  subjects' healthy-store max over their own set point: "
      f"{float((own * sl.MARGIN / setpoint).min()):.3f}-{float((own * sl.MARGIN / setpoint).max()):.3f}", flush=True)
print(f"thresholds: reference median {median_thr:.3f}; set points {float(setpoint.min()):.3f}-{float(setpoint.max()):.3f}",
      flush=True)
torch.cuda.empty_cache()

gates = {"reference median": median_thr, "own set point": setpoint.to(dev)}
for label, build in (("healthy U=10", lambda: h),
                     ("standard U=50", lambda: sl.build_store({**spec}, 50, seeds, dev)),
                     ("comparator U=100", lambda: lc.comparator_store({**spec}, 100, seeds, dev)[0])):
    st = build()
    backup = st["mem"].fiber.C.cpu()
    r0 = np.array(sl.reliability(st, dev))
    print(f"\n{label}: before sleep {r0.mean():.3f}", flush=True)
    for g, thr in gates.items():
        st["mem"].fiber.C.copy_(backup.to(dev))
        removed = lc.slept(st, 300, thr, dev)
        rel = np.array(sl.reliability(st, dev))
        print(f"  {g:17s}: {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}, lowest {rel.min():.2f}); "
              f"removed {removed:.4f}", flush=True)
    del st, backup
    h = None
    torch.cuda.empty_cache()
