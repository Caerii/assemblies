"""Exploratory (not registered): a sleep threshold one reference brain cannot move. Amendment 52's
L3 failed at (12500, 90, 0.4): one reference brain's dreams reached contrast 1.531 (pooled mean
1.380), the threshold (1.02 x the pooled MAXIMUM) rose to 1.561, and the sleep after the
comparator removed 0.16% of counts against 1.35% at the other cell. Candidate robust rules, each
x 1.02: the pooled 0.999 and 0.99 quantiles, and the median over reference brains of each brain's
maximum. For each, on the SAME brains as Amendment 52 (subjects 842-861, references 862-881, both
cells -- this re-reads a judged run's brains and is disclosed as such): sleep 300 episodes on a
healthy U = 10 store (must remove nothing and keep 1.000) and on the comparator U = 100 store
(the repair L3 asked for)."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_lifecycle as lc
from research.experiments import memory_reuse_grammar as rg
from research.experiments import memory_sleep as sl
from research.experiments import memory_threshold_law as tl

dev = "cuda"
seeds = list(lc.SEEDS)


def rules(cal_t, B):
    per_brain = cal_t.view(-1, B).max(0).values
    return {"max": float(cal_t.max()), "q999": float(torch.quantile(cal_t, 0.999)),
            "q99": float(torch.quantile(cal_t, 0.99)), "median-brain-max": float(per_brain.median())}, per_brain


for n, k, p, tau in lc.CELLS:
    spec = {"n": n, "k": k, "p": p, "tau": tau, "rho": rg.RHO, "beta": round(tl.theta(n, k, p), 5),
            "uses": [100], "episodes": 300}
    ref = sl.build_store(spec, 10, list(lc.REFERENCE_SEEDS), dev)
    cal = []
    g0 = torch.Generator(device=dev).manual_seed(sl.CAL_SEED)
    for _ in range(sl.CALIBRATION):
        sl.dream(ref["mem"], g0, None, dev, cal)
    # is the outlier a PRE-SYMPTOMATIC capture? per reference brain: cross-word token pairs
    # overlapping >= 0.3 (Amendment 49's capture criterion) in its healthy U = 10 store
    caps = []
    for i in range(len(lc.REFERENCE_SEEDS)):
        H = torch.zeros(ref["L"], n, device=dev)
        H.scatter_(1, ref["allst"][:, i], 1.0)
        O = (H @ H.T) / k
        w = ref["wordof"][:, i]
        cross = (O >= 0.3) & (w.view(-1, 1) != w.view(1, -1))
        caps.append(int(cross.sum()) // 2)
        del H, O
    bm = torch.cat(cal).float().view(-1, len(lc.REFERENCE_SEEDS)).max(0).values.tolist()
    print("  reference brains (max contrast, captured cross-word pairs): "
          + " ".join(f"{m:.3f}/{c}" for m, c in sorted(zip(bm, caps))), flush=True)
    del ref
    torch.cuda.empty_cache()
    cal_t = torch.cat(cal).float()
    base, per_brain = rules(cal_t, len(lc.REFERENCE_SEEDS))
    print(f"\n({n}, {k}, {p}) reference contrasts: mean {float(cal_t.mean()):.3f}; "
          + "  ".join(f"{r} {v:.3f}" for r, v in base.items()), flush=True)
    print("  per-brain maxima (sorted): " + " ".join(f"{v:.3f}" for v in sorted(per_brain.tolist())), flush=True)
    thresholds = {r: v * sl.MARGIN for r, v in base.items()}
    for label, build in (("healthy U=10", lambda: sl.build_store(spec, 10, seeds, dev)),
                         ("comparator U=100", lambda: lc.comparator_store(spec, 100, seeds, dev)[0])):
        st = build()
        backup = st["mem"].fiber.C.cpu()
        r0 = np.array(sl.reliability(st, dev))
        print(f"  {label}: before sleep {r0.mean():.3f}", flush=True)
        for r, thr in thresholds.items():
            st["mem"].fiber.C.copy_(backup.to(dev))
            removed = lc.slept(st, 300, thr, dev)
            rel = np.array(sl.reliability(st, dev))
            print(f"    {r:17s} threshold {thr:.3f}: {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}, "
                  f"lowest {rel.min():.2f}); removed {removed:.4f}", flush=True)
        del st, backup
        torch.cuda.empty_cache()
