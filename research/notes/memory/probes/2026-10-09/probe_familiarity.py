"""Exploratory (not registered): which LOCAL familiarity signal detects a capture? On the writes of
a standard store (no intervention), each non-initial write is previewed and scored by:
  oracle      max over stored tokens of overlap >= 0.3 -- the label (Amendment 49's check)
  comparator  |write ∩ recall| / k (Amendment 50's signal: recall = recurrence alone)
  fam2        sum over stored tokens of overlap^2 -- what a pairwise (Hebbian or anti-Hebbian)
              recognition network computes, h' W h with W = sum_S h_S h_S' (Bogacz & Brown)
  fam4, fam8  sum of overlap^4, ^8 -- higher-order recognition (dense associative memories,
              Krotov & Hopfield 2016; plausibly dendritic nonlinearity), approaching the max
Each score's AUC against the oracle label, and its hit rate at a 2% false-alarm rate.
Hypothesis: fam2 is swamped by the many weak same-word overlaps of a reused word (the fan); the
higher orders and the comparator are not. Random-successor reuse, (10000, 75, 0.48), tau = 67
(judged cell), seeds 980-999, U = 30 and 50."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_reuse_grammar as rg, memory_load_law as ml
from research.experiments import memory_write_separation as ws
from research.experiments.seq_capacity_scaling import to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._hashed import StimulusFiber

seeds = list(range(980, 1000))
n, k, p, tau = 10000, 75, 0.48, 67
dev = "cuda"
LEN = rg.LENGTH
B = len(seeds)


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def hit_at(score, label, fa=0.02):
    thr = np.quantile(score[~label], 1 - fa)
    return float((score[label] > thr).mean()) if label.any() else float("nan")


for U in (30, 50):
    M = max(1, round(0.05 * ml.unit(n, k, p) / LEN)); L = M * LEN; V = round(L / U)
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    mem = ws.build(n, k, p, tau, seeds, dev)
    area, fib = mem.area, mem.fiber
    stored, rows = [], []
    for q in range(M):
        area.inhibit()
        for e in range(LEN):
            sds = [to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A")) for i, sd in enumerate(seeds)]
            stim = StimulusFiber(sds, k, n, p, beta=mem.beta, w_max=mem.w_max, norm_init=mem.norm_init,
                                 max_rounds=1, device=dev)
            if e > 0:
                keep_b, keep_w = area.bias.clone(), area.winners.clone()
                area.bias.mul_(area.bias_decay)
                P = area.project(1, [fib, stim], freeze=True, mask_bias=False)
                area.winners = keep_w.clone()
                R = area.project(1, [fib], freeze=True, mask_bias=False)
                area.bias, area.winners = keep_b, keep_w
                hP = torch.zeros(B, n, device=dev); hP.scatter_(1, P.long(), 1.0)
                comp = torch.gather(hP, 1, R.long()).sum(1) / k
                T = torch.stack(stored)
                ov = torch.gather(hP.unsqueeze(0).expand(T.shape[0], B, n), 2, T).sum(2) / k
                rows.append(torch.stack([ov.max(0).values, comp, (ov ** 2).sum(0), (ov ** 4).sum(0), (ov ** 8).sum(0)], 1).cpu())
            win = area.project(1, [fib, stim], defer_overflow=True)
            stored.append(win.clone())
        mem.items += 1
    X = torch.cat(rows, 0).numpy()
    label = X[:, 0] >= 0.3
    print(f"\nU = {U}: {len(X)} judged writes, oracle-flagged {label.mean():.4f}", flush=True)
    for name, col in (("comparator", 1), ("fam2", 2), ("fam4", 3), ("fam8", 4)):
        s = X[:, col]
        print(f"  {name:10s} AUC {auc(s, label):.3f}; hit at 2% false alarms {hit_at(s, label):.2f}; "
              f"mean flagged {s[label].mean():.4g} vs unflagged {s[~label].mean():.4g}", flush=True)
    del mem, stored
    torch.cuda.empty_cache()
