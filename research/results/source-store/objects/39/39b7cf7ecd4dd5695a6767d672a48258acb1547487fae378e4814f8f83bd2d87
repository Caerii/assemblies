"""Exploratory (not registered): can the spurious attractor of a collapsed brain be controlled?
Random-successor reuse at 40 uses per word, (10000, 75, 0.48), tau = 67, rho = 0.05 (as
probe_collapse.py; Amendment 45's cell, judged). Replay normally runs frozen and MASKED -- no
adaptation acts on the read-out. Three replays of the ODD-indexed sequences, word-level scored:
  base      masked replay (as every registration)
  lesion    the 2k neurons the brain's failed EVEN-indexed replays converge on (8 steps after the
            cue) are suppressed during replay -- causal test, cross-validated across halves
  habit     a session adaptation: after every replay step each winner is charged c x its raw
            drive, the charge decaying by exp(-1/T) per step and kept ACROSS sequences; neurons
            the attractor reuses in replay after replay accumulate it, a correct state's do not.
Seeds 980-999."""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_reuse_grammar as rg, memory_threshold_law as tl, memory_load_law as ml
from research.experiments import memory_load_drift as md, memory_pattern_efficiency as pe
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

seeds = list(range(980, 1000))
n, k, p, tau, U = 10000, 75, 0.48, 67, 40
M = max(1, round(0.05 * ml.unit(n, k, p) / rg.LENGTH)); L = M * rg.LENGTH; V = round(L / U)
B = len(seeds)
mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX, norm_init=True,
                     rounds=1, strength=ml.STRENGTH, max_items=4, device="cuda", bias_decay=math.exp(-1 / tau))
words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                             for i, sd in enumerate(seeds)] for e in range(rg.LENGTH)])[0] for q in range(M)]
allst = torch.cat(seqs, 0).long()
wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device="cuda")
ar = torch.arange(B, device="cuda")
area = mem.area
saved_bias = area.bias.clone()


def replay(q, bias, charge=0.0, decay=1.0):
    """One sequence's word-level success [B] and its read-out 8 steps after the cue; `bias` is the
    control bias (updated in place when charge > 0)."""
    st = seqs[q]
    area.bias = bias
    area.winners = torch.stack([md.cue(st[0, i], sd, L * 100_000 + q, k, "cuda") for i, sd in enumerate(seeds)]).long()
    alive = torch.ones(B, dtype=torch.bool, device="cuda")
    end = None
    for j in range(1, rg.LENGTH):
        x, drive = area.project(1, [mem.fiber], freeze=True, mask_bias=False, return_drive=True)
        if charge:
            raw = drive + bias
            bias.mul_(decay)
            bias.scatter_add_(1, x, torch.gather(raw, 1, x) * charge)
        hot = torch.zeros(B, n, device="cuda"); hot.scatter_(1, x.long(), 1.0)
        ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
        alive &= wordof[ov.argmax(0), ar] == wordof[q * rg.LENGTH + j]
        if j == 8:
            end = x.clone()
    return alive, end


even, odd = list(range(0, M, 2)), list(range(1, M, 2))
# baseline on both halves; attractor neurons from the even half's failures
zero = torch.zeros(B, n, device="cuda")
counts = torch.zeros(B, n, device="cuda")
for q in even:
    ok, end = replay(q, zero.clone())
    counts.scatter_add_(1, end, (~ok).float().view(-1, 1).expand(-1, k).contiguous())
res = {}
base = torch.zeros(B, device="cuda")
for q in odd:
    base += replay(q, zero.clone())[0].float()
res["base"] = base / len(odd)
lesion = zero.clone()
top = torch.topk(counts, 2 * k, dim=1).indices
has = counts.sum(1) > 0
lesion.scatter_(1, top, 1e4 * has.float().view(-1, 1).expand(-1, 2 * k).contiguous())
les = torch.zeros(B, device="cuda")
for q in odd:
    les += replay(q, lesion.clone())[0].float()
res["lesion"] = les / len(odd)
for c, T in ((0.1, 50), (0.3, 50), (0.3, 500), (1.0, 500)):
    b = zero.clone(); acc = torch.zeros(B, device="cuda")
    for q in odd:
        acc += replay(q, b, charge=c, decay=math.exp(-1 / T))[0].float()
    res[f"habit c={c} T={T}"] = acc / len(odd)
area.bias = saved_bias
collapsed = res["base"] < 0.2
print(f"brains: {int(collapsed.sum())} collapsed (base < 0.2), {int((~collapsed).sum())} not; odd half, {len(odd)} sequences each")
for name, v in res.items():
    print(f"{name:18s} collapsed brains {float(v[collapsed].mean()) if collapsed.any() else float('nan'):.3f}   "
          f"other brains {float(v[~collapsed].mean()):.3f}   per brain {[round(float(x), 2) for x in v]}", flush=True)
