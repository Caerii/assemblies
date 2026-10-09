"""Exploratory (not registered): what is a collapsed brain? Random-successor reuse at 40 uses per
word (the recurrence edge), (10000, 75, 0.48), tau = 67, rho = 0.05 (Amendment 45's cell, judged).
For every brain: word-level reliability; for its FAILED sequences, the read-out 8 steps after the
cue; the mean pairwise overlap of those end states (do failures converge on a shared state?), the
overlap of the commonest neurons with the brain's most-used neurons (hubs?), and how many distinct
end states there are (clusters at overlap >= 0.5). Seeds 980-999."""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_reuse_grammar as rg, memory_threshold_law as tl, memory_load_law as ml
from research.experiments import memory_load_drift as md, memory_pattern_efficiency as pe, memory_sequences as sq
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
allst = torch.cat(seqs, 0).long()                                       # [L, B, k]
usage = torch.zeros(B, n, device="cuda")
usage.scatter_add_(1, allst.permute(1, 0, 2).reshape(B, -1), torch.ones(B, L * k, device="cuda"))
wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device="cuda")
ar = torch.arange(B, device="cuda")
ends = [[] for _ in range(B)]
ok_count = torch.zeros(B, device="cuda")
for q, st in enumerate(seqs):
    x = torch.stack([md.cue(st[0, i], sd, L * 100_000 + q, k, "cuda") for i, sd in enumerate(seeds)])
    alive = torch.ones(B, dtype=torch.bool, device="cuda")
    for j in range(1, rg.LENGTH):
        x = mem.recall(x, rounds=1)
        hot = torch.zeros(B, n, device="cuda"); hot.scatter_(1, x.long(), 1.0)
        ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
        alive &= wordof[ov.argmax(0), ar] == wordof[q * rg.LENGTH + j]
        if j == 8:
            end = x.clone()
    ok_count += alive.float()
    for i in range(B):
        if not bool(alive[i]):
            ends[i].append(end[i].tolist())
for i in range(B):
    rel = float(ok_count[i]) / M
    E = ends[i]
    if len(E) < 3:
        print(f"brain {seeds[i]}: word {rel:.2f}; {len(E)} failures", flush=True); continue
    sets = [set(e) for e in E[:150]]
    pair = [len(a & b) / k for ai, a in enumerate(sets) for b in sets[ai + 1:]]
    cnt = {}
    for s_ in sets:
        for v in s_: cnt[v] = cnt.get(v, 0) + 1
    common = sorted(cnt, key=cnt.get, reverse=True)[:k]
    hubs = set(torch.topk(usage[i], k).indices.tolist())
    clusters = []
    for s_ in sets:
        if not any(len(s_ & c) / k >= 0.5 for c in clusters): clusters.append(s_)
    print(f"brain {seeds[i]}: word {rel:.2f}; failures {len(E)}; mean pairwise end overlap {np.mean(pair):.3f} "
          f"(chance {k / n:.3f}); distinct end states {len(clusters)}; commonest-{k} vs top-{k} used overlap "
          f"{len(set(common) & hubs) / k:.2f}; usage of commonest neurons / mean usage "
          f"{float(usage[i, common].mean() / usage[i].mean()):.2f}", flush=True)
