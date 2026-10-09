"""Exploratory (not registered): is write-time capture a CASCADE? probe_store.py found the recurrence
collapse written into the store: collapsed brains hold tokens of different words written onto one
assembly (up to 31 sharing >= 0.5 with one token), healthy brains none. Writing is itself a
recurrent step (the element's stimulus fires alongside the learned recurrence), so a captured
state could pull later writes into itself. Here, in WRITE ORDER, each token is CAPTURED if it
shares >= 0.5 of its neurons with an EARLIER token of another word. Per brain: when the first
capture happens, the captured share by write decile (does it accelerate after the first?), the
largest cluster's growth, and whether sequences written BEFORE the first capture still replay.
Random-successor reuse, (10000, 75, 0.48), tau = 67, rho = 0.05 (judged cell), U = 40 and 50,
seeds 980-999."""
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
n, k, p, tau = 10000, 75, 0.48, 67
USES = (40, 50)
B = len(seeds)
dev = "cuda"
LEN = rg.LENGTH
ar = torch.arange(B, device=dev)
for U in USES:
    M = max(1, round(0.05 * ml.unit(n, k, p) / LEN)); L = M * LEN; V = round(L / U)
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX,
                         norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4, device=dev,
                         bias_decay=math.exp(-1 / tau))
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(LEN)])[0] for q in range(M)]
    allst = torch.cat(seqs, 0).long()
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    area = mem.area
    saved = area.bias.clone()
    okseq = torch.zeros(M, B, device=dev)
    for q in range(M):
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev) for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        for j in range(1, LEN):
            x = area.project(1, [mem.fiber], freeze=True, mask_bias=False)
            hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
            alive &= wordof[torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == wordof[q * LEN + j]
        okseq[q] = alive.float()
    area.bias = saved
    print(f"\nU = {U} (M {M}, L {L})", flush=True)
    print("  brain  rel | first capture (token, % of writing) | captured share by write decile | largest cluster "
          "| replay of sequences written before / after the first capture", flush=True)
    for i in range(B):
        H = torch.zeros(L, n, device=dev, dtype=torch.float16)
        H.scatter_(1, allst[:, i], 1.0)
        O = (H @ H.T).float() / k
        w = wordof[:, i]
        diffw = w.view(-1, 1) != w.view(1, -1)
        earlier = torch.ones(L, L, device=dev).tril(-1) > 0
        cap = ((O >= 0.5) & diffw & earlier).any(1).cpu().numpy()
        first = int(np.argmax(cap)) if cap.any() else None
        dec = [cap[int(L * d / 10):int(L * (d + 1) / 10)].mean() for d in range(10)]
        clus = int((O >= 0.5).sum(1).max()) - 1
        rel = float(okseq[:, i].mean())
        if first is not None:
            qf = first // LEN
            before = float(okseq[:qf, i].mean()) if qf > 0 else float("nan")
            after = float(okseq[qf:, i].mean())
            ftxt = f"{first:5d} ({100 * first / L:4.1f}%)"
        else:
            before = after = float("nan"); ftxt = "  none       "
        print(f"  {seeds[i]}  {rel:.2f} | {ftxt} | " + " ".join(f"{d:.3f}" for d in dec)
              + f" | {clus:3d} | {before:.2f} / {after:.2f}", flush=True)
        del H, O, diffw, earlier
    del mem, seqs, allst
    torch.cuda.empty_cache()
