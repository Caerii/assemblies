"""Exploratory (not registered): is the recurrence collapse written into the STORE? Random-successor
reuse, (10000, 75, 0.48), tau = 67, rho = 0.05 (judged cell; seeds 980-999), U = 10, 30, 40, 50.
probe_map2.py found that at U = 40-50 a clamped, nearly clean state (90% of the correct token)
is read out as the wrong word 6-16% of the time while still holding 0.71-0.86 of the correct
token: some token of another word must overlap the winners more. Here, with no replay at all,
every brain's stored tokens are compared pairwise (overlap = shared neurons / k):
  diff   each token's largest overlap with a token of ANOTHER word
  same   each token's largest overlap with another token of the SAME word
  dup    the share of tokens with diff >= 0.5 (written onto another word's assembly)
  hub    the largest number of tokens sharing >= 0.5 with one token (a captured state)
and set against the brain's masked replay reliability (even half, word-level)."""
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
USES = (10, 30, 40, 50)
if len(sys.argv) > 1:
    n, k, p, tau = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    USES = tuple(int(u) for u in sys.argv[5:])
B = len(seeds)
dev = "cuda"
LEN = rg.LENGTH
ar = torch.arange(B, device=dev)
rows = []
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
    # replay reliability, even half, masked
    area = mem.area
    saved = area.bias.clone()
    rel = torch.zeros(B, device=dev)
    for q in range(0, M, 2):
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev) for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        for j in range(1, LEN):
            x = area.project(1, [mem.fiber], freeze=True, mask_bias=False)
            hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
            alive &= wordof[torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == wordof[q * LEN + j]
        rel += alive.float()
    rel /= len(range(0, M, 2))
    area.bias = saved
    print(f"\nU = {U} (V {V}, L {L})", flush=True)
    for i in range(B):
        H = torch.zeros(L, n, device=dev, dtype=torch.float16)
        H.scatter_(1, allst[:, i], 1.0)
        O = (H @ H.T).float() / k
        O.fill_diagonal_(0.0)
        w = wordof[:, i]
        samew = w.view(-1, 1) == w.view(1, -1)
        diff = torch.where(samew, torch.zeros_like(O), O).max(1).values
        same = torch.where(samew, O, torch.zeros_like(O)).max(1).values
        dup = float((diff >= 0.5).float().mean())
        hub = int((O >= 0.5).sum(1).max())
        # successor overlap: tokens whose SUCCESSORS coincide although they are different words
        r = {"U": U, "seed": seeds[i], "rel": float(rel[i]), "diff": float(diff.mean()), "diff90": float(diff.quantile(0.9)),
             "same": float(same.mean()), "dup": dup, "hub": hub}
        rows.append(r)
        print(f"  brain {seeds[i]}: rel {r['rel']:.2f} | diff mean {r['diff']:.3f} p90 {r['diff90']:.3f} | same {r['same']:.3f}"
              f" | dup {dup:.3f} | hub {hub}", flush=True)
        del H, O, samew
    del mem, seqs, allst
    torch.cuda.empty_cache()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


print("\nacross brain-arms (Spearman with reliability):", flush=True)
for key in ("diff", "diff90", "same", "dup", "hub"):
    print(f"  {key}: {spearman([r[key] for r in rows], [r['rel'] for r in rows]):+.2f}; within U = 40: "
          f"{spearman([r[key] for r in rows if r['U'] == 40], [r['rel'] for r in rows if r['U'] == 40]):+.2f}", flush=True)
