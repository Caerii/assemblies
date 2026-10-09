"""Exploratory (not registered): WHERE does the logit margin go? A fan-effect hypothesis for
Amendment 48's slack. Random-successor reuse, (10000, 75, 0.48), tau = 67, rho = 0.05 (judged
cell; seeds 980-999), U = 10, 30, 40, 50; a second cell by argv as probe_basin.py.

Hypothesis. A word used U times is stored as U tokens sharing a type trace (overlap ~ ICC).
Replaying one token, the trace also drives the successors of the word's OTHER occurrences --
its fan -- so the correct successor's strongest rivals are fan successors, and the margin
falls as the trace and the fan grow. Per replay step of the even half, masked:
  fan share   is the best rival word a successor of another occurrence of the current word?
              (against chance: the fan's share of the vocabulary)
  trace       overlap of the state being projected with the current token (self) and with the
              word's other tokens (max over them)
  margin      correct word's logit minus the best rival's (all steps, won or lost), at step 1
              (straight from the cue) and later
A step-race model then asks whether the cliff is a per-step race compounded over the
sequence: P(margin_j > 0) from step 1's mean and spread, raised to the 15 steps, against the
observed reliability."""
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
if len(sys.argv) > 1:   # n k p tau U...
    n, k, p, tau = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    USES = tuple(int(u) for u in sys.argv[5:])
B = len(seeds)
dev = "cuda"
ar = torch.arange(B, device=dev)
print(f"cell ({n}, {k}, {p}), tau {tau}; beta {tl.theta(n, k, p):.4f}; kp {k * p:.1f}", flush=True)

for U in USES:
    M = max(1, round(0.05 * ml.unit(n, k, p) / rg.LENGTH)); L = M * rg.LENGTH; V = round(L / U)
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX,
                         norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4, device=dev,
                         bias_decay=math.exp(-1 / tau))
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(rg.LENGTH)])[0] for q in range(M)]
    allst = torch.cat(seqs, 0).long()                                   # [L, B, k]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()   # [L, B]
    fan = torch.zeros(B, V, V, dtype=torch.bool, device=dev)           # fan[b, w, w'] w' follows some w
    for i in range(B):
        w = torch.tensor(words[i], device=dev).long()
        fan[i, w[:, :-1].reshape(-1), w[:, 1:].reshape(-1)] = True
    area = mem.area
    saved = area.bias.clone()
    area.bias = torch.zeros(B, n, device=dev)
    rec = {j: [] for j in range(1, rg.LENGTH)}
    rel = torch.zeros(B, device=dev)
    for q in range(0, M, 2):
        area.bias.zero_()
        area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev)
                                    for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        for j in range(1, rg.LENGTH):
            prev = area.winners.clone()
            x, drive = area.project(1, [mem.fiber], freeze=True, mask_bias=False, return_drive=True)
            thr = torch.topk(drive, k, dim=1).values[:, -1].clamp_min(1e-9)
            tok = torch.gather(drive.unsqueeze(0).expand(L, B, n), 2, allst).mean(2) / thr
            wl = torch.full((B, V), -1e9, device=dev).scatter_reduce(1, wordof.T, tok.T, "amax")
            cur, want = wordof[q * rg.LENGTH + j - 1], wordof[q * rg.LENGTH + j]
            wc = wl[ar, want]
            rival_l, rival = wl.scatter(1, want.view(-1, 1), -1e9).max(1)
            in_fan = fan[ar, cur, rival].float()
            chance = (fan[ar, cur].sum(1).float() - 1).clamp_min(0) / (V - 1)
            hp = torch.zeros(B, n, device=dev); hp.scatter_(1, prev, 1.0)
            ov = torch.gather(hp.unsqueeze(0).expand(L, B, n), 2, allst).sum(2) / k        # [L, B]
            self_ov = ov[q * rg.LENGTH + j - 1]
            same = wordof == cur.view(1, -1)
            same[q * rg.LENGTH + j - 1] = False
            trace = torch.where(same, ov, torch.zeros_like(ov)).max(0).values
            hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
            ok = wordof[torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == want
            rec[j].append(torch.stack([wc - rival_l, in_fan, chance, self_ov, trace, alive.float(), wc, rival_l], 1))
            alive &= ok
        rel += alive.float()
    rel /= len(range(0, M, 2))
    area.bias = saved
    R = {j: torch.stack(v) for j, v in rec.items()}                    # [S, B, 8]
    r1 = R[1]
    m1, s1 = r1[..., 0].mean(0), r1[..., 0].std(0)
    # the race: P(margin > 0) at each step among sequences still alive, compounded
    live = [R[j][..., 5] > 0.5 for j in range(1, rg.LENGTH)]
    pj = [((R[j][..., 0] > 0) & lv).float().sum(0) / lv.float().sum(0).clamp_min(1) for j, lv in zip(range(1, rg.LENGTH), live)]
    comp = torch.stack(pj).prod(0)
    p1 = torch.distributions.Normal(0, 1).cdf(m1 / s1.clamp_min(1e-6))
    allsteps = torch.cat([R[j] for j in R], 0)
    print(f"\nU = {U} (V {V}): reliability {float(rel.mean()):.3f}", flush=True)
    print(f"  fan share of best rival {float(allsteps[..., 1].mean()):.3f} (chance {float(allsteps[..., 2].mean()):.3f}); "
          f"self overlap {float(allsteps[..., 3].mean()):.3f}, trace (max other token of the word) "
          f"{float(allsteps[..., 4].mean()):.3f}", flush=True)
    print(f"  step 1: margin {float(m1.mean()):.3f} +- {float(s1.mean()):.3f} (correct {float(r1[..., 6].mean()):.3f}, "
          f"rival {float(r1[..., 7].mean()):.3f}); trace {float(r1[..., 4].mean()):.3f}; rival in fan "
          f"{float(r1[..., 1].mean()):.3f}", flush=True)
    for j in (2, 4, 8, 15):
        t = R[j]; a = t[..., 5] > 0.5
        if a.any():
            print(f"  step {j:2d} (alive): margin {float(t[..., 0][a].mean()):.3f}, self {float(t[..., 3][a].mean()):.3f}, "
                  f"trace {float(t[..., 4][a].mean()):.3f}, rival in fan {float(t[..., 1][a].mean()):.3f}", flush=True)
    print("  per brain: rel / prod of per-step win rates / Phi(m1/s1)^15 / m1 / s1 / trace1")
    for i in range(B):
        print(f"    {seeds[i]}: {float(rel[i]):.2f} / {float(comp[i]):.2f} / {float(p1[i]) ** 15:.2f} / "
              f"{float(m1[i]):.3f} / {float(s1[i]):.3f} / {float(r1[i, :, 4].mean()):.3f}", flush=True)
    del mem, seqs, allst, fan
    torch.cuda.empty_cache()
