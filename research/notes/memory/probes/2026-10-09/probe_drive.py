"""Exploratory (not registered): an exact DRIVE DECOMPOSITION of the logit margin. Random-successor
reuse, (10000, 75, 0.48), tau = 67, rho = 0.05 (judged cell; seeds 980-999), U = 10, 40, 50; a
second cell by argv as probe_basin.py. Replay masked, the first 30 even-indexed sequences.

Drive is linear in the active presynaptic set, so each replay step's drive splits exactly by
SOURCE -- the state's neurons ON the current token (the correct track) and OFF it -- and by
WEIGHT -- the hashed connectome's baseline (connected / in-degree) and the LEARNED excess (the
store's potentiation). Each part is read, in units of the step's k-WTA threshold, on three
targets: the correct next token's neurons, the best rival token's (the top-logit token of any
other word), and the whole area (the field). The sum of the parts is checked against the
drive replay used.

Questions: is the rival's excess learned weight from ON-track neurons -- the type trace driving
the fan -- and how does each part move with reuse and along the sequence?"""
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
USES = (10, 40, 50)
if len(sys.argv) > 1:   # n k p tau U...
    n, k, p, tau = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    USES = tuple(int(u) for u in sys.argv[5:])
SEQS = 30
B = len(seeds)
dev = "cuda"
ar = torch.arange(B, device=dev)
beta = round(tl.theta(n, k, p), 5)
print(f"cell ({n}, {k}, {p}), tau {tau}; beta {beta}; kp {k * p:.1f}; "
      f"sigma/kp {math.sqrt((1 - p) / (k * p)):.3f}", flush=True)
PARTS = ("on_base", "on_learn", "off_base", "off_learn")

for U in USES:
    M = max(1, round(0.05 * ml.unit(n, k, p) / rg.LENGTH)); L = M * rg.LENGTH; V = round(L / U)
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=beta, w_max=pe.W_MAX, norm_init=True, rounds=1,
                         strength=ml.STRENGTH, max_items=4, device=dev, bias_decay=math.exp(-1 / tau))
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(rg.LENGTH)])[0] for q in range(M)]
    allst = torch.cat(seqs, 0).long()
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    area, fib = mem.area, mem.fiber
    saved = area.bias.clone()
    area.bias = torch.zeros(B, n, device=dev)
    rows = []          # per step: [B, ...] records
    worst = 0.0
    for q in list(range(0, M, 2))[:SEQS]:
        area.bias.zero_()
        area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev)
                                    for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        for j in range(1, rg.LENGTH):
            prev = area.winners.clone()
            cur_tok = allst[q * rg.LENGTH + j - 1]                      # [B, k]
            hcur = torch.zeros(B, n, device=dev); hcur.scatter_(1, cur_tok, 1.0)
            on = torch.gather(hcur, 1, prev) > 0                         # [B, m]
            parts = {nm: torch.zeros(B, n, device=dev) for nm in PARTS}
            for c in range(prev.shape[1]):
                r = prev[:, c:c + 1].contiguous()
                tot = torch.zeros(B, n, device=dev)
                fib.contribute(tot, r)
                base = fib.mod.hashed_drive(r.to(torch.int32), fib.seeds, fib.n, fib.threshold) / fib.dj
                sel = on[:, c].float().view(-1, 1)
                parts["on_base"] += base * sel; parts["on_learn"] += (tot - base) * sel
                parts["off_base"] += base * (1 - sel); parts["off_learn"] += (tot - base) * (1 - sel)
            x, drive = area.project(1, [fib], freeze=True, mask_bias=False, return_drive=True)
            worst = max(worst, float((sum(parts.values()) - drive).abs().max() / drive.abs().max()))
            thr = torch.topk(drive, k, dim=1).values[:, -1].clamp_min(1e-9).view(-1, 1)
            tok = torch.gather(drive.unsqueeze(0).expand(L, B, n), 2, allst).mean(2) / thr.view(1, -1)
            want_tok = q * rg.LENGTH + j
            want = wordof[want_tok]
            masked_tok = torch.where(wordof == want.view(1, -1), torch.full_like(tok, -1e9), tok)
            riv_tok = masked_tok.argmax(0)                                # [B]
            tc, tr = allst[want_tok], allst[riv_tok, ar]                  # [B, k]
            rec = [alive.float(), on.float().sum(1) / k]
            for nm in PARTS:
                v = parts[nm] / thr
                rec += [torch.gather(v, 1, tc).mean(1), torch.gather(v, 1, tr).mean(1), v.mean(1)]
            rows.append((j, torch.stack(rec, 1)))
            hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
            alive &= wordof[torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == want
    area.bias = saved
    print(f"\nU = {U} (V {V}); decomposition error (max relative) {worst:.1e}", flush=True)
    print("  step  n(alive)  on-frac | correct: on_b on_L off_b off_L = sum | rival: on_b on_L off_b off_L = sum"
          " | field: on_b off_b learn", flush=True)
    for js, label in (((1,), "1"), ((2,), "2"), ((3, 4), "3-4"), ((5, 6, 7, 8), "5-8"), (tuple(range(9, 16)), "9-15")):
        t = torch.cat([r for jj, r in rows if jj in js], 0)                # [S*?, B, F] -> flatten
        t = t.reshape(-1, t.shape[-1])
        a = t[:, 0] > 0.5
        if not a.any():
            continue
        t = t[a]
        f = lambda i: float(t[:, i].mean())
        c = [f(2 + 3 * s) for s in range(4)]
        rv = [f(3 + 3 * s) for s in range(4)]
        fl = [f(4 + 3 * s) for s in range(4)]
        print(f"  {label:>5} {int(a.sum()):6d}  {f(1):.3f}  | " + " ".join(f"{v:.3f}" for v in c) + f" = {sum(c):.3f}"
              + " | " + " ".join(f"{v:.3f}" for v in rv) + f" = {sum(rv):.3f}"
              + f" | {fl[0]:.3f} {fl[2]:.3f} {fl[1] + fl[3]:.3f}", flush=True)
    del mem, seqs, allst
    torch.cuda.empty_cache()
