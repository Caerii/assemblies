"""Exploratory (not registered): can SLEEP repair a captured store? Unlearning in the sense of
Hopfield, Feinstein & Palmer (1983) -- the function Crick & Mitchison (1983) proposed for REM
sleep: start the network from NOISE, let it run freely so it falls into the states it is drawn
to (in a collapsed store, the captured clusters), and apply reverse-sign plasticity to the
transitions it takes. Here a transition prev -> new taken during sleep loses one potentiation
count on every synapse prev_i -> new_j that has any (never below zero; the connectome's baseline
is untouched), from the third step of each episode on (once the state has settled).

Store: random-successor reuse, (10000, 75, 0.48), tau = 67, rho = 0.05 (judged cell), seeds
980-999, standard store_sequence. U = 50 (collapsed, 0.025) and U = 10 (healthy, as harm
control). Sleep episodes: 8 free steps from k random neurons, masked, frozen except the
unlearning; doses cumulative. After each dose: masked replay reliability over every sequence
(word-level), the share of sleep steps landing on a stored token (overlap >= 0.5) and the counts
removed."""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_reuse_grammar as rg, memory_load_law as ml
from research.experiments import memory_load_drift as md
from research.experiments import memory_write_separation as ws
from research.experiments.seq_capacity_scaling import to_i32
from neural_assemblies.core.numpy_engine import _seeding

seeds = list(range(980, 1000))
n, k, p, tau = 10000, 75, 0.48, 67
DOSES = {50: (0, 100, 300, 1000, 3000), 10: (0, 1000, 3000)}
STEPS, SETTLE = 8, 2
dev = "cuda"
LEN = rg.LENGTH
B = len(seeds)
ar = torch.arange(B, device=dev)
gen = torch.Generator(device=dev).manual_seed(4242)

for U, doses in DOSES.items():
    M = max(1, round(0.05 * ml.unit(n, k, p) / LEN)); L = M * LEN; V = round(L / U)
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    mem = ws.build(n, k, p, tau, seeds, dev)
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(LEN)])[0] for q in range(M)]
    mem.area.check_overflow()
    allst = torch.cat(seqs, 0).long()
    area, fib = mem.area, mem.fiber
    C = fib.C
    print(f"\nU = {U} (V {V}); counts held {int(C.sum())}", flush=True)

    def reliability():
        rel = torch.zeros(B, device=dev)
        for q in range(M):
            area.bias = torch.zeros(B, n, device=dev)
            area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev)
                                        for i, sd in enumerate(seeds)]).long()
            alive = torch.ones(B, dtype=torch.bool, device=dev)
            for j in range(1, LEN):
                x = area.project(1, [fib], freeze=True, mask_bias=False)
                hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
                alive &= wordof[torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == wordof[q * LEN + j]
            rel += alive.float()
        return (rel / M).cpu().numpy()

    done, removed, landed, steps = 0, 0, 0, 0
    for dose in doses:
        while done < dose:
            area.bias = torch.zeros(B, n, device=dev)
            area.winners = torch.argsort(torch.rand(B, n, device=dev, generator=gen), dim=1)[:, :k]
            for s in range(STEPS):
                prev = area.winners.clone()
                x = area.project(1, [fib], freeze=True, mask_bias=False).long()
                if s >= SETTLE:
                    bi = ar.view(B, 1, 1); ri = prev.view(B, k, 1); ci = x.view(B, 1, k)
                    old = C[bi, ri, ci]
                    dec = (old > 0).to(old.dtype)
                    C[bi, ri, ci] = old - dec
                    removed += int(dec.sum())
                    hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x, 1.0)
                    ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2) / k
                    landed += int((ov.max(0).values >= 0.5).sum()); steps += B
            done += 1
        rel = reliability()
        print(f"  sleep {dose:5d} episodes: reliability {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}); "
              f"counts removed {removed} ({removed / max(1, int(C.sum()) + removed):.4f} of held); "
              f"sleep steps landing on a stored token {landed / max(1, steps):.3f}", flush=True)
    del mem, seqs, allst, C
    torch.cuda.empty_cache()
