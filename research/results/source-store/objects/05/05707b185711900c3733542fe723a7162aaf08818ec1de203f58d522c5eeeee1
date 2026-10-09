"""Exploratory (not registered): SELF-LIMITING sleep. probe_sleep.py found unlearning repairs a
collapsed store in a window (100-300 episodes: 0.025 -> 0.70-0.74) and erases any store beyond it
(1000: 0.000) -- in a HEALTHY store too, where dreams never settle onto a stored state: the damage
there is depression of random noise transitions, not of attractors. Here unlearning is GATED by a
signal the network has: a transition is unlearned only if the dream has LOOPED -- the new state
shares >= 0.5 of its neurons with a state the same episode already visited (a stable state or a
cycle). A stored sequence runs forward and does not revisit; a spurious attractor is where free
dynamics converge. Prediction: gated sleep repairs collapsed stores, leaves healthy ones alone,
and stops by itself as the attractors dissolve (insensitive to dose).

Same store as probe_sleep.py ((10000, 75, 0.48), tau = 67, seeds 980-999, standard
store_sequence), U = 50 and U = 10; episodes of 12 free steps from k random neurons; doses
cumulative. Reported: reliability, counts removed, share of steps that looped."""
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
DOSES = {50: (0, 100, 300, 1000, 3000), 10: (0, 300, 1000, 3000)}
STEPS, LOOP = 12, 0.5
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
    held = int(C.sum())
    print(f"\nU = {U} (V {V}); counts held {held}", flush=True)

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

    done, removed, looped, steps = 0, 0, 0, 0
    for dose in doses:
        while done < dose:
            area.bias = torch.zeros(B, n, device=dev)
            area.winners = torch.argsort(torch.rand(B, n, device=dev, generator=gen), dim=1)[:, :k]
            hist = []
            for s in range(STEPS):
                prev = area.winners.clone()
                x = area.project(1, [fib], freeze=True, mask_bias=False).long()
                hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x, 1.0)
                if hist:
                    H = torch.stack(hist)                                  # [S, B, k]
                    ov = torch.gather(hot.unsqueeze(0).expand(H.shape[0], B, n), 2, H).sum(2) / k
                    gate = ov.max(0).values >= LOOP
                    if bool(gate.any()):
                        bi = ar.view(B, 1, 1); ri = prev.view(B, k, 1); ci = x.view(B, 1, k)
                        old = C[bi, ri, ci]
                        dec = ((old > 0) & gate.view(B, 1, 1)).to(old.dtype)
                        C[bi, ri, ci] = old - dec
                        removed += int(dec.sum())
                    looped += int(gate.sum()); steps += B
                hist.append(x)
            done += 1
        rel = reliability()
        print(f"  gated sleep {dose:5d} episodes: reliability {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}); "
              f"counts removed {removed} ({removed / held:.4f} of held); steps that looped {looped / max(1, steps):.3f}",
              flush=True)
    del mem, seqs, allst, C
    torch.cuda.empty_cache()
