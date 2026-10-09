"""Exploratory (not registered): sleep gated by SETTLING CONTRAST. The loop gate
(probe_sleep_gated.py) was safe but slow: dreams in a collapsed store wander through captured
clusters without revisiting a state. A captured cluster is an assembly that many writes landed
on, so it is driven through many potentiated synapses: while the dream sits in one, its winners
are driven unusually hard relative to the area. Gate: unlearn a dream transition only if the
winners' mean drive over the area's mean drive (CONTRAST) exceeds the largest contrast a HEALTHY
reference store's dreams ever reach (calibrated on 300 dreams of a U = 10 store at the same cell,
with a different noise stream), times 1.02.

Same stores as probe_sleep.py ((10000, 75, 0.48), tau = 67, seeds 980-999, standard
store_sequence): U = 50 (collapsed) and U = 10 (healthy, harm control; a store built anew).
Episodes of 8 free steps from k random neurons, unlearning from step 3; doses cumulative.
Reported: reliability, counts removed, share of steps gated, the calibrated threshold, and the
contrast distribution in each store."""
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
STEPS, SETTLE, MARGIN = 8, 2, 1.02
dev = "cuda"
LEN = rg.LENGTH
B = len(seeds)
ar = torch.arange(B, device=dev)


def build_store(U):
    M = max(1, round(0.05 * ml.unit(n, k, p) / LEN)); L = M * LEN; V = round(L / U)
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device=dev).long()
    mem = ws.build(n, k, p, tau, seeds, dev)
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(LEN)])[0] for q in range(M)]
    mem.area.check_overflow()
    return mem, seqs, torch.cat(seqs, 0).long(), wordof, M, L


def dream(mem, gen, gate_thr=None, contrasts=None):
    """One episode for every brain; unlearn gated transitions; returns counts removed, steps gated."""
    area, fib, C = mem.area, mem.fiber, mem.fiber.C
    area.bias = torch.zeros(B, n, device=dev)
    area.winners = torch.argsort(torch.rand(B, n, device=dev, generator=gen), dim=1)[:, :k]
    removed = gated = 0
    for s in range(STEPS):
        prev = area.winners.clone()
        x, drive = area.project(1, [fib], freeze=True, mask_bias=False, return_drive=True)
        x = x.long()
        if s >= SETTLE:
            con = torch.gather(drive, 1, x).mean(1) / drive.mean(1).clamp_min(1e-9)
            if contrasts is not None:
                contrasts.append(con.cpu())
            if gate_thr is not None:
                g = con >= gate_thr
                if bool(g.any()):
                    bi = ar.view(B, 1, 1); ri = prev.view(B, k, 1); ci = x.view(B, 1, k)
                    old = C[bi, ri, ci]
                    dec = ((old > 0) & g.view(B, 1, 1)).to(old.dtype)
                    C[bi, ri, ci] = old - dec
                    removed += int(dec.sum())
                gated += int(g.sum())
    return removed, gated


def reliability(mem, seqs, allst, wordof, M, L):
    area, fib = mem.area, mem.fiber
    rel = torch.zeros(B, device=dev)
    for q in range(M):
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev) for i, sd in enumerate(seeds)]).long()
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        for j in range(1, LEN):
            x = area.project(1, [fib], freeze=True, mask_bias=False)
            hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
            alive &= wordof[torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2).argmax(0), ar] == wordof[q * LEN + j]
        rel += alive.float()
    return (rel / M).cpu().numpy()


# calibration: a healthy reference store's dream contrasts
mem, *_ = build_store(10)
cal = []
g0 = torch.Generator(device=dev).manual_seed(777)
for _ in range(300):
    dream(mem, g0, None, cal)
cal = torch.cat(cal).numpy()
thr = float(cal.max()) * MARGIN
print(f"calibration (U = 10 reference, 300 dreams): contrast mean {cal.mean():.3f}, p99.9 {np.quantile(cal, 0.999):.3f}, "
      f"max {cal.max():.3f} -> threshold {thr:.3f}", flush=True)
del mem
torch.cuda.empty_cache()

for U, doses in ((50, (0, 100, 300, 1000, 3000)), (10, (0, 1000, 3000))):
    mem, seqs, allst, wordof, M, L = build_store(U)
    held = int(mem.fiber.C.sum())
    gen = torch.Generator(device=dev).manual_seed(4242)
    seen = []
    dream(mem, torch.Generator(device=dev).manual_seed(99), None, seen)
    seen = torch.cat(seen).numpy()
    print(f"\nU = {U}: dream contrast before sleep mean {seen.mean():.3f}, p90 {np.quantile(seen, 0.9):.3f}, "
          f"share above threshold {(seen >= thr).mean():.3f}", flush=True)
    done = removed = gated = steps = 0
    for dose in doses:
        while done < dose:
            r, g = dream(mem, gen, thr)
            removed += r; gated += g; steps += B * (STEPS - SETTLE); done += 1
        rel = reliability(mem, seqs, allst, wordof, M, L)
        print(f"  contrast-gated sleep {dose:5d} episodes: reliability {rel.mean():.3f} (brains < 0.2: {(rel < 0.2).sum():2d}); "
              f"counts removed {removed / held:.4f} of held; steps gated {gated / max(1, steps):.3f}", flush=True)
    del mem, seqs, allst
    torch.cuda.empty_cache()
