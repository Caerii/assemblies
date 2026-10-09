"""Exploratory (not registered): many 16-element sequences whose elements are WORDS drawn i.i.d.
from a vocabulary of V (the same word -> the same stimulus wherever it occurs), and activity noise
during replay. Survey cell (4000, 60, 0.5), tau = 33. Seeds 980-989.
args: mode (vocab|noise)."""
import os
import math, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_load_law as ml, memory_load_drift as md, memory_threshold_law as tl
from research.experiments import memory_pattern_efficiency as pe, memory_sequences as sq, memory_noise as mn
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

seeds = list(range(980, 990))
n, k, p, tau, l = 4000, 60, 0.5, 33, 16


def run(rho, V=None, nu=0.0):
    L = max(1, round(rho * ml.unit(n, k, p) / l)) * l
    M = L // l
    rng = np.random.default_rng(L * 7919 + (V or 0))
    words = rng.integers(0, V, size=(M, l)) if V else np.arange(M * l).reshape(M, l)
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX, norm_init=True,
                         rounds=1, strength=ml.STRENGTH, max_items=4, device="cuda", bias_decay=math.exp(-1 / tau))
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(w)}", "A")) for sd in seeds]
                                for w in words[q]])[0] for q in range(M)]
    gen = torch.Generator(device="cuda").manual_seed(L)
    done = torch.zeros(len(seeds), device="cuda")
    first_fail = []
    for q, st in enumerate(seqs):
        x = torch.stack([md.cue(st[0, b], sd, L * 100_000 + q, k, "cuda") for b, sd in enumerate(seeds)])
        alive = torch.ones(len(seeds), dtype=torch.bool, device="cuda")
        for j in range(1, l):
            x = mem.recall(x, rounds=1)
            if nu:
                x = mn._corrupt(x, nu, n, gen, uniform=True)
            ok = sq._overlap(x, st[j]) >= ml.MATCH
            newly = alive & ~ok
            first_fail += [j] * int(newly.sum())
            alive &= ok
        done += alive.float()
    # how context-dependent are the codes: overlap between two occurrences of the same word
    same = []
    if V:
        where = {}
        for q in range(M):
            for j in range(l):
                where.setdefault(int(words[q, j]), []).append((q, j))
        for w, occ in list(where.items())[:200]:
            if len(occ) >= 2:
                (q1, j1), (q2, j2) = occ[0], occ[1]
                same.append(float(sq._overlap(seqs[q1][j1], seqs[q2][j2]).mean()))
    fr = (done / M).tolist()
    ff = np.bincount(first_fail, minlength=l)[1:] if first_fail else []
    return L, M, sum(fr) / len(fr), (np.mean(same) if same else None), list(ff)


mode = sys.argv[1]
if mode == "vocab":
    for rho in (0.05, 0.08):
        for V in (None, 4096, 1024, 256, 64):
            t0 = time.time(); L, M, w, same, ff = run(rho, V=V)
            uses = L / V if V else 1
            print(f"rho={rho} V={V} (uses/word {uses:.1f}) L={L} M={M}: whole {w:.3f}; same-word code overlap "
                  f"{same and round(same, 3)}; first-fail by step {ff} [{time.time()-t0:.0f}s]", flush=True)
else:
    for rho in (0.05, 0.08, 0.11):
        for nu in (0.0, 0.03, 0.05, 0.1):
            t0 = time.time(); L, M, w, _, ff = run(rho, nu=nu)
            print(f"rho={rho} nu={nu} L={L} M={M}: whole {w:.3f}; first-fail by step {ff} [{time.time()-t0:.0f}s]", flush=True)
