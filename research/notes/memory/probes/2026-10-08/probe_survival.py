"""Exploratory (not registered): the survival curve along a sequence. M sequences of 256
elements per brain at total load rho; for every sequence the step of its first failure
(own overlap < 0.3), pooled over brains -> the hazard h(j) at each position j. Does the
hazard fall after a transient of a few steps (the two-phase conjecture)? Cell (8000, 80, 0.5),
tau = 50 (Amendment 40's, after it was judged). Seeds 980-989."""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_load_law as ml, memory_load_drift as md, memory_threshold_law as tl
from research.experiments import memory_pattern_efficiency as pe, memory_sequences as sq
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

seeds = list(range(980, 990))
n, k, p, tau, l = 8000, 80, 0.5, 50, 256
BANDS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 30), (31, 100), (101, 255))

for rho in (0.120, 0.130, 0.140):
    M = max(1, round(rho * ml.unit(n, k, p) / l)); L = M * l
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX, norm_init=True,
                         rounds=1, strength=ml.STRENGTH, max_items=4, device="cuda", bias_decay=math.exp(-1 / tau))
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"V{L}q{q}e{e}", "A")) for sd in seeds]
                                for e in range(l)])[0] for q in range(M)]
    first = []                                                      # first-failure step, l if none
    overl = np.zeros(8)
    for q, st in enumerate(seqs):
        x = torch.stack([md.cue(st[0, b], sd, L * 100_000 + q, k, "cuda") for b, sd in enumerate(seeds)])
        alive = torch.ones(len(seeds), dtype=torch.bool, device="cuda")
        ff = torch.full((len(seeds),), l, device="cuda")
        for j in range(1, l):
            x = mem.recall(x, rounds=1)
            o = sq._overlap(x, st[j])
            if j <= 8:
                overl[j - 1] += float(o[alive].mean()) if bool(alive.any()) else 0.0
            ok = o >= ml.MATCH
            ff = torch.where(alive & ~ok, torch.full_like(ff, j), ff)
            alive &= ok
        first += ff.tolist()
    first = np.array(first)
    N = len(first)
    out = []
    for a, b in BANDS:
        at_risk = (first >= a).sum()
        died = ((first >= a) & (first <= b)).sum()
        steps = b - a + 1
        h = 1 - (1 - died / at_risk) ** (1 / steps) if at_risk else float("nan")
        out.append(f"{a}-{b}: {h:.2e} ({died}/{at_risk})")
    print(f"rho={rho} M={M} N={N}: whole {np.mean(first == l):.3f}; per-step hazard by position: " + "; ".join(out), flush=True)
    print(f"   mean overlap of surviving read-outs at steps 1-8: {np.round(overl / M, 3).tolist()}", flush=True)
    del mem, seqs
    torch.cuda.empty_cache()
