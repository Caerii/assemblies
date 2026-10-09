"""Exploratory (not registered), probe_grammar.py with WORD-LEVEL decoding: at every replay
step the read-out is matched to the nearest stored token (any sequence) and scored by that
token's WORD, beside the token-level score. Does heavy word reuse cost because words RECUR, or because
their successors are INCONSISTENT across contexts? Each brain draws its own grammar: every
word of a vocabulary V has b allowed successors (b = 1: a deterministic chain grammar;
b = V: i.i.d. words), and sequences of 16 are random walks on it. At ~20 uses per word and
total loads rho = 0.05, 0.08, the fraction of sequences replayed whole and the same-word
code overlap. Cell (4000, 100, 0.35), tau = 40 (Amendment 42's first cell, after it was
judged). Seeds 980-989."""
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
n, k, p, tau, l, uses = 4000, 100, 0.35, 40, 16, 20


def walks(seed, M, V, b, salt):
    rng = np.random.default_rng([seed, salt])
    succ = np.stack([rng.choice(V, size=b, replace=False) for _ in range(V)]) if b < V else None
    out = np.empty((M, l), dtype=np.int64)
    for q in range(M):
        w = rng.integers(V)
        for e in range(l):
            out[q, e] = w
            w = rng.integers(V) if succ is None else succ[w, rng.integers(b)]
    return out


def run(rho, b_of_V):
    M = max(1, round(rho * ml.unit(n, k, p) / l)); L = M * l
    V = max(8, round(L / uses)); b = V if b_of_V is None else min(b_of_V, V)
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX, norm_init=True,
                         rounds=1, strength=ml.STRENGTH, max_items=4, device="cuda", bias_decay=math.exp(-1 / tau))
    words = [walks(sd, M, V, b, L * 1000 + b) for sd in seeds]
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(l)])[0] for q in range(M)]
    B = len(seeds)
    allst = torch.cat(seqs, 0)                                       # [L, B, k] every stored token
    wordof = torch.tensor(np.stack([w.reshape(-1) for w in words], 1), device="cuda")   # [L, B]
    done = torch.zeros(B, device="cuda"); wdone = torch.zeros(B, device="cuda")
    for q, st in enumerate(seqs):
        x = torch.stack([md.cue(st[0, i], sd, L * 100_000 + q, k, "cuda") for i, sd in enumerate(seeds)])
        alive = torch.ones(B, dtype=torch.bool, device="cuda"); walive = alive.clone()
        for j in range(1, l):
            x = mem.recall(x, rounds=1)
            alive &= sq._overlap(x, st[j]) >= ml.MATCH
            hot = torch.zeros(B, n, device="cuda"); hot.scatter_(1, x.long(), 1.0)
            ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst.long()).sum(2)   # [L, B]
            best = ov.argmax(0)                                      # nearest stored token per brain
            ok = (wordof[best, torch.arange(B, device="cuda")] == wordof[q * l + j]) & (ov.max(0).values >= ml.MATCH * k)
            walive &= ok
        done += alive.float(); wdone += walive.float()
    fr = (done / M).tolist(); wfr = (wdone / M).tolist()
    # same-word overlap (first two cross-sequence occurrences, brain 0)
    where = {}
    for q in range(M):
        for e in range(l):
            where.setdefault(int(words[0][q, e]), []).append((q, e))
    same = [float(sq._overlap(seqs[a[0]][a[1], 0:1], seqs[c[0]][c[1], 0:1]))
            for occ in where.values() for a, c in [next(((a, c) for a in occ for c in occ if a[0] < c[0]), (None, None))]
            if a is not None][:200]
    return L, V, b, sum(fr) / len(fr), sum(wfr) / len(wfr), (sum(same) / len(same) if same else None)


for rho in (0.05, 0.08):
    for b in (1, 2, 4, None):
        L, V, bb, w, ww, same = run(rho, b)
        print(f"rho={rho} V={V} successors b={'V' if b is None else bb}: token-level whole {w:.3f}, "
              f"WORD-level whole {ww:.3f}; same-word overlap {same and round(same, 3)}", flush=True)
