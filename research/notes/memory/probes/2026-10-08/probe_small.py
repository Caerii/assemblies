import os
import math, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import torch
from research.experiments import memory_load_drift as md, memory_load_law as ml, memory_threshold_law as tl, memory_sequences as sq
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from research.experiments import memory_pattern_efficiency as pe
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
seeds = list(range(980, 984))
n, k, p = 4000, 400, 0.5
beta = round(tl.theta(n, k, p), 5); print("beta", beta)
for tau in (5, 33, 200):
    for L in (8, 40):
        mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=beta, w_max=pe.W_MAX, norm_init=True, rounds=1,
                             strength=ml.STRENGTH, max_items=4, device="cuda", bias_decay=math.exp(-1/tau))
        els = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"L{L}e{e}", "A")) for sd in seeds] for e in range(L)]
        st = mem.store_sequence(els)[0]
        consec = [round(float(sq._overlap(st[e], st[e + 1]).mean()), 3) for e in range(3)]
        x = st[0][:, :k // 2]  # strongest half
        y = mem.recall(x, rounds=1)
        o1 = sq._overlap(y, st[1]).tolist(); o0 = sq._overlap(y, st[0]).tolist()
        full = mem.recall(st[0], rounds=1); f1 = sq._overlap(full, st[1]).tolist()
        print(f"tau={tau} L={L}: consecutive-state overlap {consec}; half-cue -> next {[round(a,2) for a in o1]} self {[round(a,2) for a in o0]}; full-cue -> next {[round(a,2) for a in f1]}")
