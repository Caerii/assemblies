"""Exploratory probe (disclosed): the one-step overlap map F(o) of the stored sequence.
Input: a fraction o of element t's winners plus (1 - o) k random neurons; output: overlap
of one masked recall round with element t+1. Positions sampled across the sequence."""
import math, sys
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from research.experiments import memory_sequences as sq

n, k, p = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
Ls = [int(x) for x in sys.argv[4].split(",")]
strength, tau = float(sys.argv[5]), (float(sys.argv[6]) if sys.argv[6] != "none" else None)
OS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
B, POS = 5, 64
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v
seeds = [i32(_seeding.fnv1a_pair_seed(900 + b, "A", "A")) for b in range(B)]
g = torch.Generator(device="cuda").manual_seed(7)
for L in Ls:
    els = [[i32(_seeding.fnv1a_pair_seed(900 + b, f"h{e}", "A")) for b in range(B)] for e in range(L)]
    mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                         strength=strength, max_items=4,
                         bias_decay=None if tau is None else math.exp(-1.0 / tau))
    st, _ = mem.store_sequence(els)
    pos = torch.linspace(0, L - 2, POS).long().tolist()
    row = []
    for o in OS:
        m = int(round(o * k))
        vals = []
        for t in pos:
            x = st[t].clone()
            perm = torch.argsort(torch.rand(B, k, device="cuda", generator=g), dim=1)
            x = torch.gather(x, 1, perm)
            if m < k:
                x[:, m:] = torch.randint(0, n, (B, k - m), device="cuda", generator=g)
            vals.append(sq._overlap(mem.recall(x, rounds=1), st[t + 1]))
        row.append(float(torch.stack(vals).mean()))
    # the chain: mean overlap of the replayed state with the stored one, by step
    x = st[0][:, :k // 2]
    traj = []
    for t in range(1, min(L, 41)):
        x = mem.recall(x, rounds=1)
        traj.append(round(float(sq._overlap(x, st[t]).mean()), 3))
    print(f"L={L}  F(o) for o={OS}: {[round(v, 3) for v in row]}")
    print(f"   F(o)-o: {[round(v - o, 3) for v, o in zip(row, OS)]}")
    print(f"   chain mean overlap, steps 1-40 (every 4th): {traj[::4]}", flush=True)
