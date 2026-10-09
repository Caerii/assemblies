"""Exploratory (not registered): is the tau peak on the SIGNAL side? For every stored
transition s_t -> s_{t+1}, the successor's structural input from its predecessor -- the
number of present synapses from s_t onto each member of s_{t+1}, in z units above a random
neuron's (k p, sd sqrt(k p (1 - p))) -- and its learned input (counts). At the twelve cell/tau
cases of probe_disp2.py, one sequence stored at rho = 0.10. Seeds 980-983."""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import torch
from research.experiments import memory_load_law as ml, memory_threshold_law as tl, memory_pattern_efficiency as pe
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

seeds = list(range(980, 984))
RHO = 0.10
CASES = [
    (6000, 300, 0.12, 10, 0.122), (6000, 300, 0.12, 20, 0.202),
    (5000, 100, 0.35, 25, 0.123), (5000, 100, 0.35, 50, 0.152),
    (10000, 100, 0.35, 50, 0.122), (10000, 100, 0.35, 100, 0.120),
    (14000, 70, 0.5, 100, 0.112), (14000, 70, 0.5, 200, 0.102),
    (21000, 70, 0.5, 64, 0.081), (21000, 70, 0.5, 150, 0.105),
    (2000, 60, 0.5, 8, 0.066), (2000, 60, 0.5, 33, 0.160),
]
for n, k, p, tau, r50 in CASES:
    beta = round(tl.theta(n, k, p), 5)
    L = int(round(RHO * ml.unit(n, k, p)))
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=beta, w_max=pe.W_MAX, norm_init=True, rounds=1,
                         strength=ml.STRENGTH, max_items=4, device="cuda", bias_decay=math.exp(-1 / tau))
    els = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"S{L}e{e}", "A")) for sd in seeds] for e in range(L)]
    st = mem.store_sequence(els)[0]                                         # [L, B, k]
    z_struct, z_late, fresh = [], [], []
    sd0 = math.sqrt(k * p * (1 - p))
    for b in range(len(seeds)):
        P = pe.presence_of(mem.fiber.pres, b, n)
        steps = torch.linspace(0, L - 2, 200).long().tolist()
        for t in steps:
            x, y = st[t, b], st[t + 1, b]
            m = P[x][:, y].float().sum(0)                                   # structural inputs onto members
            z = ((m - k * p) / sd0).mean().item()
            z_struct.append(z)
            if t > L // 2:
                z_late.append(z)
        del P
    zs, zl = sum(z_struct) / len(z_struct), sum(z_late) / len(z_late)
    print(f"({n},{k},{p}) tau={tau} ({tau / (n / k):.2f} n/k) rho50={r50}: successor structural input "
          f"z = {zs:.2f} (late half {zl:.2f}); sqrt(ln n) = {math.sqrt(math.log(n)):.2f}", flush=True)
    del mem, st
    torch.cuda.empty_cache()
