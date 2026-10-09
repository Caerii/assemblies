"""Exploratory: does tau ~ n/k minimise the over-dispersion of reuse? Store only; per-neuron use counts
and the per-pair count distribution of consecutive-state synapses. Seeds 980-989."""
import os
import math, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import torch
from research.experiments import memory_load_law as ml, memory_threshold_law as tl, memory_pattern_efficiency as pe
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
seeds = list(range(980, 990))
for n, k, p in ((4000, 200, 0.15), (2000, 60, 0.5), (15000, 50, 0.7)):
    L = int(round(0.12 * ml.unit(n, k, p)))
    for f in (0.25, 0.5, 1.0, 2.0, 4.0):
        tau = max(1, round(f * n / k))
        mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX, norm_init=True,
                             rounds=1, strength=ml.STRENGTH, max_items=4, device="cuda", bias_decay=math.exp(-1 / tau))
        els = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"D{L}e{e}", "A")) for sd in seeds] for e in range(L)]
        st = mem.store_sequence(els)[0]                                   # [L, B, k]
        use = torch.zeros(len(seeds), n, device="cuda")
        use.scatter_add_(1, st.permute(1, 0, 2).reshape(len(seeds), -1), torch.ones(len(seeds), L * k, device="cuda"))
        m, v = use.mean(1), use.var(1)
        # gaps between successive uses of a neuron: CV of gaps (1 = Poisson, 0 = periodic)
        gaps = []
        for b in range(2):
            pos = [[] for _ in range(n)]
            for t, row in enumerate(st[:, b].tolist()):
                for i in row: pos[i].append(t)
            g = [y - x for q in pos for x, y in zip(q, q[1:])]
            t = torch.tensor(g, dtype=torch.float); gaps.append((t.mean().item(), (t.std() / t.mean()).item(), (t < tau).float().mean().item()))
        C = mem.fiber.C.float()
        cmax = C.amax(dim=(1, 2)).mean().item()
        print(f"({n},{k},{p}) L={L} tau/(n/k)={f} tau={tau}: use var/mean {float((v / m).mean()):.3f}; gap mean {gaps[0][0]:.1f} CV {gaps[0][1]:.3f} "
              f"share<tau {gaps[0][2]:.3f}; max pair count {cmax:.1f}", flush=True)
        del mem, st
