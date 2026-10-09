"""Exploratory probe (disclosed): is the interference Poisson? (4000, 60, 0.5), tau 64.
From the full element A_t, split each neuron's input into the present-synapse count m_j
and the learned excess E_j = sum_i P_ij ((1+beta)^C_ij - 1) (clipped), and compare the
outsiders' and targets' statistics with the independent-Poisson model's."""
import math, sys
import numpy as np
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from research.experiments import memory_pattern_efficiency as pe

n, k, p, TAU, B = 4000, 60, 0.5, 64, 3
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
seeds = [i32(_seeding.fnv1a_pair_seed(900 + b, "A", "A")) for b in range(B)]
for L in (400, 1600, 3200):
    els = [[i32(_seeding.fnv1a_pair_seed(900 + b, f"h{e}", "A")) for b in range(B)] for e in range(L)]
    mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                         strength=0.5, max_items=4, bias_decay=math.exp(-1.0 / TAU))
    st, _ = mem.store_sequence(els)
    lam = L * k * k / (n * n)
    tab = torch.minimum((1.0 + beta) ** torch.arange(128, device="cuda", dtype=torch.float32),
                        torch.tensor(20.0, device="cuda")) - 1.0
    Eo, Et, To, Tt, own_c, other_c = [], [], [], [], [], []
    for b in range(B):
        P = pe.presence_of(mem.fiber.pres, b, n)
        C = mem.fiber.C[b].long()
        for t in range(0, L - 1, max(1, (L - 1) // 120)):
            rows = st[t, b]
            Pr, Cr = P[rows], C[rows]                      # [k, n]
            exc = (tab[Cr] * Pr).sum(0)                      # learned excess per neuron
            tgt = torch.zeros(n, dtype=torch.bool, device="cuda"); tgt[st[t + 1, b]] = True
            Eo.append(exc[~tgt]); Et.append(exc[tgt])
            # counts on present synapses: targets' own synapses vs everything else
            own = Cr[:, tgt][Pr[:, tgt]] ; oth = Cr[:, ~tgt][Pr[:, ~tgt]]
            own_c.append(own.float()); other_c.append(oth.float())
    Eo, Et = torch.cat(Eo), torch.cat(Et)
    oc, tc = torch.cat(other_c), torch.cat(own_c)
    # independent-Poisson prediction for an outsider: m ~ Bin(k, p) synapses, each (1+b)^Pois(lam) - 1
    rng = np.random.default_rng(0)
    m = rng.binomial(k, p, 200000)
    cnt = rng.poisson(lam, (200000, k)); mask = np.arange(k)[None, :] < m[:, None]
    pe_ = ((np.minimum((1 + beta) ** cnt, 20.0) - 1) * mask).sum(1)
    print(f"L={L} lam={lam:.3f}")
    print(f"   outsider excess: mean {float(Eo.mean()):.3f} var {float(Eo.var()):.3f}   Poisson model: mean {pe_.mean():.3f} var {pe_.var():.3f}")
    print(f"   target excess:   mean {float(Et.mean()):.3f} var {float(Et.var()):.3f}")
    print(f"   counts on outsiders' present synapses: mean {float(oc.mean()):.3f} var/mean {float(oc.var() / oc.mean()):.2f} "
          f"P(>=2) {float((oc >= 2).float().mean()):.4f} (Poisson {1 - math.exp(-lam) * (1 + lam):.4f})")
    print(f"   counts on targets' own synapses: mean {float(tc.mean()):.3f} (model 1 + lam = {1 + lam:.3f})", flush=True)
