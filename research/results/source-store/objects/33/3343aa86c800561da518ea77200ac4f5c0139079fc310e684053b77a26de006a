"""Exploratory (H2): does every knob act through the dispersion of interference?
At cells/tau whose rho_50 is measured, store one sequence at rho = 0.10 and measure the learned
excess drive S_j = sum_{i in x} P_ij g(C_ij), g(c) = min((1+beta)^c, w_max) - 1, onto NON-target
neurons j (not in the next state), from x = the stored state s_t, over many t. Compare its variance
with (a) independent Poisson counts of the same mean, (b) independent counts with the EMPIRICAL
count distribution (shuffled pairs). D_total = measured/(a); D_count = (b)/(a); D_corr = measured/(b).
Seeds 980-983 (4 brains), 64 steps sampled per brain."""
import os
import math, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import torch
from research.experiments import memory_load_law as ml, memory_threshold_law as tl, memory_pattern_efficiency as pe
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

seeds = list(range(980, 984))
RHO = 0.10
# (n, k, p, tau, measured rho_50, source)
CASES = [
    (6000, 300, 0.12, 10, 0.122, "A41"), (6000, 300, 0.12, 20, 0.202, "A41"),
    (5000, 100, 0.35, 25, 0.123, "A41"), (5000, 100, 0.35, 50, 0.152, "A41"),
    (10000, 100, 0.35, 50, 0.122, "A41"), (10000, 100, 0.35, 100, 0.120, "A41"),
    (14000, 70, 0.5, 100, 0.112, "A41"), (14000, 70, 0.5, 200, 0.102, "A41"),
    (21000, 70, 0.5, 64, 0.081, "A39"), (21000, 70, 0.5, 150, 0.105, "A39"),
    (2000, 60, 0.5, 8, 0.066, "probe"), (2000, 60, 0.5, 33, 0.160, "probe"),
]


def poisson_var(lam, beta, wmax):
    # E[g], E[g^2] for C ~ Poisson(lam), g = min((1+beta)^C, wmax) - 1
    eg = eg2 = 0.0
    pk = math.exp(-lam)
    for c in range(0, 200):
        g = min((1 + beta) ** c, wmax) - 1
        eg += pk * g; eg2 += pk * g * g
        pk *= lam / (c + 1)
        if pk < 1e-15 and c > lam: break
    return eg, eg2


for n, k, p, tau, r50, src in CASES:
    beta = round(tl.theta(n, k, p), 5)
    L = int(round(RHO * ml.unit(n, k, p)))
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=beta, w_max=pe.W_MAX, norm_init=True, rounds=1,
                         strength=ml.STRENGTH, max_items=4, device="cuda", bias_decay=math.exp(-1 / tau))
    els = [[to_i32(_seeding.fnv1a_pair_seed(sd, f"D{L}e{e}", "A")) for sd in seeds] for e in range(L)]
    st = mem.store_sequence(els)[0]                                         # [L, B, k]
    res = []
    for b in range(len(seeds)):
        Cb = mem.fiber.C[b]                                                 # [n, n] int counts (pre, post)
        P = pe.presence_of(mem.fiber.pres, b, n)                            # [n, n] bool

        def g_rows(rows):
            c = Cb[rows].to(torch.float32)
            return (torch.clamp((1 + beta) ** c, max=pe.W_MAX) - 1) * P[rows].to(torch.float32)
        steps = torch.linspace(0, L - 2, 64).long()
        meas = []
        for t in steps.tolist():
            x, nxt = st[t, b], st[t + 1, b]
            S = g_rows(x).sum(0)                                            # [n] excess onto every j
            mask = torch.ones(n, dtype=torch.bool, device=S.device); mask[nxt] = False
            meas.append(S[mask].var().item())
        sample = torch.randperm(n, device=Cb.device)[:min(n, 2000)]
        Gs, Ps = g_rows(sample), P[sample]
        lam = (Cb[sample].to(torch.float32).sum() / Ps.sum()).item()        # mean count per present pair
        eg, eg2 = poisson_var(lam, beta, pe.W_MAX)
        indep_poisson = k * (p * eg2 - (p * eg) ** 2)
        vals = Gs[Ps]
        e1, e2 = vals.mean().item(), (vals ** 2).mean().item()
        indep_emp = k * (p * e2 - (p * e1) ** 2)
        cmax = Cb.max().item()
        del P
        res.append((sum(meas) / len(meas), indep_poisson, indep_emp, lam, cmax))
    m = [sum(r[i] for r in res) / len(res) for i in range(5)]
    print(f"{src} ({n},{k},{p}) tau={tau} ({tau/(n/k):.2f} n/k) rho50={r50}: lambda {m[3]:.3f} maxC {m[4]:.0f} | "
          f"D_total {m[0]/m[1]:.2f} D_count {m[2]/m[1]:.2f} D_corr {m[0]/m[2]:.2f} | rho50*D_total {r50*m[0]/m[1]:.3f}",
          flush=True)
    del mem, st
    torch.cuda.empty_cache()
