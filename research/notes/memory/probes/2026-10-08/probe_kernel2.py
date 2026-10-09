"""Exploratory probe (disclosed): does a one-variable Markov reduction of replay
reproduce Amendment 34's noise horizons?

At each A34 cell (tau 33, strength 0.5, beta theta, L = 400), on 20 probe brains
(seeds 920-939, not A34's 482-501):
  KERNEL  P(o' | o): input = m = round(o k) of element t's winners + (k - m) random
          neurons; o' = overlap of one masked recall with element t + 1.
  SIM     Markov chain on o with that kernel and A34's noise (nu k of each output
          replaced at random), the half-cue first step measured directly; first
          miss when o < 0.3; per "brain" the mean of 4 draws; median over brains.
  REAL    A34's own replay on the probe brains (same noise model, 4 draws), and
          the real (o_t, o_t+1) pairs, to compare with the kernel's conditional mean.
"""
import math, statistics, sys
import numpy as np
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from research.experiments import memory_sequences as sq

CELLS = [(4000, 60, 0.5, 39.0), (8000, 60, 0.5, 232.0)]
NU, L, TAU, DRAWS, B = 0.1, 400, 33, 4, 20
OGRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
POS = 40
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v

def corrupt(x, frac, n, gen, uniform=False):
    m = int(round(frac * x.shape[1]))
    x = x.clone()
    if uniform:
        perm = torch.argsort(torch.rand(x.shape, device=x.device, generator=gen), dim=1)
        x = torch.gather(x, 1, perm)
    if m:
        x[:, :m] = torch.randint(0, n, (x.shape[0], m), device=x.device, generator=gen)
    return x

for n, k, p, a34 in CELLS:
    beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
    seeds = [i32(_seeding.fnv1a_pair_seed(920 + b, "A", "A")) for b in range(B)]
    els = [[i32(_seeding.fnv1a_pair_seed(920 + b, f"r{e}", "A")) for b in range(B)] for e in range(L)]
    mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                         strength=0.5, max_items=4, bias_decay=math.exp(-1.0 / TAU))
    st, _ = mem.store_sequence(els)
    g = torch.Generator(device="cuda").manual_seed(11)
    # KERNEL
    kernel = {}
    pos = torch.linspace(0, L - 2, POS).long().tolist()
    for o in OGRID:
        m = int(round(o * k))
        out = []
        for t in pos:
            perm = torch.argsort(torch.rand(B, k, device="cuda", generator=g), dim=1)
            x = torch.gather(st[t], 1, perm)
            if m < k:
                x[:, m:] = torch.randint(0, n, (B, k - m), device="cuda", generator=g)
            out.append(sq._overlap(mem.recall(x, rounds=1), st[t + 1]))
        kernel[float(o)] = torch.stack(out).flatten().cpu().numpy()
    first_step = sq._overlap(mem.recall(st[0][:, :k // 2], rounds=1), st[1]).cpu().numpy()
    # REAL chains (A34's replay: top slots replaced) and with uniformly random slots
    real = {}
    for uniform in (False, True):
      real_first, pairs = [], []
      for d in range(DRAWS):
        x = st[0][:, :k // 2]
        alive = torch.ones(B, dtype=torch.bool, device="cuda")
        steps = torch.zeros(B, device="cuda")
        for j in range(1, L):
            x = corrupt(mem.recall(x, rounds=1), NU, n, g, uniform)
            o = sq._overlap(x, st[j])
            alive &= o >= 0.3
            steps += alive.float()
        real_first.append(steps.cpu().numpy())
      real[uniform] = np.mean(real_first, axis=0)
    # SIM: Markov chain on o
    rng = np.random.default_rng(3)
    keys = np.array(sorted(kernel))
    mrep = int(round(NU * k))
    def corr(o1, uniform):
        true = int(round(o1 * k))
        if uniform:
            return rng.hypergeometric(true, k - true, k - mrep) / k
        return max(true - mrep, 0) / k                          # the top slots are true members
    sim = {}
    for uniform in (False, True):
        out = []
        for _ in range(2000):
            o = corr(rng.choice(first_step), uniform)
            t = 0
            while o >= 0.3 and t < L - 1:
                t += 1
                key = keys[np.abs(keys - o).argmin()]
                o = corr(rng.choice(kernel[key]), uniform)
            out.append(t if o < 0.3 else L - 1)
        sim[uniform] = np.array(out).reshape(-1, DRAWS).mean(axis=1)
    print(f"({n}, {k}, {p}) nu={NU}: A34 recorded median {a34:.0f}")
    for uniform, name in ((False, "TOP slots (A34's model)"), (True, "UNIFORM random slots")):
        print(f"   {name}: REAL median {statistics.median(real[uniform]):.1f} "
              f"{np.percentile(real[uniform], [25, 75]).round(1).tolist()}   "
              f"SIM median {np.median(sim[uniform]):.1f} {np.percentile(sim[uniform], [25, 75]).round(1).tolist()}", flush=True)
