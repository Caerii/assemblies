"""Exploratory probe (disclosed): is the noisy replay's wrong part a self-sustaining set?
(4000, 60, 0.5), tau 33, L = 400, 20 probe brains (seeds 920-939), nu = 0.1."""
import math
import numpy as np
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

n, k, p, L, TAU, B, NU = 4000, 60, 0.5, 400, 33, 20, 0.1
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
seeds = [i32(_seeding.fnv1a_pair_seed(920 + b, "A", "A")) for b in range(B)]
els = [[i32(_seeding.fnv1a_pair_seed(920 + b, f"r{e}", "A")) for b in range(B)] for e in range(L)]
mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                     strength=0.5, max_items=4, bias_decay=math.exp(-1.0 / TAU))
st, _ = mem.store_sequence(els)
g = torch.Generator(device="cuda").manual_seed(9)
def mask(ix):
    v = torch.zeros(B, n, dtype=torch.bool, device="cuda"); v.scatter_(1, ix, True); return v
x = st[0][:, :k // 2]
wrong, injected = {}, {}
for j in range(1, 61):
    y = mem.recall(x, rounds=1)
    m = int(round(NU * k))
    x = y.clone(); x[:, :m] = torch.randint(0, n, (B, m), device="cuda", generator=g)
    inj = mask(x[:, :m]) & ~mask(y)                 # neurons that are there only by injection
    w = mask(x) & ~mask(st[j])                       # the wrong part
    wrong[j], injected[j] = w, inj
C = mem.fiber.C.float()                              # [B, n, n] counts (pre, post)
pot_in = (C > 0).float().sum(dim=1)                  # potentiated in-degree per neuron [B, n]
for j in (10, 20, 30, 40, 50):
    w, w1, w5 = wrong[j], wrong[j + 1], wrong[j + 5] if j + 5 <= 60 else wrong[60]
    size = w.sum(1).float()
    per1 = ((w & w1).sum(1).float() / size.clamp_min(1)).mean()
    per5 = ((w & w5).sum(1).float() / size.clamp_min(1)).mean()
    chance = float(size.mean()) / n
    # what fraction of the next step's wrong part was driven, not injected
    driven = ((wrong[j + 1] & ~injected[j + 1]).sum(1).float() / wrong[j + 1].sum(1).float().clamp_min(1)).mean()
    # potentiated in-degree of the wrong neurons vs the area
    deg_w = (pot_in * w).sum(1) / size.clamp_min(1)
    # recurrent counts WITHIN the wrong set vs from the wrong set to random neurons
    within = torch.stack([C[b][w[b]][:, w[b]].mean() for b in range(B)]).mean()
    to_all = torch.stack([C[b][w[b]].mean() for b in range(B)]).mean()
    print(f"step {j}: |wrong| {float(size.mean()):.1f}  persists 1 step {float(per1):.2f}, 5 steps {float(per5):.2f} "
          f"(chance {chance:.3f})  driven (not injected) {float(driven):.2f}  "
          f"pot in-degree {float(deg_w.mean()):.0f} vs area {float(pot_in.mean()):.0f}  "
          f"counts within wrong {float(within):.4f} vs wrong->all {float(to_all):.4f}", flush=True)
