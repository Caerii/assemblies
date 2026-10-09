"""Exploratory probe (disclosed): position vs structured errors in the noisy replay.
(4000, 60, 0.5), tau 33, L = 400, 20 probe brains (seeds 920-939), nu = 0.1."""
import math
import numpy as np
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from research.experiments import memory_sequences as sq

n, k, p, L, TAU, NU, B, DRAWS = 4000, 60, 0.5, 400, 33, 0.1, 20, 8
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
seeds = [i32(_seeding.fnv1a_pair_seed(920 + b, "A", "A")) for b in range(B)]
els = [[i32(_seeding.fnv1a_pair_seed(920 + b, f"r{e}", "A")) for b in range(B)] for e in range(L)]
mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                     strength=0.5, max_items=4, bias_decay=math.exp(-1.0 / TAU))
st, _ = mem.store_sequence(els)
g = torch.Generator(device="cuda").manual_seed(5)

def fill(t, o):
    m = int(round(o * k))
    perm = torch.argsort(torch.rand(B, k, device="cuda", generator=g), dim=1)
    x = torch.gather(st[t], 1, perm)
    if m < k:
        x[:, m:] = torch.randint(0, n, (B, k - m), device="cuda", generator=g)
    return x

# teacher-forced F(0.7) and F(1.0) at EVERY position
F7 = torch.stack([sq._overlap(mem.recall(fill(t, 0.7), rounds=1), st[t + 1]) for t in range(L - 1)])  # [L-1, B]
F10 = torch.stack([sq._overlap(mem.recall(st[t], rounds=1), st[t + 1]) for t in range(L - 1)])
bins = [(0, 10), (10, 20), (20, 40), (40, 70), (70, 140), (140, 270), (270, 399)]
print("teacher-forced F(0.7) by position:", [(a, b, round(float(F7[a:b].mean()), 3)) for a, b in bins])
print("teacher-forced F(1.0) by position:", [(a, b, round(float(F10[a:b].mean()), 3)) for a, b in bins])
print("weakest positions F(0.7) (brain-mean):", torch.topk(-F7.mean(1), 8).indices.tolist(),
      [round(-v, 3) for v in torch.topk(-F7.mean(1), 8).values.tolist()])

# real noisy chains: o_t by step, raw first misses
O = torch.full((DRAWS, L - 1, B), float("nan"), device="cuda")
first = []
for d in range(DRAWS):
    x = st[0][:, :k // 2]
    alive = torch.ones(B, dtype=torch.bool, device="cuda")
    fm = torch.full((B,), L - 1.0, device="cuda")
    for j in range(1, L):
        x = mem.recall(x, rounds=1)
        mrep = int(round(NU * k))
        x = x.clone(); x[:, :mrep] = torch.randint(0, n, (B, mrep), device="cuda", generator=g)
        o = sq._overlap(x, st[j])
        O[d, j - 1] = torch.where(alive, o, torch.full_like(o, float("nan")))
        newly = alive & (o < 0.3)
        fm = torch.where(newly, torch.full_like(fm, j - 1.0), fm)
        alive &= o >= 0.3
    first.append(fm)
first = torch.stack(first).cpu().numpy()          # [DRAWS, B]
print("raw first-miss steps: quartiles", np.percentile(first, [10, 25, 50, 75, 90]).round(1).tolist(),
      " CV", round(float(first.std() / first.mean()), 2))
print("   histogram (bins of 10 steps up to 100):", np.histogram(first, bins=range(0, 110, 10))[0].tolist())
Om = torch.nanmean(O, dim=(0, 2)).cpu().numpy()
print("real chain mean overlap (alive chains) at steps 1,2,4,8,16,24,32,40,48:",
      [round(float(Om[s - 1]), 3) for s in (1, 2, 4, 8, 16, 24, 32, 40, 48)])
# is the drop position-locked? correlation of brain's first miss with its weakest early position
