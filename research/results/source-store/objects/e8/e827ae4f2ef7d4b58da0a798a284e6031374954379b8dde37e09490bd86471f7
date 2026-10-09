"""Exploratory probe (disclosed): what breaks Hebbian sequence replay at L_H?
(4000, 60, 0.5), beta = theta, strength 0, one round per element, 5 brains (seeds 900-904)."""
import math, sys
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from research.experiments import memory_sequences as sq

n, k, p = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
Ls = [int(x) for x in sys.argv[4].split(",")]
strength = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
tau = float(sys.argv[6]) if len(sys.argv) > 6 else None
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
B = 5
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v
seeds = [i32(_seeding.fnv1a_pair_seed(900 + b, "A", "A")) for b in range(B)]

for L in Ls:
    els = [[i32(_seeding.fnv1a_pair_seed(900 + b, f"h{e}", "A")) for b in range(B)] for e in range(L)]
    mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                         strength=strength, max_items=4,
                         bias_decay=None if tau is None else math.exp(-1.0 / tau))
    st, _ = mem.store_sequence(els)                     # [L, B, k]
    # write side: win counts, state overlaps, loops
    wins = torch.zeros(B, n, device=st.device)
    for e in range(L):
        wins.scatter_add_(1, st[e], torch.ones(B, k, device=st.device))
    lam = L * k / n
    maxwin = wins.max(dim=1).values.tolist()
    var_ratio = (wins.var(dim=1) / lam).tolist()      # binomial ~ (1 - k/n)
    # max overlap of each state with any earlier state
    loops, mean_pair = "-", "-"
    # replay side: teacher-forced one step from full and half, chained replay
    full = torch.stack([sq._overlap(mem.recall(st[t], rounds=1), st[t + 1]) for t in range(L - 1)])  # [L-1, B]
    half = torch.stack([sq._overlap(mem.recall(st[t][:, :k // 2], rounds=1), st[t + 1]) for t in range(L - 1)])
    x = st[0][:, :k // 2]
    alive = torch.ones(B, dtype=torch.bool, device=st.device)
    first = torch.full((B,), L - 1, device=st.device)
    for t in range(1, L):
        x = mem.recall(x, rounds=1)
        ok = sq._overlap(x, st[t]) >= 0.3
        newly = alive & ~ok
        first = torch.where(newly, torch.full_like(first, t), first)
        alive &= ok
    print(f"L={L} lam(win)={lam:.2f} maxwin={maxwin} var/lam={[round(v, 2) for v in var_ratio]}")
    fq = torch.quantile(full.flatten(), torch.tensor([0.01, 0.1, 0.5], device=full.device)).tolist()
    hq = torch.quantile(half.flatten(), torch.tensor([0.01, 0.1, 0.5], device=half.device)).tolist()
    print(f"   one-step full q01/q10/q50 {[round(v, 3) for v in fq]}  half {[round(v, 3) for v in hq]}  "
          f"half<0.3 at {int((half < 0.3).sum())} of {half.numel()} steps")
    bad = (half < 0.3).nonzero()[:, 0].tolist()
    print(f"   positions of half<0.3 (first 20, as multiples of n/k): {[round(t * k / n, 2) for t in bad[:20]]}")
    print(f"   first miss: {first.tolist()}", flush=True)
