"""Exploratory probe (disclosed): ghost tracks in the noisy replay.
(4000, 60, 0.5), tau 33, L = 400, 20 probe brains (seeds 920-939), nu = 0.1 and 0."""
import math
import numpy as np
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

n, k, p, L, TAU, B = 4000, 60, 0.5, 400, 33, 20
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
seeds = [i32(_seeding.fnv1a_pair_seed(920 + b, "A", "A")) for b in range(B)]
els = [[i32(_seeding.fnv1a_pair_seed(920 + b, f"r{e}", "A")) for b in range(B)] for e in range(L)]
mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                     strength=0.5, max_items=4, bias_decay=math.exp(-1.0 / TAU))
st, _ = mem.store_sequence(els)
X = torch.zeros(B, L, n, device="cuda")
for e in range(L):
    X[:, e].scatter_(1, st[e], 1.0)
# the code's own structure: overlap of A_t with A_{t+d}
G = torch.einsum("bln,bmn->blm", X, X) / k                          # [B, L, L]
ac = [float(torch.diagonal(G, offset=d, dim1=1, dim2=2).mean()) for d in range(1, 141)]
top = np.argsort(ac)[::-1][:8]
print("code overlap A_t vs A_t+d: d=1..5", [round(v, 3) for v in ac[:5]],
      " largest at d =", [int(d) + 1 for d in top], [round(ac[d], 3) for d in top])
g = torch.Generator(device="cuda").manual_seed(9)
for NU in (0.1, 0.0):
    x = st[0][:, :k // 2]
    rows = []
    for j in range(1, 61):
        x = mem.recall(x, rounds=1)
        m = int(round(NU * k))
        if m:
            x = x.clone(); x[:, :m] = torch.randint(0, n, (B, m), device="cuda", generator=g)
        if j in (5, 10, 20, 30, 40, 50, 60):
            v = torch.zeros(B, n, device="cuda"); v.scatter_(1, x, 1.0)
            ov = torch.einsum("bn,bln->bl", v, X) / k                   # [B, L] overlap with each stored element
            true = ov[:, j].clone()
            ov[:, j] = -1
            best, arg = ov.max(dim=1)
            off = (arg - j).cpu().numpy()
            # how much of the state is explained by the true + best ghost
            rows.append(f"step {j}: true {float(true.mean()):.2f}  best other {float(best.mean()):.2f} "
                        f"at offsets {np.round(np.percentile(off, [25, 50, 75])).astype(int).tolist()}")
    print(f"nu={NU}:"); print("   " + "\n   ".join(rows), flush=True)
