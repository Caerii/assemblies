import math, torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from research.experiments import memory_sequences as sq
n, k, p, L, B = 4000, 60, 0.5, 400, 20
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
seeds = [i32(_seeding.fnv1a_pair_seed(920 + b, "A", "A")) for b in range(B)]
els = [[i32(_seeding.fnv1a_pair_seed(920 + b, f"r{e}", "A")) for b in range(B)] for e in range(L)]
mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                     strength=0.5, max_items=4, bias_decay=math.exp(-1.0 / 33))
st, _ = mem.store_sequence(els)
y = mem.recall(st[10][:, :k // 2], rounds=1)            # [B, k]
true = (y.unsqueeze(2) == st[11].unsqueeze(1)).any(2).float()   # [B, k] slot is a true member
print("index-sorted ascending?", bool((y[:, 1:] > y[:, :-1]).all()))
print("true-member rate by slot sixth:", [round(float(true[:, i*10:(i+1)*10].mean()), 2) for i in range(6)])
# drive order check
raw = torch.zeros(B, n, device="cuda"); mem.fiber.contribute(raw, st[10][:, :k // 2])
d = torch.gather(raw, 1, y)
print("drive-sorted descending?", bool((d[:, 1:] <= d[:, :-1] + 1e-9).all()),
      " mean drive first 6 slots", round(float(d[:, :6].mean()), 4), " last 6", round(float(d[:, -6:].mean()), 4))
