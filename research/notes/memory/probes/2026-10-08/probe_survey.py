"""Exploratory survey (disclosed): replay RELIABILITY against load, tau = 64.
10 probe brains (seeds 960-969) per cell; for each L on a geometric ladder a fresh
sequence of L elements is stored per brain and replayed noiselessly from a uniformly
random half of element 0. Reported per L: the fraction of brains replaying all L - 1
steps (reliability) and the mean replay fraction (A29's statistic).
Held out, never run here: (6000, 90, 0.4) and (12000, 80, 0.5)."""
import json, math, sys, time
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from research.experiments import memory_sequences as sq

CELLS = [(2000, 60, 0.5), (4000, 60, 0.5), (8000, 60, 0.5), (4000, 120, 0.5),
         (8000, 120, 0.5), (8000, 120, 0.25), (16000, 120, 0.5), (4000, 30, 0.5)]
if len(sys.argv) > 1:
    CELLS = [tuple(float(x) if i == 2 else int(x) for i, x in enumerate(c.split(":"))) for c in sys.argv[1].split(",")]
TAU, B, MATCH = 64, 10, 0.3
HELD_OUT = {(6000, 90, 0.4), (12000, 80, 0.5)}
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v

def run(n, k, p, L, seeds):
    beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
    els = [[i32(_seeding.fnv1a_pair_seed(960 + b, f"s{L}e{e}", "A")) for b in range(B)] for e in range(L)]
    mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                         strength=0.5, max_items=4, bias_decay=math.exp(-1.0 / TAU))
    st, _ = mem.store_sequence(els)
    g = torch.Generator(device="cuda").manual_seed(L)
    perm = torch.argsort(torch.rand(B, k, device="cuda", generator=g), dim=1)
    x = torch.gather(st[0], 1, perm)[:, :k // 2]
    alive = torch.ones(B, dtype=torch.bool, device="cuda")
    steps = torch.zeros(B, device="cuda")
    for j in range(1, L):
        x = mem.recall(x, rounds=1)
        alive &= sq._overlap(x, st[j]) >= MATCH
        steps += alive.float()
        if not bool(alive.any()):
            break
    del mem
    torch.cuda.empty_cache()
    return steps.cpu().tolist()

out = {}
for n, k, p in CELLS:
    assert (n, k, p) not in HELD_OUT
    seeds = [i32(_seeding.fnv1a_pair_seed(960 + b, "A", "A")) for b in range(B)]
    unit = n * n * p / (k * math.log(n))                 # L at rho = 1
    rows = []
    L = max(16, int(0.04 * unit))
    zeros = 0
    while True:
        t0 = time.time()
        steps = run(n, k, p, L, seeds)
        full = sum(s >= L - 1 for s in steps) / B
        frac = sum(steps) / (B * (L - 1))
        rows.append({"L": L, "rho": L / unit, "full": full, "frac": frac})
        print(f"({n}, {k}, {p}) L={L} rho={L / unit:.3f}: full {full:.1f}  mean fraction {frac:.2f}  [{time.time() - t0:.0f}s]", flush=True)
        zeros = zeros + 1 if frac < 0.2 else 0
        if zeros >= 2 or L > 0.6 * unit:
            break
        L = int(round(L * 2 ** 0.25))
    out[f"{n}/{k}/{p:g}"] = rows
    json.dump(out, open("survey.json", "w"), indent=1)
