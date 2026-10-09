"""Exploratory probe (disclosed): is noisy replay Markov in the overlap o?
(n, k, 0.5), tau 33, L = 400, 20 probe brains (seeds 940-959), uniform noise.
Kernel from REAL chain transitions (o_in -> pre-noise o_out) at the training nus;
a Markov chain on o with that kernel predicts the held-out nu's horizon."""
import math, statistics, sys
import numpy as np
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory
from research.experiments import memory_sequences as sq

n, k = int(sys.argv[1]), int(sys.argv[2])
TRAIN = [float(v) for v in sys.argv[3].split(",")]
TEST = float(sys.argv[4])
p, L, TAU, B, DRAWS = 0.5, 400, 33, 20, 4
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v
beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
seeds = [i32(_seeding.fnv1a_pair_seed(940 + b, "A", "A")) for b in range(B)]
els = [[i32(_seeding.fnv1a_pair_seed(940 + b, f"m{e}", "A")) for b in range(B)] for e in range(L)]
mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                     strength=0.5, max_items=4, bias_decay=math.exp(-1.0 / TAU))
st, _ = mem.store_sequence(els)
g = torch.Generator(device="cuda").manual_seed(21)

def chains(nu, draws):
    """Real replay; returns first-miss per brain (mean of draws) and (o_in, o_out) pairs."""
    m = int(round(nu * k))
    firsts, pairs = [], []
    for _ in range(draws):
        x = st[0][:, :k // 2]
        alive = torch.ones(B, dtype=torch.bool, device="cuda")
        steps = torch.zeros(B, device="cuda")
        for j in range(1, L):
            o_in = sq._overlap(x, st[j - 1]) if j > 1 else None
            y = mem.recall(x, rounds=1)
            o_out = sq._overlap(y, st[j])
            if o_in is not None:
                a = alive.cpu().numpy()
                pairs += list(zip(o_in.cpu().numpy()[a], o_out.cpu().numpy()[a]))
            perm = torch.argsort(torch.rand(B, k, device="cuda", generator=g), dim=1)
            x = torch.gather(y, 1, perm)
            if m:
                x[:, :m] = torch.randint(0, n, (B, m), device="cuda", generator=g)
            alive &= sq._overlap(x, st[j]) >= 0.3
            steps += alive.float()
        firsts.append(steps.cpu().numpy())
    return np.mean(firsts, axis=0), np.array(pairs)

train_pairs = []
for nu in TRAIN:
    _, pr = chains(nu, DRAWS)
    train_pairs.append(pr)
pairs = np.concatenate(train_pairs)
real_test, test_pairs = chains(TEST, DRAWS)
# first-step distribution (half cue)
first_step = sq._overlap(mem.recall(st[0][:, :k // 2], rounds=1), st[1]).cpu().numpy()

# empirical kernel: bin o_in to the grid of k-ths
bins = {}
for oi, oo in pairs:
    bins.setdefault(int(round(oi * k)), []).append(oo)
keys = np.array(sorted(bins))
def kernel(true_in):
    key = keys[np.abs(keys - true_in).argmin()]
    return bins[key]
rng = np.random.default_rng(4)
m = int(round(TEST * k))
def corrupt(o_out):
    true = int(round(o_out * k))
    return rng.hypergeometric(true, k - true, k - m)            # true members kept
sims = []
for _ in range(4000):
    t_in = corrupt(rng.choice(first_step)); t = 1
    if t_in / k < 0.3:
        sims.append(0); continue
    while t < L - 1:
        t_in = corrupt(rng.choice(kernel(t_in)));
        if t_in / k < 0.3:
            break
        t += 1
    sims.append(t)
sims = np.array(sims).reshape(-1, DRAWS).mean(axis=1)
# held-out check of the kernel's conditional mean on the test chains
dev = []
for oi, oo in test_pairs:
    dev.append(oo - np.mean(kernel(int(round(oi * k)))))
print(f"({n}, {k}) train nu={TRAIN} test nu={TEST}: kernel from {len(pairs)} real transitions, bins {len(keys)}")
print(f"   REAL test median {statistics.median(real_test):.1f} {np.percentile(real_test, [25, 75]).round(1).tolist()}"
      f"   SIM (real kernel) median {np.median(sims):.1f} {np.percentile(sims, [25, 75]).round(1).tolist()}")
print(f"   held-out conditional-mean residual: mean {np.mean(dev):+.4f} sd {np.std(dev):.4f} (n={len(dev)})")
lo = np.array(test_pairs)
for a in (0.4, 0.5, 0.6, 0.7):
    sel = (lo[:, 0] >= a) & (lo[:, 0] < a + 0.1)
    if sel.sum() > 30:
        print(f"   o_in {a:.1f}-{a + 0.1:.1f}: test E[o_out] {lo[sel, 1].mean():.3f}  kernel {np.mean([np.mean(kernel(int(round(v * k)))) for v in lo[sel, 0]]):.3f}  n={int(sel.sum())}")
