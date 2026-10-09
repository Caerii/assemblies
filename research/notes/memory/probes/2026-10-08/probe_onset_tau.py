"""Exploratory probe (disclosed): the write-time collapse onset at A27's six cells.
One long Hebbian write per brain (5 brains, seeds 900-904); writing is causal, so the
first element whose state overlaps an earlier one >= 0.3 is the collapse onset for any L."""
import math, sys, statistics
import torch
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

CELLS = [(2000, 60, 0.5, 38), (4000, 120, 0.5, 49), (4000, 60, 0.5, 193), (4000, 60, 0.25, 97),
         (8000, 60, 0.5, 939), (4000, 30, 0.5, 562)]
if len(sys.argv) > 1:
    CELLS = [tuple(float(x) if i == 2 else int(x) for i, x in enumerate(c.split(":"))) for c in sys.argv[1].split(",")]
B = 5
def i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v

TAUS = [float(t) for t in sys.argv[2].split(",")] if len(sys.argv) > 2 else [None]
STRENGTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
for (n, k, p, LH), tau in [(c, t) for c in CELLS for t in TAUS]:
    beta = round(math.sqrt((1 - p) * math.log(n) / (p * k)), 5)
    L = int(min(3.5 * LH, 24_000_000 // n))
    seeds = [i32(_seeding.fnv1a_pair_seed(900 + b, "A", "A")) for b in range(B)]
    els = [[i32(_seeding.fnv1a_pair_seed(900 + b, f"h{e}", "A")) for b in range(B)] for e in range(L)]
    mem = AssemblyMemory(seeds, n, k, p, beta=beta, w_max=20.0, norm_init=True, rounds=1,
                         strength=STRENGTH, max_items=4,
                         bias_decay=None if tau is None or tau == 0 else (None if tau > 1e8 else math.exp(-1.0 / tau)))
    if tau == 0:
        mem.area.bias_decay = 0.0
    st, _ = mem.store_sequence(els)                     # [L, B, k]
    onset, disp_at = [], []
    for b in range(B):
        X = torch.zeros(L, n, device=st.device, dtype=torch.float16)
        X.scatter_(1, st[:, b], 1.0)
        G = (X @ X.T).float() / k                       # [L, L]
        G = torch.tril(G, -1)
        loop = (G.max(dim=1).values >= 0.3).nonzero()
        e_c = int(loop[0]) if len(loop) else L
        onset.append(e_c)
        # overdispersion of win counts in the first e_c / 2 elements
        h = max(e_c // 2, 2)
        w = X[:h].float().sum(0)
        disp_at.append(round(float(w.var() / max(w.mean(), 1e-9)), 2))
    med = statistics.median(onset)
    print(f"tau={tau} ({n}, {k}, {p}) beta={beta} A27 L_H={LH}  onsets={onset} median={med}  "
          f"onset/L_H={med / LH:.2f}  var/mean at onset/2={disp_at}", flush=True)
