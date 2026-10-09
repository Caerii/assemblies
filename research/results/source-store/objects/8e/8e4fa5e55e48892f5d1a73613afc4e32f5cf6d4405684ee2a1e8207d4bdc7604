"""Exploratory (not registered): the TWO-CHAIN REDUCTION of replay, measured by clamping.
Random-successor reuse, (10000, 75, 0.48), tau = 67, rho = 0.05 (judged cell; seeds 980-999),
U = 10, 30, 40, 50; a second cell by argv as probe_basin.py. Replay frozen, masked, bias zero:
a replay step is a deterministic function of the winner set, so a Markov chain on a coarse
state approximates it.

The map. At a random position (sequence q, element j) of each brain, an input of k neurons is
clamped: a fraction a from the current token T, a fraction b from a FAN INTRUDER R (a token that
follows another occurrence of T's predecessor word, of a different word than T -- the rival the
previous step's fan admits), the rest random. One projection, then read:
  a'   overlap with T's successor (the correct chain)
  b'   overlap with R's successor (the intruder's own chain)
  f'   max overlap with any fan successor of T's word (a fresh intruder seeded now)
  ok   word-level read-out correct (as every registration scores it)
on a grid of (a, b), ten positions per brain per point.

The chain. Step 1 is measured from real cues (a half of the first token); from step 2 the state
(a, b) moves by sampling the measured outcome at the nearest grid point, the next intruder
being the larger of b' (chain) and f' (seed). Reliability = P(every step ok over 15 steps), set
against the observed replay (probe_fan.log). The ONE-variable chain uses only b = 0 clamps (the
state is a alone, as the earlier overlap map) -- does it miss the edge?"""
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import numpy as np
import torch
from research.experiments import memory_reuse_grammar as rg, memory_threshold_law as tl, memory_load_law as ml
from research.experiments import memory_load_drift as md, memory_pattern_efficiency as pe
from research.experiments.seq_capacity_scaling import seeds_for, to_i32
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.core.torch_engine._memory import AssemblyMemory

seeds = list(range(980, 1000))
n, k, p, tau = 10000, 75, 0.48, 67
USES = (10, 30, 40, 50)
if len(sys.argv) > 1:   # n k p tau U...
    n, k, p, tau = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    USES = tuple(int(u) for u in sys.argv[5:])
B = len(seeds)
dev = "cuda"
LEN = rg.LENGTH
A_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
B_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
POS = 10            # positions per brain per grid point
SIMS = 20000
rng = np.random.default_rng(7)
print(f"cell ({n}, {k}, {p}), tau {tau}", flush=True)


def chain(init, table, two):
    """Simulated reliability: init [(a, b, ok)], table {(ag, bg): [(a', b', f', ok)]}."""
    keys = np.array(list(table.keys()))
    ok_all = 0
    for _ in range(SIMS):
        a, b, ok = init[rng.integers(len(init))]
        if not ok:
            continue
        good = True
        for _s in range(2, LEN):
            bb = b if two else 0.0
            g = keys[np.argmin((keys[:, 0] - a) ** 2 + (keys[:, 1] - bb) ** 2)]
            out = table[(g[0], g[1])]
            a2, b2, f2, ok2 = out[rng.integers(len(out))]
            if not ok2:
                good = False
                break
            a, b = a2, max(b2, f2)
        ok_all += good
    return ok_all / SIMS


for U in USES:
    M = max(1, round(0.05 * ml.unit(n, k, p) / LEN)); L = M * LEN; V = round(L / U)
    mem = AssemblyMemory(seeds_for(seeds), n, k, p, beta=round(tl.theta(n, k, p), 5), w_max=pe.W_MAX,
                         norm_init=True, rounds=1, strength=ml.STRENGTH, max_items=4, device=dev,
                         bias_decay=math.exp(-1 / tau))
    words = [rg.walks(sd, M, V, None, L * 1000 + U) for sd in seeds]
    seqs = [mem.store_sequence([[to_i32(_seeding.fnv1a_pair_seed(sd, f"w{int(words[i][q, e])}", "A"))
                                 for i, sd in enumerate(seeds)] for e in range(LEN)])[0] for q in range(M)]
    allst = torch.cat(seqs, 0).long()                                   # [L, B, k]
    allnp = allst.cpu().numpy()
    wflat = [w.reshape(-1) for w in words]
    wordof = torch.tensor(np.stack(wflat, 1), device=dev).long()
    occ = []
    for i in range(B):
        d = {}
        for t, w in enumerate(wflat[i]):
            d.setdefault(int(w), []).append(t)
        occ.append(d)
    area, fib = mem.area, mem.fiber
    saved = area.bias.clone()
    ar = torch.arange(B, device=dev)

    def step(win):
        area.bias = torch.zeros(B, n, device=dev)
        area.winners = win
        x = area.project(1, [fib], freeze=True, mask_bias=False)
        hot = torch.zeros(B, n, device=dev); hot.scatter_(1, x.long(), 1.0)
        return hot

    def ovl(hot, toks):
        """overlap of each brain's winners with its token toks[i] -> [B]"""
        t = torch.as_tensor(toks, device=dev).long()
        return torch.gather(hot, 1, allst[t, ar]).sum(1) / k

    def fanmax(hot, fans):
        out = []
        for i in range(B):
            if not fans[i]:
                out.append(0.0); continue
            t = torch.as_tensor(fans[i], device=dev).long()
            out.append(float(hot[i, allst[t, i]].sum(1).max()) / k)
        return np.array(out)

    def correct(hot, want):
        ov = torch.gather(hot.unsqueeze(0).expand(L, B, n), 2, allst).sum(2)
        return (wordof[ov.argmax(0), ar] == wordof[torch.as_tensor(want, device=dev).long(), ar]).cpu().numpy()

    def fan_of(i, t_cur, exclude):
        w = int(wflat[i][t_cur])
        return [t + 1 for t in occ[i][w] if t % LEN <= LEN - 2 and t + 1 != exclude]

    # step 1 from real cues
    init = []
    for q in range(0, M, 2):
        win = torch.stack([md.cue(seqs[q][0, i], sd, L * 100_000 + q, k, dev) for i, sd in enumerate(seeds)]).long()
        hot = step(win)
        t0 = q * LEN
        a1 = ovl(hot, [t0 + 1] * B).cpu().numpy()
        f1 = fanmax(hot, [fan_of(i, t0, t0 + 1) for i in range(B)])
        ok = correct(hot, [t0 + 1] * B)
        init += list(zip(a1, f1, ok))
    # the clamped map
    table = {}
    for ag in A_GRID:
        for bg in B_GRID:
            if ag + bg > 1.0 + 1e-9:
                continue
            outs = []
            for _ in range(POS):
                wins, cur, nxt, ri, rn, fans = [], [], [], [], [], []
                for i in range(B):
                    while True:
                        q = int(rng.integers(M)); j = int(rng.integers(1, LEN - 1))
                        tc = q * LEN + j
                        cands = [t for t in occ[i][int(wflat[i][tc - 1])]
                                 if t != tc - 1 and t % LEN <= LEN - 3 and wflat[i][t + 1] != wflat[i][tc]]
                        if cands:
                            t2 = cands[int(rng.integers(len(cands)))]
                            break
                    T, R = allnp[tc, i], allnp[t2 + 1, i]
                    na, nb = int(round(ag * k)), int(round(bg * k))
                    Ronly = np.setdiff1d(R, T)
                    pick_a = rng.choice(T, na, replace=False)
                    pick_b = rng.choice(Ronly, min(nb, len(Ronly)), replace=False)
                    used = set(T.tolist()) | set(R.tolist())
                    rest = []
                    while len(rest) < k - len(pick_a) - len(pick_b):
                        c = int(rng.integers(n))
                        if c not in used:
                            used.add(c); rest.append(c)
                    wins.append(np.concatenate([pick_a, pick_b, np.array(rest, dtype=np.int64)]))
                    cur.append(tc); nxt.append(tc + 1); ri.append(t2 + 1); rn.append(t2 + 2)
                    fans.append(fan_of(i, tc, tc + 1))
                hot = step(torch.as_tensor(np.stack(wins), device=dev).long())
                a2 = ovl(hot, nxt).cpu().numpy(); b2 = ovl(hot, rn).cpu().numpy()
                f2 = fanmax(hot, fans); ok = correct(hot, nxt)
                outs += list(zip(a2, b2, f2, ok))
            table[(ag, bg)] = outs
    area.bias = saved
    two, one = chain(init, table, True), chain(init, table, False)
    i1 = np.array(init)
    print(f"\nU = {U} (V {V}): step 1 a {i1[:, 0].mean():.3f} fan seed {i1[:, 1].mean():.3f} ok {i1[:, 2].mean():.3f}"
          f" | predicted reliability: two-chain {two:.3f}, one-variable {one:.3f}", flush=True)
    print("  map (mean a' / b' / f' / ok) at selected points:", flush=True)
    for ag in (0.5, 0.7, 0.9):
        line = []
        for bg in (0.0, 0.1, 0.2, 0.3):
            if (ag, bg) not in table:
                continue
            o = np.array(table[(ag, bg)])
            line.append(f"b{bg:.1f}: {o[:, 0].mean():.2f}/{o[:, 1].mean():.2f}/{o[:, 2].mean():.2f}/{o[:, 3].mean():.2f}")
        print(f"   a {ag:.1f}  " + "  ".join(line), flush=True)
    del mem, seqs, allst
    torch.cuda.empty_cache()
