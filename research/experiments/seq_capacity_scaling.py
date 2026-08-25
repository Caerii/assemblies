"""How does the M-ceiling scale with n?  Registered in PREREG_capacity_scaling.md.

The ceiling study closed with "What is NOT established: how any of this scales
with n", blocked on compute. This runs the sweep on `batched_project_hashed`,
which trains 16 independent brains at once with a generated connectome and is
verified against `numpy_sparse` on all four substrate arms.

PROTOCOL DIFFERENCE, restated here because the numbers must not be read as
comparable to `PREREG_substrate_ceiling.md`: that study cued each assembly with
a STIMULUS whose own fiber also learns. This path has no stimulus fiber, so the
cue is a fixed initial winner set. The deliverable is the SCALING with n.

    python research/experiments/seq_capacity_scaling.py [--smoke]

`--smoke` checks the API only. Its numbers are VOID.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_assemblies.core.numpy_engine import _seeding                # noqa: E402
from neural_assemblies.core.torch_engine._batched import (              # noqa: E402
    batched_project_hashed)
from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from _substrate import ceiling_from_curve                               # noqa: E402

DEV = "cuda"
K = 60
P = 0.50
T = 8
BETA = 0.10
W_MAX = 20.0
HALF_BAR = 0.50
DISTINCT_GATE = 3.0
NS = (1000, 2000, 4000, 8000)
MS = (8, 16, 32, 64, 128, 256)
NBRAIN = 16
RECALL_SAMPLE = 32
PAIR_SAMPLE = 200
ARMS = {"B": dict(norm_init=True, synaptic_scaling=False),
        "G": dict(norm_init=True, synaptic_scaling=True)}


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def seeds_for(nbrain):
    return [to_i32(_seeding.fnv1a_pair_seed(42 + b, "A", "A"))
            for b in range(nbrain)]


def _fill(state, n):
    """rows/n -- the fraction of the area that has EVER fired.

    CAP3's censoring guard reads this. `colmask` has a bit set for every round
    a neuron won, so a nonzero word means it fired at least once.
    """
    cm = state.get("colmask")
    if cm is None:
        return float("nan")
    ever = (cm != 0).any(dim=1)                 # [B, n] over the word axis
    return (ever.sum(dim=1).float() / n).cpu().numpy()


def run_cell(n, arm, m_max, nbrain, rng):
    """Train up to `m_max` assemblies, checkpointing at every M in MS."""
    cfg = ARMS[arm]
    sd = seeds_for(nbrain)
    total_rounds = m_max * T
    state = None
    stored = []
    out = {}
    for a in range(m_max):
        cue = torch.from_numpy(np.stack(
            [np.sort(rng.choice(n, K, replace=False)) for _ in range(nbrain)]
        )).to(DEV)
        res = batched_project_hashed(
            n, K, P, sd, cue, T, beta=BETA, w_max=W_MAX,
            state=state, max_rounds=total_rounds, return_state=True, **cfg)
        win, state = res
        stored.append(win)
        M = a + 1
        if M in MS:
            out[M] = measure(n, arm, sd, state, stored, nbrain, rng, cfg)
    return out


def measure(n, arm, sd, state, stored, nbrain, rng, cfg):
    M = len(stored)
    St = torch.stack(stored)                    # [M, B, K]

    # -- distinctness: EXACT duplicates, which spread is nearly blind to
    key = torch.sort(St, dim=2).values
    dist = []
    for b in range(nbrain):
        rows = [tuple(key[a, b].tolist()) for a in range(M)]
        dist.append(len(set(rows)) / M)

    # -- pairwise overlap on a sample of pairs
    npair = min(PAIR_SAMPLE, M * (M - 1) // 2) if M > 1 else 0
    pw = np.zeros(nbrain)
    if npair:
        ia = rng.integers(0, M, npair)
        ib = rng.integers(0, M, npair)
        keep = ia != ib
        ia, ib = ia[keep], ib[keep]
        for b in range(nbrain):
            ov = [len(set(St[x, b].tolist()) & set(St[y, b].tolist())) / K
                  for x, y in zip(ia, ib)]
            pw[b] = float(np.mean(ov)) if ov else 0.0
    chance = K / n
    pw_x = pw / chance

    # -- half-cue rank-1, frozen (the probe equivalent)
    samp = rng.choice(M, min(RECALL_SAMPLE, M), replace=False)
    hits = np.zeros(nbrain)
    for a in samp:
        half = St[a][:, : K // 2].contiguous()
        rec = batched_project_hashed(
            n, K, P, sd, half, T, beta=BETA, w_max=W_MAX, state=state,
            freeze=True, **cfg)
        mask = torch.zeros(nbrain, n, dtype=torch.bool, device=DEV)
        mask.scatter_(1, rec, True)
        ov = torch.stack([mask.gather(1, St[x].long()).sum(1)
                          for x in range(M)])          # [M, B]
        hits += (ov.argmax(dim=0) == a).cpu().numpy()
    rank1 = hits / len(samp)
    return dict(rank1=rank1.tolist(), pairwise_x=pw_x.tolist(),
                distinct=dist, fill=_fill(state, n).tolist())


def gated(cell):
    r = ensemble_from_values(cell["rank1"])
    x = ensemble_from_values(cell["pairwise_x"])
    d = ensemble_from_values(cell["distinct"])
    ok = (x.high <= DISTINCT_GATE and d.low >= 0.9)
    return (r.mean if ok else 0.0), r, x, d


def main():
    global MS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ns = (1000, 2000) if args.smoke else NS
    ms = (4, 8) if args.smoke else MS
    nb = 4 if args.smoke else NBRAIN
    MS = ms
    if args.smoke:
        print("*** SMOKE: API only. THESE NUMBERS ARE VOID. ***")

    print(f"k={K} p={P} T={T} beta={BETA} w_max={W_MAX} brains={nb}")
    for n in ns:
        print(f"  n={n:>6}  kp={K*P:.1f} vs floor 3ln n={3*math.log(n):.1f}"
              f"  {'IN REGIME' if K*P >= 3*math.log(n) else 'OUT OF REGIME'}")

    res, t0 = {}, time.perf_counter()
    for arm in ARMS:
        for n in ns:
            rng = np.random.default_rng(1234)
            cells = run_cell(n, arm, max(ms), nb, rng)
            res[f"{arm}/{n}"] = cells
            curve = []
            for M in sorted(cells):
                g, r, x, d = gated(cells[M])
                curve.append((M, g))
                print(f"    {arm} n={n:>5} M={M:>4}  rank1 {r.mean:.3f} "
                      f"pw/chance {x.mean:6.2f}  distinct {d.mean:.3f}  "
                      f"fill {np.mean(cells[M]['fill']):.3f}  gated {g:.3f}")
            c = ceiling_from_curve(curve, threshold=HALF_BAR)
            fill_at = np.mean(cells[max(cells)]["fill"])
            print(f"    {arm} n={n:>5}  CEILING {c}  fill@max {fill_at:.3f}"
                  f"  {'CENSORED' if fill_at >= 0.95 else 'ok'}")
            res[f"{arm}/{n}/ceiling"] = dict(
                m_star=c.m_star, supported=bool(c.supported),
                fill=float(fill_at), censored=bool(fill_at >= 0.95))
    print(f"\n  elapsed {time.perf_counter()-t0:.1f}s")

    print("\n--- CAP2: fit log M* = a + b log n over UNCENSORED n ---")
    for arm in ARMS:
        pts = [(n, res[f"{arm}/{n}/ceiling"]) for n in ns
               if f"{arm}/{n}/ceiling" in res]
        good = [(n, c["m_star"]) for n, c in pts
                if c["supported"] and not c["censored"] and c["m_star"] > 0]
        if len(good) < 3:
            print(f"    {arm}: only {len(good)} uncensored supported n "
                  f"-- NOT ANSWERED at this k, p. No line is fitted.")
            continue
        x = np.log(np.array([g[0] for g in good], dtype=float))
        y = np.log(np.array([g[1] for g in good], dtype=float))
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        se = (np.sqrt((resid**2).sum() / max(len(x) - 2, 1))
              / max(np.sqrt(((x - x.mean())**2).sum()), 1e-12))
        lo, hi = b - 1.96 * se, b + 1.96 * se
        verdict = ("EXTENSIVE (CI contains 1)" if lo <= 1.0 <= hi
                   else ("SUBLINEAR" if hi < 1.0 else "SUPERLINEAR"))
        print(f"    {arm}: b = {b:.3f} [{lo:.3f}, {hi:.3f}] over "
              f"{len(good)} points -> {verdict}")

    if not args.smoke:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "capacity_scaling_results.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
