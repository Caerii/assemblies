"""How does the M-ceiling scale with n?  Registered in PREREG_capacity_scaling.md.

The ceiling study closed with "What is NOT established: how any of this scales
with n", blocked on compute. This runs the sweep on `batched_project_hashed`,
which trains 16 independent brains at once with a generated connectome and is
verified against `numpy_sparse` on all four substrate arms.

PROTOCOL. As registered: the area is inhibited between assemblies, and each
assembly is trained by firing its own STIMULUS every round alongside recurrence
-- `project({s: [AREA]}, {AREA: [AREA]})`. An earlier version substituted a
fixed initial winner set for the stimulus and that removed the ANCHOR, not just
the fiber: rank1 was 0.19 at M=4 (Amendment 1). The stimulus fiber is now
hash-generated and priced as the engine prices it.

ONE REMAINING DIFFERENCE from `PREREG_substrate_ceiling.md`: the stimulus fiber
here does not itself learn a per-cell connectome -- it stores pre-summed input,
which is what the engine stores, but its base is generated rather than drawn in
RNG order. Absolute ceilings are therefore not bit-comparable to that study's
M* = 41 / 104 at n=2000; the SCALING with n is the deliverable.

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
        # INHIBITED between assemblies: the area starts with no winners, so
        # round 1 is stimulus-only and recurrence joins from round 2. That is
        # `brain.inhibit_areas([AREA])` followed by T rounds of
        # `project({s: [AREA]}, {AREA: [AREA]})`.
        cue = torch.zeros(nbrain, 0, dtype=torch.int64, device=DEV)
        ss = [to_i32(_seeding.fnv1a_pair_seed(42 + b, f"s{a}", "A"))
              for b in range(nbrain)]
        res = batched_project_hashed(
            n, K, P, sd, cue, T, beta=BETA, w_max=W_MAX,
            stim_seeds=ss, stim_size=K,
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


def _fill_at(cells, m_star):
    """rows/n interpolated at M*, in log2(M)."""
    pts = sorted((M, float(np.mean(cells[M]["fill"]))) for M in cells)
    if not pts or m_star is None or m_star <= 0:
        return float("nan")
    if m_star <= pts[0][0]:
        return pts[0][1]
    if m_star >= pts[-1][0]:
        return pts[-1][1]
    for (m0, f0), (m1, f1) in zip(pts, pts[1:]):
        if m0 <= m_star <= m1:
            t = ((math.log2(m_star) - math.log2(m0))
                 / max(math.log2(m1) - math.log2(m0), 1e-12))
            return f0 + t * (f1 - f0)
    return pts[-1][1]


def gated(cell):
    r = ensemble_from_values(cell["rank1"])
    x = ensemble_from_values(cell["pairwise_x"])
    d = ensemble_from_values(cell["distinct"])
    ok = (x.high <= DISTINCT_GATE and d.low >= 0.9)
    return (r.mean if ok else 0.0), r, x, d


def main():
    global MS, K
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ns", type=str, default=None,
                    help="comma-separated n values (default: the grid)")
    ap.add_argument("--ms", type=str, default=None,
                    help="comma-separated M checkpoints")
    ap.add_argument("--brains", type=int, default=None)
    ap.add_argument("--arms", type=str, default=None)
    ap.add_argument("--nk", type=str, default=None,
                    help="explicit n:k pairs, e.g. 4000:60,8000:120")
    ap.add_argument("--ksqrt", action="store_true",
                    help="set k = round(sqrt(n)) per n, which holds the chance "
                         "overlap k*k/n at 1 while n varies -- the probe that "
                         "separates interference-limited from tiling-limited")
    args = ap.parse_args()
    ns = (1000, 2000) if args.smoke else NS
    ms = (4, 8) if args.smoke else MS
    nb = 4 if args.smoke else NBRAIN
    if args.ns:
        ns = tuple(int(x) for x in args.ns.split(","))
    if args.ms:
        ms = tuple(int(x) for x in args.ms.split(","))
    if args.brains:
        nb = args.brains
    nk = None
    if args.nk:
        nk = [tuple(int(v) for v in pair.split(":"))
              for pair in args.nk.split(",")]
        ns = [a for a, _ in nk]
    if args.arms:
        keep = set(args.arms.split(","))
        for kk in list(ARMS):
            if kk not in keep:
                del ARMS[kk]
    MS = ms
    if args.smoke:
        print("*** SMOKE: API only. THESE NUMBERS ARE VOID. ***")

    print(f"k={'sqrt(n)' if args.ksqrt else K} p={P} T={T} "
          f"beta={BETA} w_max={W_MAX} brains={nb}")
    for i, n in enumerate(ns):
        kk = (nk[i][1] if nk else
              (int(round(math.sqrt(n))) if args.ksqrt else K))
        print(f"  n={n:>6} k={kk:>4} kp={kk*P:.1f} vs floor "
              f"3ln n={3*math.log(n):.1f}  "
              f"{'IN REGIME' if kk*P >= 3*math.log(n) else 'OUT OF REGIME'}"
              f"   k*k/n={kk*kk/n:.2f}")

    res, t0 = {}, time.perf_counter()
    k_base = K
    for arm in ARMS:
        for i, n in enumerate(ns):
            if nk:
                K = nk[i][1]
            else:
                K = int(round(math.sqrt(n))) if args.ksqrt else k_base
            rng = np.random.default_rng(1234)
            try:
                cells = run_cell(n, arm, max(ms), nb, rng)
            except RuntimeError as exc:
                # `batched_project_hashed` REFUSES rather than diverging when
                # the w_max clip could bind under column scaling. That is a
                # cell this method cannot measure, not a cell that failed.
                print(f"    {arm} n={n:>5}  UNMEASURABLE: {exc}")
                res[f"{arm}/{n}/ceiling"] = dict(
                    m_star=None, supported=False, fill=float("nan"),
                    censored=False, unmeasurable=str(exc))
                continue
            res[f"{arm}/{n}"] = cells
            curve = []
            for M in sorted(cells):
                g, r, x, d = gated(cells[M])
                curve.append((M, g))
                print(f"    {arm} n={n:>5} M={M:>4}  rank1 {r.mean:.3f} "
                      f"pw/chance {x.mean:6.2f}  distinct {d.mean:.3f}  "
                      f"fill {np.mean(cells[M]['fill']):.3f}  gated {g:.3f}")
            c = ceiling_from_curve(curve, threshold=HALF_BAR)
            alpha = (c.m_star * K / n) if c.m_star else float("nan")
            # CAP3 says rows/n AT THE CEILING, not at the largest M on the
            # grid. Taking it at max(M) censors every point, because the grid
            # deliberately runs past the ceiling to bracket it. Interpolated in
            # log2(M) to match `ceiling_from_curve`'s own interpolation.
            fill_at = _fill_at(cells, c.m_star)
            print(f"    {arm} n={n:>5} k={K:>4}  CEILING {c}  "
                  f"fill@M* {fill_at:.3f}  M*k/n {alpha:.3f}"
                  f"  {'CENSORED' if fill_at >= 0.95 else 'ok'}")
            res[f"{arm}/{n}/ceiling"] = dict(
                m_star=c.m_star, supported=bool(c.supported), k=int(K),
                alpha=float(alpha),
                fill=float(fill_at), censored=bool(fill_at >= 0.95))
    print(f"\n  elapsed {time.perf_counter()-t0:.1f}s")

    print("\n--- CAP2: fit log M* = a + b log n over UNCENSORED n ---")
    for arm in ARMS:
        pts = [(n, res[f"{arm}/{n}/ceiling"]) for n in ns
               if f"{arm}/{n}/ceiling" in res]
        good = [(n, c["m_star"]) for n, c in pts
                if c["supported"] and not c["censored"]
                and c["m_star"] is not None and c["m_star"] > 0]
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
