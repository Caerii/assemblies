"""F3 of PREREG_formation_interference: is failure CONCENTRATED capture?

Trains M = 32 items (~1.4x the M* = 23.5 ceiling) at the baseline operating
point, then per stored item: frozen rank-1 hit/miss, the item's MAX overlap
with any other stored assembly, and WHICH item that partner is -- capture has
a direction (a later item falls into an earlier attractor), diffuse crosstalk
does not.

    python research/experiments/seq_formation_forensics.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.core.numpy_engine import _seeding                # noqa: E402
from neural_assemblies.core.torch_engine._batched import (              # noqa: E402
    batched_project_hashed)

DEV = "cuda"
N, K, P, T, BETA, W_MAX, M = 4000, 100, 0.5, 8, 0.10, 20.0, 32
NBRAIN = 16


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def main():
    sd = [to_i32(_seeding.fnv1a_pair_seed(42 + b, "A", "A"))
          for b in range(NBRAIN)]
    state, stored = None, []
    for a in range(M):
        cue = torch.zeros(NBRAIN, 0, dtype=torch.int64, device=DEV)
        ss = [to_i32(_seeding.fnv1a_pair_seed(42 + b, f"s{a}", "A"))
              for b in range(NBRAIN)]
        win, state = batched_project_hashed(
            N, K, P, sd, cue, T, beta=BETA, w_max=W_MAX, norm_init=True,
            stim_seeds=ss, stim_size=K, state=state, max_rounds=M * T,
            return_state=True)
        stored.append(win)
    St = torch.stack(stored).long()                       # [M, B, K]
    Ks = torch.sort(St, dim=2).values

    # pairwise overlap matrix per brain, full M x M
    masks = torch.zeros(M, NBRAIN, N, dtype=torch.bool, device=DEV)
    masks.scatter_(2, St, True)
    fm = masks.float().reshape(M, -1)
    ov = torch.einsum("ax,bx->ab", fm, fm)                # sums over B*n
    # per-brain needed: do it brain-wise instead
    per = torch.zeros(M, M, NBRAIN, device=DEV)
    for b in range(NBRAIN):
        f = masks[:, b].float()
        per[:, :, b] = (f @ f.T) / K
    eye = torch.eye(M, device=DEV, dtype=torch.bool).unsqueeze(2)
    per_off = per.masked_fill(eye, -1.0)
    maxov, argmax = per_off.max(dim=1)                    # [M, B]

    # frozen rank-1 per item
    hits = torch.zeros(M, NBRAIN, dtype=torch.bool, device=DEV)
    off = (torch.arange(NBRAIN, device=DEV, dtype=torch.int64)
           * N).view(1, NBRAIN, 1)
    flat = (St + off).reshape(-1)
    for a in range(M):
        half = St[a][:, : K // 2].to(torch.int32).contiguous()
        rec = batched_project_hashed(
            N, K, P, sd, half, T, beta=BETA, w_max=W_MAX, norm_init=True,
            state=state, freeze=True)
        mask = torch.zeros(NBRAIN * N, dtype=torch.bool, device=DEV)
        mask[(rec + off[0]).reshape(-1)] = True
        o = mask[flat].view(M, NBRAIN, K).sum(2)
        hits[a] = o.argmax(dim=0) == a

    hits_f = hits.float().cpu().numpy()
    mo = maxov.cpu().numpy()
    am = argmax.cpu().numpy()

    print(f"n={N} k={K} p={P} beta={BETA} T={T} M={M} brains={NBRAIN}"
          f"   (M* at this operating point: 23.5)")
    print(f"  overall rank-1: {hits_f.mean():.3f}")

    failed = hits_f.reshape(-1) < 0.5
    ok = ~failed
    mo_flat = mo.reshape(-1)
    med_f = float(np.median(mo_flat[failed])) if failed.any() else float("nan")
    med_o = float(np.median(mo_flat[ok])) if ok.any() else float("nan")
    print(f"  max-overlap median: FAILED items {med_f:.3f}   "
          f"RECALLED items {med_o:.3f}   ratio {med_f/max(med_o,1e-9):.2f} "
          f"(bar: >= 2)")

    # direction: for failed items, is the max partner EARLIER in formation?
    item_idx = np.repeat(np.arange(M), NBRAIN)
    part = am.reshape(-1)
    f_earlier = float((part[failed] < item_idx[failed]).mean()) if failed.any() else float("nan")
    o_earlier = float((part[ok] < item_idx[ok]).mean()) if ok.any() else float("nan")
    print(f"  P(max partner formed EARLIER | failed)   = {f_earlier:.3f}")
    print(f"  P(max partner formed EARLIER | recalled) = {o_earlier:.3f}")
    print(f"  (uniform null = mean of i/(M-1) over items ~ 0.5)")

    # failure by formation position, halves
    by_item = hits_f.mean(axis=1)
    print(f"  rank-1 by formation half: first {by_item[:M//2].mean():.3f}   "
          f"second {by_item[M//2:].mean():.3f}")
    print(f"  n failed cells: {int(failed.sum())} of {M*NBRAIN}")


if __name__ == "__main__":
    main()
