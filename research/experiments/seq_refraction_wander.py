"""P2 of PREREG_refraction_capacity, and what it turned into.

REGISTERED: a refracted assembly must WANDER once w_max binds, at about
ln(w_max)/ln(1+beta) + (1-1/w_max)/beta ~ 41 rounds (w_max=20, beta=0.1).

MEASURED FIRST: the recurrent refracted arm departs at round 11 -- every brain,
to below-chance overlap -- while a FEEDFORWARD refracted arm holds to ~20 and
then settles at 0.63. So the recurrent churn is not the clip. In the stable
state the identity says net drive is CONSTANT under repetition, i.e.
refraction at strength = beta removes the Hebbian convergence force. A
feedforward area needs none (its input ranking is fixed); a recurrent assembly
converges only through rich-get-richer, so with the force cancelled the
changing recurrent input reshuffles the winners every round.

This script therefore sweeps the strength and measures CONVERGENCE
(consecutive-round stability), predicting convergence time ~ 1/(beta - s).

    python research/experiments/seq_refraction_wander.py
"""
from __future__ import annotations

import math
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
from neural_assemblies.core.torch_engine._hashed import (               # noqa: E402
    AreaFiber, HashedArea, StimulusFiber)

DEV = "cuda"
N, K, P, BETA, W_MAX = 4000, 100, 0.5, 0.10, 20.0
ROUNDS, REF_ROUND, NBRAIN = 240, 10, 16


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def run(refracted, *, recurrent=True, strength_factor=1.0):
    sd = [to_i32(_seeding.fnv1a_pair_seed(42 + b, "A", "A"))
          for b in range(NBRAIN)]
    ss = [to_i32(_seeding.fnv1a_pair_seed(42 + b, "s0", "A"))
          for b in range(NBRAIN)]
    area = HashedArea(N, K, sd, device=DEV,
                      refracted_strength=(BETA * strength_factor
                                          if refracted else 0.0))
    fiber = AreaFiber(sd, N, N, P, beta=BETA, w_max=W_MAX, norm_init=True,
                      synaptic_scaling=False, max_rounds=ROUNDS, device=DEV)
    stim = StimulusFiber(ss, K, N, P, beta=BETA, w_max=W_MAX, norm_init=True,
                         max_rounds=ROUNDS, device=DEV)
    ref, overlaps, consec, prev = None, [], [], None
    for r in range(1, ROUNDS + 1):
        # one-round episodes so every round's winners are observable; the
        # stimulus fiber's potentiation counter and the area fiber's store
        # both persist across episodes.
        w = area.project(1, [fiber, stim] if recurrent else [stim])
        if r == REF_ROUND:
            ref = torch.zeros(NBRAIN, N, dtype=torch.bool, device=DEV)
            ref.scatter_(1, w, True)
        if ref is not None:
            overlaps.append((torch.gather(ref, 1, w).sum(1).float() / K)
                            .cpu().numpy())
        if prev is not None:
            pm = torch.zeros(NBRAIN, N, dtype=torch.bool, device=DEV)
            pm.scatter_(1, prev, True)
            consec.append((torch.gather(pm, 1, w).sum(1).float() / K)
                          .cpu().numpy())
        prev = w
    ov = np.stack(overlaps)                                # [rounds-9, B]
    cs = np.stack(consec)                                  # [rounds-1, B]
    departed = ov < 0.5
    first = np.where(departed.any(0),
                     departed.argmax(0) + REF_ROUND, -1)
    conv = np.full(NBRAIN, -1)
    for b in range(NBRAIN):
        ok = cs[:, b] >= 0.95
        for r in range(len(ok)):
            if ok[r:].all():
                conv[b] = r + 2          # cs[r] compares rounds r+1 and r+2
                break
    return ov, first, cs, conv, area.fill.cpu().numpy()


def main():
    pred = math.log(W_MAX) / math.log(1 + BETA) + (1 - 1 / W_MAX) / BETA
    print(f"n={N} k={K} p={P} beta={BETA} w_max={W_MAX} rounds={ROUNDS} "
          f"brains={NBRAIN}   registered saturation prediction ~ {pred:.1f}")
    print("  'conv' = first round after which consecutive-round overlap stays "
          ">= 0.95; 'late' = mean consecutive overlap over rounds 200-240")
    arms = [("control (rec)", False, True, 1.0),
            ("refracted FEEDFORWARD only", True, False, 1.0)]
    arms += [(f"refracted (rec) s = {f:.2f} beta", True, True, f)
             for f in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)]
    marks = [10, 20, 30, 40, 60, 100, 150, 200, 240]
    for name, refr, rec, sf in arms:
        ov, first, cs, conv, fill = run(refr, recurrent=rec,
                                        strength_factor=sf)
        cv = conv[conv >= 0]
        head = f"  {name}: converged {len(cv)}/{NBRAIN}"
        if len(cv):
            head += (f", median conv round {np.median(cv):.0f} "
                     f"[{cv.min()}, {cv.max()}]")
        head += f"; late {cs[198:].mean():.3f}; fill {fill.mean():.3f}"
        print()
        print(head)
        print("    round          " + " ".join(f"{m:>5d}" for m in marks))
        print("    vs round-10    " + " ".join(
            f"{ov[m - REF_ROUND].mean():5.3f}" for m in marks))
        print("    consecutive    " + " ".join(
            f"{cs[m - 2].mean():5.3f}" for m in marks))


if __name__ == "__main__":
    main()
