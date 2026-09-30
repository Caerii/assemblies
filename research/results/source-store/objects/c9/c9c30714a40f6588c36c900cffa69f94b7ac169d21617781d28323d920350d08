"""Does the hash-generated STIMULUS fiber reproduce `numpy_sparse`?

The stimulus is where the pricing law bites: its weights store PRE-SUMMED
input, so both the norm_init divisor (`tgt.n`, not the stimulus size) and the
w_max cap (`w_max * stim_size * p`, not a raw `w_max`) differ from the area
fiber's. Getting either wrong is silent and has a direction -- the stimulus
either decides every winner or gets pinned at its cap.

Same method as the substrate parity: capture the ENGINE's winner trajectory,
replay it through the kernels, compare the DRIVE every round.
"""
import os
import random
import sys

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from neural_assemblies.core.brain import Brain               # noqa: E402
from neural_assemblies.core.numpy_engine import _seeding     # noqa: E402
from neural_assemblies.core.torch_engine import _fused_cuda  # noqa: E402
from neural_assemblies.core.torch_engine._batched import (   # noqa: E402
    _chain_table, _gain_table)

AREA, STIM, DEV = "A", "s0", "cuda"


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def engine_trace(n, k, p, beta, T, seed, norm_init, w_max):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=p, seed=seed, engine="numpy_sparse", w_max=w_max,
                  recurrent_projection=True, norm_init=norm_init,
                  synaptic_scaling=False)
    brain.add_area(AREA, n, k, beta)
    brain.add_stimulus(STIM, k)
    eng = brain._engine_for(brain.areas[AREA])
    eng.materialize_area(AREA, storage="dense")
    rng = np.random.default_rng(seed)
    eng.set_winners(AREA, np.sort(
        rng.choice(n, k, replace=False)).astype(np.uint32))
    stim0 = np.asarray(eng._stim_conns[STIM][AREA].weights,
                       dtype=np.float64).copy()
    drives, prevs, news = [], [], []
    for _ in range(T):
        prevs.append(np.asarray(eng.get_winners(AREA), dtype=np.int64))
        res = eng.project_into(AREA, [STIM], [AREA], plasticity_enabled=True,
                               record_activation=True)
        drives.append(np.asarray(res.pre_kwta_inputs, dtype=np.float64))
        news.append(np.asarray(eng.get_winners(AREA), dtype=np.int64))
    return (drives, prevs, news, stim0,
            _seeding.fnv1a_pair_seed(seed, AREA, AREA))


def replay(n, k, p, beta, T, apair, stim0, prevs, news, norm_init, w_max):
    mod = _fused_cuda.load()
    thr = _fused_cuda.threshold_for(p)
    a_s = torch.tensor([to_i32(apair)], dtype=torch.int32, device=DEV)
    rowmask = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    colmask = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    tab = torch.from_numpy(_chain_table(beta, w_max, T)).to(DEV)
    gpow = torch.from_numpy(_gain_table(beta, T)).to(DEV)
    colids = torch.arange(n, dtype=torch.int32, device=DEV).view(1, n)
    dj = mod.hashed_indegree(a_s, n, thr, 1.0) if norm_init else None

    # THE ENGINE'S OWN stim base, injected. Its stimulus connectome is drawn
    # by `self._rng.binomial(...)` consumed in stimulus insertion order, so it
    # is NOT a pure function of position and a generated connectome cannot
    # reproduce it. Injecting it isolates the question this test can answer:
    # is the PRICING right?
    stim_base = torch.from_numpy(
        stim0.astype(np.float32)).to(DEV).view(1, -1)
    stim_pot = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    stim_dj = ((stim_base + p * (n - k)).clamp_min(1.0)
               if norm_init else None)
    stim_hi = (w_max * max(1.0, k * p)) if w_max is not None else float("inf")

    out = []
    for t in range(T):
        rows = torch.from_numpy(
            prevs[t].astype(np.int32)).to(DEV).view(1, -1).contiguous()
        d = mod.hashed_drive(rows, a_s, n, thr)
        if t:
            mod.dev_correct(rows, rowmask, colids, colmask, tab, a_s, thr, d)
        if dj is not None:
            d = d / dj
        sd = stim_base * gpow[stim_pot.clamp_max(gpow.numel() - 1)]
        if stim_hi != float("inf"):
            sd = sd.clamp_max(stim_hi)
        if stim_dj is not None:
            sd = sd / stim_dj
        d = d + sd
        out.append(d[0].cpu().numpy().astype(np.float64))
        bit = 1 << t
        pidx = torch.from_numpy(prevs[t]).to(DEV).view(1, -1)
        sidx = torch.from_numpy(news[t]).to(DEV).view(1, -1)
        rowmask.scatter_(1, pidx, rowmask.gather(1, pidx) | bit)
        colmask.scatter_(1, sidx, colmask.gather(1, sidx) | bit)
        stim_pot.scatter_add_(1, sidx, torch.ones_like(sidx))
    return out


def arm(name, norm_init, n=1024, k=30, p=0.1, beta=0.1, T=6, seed=7,
        w_max=20.0):
    d_cpu, prevs, news, stim0, apair = engine_trace(
        n, k, p, beta, T, seed, norm_init, w_max)
    d_gpu = replay(n, k, p, beta, T, apair, stim0, prevs, news, norm_init,
                   w_max)
    worst = 0.0
    for t in range(T):
        m = min(len(d_cpu[t]), len(d_gpu[t]))
        a, b = d_cpu[t][:m], d_gpu[t][:m]
        e = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-12)
        nbad = int((np.abs(a - b) > 1e-6 * max(np.abs(a).max(), 1e-12)).sum())
        print(f"      t={t} rel {e:.4g}  cells off {nbad}/{m}")
        worst = max(worst, e)
    print(f"  {name:<22} n={n:>5} k={k:>3} p={p}  rel {worst:.4g}"
          f"   cpu mean {d_cpu[-1].mean():.5g}")


def main():
    print("=== stimulus PRICING (engine stim base injected) ===")
    arm("NONE + stim", False)
    arm("B (norm_init) + stim", True)
    arm("NONE + stim", False, n=2048, k=50, p=0.5)
    arm("B (norm_init) + stim", True, n=2048, k=50, p=0.5)
    arm("B, w_max=None", True, w_max=None)


if __name__ == "__main__":
    main()
