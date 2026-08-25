"""Do the SUBSTRATES reproduce `numpy_sparse`, round by round?

The selector's tie policy differs (canonical vs argpartition+argsort) and the
drive is an integer Bernoulli sum on round 1, so letting both run freely would
conflate a tie-order divergence with a substrate error. Instead the ENGINE's
winner trajectory is captured and REPLAYED through the kernels, and the DRIVE
is compared at every round. That isolates the substrate arithmetic exactly.

Arms: NONE, B (norm_init), C (synaptic_scaling), G (both).
"""
import os
import random

os.environ.setdefault(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

import sys                                                  # noqa: E402
sys.path.insert(0, 'F:/Github/assemblies')
from neural_assemblies.core.brain import Brain               # noqa: E402
from neural_assemblies.core.numpy_engine import _seeding     # noqa: E402
from neural_assemblies.core.torch_engine import _fused_cuda  # noqa: E402
from neural_assemblies.core.torch_engine._batched import (   # noqa: E402
    _chain_table, _column_index)

AREA = "A"
DEV = 'cuda'


def to_i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def engine_trace(n, k, p, beta, T, seed, norm_init, scaling, w_max):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=p, seed=seed, engine="numpy_sparse", w_max=w_max,
                  recurrent_projection=True, norm_init=norm_init,
                  synaptic_scaling=scaling)
    brain.add_area(AREA, n, k, beta)
    eng = brain._engine_for(brain.areas[AREA])
    eng.materialize_area(AREA, storage="dense")
    rng = np.random.default_rng(seed)
    w0 = np.sort(rng.choice(n, k, replace=False)).astype(np.uint32)
    eng.set_winners(AREA, w0)
    drives, prevs, news = [], [], []
    for _ in range(T):
        prev = np.asarray(eng.get_winners(AREA), dtype=np.int64)
        res = eng.project_into(AREA, [], [AREA], plasticity_enabled=True,
                               record_activation=True)
        drives.append(np.asarray(res.pre_kwta_inputs, dtype=np.float64))
        prevs.append(prev)
        news.append(np.asarray(eng.get_winners(AREA), dtype=np.int64))
    pair = _seeding.fnv1a_pair_seed(seed, AREA, AREA)
    return drives, prevs, news, pair


def replay(n, k, p, beta, T, pair, prevs, news, norm_init, scaling, w_max):
    mod = _fused_cuda.load()
    thr = _fused_cuda.threshold_for(p)
    seeds_t = torch.tensor([to_i32(pair)], dtype=torch.int32, device=DEV)
    rowmask = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    colmask = torch.zeros(1, n, dtype=torch.int64, device=DEV)
    tab = torch.from_numpy(_chain_table(beta, w_max, T)).to(DEV)
    scale = torch.ones(1, n, dtype=torch.float32, device=DEV) if scaling else None
    setpoint = max(float(n) * float(p), 1e-12)
    dj = mod.hashed_indegree(seeds_t, n, thr, 1.0) if norm_init else None
    hist, out = [], []
    for t in range(T):
        rows = torch.from_numpy(prevs[t].astype(np.int32)).to(DEV).view(1, -1)
        d = mod.hashed_drive(rows.contiguous(), seeds_t, n, thr)
        if hist:
            cid, cmask = _column_index(hist)
            mod.dev_correct(rows.contiguous(), rowmask, cid, cmask, tab,
                            seeds_t, thr, d)
        if scale is not None:
            d = d * scale
        if dj is not None:
            d = d / dj
        out.append(d[0].cpu().numpy().astype(np.float64))
        # commit this round exactly as the engine does
        bit = 1 << t
        pidx = torch.from_numpy(prevs[t]).to(DEV).view(1, -1)
        sidx = torch.from_numpy(news[t]).to(DEV).view(1, -1)
        rowmask.scatter_(1, pidx, rowmask.gather(1, pidx) | bit)
        colmask.scatter_(1, sidx, colmask.gather(1, sidx) | bit)
        hist.append(sidx.to(torch.int32))
        if scale is not None:
            mass = mod.column_mass(sidx.to(torch.int32).contiguous(), rowmask,
                                   colmask, tab, seeds_t, thr)
            scale.scatter_(1, sidx, setpoint / mass.clamp_min(1e-12))
    return out


def arm(name, norm_init, scaling, n=1024, k=30, p=0.1, beta=0.1, T=6,
        seed=7, w_max=20.0):
    d_cpu, prevs, news, pair = engine_trace(
        n, k, p, beta, T, seed, norm_init, scaling, w_max)
    d_gpu = replay(n, k, p, beta, T, pair, prevs, news, norm_init, scaling,
                   w_max)
    worst, worst_rel = 0.0, 0.0
    for t in range(T):
        m = min(len(d_cpu[t]), len(d_gpu[t]))
        a, b = d_cpu[t][:m], d_gpu[t][:m]
        e = float(np.abs(a - b).max())
        worst = max(worst, e)
        worst_rel = max(worst_rel, e / max(float(np.abs(a).max()), 1e-12))
    print(f"  arm {name:<4} norm_init={norm_init!s:<5} scaling={scaling!s:<5} "
          f"T={T}  max|diff| {worst:.6g}  rel {worst_rel:.3g}")


def main():
    print("=== drive per round, engine trajectory replayed (n=1024 k=30 p=0.1) ===")
    arm("NONE", False, False)
    arm("B", True, False)
    arm("C", False, True)
    arm("G", True, True)
    print("\n=== denser fiber (n=2048 k=50 p=0.5) ===")
    arm("NONE", False, False, n=2048, k=50, p=0.5)
    arm("B", True, False, n=2048, k=50, p=0.5)
    arm("C", False, True, n=2048, k=50, p=0.5)
    arm("G", True, True, n=2048, k=50, p=0.5)


if __name__ == "__main__":
    main()
