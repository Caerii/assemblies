"""Exploratory (not registered): the load on the C -> S links of Amendment 33. For every plan
position, does its (stored) C state evoke its chunk's first S state in one masked round?
Variable: rho_x = A k ln n_S / (n_C n_S p), A = plan positions linked. Seeds 980-989."""
import os
import math, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import torch
from research.experiments import memory_hierarchy as mh, memory_sequences as sq, memory_threshold_law as tl

seeds = list(range(980, 990))
K, P, PLEN = 60, 0.5, 8


def hits(ns, nc, chunks, n_plans, link=2):
    spec = {"n_s": ns, "n_c": nc, "k": K, "p": P, "beta_s": round(tl.theta(ns, K, P), 5),
            "beta_c": round(tl.theta(nc, K, P), 5), "chunk_len": 2}
    S, C, CS, ch, orders, pstates = mh.build(spec, link, seeds, "cuda", (chunks, n_plans))
    got, wrong = [], []
    for order, st in zip(orders, pstates):
        for i, c in enumerate(order):
            s = S.area.project(1, [CS], rows_for={id(CS): st[i]}, freeze=True, mask_bias=True,
                               manage_episodes=False)
            got.append((sq._overlap(s, ch[c][0]) >= 0.3).float())
            others = torch.stack([sq._overlap(s, ch[o][0]) for o in range(chunks) if o != c]).amax(0)
            wrong.append(others)
    g = torch.stack(got).mean().item(); w = torch.stack(wrong).mean().item()
    return g, w, len(orders) * PLEN


for ns, nc in ((4000, 2000), (8000, 4000), (4000, 4000), (8000, 2000)):
    unit = nc * ns * P / (K * math.log(ns))
    for chunks in (16, 64):
        zeros = 0
        for j in range(0, 20):
            rho = 0.03 * 2 ** (j / 4)
            n_plans = max(2, round(rho * unit / PLEN))
            t0 = time.time()
            g, w, A = hits(ns, nc, chunks, n_plans)
            print(f"S={ns} C={nc} chunks={chunks} plans={n_plans} A={A} rho_x={A / unit:.3f} fan-in={A / chunks:.0f}: "
                  f"start hit {g:.3f}, best wrong start overlap {w:.3f} [{time.time() - t0:.0f}s]", flush=True)
            zeros = zeros + 1 if g < 0.05 else 0
            if zeros >= 2 or n_plans > 3000:
                break
