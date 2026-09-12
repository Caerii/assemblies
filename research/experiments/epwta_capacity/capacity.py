"""Phase 2 -- capacity of one area under fixed-k WTA vs E%-WTA.

Operating point is chosen from ``regime_map.py``: p_s=0.5, k_s=100, n=1000,
where BOTH window rules form assemblies of usable size on a fresh substrate
(epsilon 10.5 +/- 4.3, formed 0.90; sigma 28.2 +/- 13.9, formed 1.00), so the
comparison is not a window-collapse artefact.

Fixed-k arms are matched to the V=1 emergent mean size of each E% arm
(k=9 ~ epsilon, k=25 ~ sigma) plus the repo's default k=30 for reference.

All items live on ONE shared substrate (see ``shared_substrate.py``).
Output: ``results_capacity.json``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from research.json_documents import write_new_document

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.epwta_capacity.shared_substrate import (  # noqa: E402
    build_substrate, cosine_overlap, retrieve, store,
)

HERE = Path(__file__).resolve().parent

N = 1000
POOL = 1000
K_S = 100
P_S = 0.5
P_I = 0.2
W_INH = -0.2
BETA = 0.01
MIN_SIZE = 6
V_MAX = 128
CHECKPOINTS = [2, 4, 8, 16, 32, 64, 96, 128]
SEEDS = list(range(8))

ARMS = [
    ("epsilon", 0),
    ("sigma", 0),
    ("topk", 9),     # matched to epsilon's V=1 emergent size
    ("topk", 25),    # matched to sigma's V=1 emergent size
    ("topk", 30),    # repo default
]
METAS = [0.0, 5.0]


def measure(sub, stored, policy, k) -> dict:
    """Retrieval / identification / overlap over everything stored so far."""
    V = len(stored)
    tmpl = [s["winners"] for s in stored]
    rec, ident, rsize = [], [], []
    for i in range(V):
        r = retrieve(sub, i, policy=policy, k=k)
        rsize.append(len(r))
        tgt = tmpl[i]
        rec.append(len(np.intersect1d(r, tgt)) / len(tgt) if len(tgt) else 0.0)
        sims = [cosine_overlap(r, t) for t in tmpl]
        best = int(np.argmax(sims))
        ident.append(1.0 if (best == i and sims[i] > 0) else 0.0)
    # pairwise overlap of stored assemblies, with its own chance level
    ov, ch = [], []
    pairs = [(i, j) for i in range(V) for j in range(i + 1, V)]
    if len(pairs) > 2000:
        rs = np.random.default_rng(0)
        pairs = [pairs[x] for x in rs.choice(len(pairs), 2000, replace=False)]
    for i, j in pairs:
        ov.append(cosine_overlap(tmpl[i], tmpl[j]))
        ch.append(np.sqrt(len(tmpl[i]) * len(tmpl[j])) / N)
    q = max(1, V // 5)
    sizes = [s["size"] for s in stored]
    return {
        "V": V,
        "recovery": float(np.mean(rec)),
        "recovery_early": float(np.mean(rec[:q])),
        "recovery_late": float(np.mean(rec[-q:])),
        "ident": float(np.mean(ident)),
        "ident_chance": 1.0 / V,
        "pairwise": float(np.mean(ov)) if ov else 0.0,
        "pairwise_chance": float(np.mean(ch)) if ch else 0.0,
        "size_mean": float(np.mean(sizes)),
        "size_sd": float(np.std(sizes, ddof=1)) if V > 1 else 0.0,
        "size_early": float(np.mean(sizes[:q])),
        "size_late": float(np.mean(sizes[-q:])),
        "retr_size_mean": float(np.mean(rsize)),
        "formed_rate": float(np.mean([s["formed"] for s in stored])),
        "collapsed_frac": float(np.mean([s["size"] < MIN_SIZE for s in stored])),
    }


def run(policy: str, k: int, meta: float, seed: int) -> list:
    sub = build_substrate(n=N, pool=POOL, k_s=K_S, n_items=V_MAX,
                          p_s=P_S, p_i=P_I, w_inh=W_INH, seed=seed)
    stored, out = [], []
    for i in range(V_MAX):
        stored.append(store(sub, i, policy=policy, k=k, beta=BETA,
                            metaplasticity=meta, min_size=MIN_SIZE))
        if (i + 1) in CHECKPOINTS:
            out.append(measure(sub, stored, policy, k))
    return out


def main() -> None:
    t0 = time.time()
    rows = []
    for meta in METAS:
        for policy, k in ARMS:
            arm = policy if policy != "topk" else f"topk{k}"
            per_seed = [run(policy, k, meta, 100 + s) for s in SEEDS]
            for ci in range(len(CHECKPOINTS)):
                cells = [ps[ci] for ps in per_seed]
                agg = {"arm": arm, "policy": policy, "k": k, "meta": meta,
                       "V": cells[0]["V"]}
                for key in cells[0]:
                    if key == "V":
                        continue
                    vals = [c[key] for c in cells]
                    agg[key] = float(np.mean(vals))
                    agg[key + "_sd"] = float(np.std(vals, ddof=1))
                    agg[key + "_vals"] = vals
                rows.append(agg)
            last = rows[-1]
            print(f"meta={meta} {arm:8s} V=128 size={last['size_mean']:5.1f} "
                  f"ident={last['ident']:.2f} rec={last['recovery']:.2f} "
                  f"formed={last['formed_rate']:.2f}", flush=True)
    out = {"params": {"n": N, "pool": POOL, "k_s": K_S, "p_s": P_S, "p_i": P_I,
                      "w_inh": W_INH, "beta": BETA, "min_size": MIN_SIZE,
                      "V_max": V_MAX, "checkpoints": CHECKPOINTS,
                      "seeds": len(SEEDS)},
           "rows": rows, "duration_s": time.time() - t0}
    write_new_document(HERE / "results_capacity.json", out)
    print(f"done in {out['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
