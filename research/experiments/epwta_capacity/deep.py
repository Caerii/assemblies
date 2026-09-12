"""Phase 4 -- push V far enough to find the fixed-k ceiling too.

At V=128 (``capacity.py``) topk9 is still at identification 1.00, so its
capacity is only bounded below.  This runs the same operating point out to
V=512 for the four size-matched arms.  Output: ``results_deep.json``.
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

from research.experiments.epwta_capacity.capacity import (  # noqa: E402
    BETA, K_S, MIN_SIZE, N, P_I, P_S, POOL, W_INH, measure,
)
from research.experiments.epwta_capacity.shared_substrate import (  # noqa: E402
    build_substrate, store,
)

HERE = Path(__file__).resolve().parent

V_MAX = 512
CHECKPOINTS = [128, 192, 256, 384, 512]
ARMS = [("epsilon", 0), ("sigma", 0), ("topk", 9), ("topk", 25)]
METAS = [0.0, 5.0]
SEEDS = list(range(4))


def run(policy, k, meta, seed):
    sub = build_substrate(n=N, pool=POOL, k_s=K_S, n_items=V_MAX, p_s=P_S,
                          p_i=P_I, w_inh=W_INH, seed=seed)
    stored, out = [], []
    for i in range(V_MAX):
        stored.append(store(sub, i, policy=policy, k=k, beta=BETA,
                            metaplasticity=meta, min_size=MIN_SIZE))
        if (i + 1) in CHECKPOINTS:
            out.append(measure(sub, stored, policy, k))
    return out


def main():
    t0 = time.time()
    rows = []
    for meta in METAS:
        for policy, k in ARMS:
            arm = policy if policy != "topk" else f"topk{k}"
            per_seed = [run(policy, k, meta, 300 + s) for s in SEEDS]
            for ci in range(len(CHECKPOINTS)):
                cells = [ps[ci] for ps in per_seed]
                agg = {"arm": arm, "meta": meta, "V": cells[0]["V"]}
                for key in cells[0]:
                    if key == "V":
                        continue
                    vals = [c[key] for c in cells]
                    agg[key] = float(np.mean(vals))
                    agg[key + "_sd"] = float(np.std(vals, ddof=1))
                    agg[key + "_vals"] = vals
                rows.append(agg)
                print(f"meta={meta} {arm:8s} V={agg['V']:4d} "
                      f"size={agg['size_mean']:5.1f} ident={agg['ident']:.2f} "
                      f"rec={agg['recovery']:.2f}", flush=True)
    out = {"params": {"n": N, "pool": POOL, "k_s": K_S, "p_s": P_S,
                      "beta": BETA, "V_max": V_MAX,
                      "checkpoints": CHECKPOINTS, "seeds": len(SEEDS)},
           "rows": rows, "duration_s": time.time() - t0}
    write_new_document(HERE / "results_deep.json", out)
    print(f"done in {out['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
