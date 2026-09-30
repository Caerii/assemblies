"""Phase 3 -- is the E%-WTA deficit intrinsic, or induced by loading?

Phase 2 shows the E% window narrowing as items accumulate: potentiation grows
``h_max`` faster than the bulk of ``h``, so a window defined relative to the
peak keeps shrinking.  If that is the whole story, then at ``beta = 0`` (no
plasticity: assemblies are read off a static substrate) E%-WTA should hold its
size and its capacity deficit should vanish.

Same operating point and substrate as ``capacity.py``; V is fixed at 64.
Output: ``results_beta_control.json``.
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

from research.experiments.epwta_capacity.capacity import measure  # noqa: E402
from research.experiments.epwta_capacity.shared_substrate import (  # noqa: E402
    build_substrate, store,
)

HERE = Path(__file__).resolve().parent

N, POOL, K_S, P_S, P_I, W_INH, MIN_SIZE = 1000, 1000, 100, 0.5, 0.2, -0.2, 6
V = 64
BETAS = [0.0, 0.005, 0.01, 0.02]
ARMS = [("epsilon", 0), ("sigma", 0), ("topk", 9), ("topk", 25)]
SEEDS = list(range(6))


def run(policy, k, beta, seed):
    sub = build_substrate(n=N, pool=POOL, k_s=K_S, n_items=V, p_s=P_S,
                          p_i=P_I, w_inh=W_INH, seed=seed)
    stored = [store(sub, i, policy=policy, k=k, beta=beta, metaplasticity=0.0,
                    min_size=MIN_SIZE) for i in range(V)]
    return measure(sub, stored, policy, k)


def main():
    t0 = time.time()
    rows = []
    for beta in BETAS:
        for policy, k in ARMS:
            arm = policy if policy != "topk" else f"topk{k}"
            cells = [run(policy, k, beta, 200 + s) for s in SEEDS]
            agg = {"arm": arm, "beta": beta, "V": V}
            for key in cells[0]:
                if key == "V":
                    continue
                vals = [c[key] for c in cells]
                agg[key] = float(np.mean(vals))
                agg[key + "_sd"] = float(np.std(vals, ddof=1))
                agg[key + "_vals"] = vals
            rows.append(agg)
            print(f"beta={beta:<6} {arm:8s} size={agg['size_mean']:5.1f} "
                  f"ident={agg['ident']:.2f} rec={agg['recovery']:.2f} "
                  f"pairwise={agg['pairwise']:.3f} formed={agg['formed_rate']:.2f}",
                  flush=True)
    out = {"params": {"n": N, "pool": POOL, "k_s": K_S, "p_s": P_S, "V": V,
                      "seeds": len(SEEDS), "metaplasticity": 0.0},
           "rows": rows, "duration_s": time.time() - t0}
    write_new_document(HERE / "results_beta_control.json", out)
    print(f"done in {out['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
