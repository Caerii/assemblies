"""Phase 1 -- map the drive regime in which E%-WTA actually forms assemblies.

The prior capacity experiment (``research/experiments/capacity/``) refused to
answer the fixed-k vs E%-WTA capacity question because at k_s=30, p=0.05 BOTH
the ``epsilon`` and ``sigma`` windows collapsed to a mean assembly size of
1.1-1.6.  Before any capacity comparison can mean anything we need the region
of (p_s, k_s) space where a window of usable width exists.

This sweeps the paper's own dense reference model (``epwta.form_assembly``,
one fresh substrate per run, one assembly) over p_s x k_s and records, per
window rule:

  * formation rate (all four conditions of Eq. 8-11)
  * mean/sd emergent assembly size (over ALL runs, formed or not)
  * which condition failed when it failed

Output: ``results_regime_map.json``.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_assemblies.assembly_calculus.epwta import form_assembly  # noqa: E402

HERE = Path(__file__).resolve().parent

N = 1000
P_S_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]
K_S_GRID = [30, 60, 100, 200, 400]
MODES = ["epsilon", "sigma"]
SEEDS = list(range(10))
BETA = 0.01          # paper default
P_I = 0.2
W_INH = -0.2


def run_cell(mode: str, p_s: float, k_s: int, seeds) -> dict:
    sizes, formed, fails = [], [], []
    iters = []
    for s in seeds:
        r = form_assembly(
            n=N, stimulus_size=k_s, p_s=p_s, p_i=P_I, w_inh=W_INH,
            beta=BETA, selection=mode, seed=1000 * s + k_s, max_iters=100,
        )
        sizes.append(r.size)
        formed.append(1.0 if r.formed else 0.0)
        iters.append(r.iterations)
        if not r.formed:
            fails.append(r.failure or "?")
    return {
        "mode": mode, "p_s": p_s, "k_s": k_s, "n": N,
        "size_mean": float(np.mean(sizes)), "size_sd": float(np.std(sizes, ddof=1)),
        "sizes": sizes,
        "formed_rate": float(np.mean(formed)),
        "iters_mean": float(np.mean(iters)),
        "failures": dict(Counter(fails)),
    }


def main() -> None:
    t0 = time.time()
    rows = []
    for mode in MODES:
        for p_s in P_S_GRID:
            for k_s in K_S_GRID:
                r = run_cell(mode, p_s, k_s, SEEDS)
                rows.append(r)
                print(f"{mode:8s} p_s={p_s:<5} k_s={k_s:<4} "
                      f"size={r['size_mean']:8.1f}+/-{r['size_sd']:<7.1f} "
                      f"formed={r['formed_rate']:.2f} {r['failures']}",
                      flush=True)
    out = {"params": {"n": N, "beta": BETA, "p_i": P_I, "w_inh": W_INH,
                      "seeds": len(SEEDS), "min_size": 6},
           "rows": rows, "duration_s": time.time() - t0}
    (HERE / "results_regime_map.json").write_text(json.dumps(out, indent=1))
    print(f"done in {out['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
