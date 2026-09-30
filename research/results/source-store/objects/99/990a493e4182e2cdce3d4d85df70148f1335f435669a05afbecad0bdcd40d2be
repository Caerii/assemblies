"""Confirmation seeds for the corrected-substrate beta control (Task 2).

Adds seeds 4,5 for the norm_init=True beta cells in rec mode so the headline
V*(n,beta) statistics rest on 6 seeds, and re-verifies reproducibility of the
salvaged 4-seed data in results_norm_all_rec.json. REUSES
norm_init_capacity.run_cell unchanged; writes incrementally so a crash loses
nothing. Results land in results_confirm_beta_rec.json (kept separate from the
salvaged json).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from research.experiments.capacity import lexicon_capacity as lc
from research.experiments.capacity import norm_init_capacity as ni
from research.json_documents import write_checkpoint_document

HERE = Path(__file__).parent
OUT = HERE / "results_confirm_beta_rec.json"

NS = (1000, 3000, 10000)
BETAS = (0.0, 0.01, 0.05, 0.2)
SEEDS = (4, 5)
MODE = "rec"
VOCAB = 400
CPS = lc.DEFAULT_CHECKPOINTS


def main() -> int:
    results = []
    if OUT.exists():
        results = json.loads(OUT.read_text())
    done = {(r["config"]["n"], r["config"]["beta"], r["config"]["seed"])
            for r in results}
    t0 = time.perf_counter()
    # smallest n first so partial coverage still spans all betas cheaply
    for n in NS:
        for beta in BETAS:
            for s in SEEDS:
                if (n, beta, s) in done:
                    continue
                res = ni.run_cell(n, beta, MODE, True, s, VOCAB, CPS)
                results.append(res)
                write_checkpoint_document(OUT, results)
                print(f"  n={n:>6} beta={beta:<5} seed={s} "
                      f"V*={res['vstar']:>3} (censored={res['vstar_censored']}) "
                      f"vstar_recruit={res['vstar_recruit']:>3} "
                      f"ident@400={res['checkpoints'][-1]['identification_acc']:.3f} "
                      f"({res['wall_seconds']:.1f}s)", flush=True)
    print(f"confirm done in {time.perf_counter()-t0:.1f}s -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
