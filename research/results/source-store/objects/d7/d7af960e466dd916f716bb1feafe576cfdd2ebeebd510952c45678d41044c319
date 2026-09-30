"""#150 POST-HOC arms D and E -- labeled as such; no registered bar.

These arms were designed AFTER Phase 1's registered run, each with its
prediction stated in the session transcript before its data:

  D  split, NO homeostatic scaling (same slice/teacher/exam as arm A).
     Prediction: if the PL inversion is caused by per-column
     NORMALIZATION under the 90/10 class imbalance, D recovers SG (as
     the #149 gate did at 70/30 without scaling); if raw form-load
     dilution suffices, D stays inverted.
     RESULT: SG 0.036 -> 0.513 +/- 0.057; scaling is the cause.
     With rows: F1 spearman(exposure, correct) = +0.069 +/- 0.044 (CI
     excludes 0 -- the exposure law transfers on a WORKING readout) and
     the buckets reproduce the law's shape: 1 exposure = 0.4995
     (chance), 2-3 = 0.560, 4+ = 0.575.

  E  split, no scaling, n=10000 (capacity control). The critical-load
     law (alpha* ~ 1.15 n/k) puts per-area capacity at n=3000/k=30 at
     ~115 forms; the SG area carries ~363 (3x OVER), the PL area 46
     (under) -- predicting exactly the observed asymmetry (SG 0.51,
     PL 0.71). At n=10000 capacity ~383 > 363. Prediction: SG recovers
     toward PL's level; SG staying ~0.5 refutes the over-capacity
     account.

First executions ran as scratchpad transcriptions of this code (results
in childes_arm_d_results.json / childes_arm_e_results.json and the
session log); this file is the committed, re-runnable record.

Run: python childes_posthoc_arms.py [D|E]
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
from pathlib import Path

from research.json_documents import write_new_document

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from childes_graduation import build_number_slice, load_corpus  # noqa: E402

SEEDS = list(range(42, 52))


def run_arm(n: int, out_name: str):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )
    from neural_assemblies.diagnostics import spearman

    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)
    sgs, pls, f1 = [], [], []
    buckets = {"1": [], "2-3": [], "4+": []}
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        p = EmergentParser(n=n, k=30, seed=seed, fast_training=True)
        for w in labels:
            if w not in p.stim_map:
                p.register_word(w)
            p.word_grounding[w] = GroundingContext(visual=[w])
        p.train_number(sentences, labels=labels)
        rows = []
        for w, lab in labels.items():
            if w in ambiguous:
                continue
            got, diag = p.recall_number(w)
            rows.append((exposures[w], lab, diag.get("mi_answer", got) == lab))
        f1.append(spearman([r[0] for r in rows], [r[2] for r in rows]))
        for e, _lab, ok in rows:
            b = "1" if e == 1 else ("2-3" if e <= 3 else "4+")
            buckets[b].append(ok)
        sgs.append(st.mean(ok for e, lab, ok in rows if lab == "SG"))
        pls.append(st.mean(ok for e, lab, ok in rows if lab == "PL"))
        print(f"seed={seed} sg={sgs[-1]:.3f} pl={pls[-1]:.3f} "
              f"rho={f1[-1]:.4f}", flush=True)

    def mci(xs):
        return {"mean": st.mean(xs),
                "ci": 2.262 * st.stdev(xs) / len(xs) ** 0.5}

    out = {"n": n, "sg": mci(sgs), "pl": mci(pls), "F1": mci(f1),
           "buckets": {b: st.mean(v) for b, v in buckets.items()},
           "per_seed": {"sg": sgs, "pl": pls, "f1": f1}}
    path = os.path.join(os.path.dirname(__file__), out_name)
    write_new_document(Path(path), out)
    print(json.dumps({k: out[k] for k in ("sg", "pl", "F1", "buckets")},
                     indent=2))


if __name__ == "__main__":
    arm = (sys.argv[1] if len(sys.argv) > 1 else "D").upper()
    if arm == "D":
        run_arm(3000, "childes_arm_d_results.json")
    elif arm == "E":
        run_arm(10000, "childes_arm_e_results.json")
    else:
        raise SystemExit("arm must be D or E")
