"""#151 S-unit: the collision-load law's scale cell -- n=100000, L0G4.

REGISTERED PREDICTION (before any n=1e5 cell has ever run): the 2x2's
residual PL deficit tracks the CROSS-WORD ROW-COLLISION LOAD
V*k^2/n -- shared core rows leak the frequent class's boosted mass into
the MI comparison:

    n=3000:   load 109 >> k=30  -> PL 0.11-0.41 (measured)
    n=10000:  load  33 ~  k=30  -> PL 0.29-0.49 (measured)
    n=100000: load 3.3 << k=30  -> BAR: PL >= 0.75 with SG >= 0.90,
                                    balanced >= 0.85.

If PL stays ~0.5 at n=1e5 the collision account is refuted and the
residual must be re-attributed. n=1e5 is also the first cell inside the
papers' own operating range (they run 1e5..1e7); margin factor
sqrt(2 ln(n/k)/kp) = 3.3 vs G4 boost ~15 -- comfortably cleared.

Monolithic training (no per-episode snapshots -- P1 is already
established; this cell tests the score, and monolithic == decomposed by
the attribution unit's 409/409 gate). Seeds 42-44 first (n=1e5 cells
are expensive); extend if the verdict is marginal.

Run: python paper_regime_scale.py
"""
from __future__ import annotations


import os
import random
import statistics as st
import sys
from pathlib import Path
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from research.json_documents import write_new_document

from childes_graduation import build_number_slice, load_corpus  # noqa: E402

SEEDS = [42, 43, 44]
N = 100000
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "paper_regime_scale_results.json")


def cell(seed, sentences, labels, exposures, ambiguous):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )

    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    p = EmergentParser(n=N, k=30, seed=seed, fast_training=True)
    p.morph_label_stim = False
    p.morph_beta_gain = 4.0
    for w in labels:
        if w not in p.stim_map:
            p.register_word(w)
        p.word_grounding[w] = GroundingContext(visual=[w])
    p.train_number(sentences, labels=labels)
    t_train = time.time() - t0

    ok = {"SG": [0, 0], "PL": [0, 0]}
    for w, lab in labels.items():
        if w in ambiguous:
            continue
        got, diag = p.recall_number(w)
        ok[lab][0] += (diag.get("mi_answer", got) == lab)
        ok[lab][1] += 1
    sg = ok["SG"][0] / ok["SG"][1]
    pl = ok["PL"][0] / ok["PL"][1]
    return {"sg": sg, "pl": pl, "bal": (sg + pl) / 2,
            "train_seconds": round(t_train, 1)}


def main():
    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)
    out = {"n": N, "cells": {}}
    for seed in SEEDS:
        r = cell(seed, sentences, labels, exposures, ambiguous)
        out["cells"][seed] = r
        print(f"n={N} L0G4 seed={seed} sg={r['sg']:.3f} pl={r['pl']:.3f} "
              f"bal={r['bal']:.3f} ({r['train_seconds']}s train)",
              flush=True)
    vals = list(out["cells"].values())
    if len(vals) >= 2:
        for key in ("sg", "pl", "bal"):
            xs = [v[key] for v in vals]
            out[key] = {"mean": st.mean(xs),
                        "ci": (4.303 if len(xs) == 3 else 2.776)
                        * st.stdev(xs) / len(xs) ** 0.5}
            print(key, out[key])
    write_new_document(Path(OUT_PATH), out)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
