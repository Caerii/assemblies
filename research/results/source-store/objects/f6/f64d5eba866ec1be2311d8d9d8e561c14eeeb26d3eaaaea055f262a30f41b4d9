"""#151 close-out: the CHILDES Phase 1 headline ON THE PRODUCTION RECIPE
(fixed engine), with per-word rows -- the exposure law re-priced and the
E=1 residual instrumented.

Context: the adoption gate passed at ceiling on the fixed engine
(145288c: min(SG,PL) E>=2 = 1.000 x10 seeds, PL E=1 0.848). The #150
Phase 1 numbers and the exposure law's measured exchange rate predate
the growth-ratchet fix (53e9808) and are STALE. This unit re-prices
them and records the correlates for the new theory question (E=1 reads
0.85 where naive margin arithmetic says ~0.55 cannot be beaten).

REGISTERED BARS AND INTERPRETATION RULES (before the run; Brown at
n=1e5, full recipe = split default + morph_label_stim=False +
morph_beta_gain=4 + scaling untouched, seeds 42-51, byte-identical
construction to adoption_gate.brown_cell):
  P1 HEADLINE REPLICATION (determinism across scripts): per-seed
     sg_e2p/pl_e2p/pl_e1 must MATCH adoption_gate_results.json exactly.
     Any mismatch is a hidden nondeterminism and the unit STOPS there.
  P2 EXPOSURE LAW RE-PRICED: Spearman(per-form exposure, correctness)
     over PL forms, per seed, mean +/- CI over 10 seeds.
     PREDICTION: positive but SMALLER than the pre-fix +0.61 -- ceiling
     accuracy restricts the range. INTERPRETATION RULE (stated now):
     the law is REFUTED only if rho is significantly NEGATIVE (failures
     concentrating at high exposure). A CI containing 0 at ceiling
     accuracy is attenuation-by-success and is reported as such, not as
     transfer failure.
  P3 E=1 RESIDUAL ATTRIBUTION (exploratory, no bar): for every failing
     probe, record exposure, shared-row count with SG cores (the A3
     collision modifier), and own/other drive. WEAK PREDICTION:
     failures sit at E=1 AND high shared-row count. This table feeds
     the registered theory unit on why E=1 clears the naive margin; it
     decides nothing here.
  P4 MECHANISM LIVENESS: record synaptic_scaling_deferred and whether
     any flush ran. PREDICTION: scaling is DORMANT in this recipe (the
     #150 scope limit keeps it off on natural imbalance), so the E19
     schedule-wall claim is scoped to the synthetic config and is NOT
     part of the production-recipe claim (an unmoved metric on a
     dormant mechanism would be vacuous -- census before claiming).
  GUARD: SG >= 0.8 every seed (gate read 1.000).

Run: python childes_phase1_recipe.py
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from research.json_documents import write_new_document

from childes_graduation import build_number_slice, load_corpus  # noqa: E402

SEEDS = list(range(42, 52))
N = 100000
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "childes_phase1_recipe_results.json")


def core_rows(p, word):
    from neural_assemblies.assembly_calculus.ops import (
        project as _ops_project,
    )

    brain = p.brain
    core = p._word_core_area(word)
    phon = p.stim_map.get(word)
    if core is None or phon is None:
        return None
    with brain.read_only():
        _ops_project(brain, phon, core, rounds=max(1, int(p.rounds)))
        return [int(r) for r in brain.areas[core].winners]


def cell(seed, sentences, labels, exposures, ambiguous):
    from collections import Counter

    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )

    random.seed(seed)
    np.random.seed(seed)
    p = EmergentParser(n=N, k=30, seed=seed, fast_training=True)
    p.morph_label_stim = False
    p.morph_beta_gain = 4.0
    for w in labels:
        if w not in p.stim_map:
            p.register_word(w)
        p.word_grounding[w] = GroundingContext(visual=[w])
    p.train_number(sentences, labels=labels)

    eng = p.brain._engine
    liveness = {
        "scaling_deferred": bool(getattr(eng, "synaptic_scaling_deferred",
                                         False)),
        "scaling_areas": sorted(getattr(eng, "synaptic_scaling", []) or []),
    }

    sg_row_count: Counter = Counter()
    for w, lab in labels.items():
        if lab == "SG":
            rows = core_rows(p, w)
            if rows:
                sg_row_count.update(rows)

    rows_out = []
    for w, lab in labels.items():
        if w in ambiguous:
            continue
        got, diag = p.recall_number(w)
        s = diag.get("scores") or {}
        rec = {"form": w, "label": lab, "exposure": exposures.get(w, 0),
               "ok": diag.get("mi_answer", got) == lab,
               "own": s.get(lab),
               "other": s.get("SG" if lab == "PL" else "PL")}
        if lab == "PL":
            cr = core_rows(p, w) or []
            rec["shared_rows"] = sum(
                1 for r in cr if sg_row_count.get(r, 0) > 0)
            rec["shared_weighted"] = sum(sg_row_count.get(r, 0) for r in cr)
        rows_out.append(rec)
    return {"rows": rows_out, "liveness": liveness}


def strata(rows, lab):
    e1 = [r["ok"] for r in rows if r["label"] == lab and r["exposure"] == 1]
    e2 = [r["ok"] for r in rows if r["label"] == lab and r["exposure"] >= 2]
    return (st.mean(e1) if e1 else None, st.mean(e2) if e2 else None)


def mci(xs):
    xs = [x for x in xs if x is not None and x == x]
    if len(xs) < 2:
        return {"mean": xs[0] if xs else None, "ci": None, "n": len(xs)}
    return {"mean": st.mean(xs),
            "ci": 2.262 * st.stdev(xs) / len(xs) ** 0.5, "n": len(xs)}


def main():
    from neural_assemblies.diagnostics import spearman

    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)

    gate_path = os.path.join(os.path.dirname(__file__),
                             "adoption_gate_results.json")
    gate = json.load(open(gate_path))["brown"]

    cells, p1_mismatch = {}, []
    for seed in SEEDS:
        c = cell(seed, sentences, labels, exposures, ambiguous)
        cells[seed] = c
        sg_e1, sg_e2 = strata(c["rows"], "SG")
        pl_e1, pl_e2 = strata(c["rows"], "PL")
        g = gate[str(seed)] if str(seed) in gate else gate[seed]
        for k, v in (("sg_e2p", sg_e2), ("pl_e2p", pl_e2),
                     ("pl_e1", pl_e1)):
            if v is not None and abs(v - g[k]) > 1e-9:
                p1_mismatch.append((seed, k, v, g[k]))
        print(f"seed={seed} sg_e2p={sg_e2:.3f} pl_e2p={pl_e2:.3f} "
              f"pl_e1={pl_e1:.3f} scaling_deferred="
              f"{c['liveness']['scaling_deferred']}", flush=True)

    # P2: exposure law, per seed over PL forms.
    rhos = []
    for seed in SEEDS:
        pl = [r for r in cells[seed]["rows"] if r["label"] == "PL"]
        rhos.append(spearman([r["exposure"] for r in pl],
                             [r["ok"] for r in pl]))
    # P3: pooled failure table.
    fails = [dict(r, seed=seed) for seed in SEEDS
             for r in cells[seed]["rows"] if not r["ok"]]

    analysis = {
        "P1_headline_match": not p1_mismatch,
        "P1_mismatches": p1_mismatch,
        "P2_rho_per_seed": [round(r, 3) if r == r else None for r in rhos],
        "P2_rho": mci(rhos),
        "P3_n_failures": len(fails),
        "P3_failure_exposures": sorted(
            r["exposure"] for r in fails if r["label"] == "PL"),
        "P3_fail_shared_weighted": mci(
            [r.get("shared_weighted") for r in fails if r["label"] == "PL"]),
        "P3_pass_shared_weighted": mci(
            [r.get("shared_weighted") for seed in SEEDS
             for r in cells[seed]["rows"]
             if r["label"] == "PL" and r["ok"]]),
        "P4_liveness": cells[SEEDS[0]]["liveness"],
        "guard_sg_min": min(
            st.mean([r["ok"] for r in cells[s]["rows"]
                     if r["label"] == "SG"]) for s in SEEDS),
    }
    out = {"n": N, "cells": {s: cells[s] for s in SEEDS},
           "analysis": analysis}
    write_new_document(Path(OUT_PATH), out)
    print(json.dumps(analysis, indent=2, default=str))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
