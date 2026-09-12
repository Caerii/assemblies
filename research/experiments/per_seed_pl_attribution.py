"""#151 A2-unit: WHY is PL bimodal per seed at n=1e5 under the paper
regime? Attribution before any mechanism change (the standing rule).

Context (the_papers_regime_works...md): L0G4 at n=1e5 read balanced
0.946 / 0.640 / 0.750 across seeds 42-44 -- PL 0.891 / 0.283 / 0.500.
The gate is gain-invariant (G2 reproduces it) and LOOKED config- and
n-invariant at 3 seeds. Suspect: the PL words' core-row draw. This unit
measures instead of pattern-matching.

REGISTERED QUESTIONS AND SIGNATURES (before the run; 12 seeds 42-53,
n=1e5, L0G4, Brown slice, scaling OFF):
  A1 WORD OR SEED: per-word failure correlated across seeds?
     - If the SAME words fail in every seed -> word property; check
       exposure first (1-exposure PL words are at chance BY LAW).
     - If bad seeds fail on RANDOM words -> seed-global factor.
     Signature: split-half correlation of per-word failure rates across
     seed halves; and per-word failure vs exposure.
  A2 WHICH DRIVE MOVES: for failing vs passing PL words (within and
     across seeds), is own-drive LOWER or other-drive HIGHER? Paired
     medians on diag scores.
  A3 COLLISION AT WORD LEVEL: per (seed, word), count the word's core
     rows shared with >=1 SG word's core assembly (geometry, no
     thresholds) and the multiplicity-weighted count. Signature:
     spearman(shared_rows, correct) < 0 with the n=12-seed CI excluding
     0 -> the collision-load law confirmed at word level. Null result
     -> collisions are NOT the gate and the account is refuted.
  A4 IS THE BIMODALITY REAL: 12-seed distribution of per-seed PL
     accuracy (report the distribution, never a bare mean -- the
     bimodal lesson); and correlate per-seed PL at n=1e5 with the same
     seeds' L0G4 n=1e4 values (42-46, from paper_regime_2x2): a strong
     cross-n correlation would argue AGAINST pure geometry (geometry
     redraws with n) and constrain the hypothesis space sharply.

Run: python per_seed_pl_attribution.py
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

SEEDS = list(range(42, 54))
N = 100000
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "per_seed_pl_attribution_results.json")


def core_rows(p, word):
    from neural_assemblies.assembly_calculus.ops import (
        project as _ops_project,
    )

    brain = p.brain
    core = p._word_core_area(word)
    phon = p.stim_map.get(word)
    if core is None or phon is None:
        return None
    rounds = max(1, int(p.rounds))
    with brain.read_only():
        _ops_project(brain, phon, core, rounds=rounds)
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

    # Geometry: every SG word's core rows, with multiplicity.
    sg_row_count: Counter = Counter()
    for w, lab in labels.items():
        if lab != "SG":
            continue
        rows = core_rows(p, w)
        if rows:
            sg_row_count.update(rows)

    pl_rows = []
    for w, lab in labels.items():
        if lab != "PL" or w in ambiguous:
            continue
        rows = core_rows(p, w) or []
        got, diag = p.recall_number(w)
        s = diag.get("scores") or {}
        pl_rows.append({
            "form": w, "exposure": exposures.get(w),
            "ok": diag.get("mi_answer", got) == "PL",
            "own": s.get("PL"), "other": s.get("SG"),
            "shared_rows": sum(1 for r in rows if sg_row_count.get(r, 0) > 0),
            "shared_weighted": sum(sg_row_count.get(r, 0) for r in rows),
        })
    sg_ok = []
    for w, lab in labels.items():
        if lab != "SG" or w in ambiguous:
            continue
        got, diag = p.recall_number(w)
        sg_ok.append(diag.get("mi_answer", got) == "SG")
    return {"pl_rows": pl_rows,
            "pl_acc": st.mean(r["ok"] for r in pl_rows),
            "sg_acc": st.mean(sg_ok)}


def main():
    from neural_assemblies.diagnostics import spearman

    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)

    cells = {}
    for seed in SEEDS:
        r = cell(seed, sentences, labels, exposures, ambiguous)
        cells[seed] = r
        print(f"seed={seed} pl={r['pl_acc']:.3f} sg={r['sg_acc']:.3f}",
              flush=True)

    # A1: per-word failure rates, split-half across seeds; exposure link.
    forms = [row["form"] for row in cells[SEEDS[0]]["pl_rows"]]
    fail = {f: [not row["ok"]
                for s in SEEDS
                for row in cells[s]["pl_rows"] if row["form"] == f]
            for f in forms}
    half_a = {f: st.mean(v[:6]) for f, v in fail.items()}
    half_b = {f: st.mean(v[6:]) for f, v in fail.items()}
    a1_split_half = spearman([half_a[f] for f in forms],
                             [half_b[f] for f in forms])
    a1_exposure = spearman(
        [exposures.get(f, 0) for f in forms],
        [st.mean(fail[f]) for f in forms])

    # A2: drive medians for failing vs passing PL probes (pooled).
    pooled = [row for s in SEEDS for row in cells[s]["pl_rows"]
              if row["own"] is not None]
    med = lambda xs: st.median(xs) if xs else None  # noqa: E731
    a2 = {
        "own_fail": med([r["own"] for r in pooled if not r["ok"]]),
        "own_pass": med([r["own"] for r in pooled if r["ok"]]),
        "other_fail": med([r["other"] for r in pooled if not r["ok"]]),
        "other_pass": med([r["other"] for r in pooled if r["ok"]]),
    }

    # A3: collision vs correctness, per seed then aggregated.
    a3_per_seed = []
    for s in SEEDS:
        rows = cells[s]["pl_rows"]
        rho = spearman([r["shared_weighted"] for r in rows],
                       [r["ok"] for r in rows])
        a3_per_seed.append(rho)
    a3_clean = [x for x in a3_per_seed if x == x]
    a3 = {"per_seed": a3_per_seed,
          "mean": st.mean(a3_clean) if a3_clean else None,
          "ci": (2.201 * st.stdev(a3_clean) / len(a3_clean) ** 0.5
                 if len(a3_clean) > 2 else None)}

    # A4: distribution + cross-n correlation with the 2x2's L0G4 n=1e4.
    pl_dist = sorted(round(cells[s]["pl_acc"], 3) for s in SEEDS)
    try:
        d2 = json.load(open(os.path.join(os.path.dirname(__file__),
                                         "paper_regime_2x2_results.json")))
        common = [s for s in SEEDS if f"10000-L0G4-{s}" in d2["cells"]]
        a4_cross_n = spearman(
            [cells[s]["pl_acc"] for s in common],
            [d2["cells"][f"10000-L0G4-{s}"]["pl"] for s in common])
        a4_common = len(common)
    except Exception as e:  # noqa: BLE001
        a4_cross_n, a4_common = None, f"unavailable: {e}"

    analysis = {
        "A1_split_half_word_corr": a1_split_half,
        "A1_exposure_vs_failure": a1_exposure,
        "A2_drive_medians": a2,
        "A3_collision_vs_correct": a3,
        "A4_pl_distribution": pl_dist,
        "A4_cross_n_corr": a4_cross_n,
        "A4_cross_n_seeds": a4_common,
    }
    out = {"cells": {s: {"pl_acc": cells[s]["pl_acc"],
                         "sg_acc": cells[s]["sg_acc"],
                         "pl_rows": cells[s]["pl_rows"]} for s in SEEDS},
           "analysis": analysis}
    write_new_document(Path(OUT_PATH), out)
    print(json.dumps(analysis, indent=2, default=str))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
