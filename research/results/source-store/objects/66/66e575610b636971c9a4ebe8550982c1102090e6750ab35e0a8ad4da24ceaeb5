"""#151 Z-unit: four decision rules on identical substrates, per-word rows.

The readout campaign so far, each gate honestly failed forward:
  MI          compares boosted-typical vs selected-extreme -> inverts
              with n (attribution unit).
  RAW MASS    fixed columns kill the extreme term, but shared-row
              inflation reads CLASS mass -> bias flips class with n
              (mass_readout_gate raw).
  EXCESS      base-rate subtraction kills the mean bias -> best Brown
              readout so far (bal 0.66/0.61 vs MI 0.65/0.53) but SG
              stays under 0.60 at both n.

HYPOTHESIS H-VAR (registered before this run): the residual is a
VARIANCE asymmetry -- the frequent class's image carries ~8x the total
mass, so the shared-row noise term of excess has much larger variance
in the SG image than the PL image; a mean-corrected comparison still
loses SG words into that noise. Signatures:
  Z1  per-class spread check: std over words of excess-into-SG-image
      exceeds std of excess-into-PL-image by > 2x (the variance model
      is REAL, else the z-rule has no foundation and H-VAR is refuted);
  Z2  the z-rule (excess / sqrt(baseline), Poisson-family normalizer)
      symmetrizes the classes: |sg - pl| gap under z < gap under
      excess at both n;
  Z3  z balanced >= excess balanced at both n (paired, 5 seeds).
If Z1 holds but Z2/Z3 fail, the noise is not sqrt(baseline)-scaled and
the next normalizer must be MEASURED (per-image empirical std), not
guessed. If Z1 fails, the readout family is exhausted at the mean level
and the registered continuation moves to the TRAINING side: raise the
word-core drive share during value-area training (the phon_weight=6
lever family) so images become word-conditioned in WHICH neurons fire.

Arms: Brown slice, split, NO scaling, n in {3000, 10000}, seeds 42-46.
Per-word rows saved (label, exposure, per-label raw/baseline/excess,
answers under each rule).

Run: python readout_decision_rules.py
"""
from __future__ import annotations

import json
import math
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

SEEDS = [42, 43, 44, 45, 46]
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "readout_decision_rules_results.json")


def cell(n, seed, sentences, labels, exposures, ambiguous):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )

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
        raw = diag.get("mass_raw") or {}
        base = diag.get("mass_baseline") or {}
        exc = diag.get("mass_scores") or {}
        if set(raw) != {"SG", "PL"}:
            continue
        z = {L: exc[L] / math.sqrt(base[L] + 1e-9) for L in ("SG", "PL")}
        rows.append({
            "form": w, "label": lab, "exposure": exposures.get(w),
            "mi": diag.get("mi_answer"),
            "excess": max(exc, key=exc.get),
            "z": max(z, key=z.get),
            "exc_SG": exc["SG"], "exc_PL": exc["PL"],
            "base_SG": base["SG"], "base_PL": base["PL"],
        })
    return rows


def per_class_acc(rows, rule):
    out = {}
    for lab in ("SG", "PL"):
        xs = [r[rule] == lab for r in rows if r["label"] == lab]
        out[lab.lower()] = st.mean(xs) if xs else None
    out["bal"] = (out["sg"] + out["pl"]) / 2
    return out


def main():
    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)

    all_cells = {}
    for n in (3000, 10000):
        for seed in SEEDS:
            rows = cell(n, seed, sentences, labels, exposures, ambiguous)
            key = f"{n}-{seed}"
            spread_sg = st.stdev(r["exc_SG"] for r in rows)
            spread_pl = st.stdev(r["exc_PL"] for r in rows)
            summ = {
                "mi": per_class_acc(rows, "mi"),
                "excess": per_class_acc(rows, "excess"),
                "z": per_class_acc(rows, "z"),
                "Z1_spread_ratio": spread_sg / max(spread_pl, 1e-12),
            }
            all_cells[key] = {"summary": summ, "rows": rows}
            print(f"n={n} seed={seed} "
                  f"exc sg/pl {summ['excess']['sg']:.3f}/"
                  f"{summ['excess']['pl']:.3f} "
                  f"z sg/pl {summ['z']['sg']:.3f}/{summ['z']['pl']:.3f} "
                  f"spreadSG/PL {summ['Z1_spread_ratio']:.2f}", flush=True)

    def mci(xs):
        xs = list(xs)
        return {"mean": st.mean(xs),
                "ci": 2.776 * st.stdev(xs) / len(xs) ** 0.5}

    analysis = {}
    for n in (3000, 10000):
        cs = [all_cells[f"{n}-{s}"]["summary"] for s in SEEDS]
        analysis[f"Z1_spread_ratio_{n}"] = mci(c["Z1_spread_ratio"]
                                               for c in cs)
        analysis[f"Z2_gap_excess_{n}"] = mci(abs(c["excess"]["sg"]
                                                 - c["excess"]["pl"])
                                             for c in cs)
        analysis[f"Z2_gap_z_{n}"] = mci(abs(c["z"]["sg"] - c["z"]["pl"])
                                        for c in cs)
        analysis[f"Z3_zbal_minus_excbal_{n}"] = mci(c["z"]["bal"]
                                                    - c["excess"]["bal"]
                                                    for c in cs)
        analysis[f"z_bal_{n}"] = mci(c["z"]["bal"] for c in cs)
        analysis[f"excess_bal_{n}"] = mci(c["excess"]["bal"] for c in cs)
    out = {"cells": {k: v["summary"] for k, v in all_cells.items()},
           "rows": {k: v["rows"] for k, v in all_cells.items()},
           "analysis": analysis}
    write_new_document(Path(OUT_PATH), out)
    print(json.dumps(analysis, indent=2))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
