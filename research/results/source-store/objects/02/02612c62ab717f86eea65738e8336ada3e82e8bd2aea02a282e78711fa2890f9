"""#151 P-unit: the papers' regime, 2x2 -- {label stim: on, OFF} x
{fiber beta gain: 1, 4} on the Brown substrate.

REGISTERED (what_the_papers_actually_prescribe.md; mechanism e4f6da8
landed with unit smoke only, direction unread). One DEVIATION from the
note, stated: the note registered a kp axis; the engine has no
per-fiber p, and COLT22 Remark 2's margin is a beta-vs-kp TRADEOFF, so
axis 2 is the fiber beta gain through the existing tested bracket
(gain 4 -> beta_eff 0.2 -> boost (1.2)^15 ~ 15 vs extreme factor
2.5-2.8 at kp=1.5).

BARS (before any cell; seeds 42-46, n in {3000, 10000}, split default,
scaling OFF -- the D/E substrate; MI is THE readout under test, since
the papers' construction is supposed to make it self-calibrating):
  P1 MECHANISM: per-word training-image x class-attractor overlap
     < 0.5 in the L0 (routing-only) arms at both n -- the images must
     actually become word-conditioned (baseline measured 0.998). A
     score gain without P1 is not the papers' mechanism.
  P2 RECOVERY: MI balanced >= 0.70 AND SG >= 0.60 in L0G4 at both n.
  P3 COMPOSITION: paired over seeds, L0G1 > L1G1 and L1G4 > L1G1 and
     L0G4 >= max(L0G1, L1G4) on balanced MI -- the two axes fix
     DIFFERENT parts of the mechanism (winner selection vs margin) and
     must compose.
  P4 CAPACITY (directional): once images are word-specific the value
     areas must STORE ~363 SG assemblies; alpha* ~ 1.15 n/k = 115 at
     n=3000 vs 383 at n=10000. Prediction: in L0 arms the (PL - SG)
     gap SHRINKS from n=3000 to n=10000 -- the capacity law becomes
     binding exactly when the attractor stops carrying the load.
  GUARD: the synthetic #149-gate protocol (uniform-50 curriculum,
     n=3000 k=30, seeds 42-46) under L0G4: balanced number not worse
     than the L1G1 baseline by more than 0.05 paired.

Training is DECOMPOSED into single-word calls (equivalence 409/409
verified in the attribution unit; scaling OFF makes per-call flush a
no-op) so per-word final images can be snapped for P1.

Run: python paper_regime_2x2.py
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from research.json_documents import write_new_document

from childes_graduation import build_number_slice, load_corpus  # noqa: E402
from imbalance_attribution import episode_stream  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ARMS = {"L1G1": (True, 1.0), "L0G1": (False, 1.0),
        "L1G4": (True, 4.0), "L0G4": (False, 4.0)}
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "paper_regime_2x2_results.json")


def brown_cell(n, seed, label_on, gain, sentences, labels, exposures,
               ambiguous):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        NUMBER, feature_value_area,
    )
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )

    random.seed(seed)
    np.random.seed(seed)
    p = EmergentParser(n=n, k=30, seed=seed, fast_training=True)
    p.morph_label_stim = label_on
    p.morph_beta_gain = gain
    for w in labels:
        if w not in p.stim_map:
            p.register_word(w)
        p.word_grounding[w] = GroundingContext(visual=[w])

    area_of = {lab: feature_value_area(NUMBER, lab) for lab in ("SG", "PL")}
    final_img = {}
    col_counts = {"SG": Counter(), "PL": Counter()}
    for w, lab in episode_stream(sentences, labels):
        p.train_number([[w]], labels={w: lab})
        ws = frozenset(p.brain.areas[area_of[lab]].winners)
        final_img[w] = ws
        col_counts[lab].update(ws)
    attractor = {lab: frozenset(c for c, _ in col_counts[lab].most_common(p.k))
                 for lab in ("SG", "PL")}
    img_att = {"SG": [], "PL": []}
    for w, ws in final_img.items():
        lab = labels[w]
        img_att[lab].append(len(ws & attractor[lab]) / p.k)

    rows = []
    for w, lab in labels.items():
        if w in ambiguous:
            continue
        got, diag = p.recall_number(w)
        rows.append({"label": lab, "exposure": exposures.get(w),
                     "mi": diag.get("mi_answer", got) == lab})

    def acc(lab):
        xs = [r["mi"] for r in rows if r["label"] == lab]
        return st.mean(xs) if xs else None

    sg, pl = acc("SG"), acc("PL")
    return {"sg": sg, "pl": pl, "bal": (sg + pl) / 2,
            "img_att_sg": st.mean(img_att["SG"]),
            "img_att_pl": st.mean(img_att["PL"])}


def synthetic_cell(seed, label_on, gain):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.curriculum import (
        CurriculumTrainer,
    )
    from neural_assemblies.assembly_calculus.emergent.vocabulary_builder \
        import build_vocabulary_preset
    from neural_assemblies.lexicon.data import NOUNS

    random.seed(seed)
    np.random.seed(seed)
    p = EmergentParser(n=3000, k=30, seed=seed,
                      vocabulary=build_vocabulary_preset("core"),
                      fast_training=True)
    p.morph_label_stim = label_on
    p.morph_beta_gain = gain
    t = CurriculumTrainer(p)
    for stage in ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
                  "SENTENCES"):
        t.train_stage(stage)
    labels = {}
    for e in NOUNS:
        pl_form = e.get("forms", {}).get("plural")
        if not pl_form:
            continue
        if e["lemma"] in p.stim_map:
            labels[e["lemma"]] = "SG"
        if pl_form in p.stim_map:
            labels[pl_form] = "PL"
    ok = {"SG": [0, 0], "PL": [0, 0]}
    for w, lab in labels.items():
        got, diag = p.recall_number(w)
        ok[lab][0] += (diag.get("mi_answer", got) == lab)
        ok[lab][1] += 1
    sg = ok["SG"][0] / ok["SG"][1]
    pl = ok["PL"][0] / ok["PL"][1]
    return {"sg": sg, "pl": pl, "bal": (sg + pl) / 2}


def mci(xs):
    xs = list(xs)
    return {"mean": st.mean(xs),
            "ci": 2.776 * st.stdev(xs) / len(xs) ** 0.5}


def main():
    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)

    cells = {}
    for n in (3000, 10000):
        for arm, (label_on, gain) in ARMS.items():
            for seed in SEEDS:
                r = brown_cell(n, seed, label_on, gain, sentences, labels,
                               exposures, ambiguous)
                cells[f"{n}-{arm}-{seed}"] = r
                print(f"n={n} {arm} seed={seed} sg={r['sg']:.3f} "
                      f"pl={r['pl']:.3f} bal={r['bal']:.3f} "
                      f"imgatt sg/pl {r['img_att_sg']:.3f}/"
                      f"{r['img_att_pl']:.3f}", flush=True)

    synth = {}
    for arm in ("L1G1", "L0G4"):
        label_on, gain = ARMS[arm]
        for seed in SEEDS:
            synth[f"{arm}-{seed}"] = synthetic_cell(seed, label_on, gain)
            print(f"synth {arm} seed={seed} "
                  f"bal={synth[f'{arm}-{seed}']['bal']:.3f}", flush=True)

    def arm_stat(n, arm, key):
        return mci(cells[f"{n}-{arm}-{s}"][key] for s in SEEDS)

    def paired(n, a, b):
        return mci(cells[f"{n}-{a}-{s}"]["bal"] - cells[f"{n}-{b}-{s}"]["bal"]
                   for s in SEEDS)

    analysis = {}
    for n in (3000, 10000):
        analysis[f"P1_imgatt_sg_L0G4_{n}"] = arm_stat(n, "L0G4", "img_att_sg")
        analysis[f"P2_bal_L0G4_{n}"] = arm_stat(n, "L0G4", "bal")
        analysis[f"P2_sg_L0G4_{n}"] = arm_stat(n, "L0G4", "sg")
        analysis[f"P3_L0G1_minus_L1G1_{n}"] = paired(n, "L0G1", "L1G1")
        analysis[f"P3_L1G4_minus_L1G1_{n}"] = paired(n, "L1G4", "L1G1")
        analysis[f"P3_L0G4_minus_L1G1_{n}"] = paired(n, "L0G4", "L1G1")
        analysis[f"P4_gap_L0G4_{n}"] = mci(
            cells[f"{n}-L0G4-{s}"]["pl"] - cells[f"{n}-L0G4-{s}"]["sg"]
            for s in SEEDS)
    analysis["GUARD_synth_L0G4_minus_L1G1"] = mci(
        synth[f"L0G4-{s}"]["bal"] - synth[f"L1G1-{s}"]["bal"] for s in SEEDS)

    out = {"cells": cells, "synthetic": synth, "analysis": analysis}
    write_new_document(Path(OUT_PATH), out)
    print(json.dumps(analysis, indent=2))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
