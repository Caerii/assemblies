"""#151 R-unit: the adoption gate -- the full recipe's headline on
CHILDES, stratified by exposure, plus the synthetic-guard root-cause.

REGISTERED BARS (before any cell; the_bimodality_was_27_coin_flips.md
set the frame):
  R1 ADOPTION: on Brown at n=1e5 under the full recipe (split +
     routing-only + gain 4, scaling OFF), seeds 42-51:
     min(SG, PL) accuracy on the E>=2 exam >= 0.75 (mean over seeds,
     per-seed distribution reported -- never a bare mean after the
     bimodality lesson). The E=1 stratum is REPORTED, not barred: it
     is COLT22's own O(log k) floor.
  R2 GUARD ROOT-CAUSE: the paper regime's synthetic deficit (-0.071
     overall) is PREDICTED to be the same margin arithmetic at the
     synthetic corpus's tiny exposures. Verify on the #149-gate
     protocol (uniform-50 curriculum, n=3000, seeds 42-46), per-word
     exposures read from the parser's own _morph_exposure counters:
       (a) paired (L0G4 - L1G1) on the E>=2 subset >= -0.02;
       (b) the deficit CONCENTRATES at E=1: delta(E=1) < delta(E>=2).
     If R2 fails, the deficit is NOT exposure arithmetic and adoption
     stops for investigation -- verify, don't assume.
  ADOPTION FORM (decided at registration, not after data): defaults do
     NOT flip -- the paper regime becomes the documented PRODUCTION
     recipe (set at scale), because the default corpus's tiny
     exposures sit under the raised margin by R2's own account. The
     E=1 floor is priced, not hidden: production reporting is
     stratified by exposure.

Run: python adoption_gate.py
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from childes_graduation import build_number_slice, load_corpus  # noqa: E402

SEEDS = list(range(42, 52))
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "adoption_gate_results.json")


def strat(rows):
    """Per-class x per-stratum accuracies from per-word rows."""
    out = {}
    for lab in ("SG", "PL"):
        for name, f in (("e1", lambda e: e == 1), ("e2p", lambda e: e >= 2)):
            xs = [r["ok"] for r in rows
                  if r["label"] == lab and f(r["exposure"])]
            out[f"{lab.lower()}_{name}"] = st.mean(xs) if xs else None
            out[f"n_{lab.lower()}_{name}"] = len(xs)
    both = [r["ok"] for r in rows]
    out["overall"] = st.mean(both)
    sg = [r["ok"] for r in rows if r["label"] == "SG"]
    pl = [r["ok"] for r in rows if r["label"] == "PL"]
    out["bal"] = (st.mean(sg) + st.mean(pl)) / 2
    return out


def brown_cell(seed, sentences, labels, exposures, ambiguous):
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )

    random.seed(seed)
    np.random.seed(seed)
    p = EmergentParser(n=100000, k=30, seed=seed, fast_training=True)
    p.morph_label_stim = False
    p.morph_beta_gain = 4.0
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
        rows.append({"label": lab, "exposure": exposures.get(w, 0),
                     "ok": diag.get("mi_answer", got) == lab})
    return strat(rows)


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
    exposure = {k.split(":", 1)[1]: v
                for k, v in getattr(p, "_morph_exposure", {}).items()
                if k.startswith("NUMBER:")}
    rows = []
    for e in NOUNS:
        pl_form = e.get("forms", {}).get("plural")
        if not pl_form:
            continue
        for w, lab in ((e["lemma"], "SG"), (pl_form, "PL")):
            if w not in p.stim_map:
                continue
            got, diag = p.recall_number(w)
            rows.append({"label": lab, "exposure": exposure.get(w, 0),
                         "ok": diag.get("mi_answer", got) == lab})
    return strat(rows), rows


def mci(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return {"mean": xs[0] if xs else None, "ci": None, "n": len(xs)}
    t = {5: 2.776, 10: 2.262}.get(len(xs), 2.262)
    return {"mean": st.mean(xs), "ci": t * st.stdev(xs) / len(xs) ** 0.5,
            "n": len(xs)}


def main():
    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)

    brown = {}
    for seed in SEEDS:
        r = brown_cell(seed, sentences, labels, exposures, ambiguous)
        brown[seed] = r
        print(f"brown seed={seed} sg_e2p={r['sg_e2p']:.3f} "
              f"pl_e2p={r['pl_e2p']:.3f} pl_e1={r['pl_e1']:.3f} "
              f"bal={r['bal']:.3f}", flush=True)

    synth = {}
    for arm, (label_on, gain) in (("L1G1", (True, 1.0)),
                                  ("L0G4", (False, 4.0))):
        for seed in SEEDS[:5]:
            s, _rows = synthetic_cell(seed, label_on, gain)
            synth[f"{arm}-{seed}"] = s
            print(f"synth {arm} seed={seed} overall={s['overall']:.3f} "
                  f"bal={s['bal']:.3f}", flush=True)

    r1_min_e2p = [min(brown[s]["sg_e2p"], brown[s]["pl_e2p"])
                  for s in SEEDS]

    def synth_delta(key):
        ds = []
        for s in SEEDS[:5]:
            a, b = synth[f"L0G4-{s}"], synth[f"L1G1-{s}"]
            va = [a[f"sg_{key}"], a[f"pl_{key}"]]
            vb = [b[f"sg_{key}"], b[f"pl_{key}"]]
            va = [x for x in va if x is not None]
            vb = [x for x in vb if x is not None]
            if va and vb:
                ds.append(st.mean(va) - st.mean(vb))
        return mci(ds)

    analysis = {
        "R1_min_e2p": mci(r1_min_e2p),
        "R1_per_seed_min_e2p": [round(x, 3) for x in r1_min_e2p],
        "headline_bal": mci(brown[s]["bal"] for s in SEEDS),
        "headline_sg_e2p": mci(brown[s]["sg_e2p"] for s in SEEDS),
        "headline_pl_e2p": mci(brown[s]["pl_e2p"] for s in SEEDS),
        "headline_pl_e1": mci(brown[s]["pl_e1"] for s in SEEDS),
        "R2a_delta_e2p": synth_delta("e2p"),
        "R2b_delta_e1": synth_delta("e1"),
    }
    out = {"brown": brown, "synthetic": synth, "analysis": analysis}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(analysis, indent=2))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
