"""#151 M-unit: does the fixed-column (mass) readout close the inversion?

REGISTERED BARS (before any cell; the mechanism landed at 4540ce6 with
only unit smoke, direction unread):

  M1 RECOVERY on Brown (split, NO scaling, the D/E substrate,
     seeds 42-46): mass balanced-number >= 0.65 at n=3000, AND mass
     SG >= 0.60 at BOTH n (MI reads SG 0.513 / 0.215 -- the readout
     must fix the class the competition was losing, not trade it).
  M2 N-ROBUSTNESS: |mass_bal(n=1e4) - mass_bal(n=3e3)| < 0.10, and the
     paired |delta| is SMALLER than MI's (the extreme-value term is the
     n-dependence; removing it must remove the scaling).
  M3 SYNTHETIC GUARD (the #149-gate protocol: default uniform-50
     curriculum through SENTENCES, n=3000 k=30, seeds 42-46): mass
     balanced-number not worse than MI by more than 0.05 paired --
     the fix must not buy Brown by selling the regime every adopted
     default was measured in.

One exam pass per substrate: mi/overlap/mass all ride in diag, so the
three readouts are compared on IDENTICAL substrates and probes --
paired by construction at the item level.

Run: python mass_readout_gate.py
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

SEEDS = [42, 43, 44, 45, 46]
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "mass_readout_gate_results.json")


def exam(p, labels, exposures, ambiguous):
    per = {}
    for w, lab in labels.items():
        if w in ambiguous:
            continue
        got, diag = p.recall_number(w)
        mi = diag.get("mi_answer", got)
        ov = diag.get("overlap_answer")
        ms = diag.get("mass_answer")
        per[w] = {"label": lab, "exposure": exposures.get(w),
                  "mi": mi == lab,
                  "ov": (ov if ov is not None else mi) == lab,
                  "mass": (ms if ms is not None else mi) == lab}
    out = {}
    for key in ("mi", "ov", "mass"):
        for lab in ("SG", "PL"):
            xs = [r[key] for r in per.values() if r["label"] == lab]
            out[f"{key}_{lab.lower()}"] = st.mean(xs) if xs else None
        out[f"{key}_bal"] = (out[f"{key}_sg"] + out[f"{key}_pl"]) / 2
    return out, per


def brown_cell(n, seed, sentences, labels, exposures, ambiguous):
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
    out, _per = exam(p, labels, exposures, ambiguous)
    return out


def synthetic_cell(seed):
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
    t = CurriculumTrainer(p)
    for stage in ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD",
                  "SENTENCES"):
        t.train_stage(stage)
    labels = {}
    for e in NOUNS:
        pl = e.get("forms", {}).get("plural")
        if not pl:
            continue
        if e["lemma"] in p.stim_map:
            labels[e["lemma"]] = "SG"
        if pl in p.stim_map:
            labels[pl] = "PL"
    out, _per = exam(p, labels, {}, set())
    return out


def mci(xs):
    xs = list(xs)
    if len(xs) < 2:
        return {"mean": xs[0] if xs else None, "ci": None}
    return {"mean": st.mean(xs),
            "ci": 2.776 * st.stdev(xs) / len(xs) ** 0.5}


def main():
    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)

    cells = {}
    for n in (3000, 10000):
        for seed in SEEDS:
            r = brown_cell(n, seed, sentences, labels, exposures, ambiguous)
            cells[f"{n}-{seed}"] = r
            print(f"brown n={n} seed={seed} "
                  f"mi sg/pl {r['mi_sg']:.3f}/{r['mi_pl']:.3f} "
                  f"mass sg/pl {r['mass_sg']:.3f}/{r['mass_pl']:.3f}",
                  flush=True)
    synth = {}
    for seed in SEEDS:
        r = synthetic_cell(seed)
        synth[seed] = r
        print(f"synth seed={seed} mi_bal {r['mi_bal']:.3f} "
              f"mass_bal {r['mass_bal']:.3f}", flush=True)

    d_mass = [cells[f"10000-{s}"]["mass_bal"] - cells[f"3000-{s}"]["mass_bal"]
              for s in SEEDS]
    d_mi = [cells[f"10000-{s}"]["mi_bal"] - cells[f"3000-{s}"]["mi_bal"]
            for s in SEEDS]
    analysis = {
        "M1_mass_bal_3000": mci(cells[f"3000-{s}"]["mass_bal"]
                                for s in SEEDS),
        "M1_mass_sg_3000": mci(cells[f"3000-{s}"]["mass_sg"] for s in SEEDS),
        "M1_mass_sg_10000": mci(cells[f"10000-{s}"]["mass_sg"]
                                for s in SEEDS),
        "M2_delta_mass": mci(d_mass),
        "M2_delta_mi": mci(d_mi),
        "M2_abs_shrinks": st.mean(abs(x) for x in d_mass)
        < st.mean(abs(x) for x in d_mi),
        "M3_paired_mass_minus_mi": mci(synth[s]["mass_bal"]
                                       - synth[s]["mi_bal"] for s in SEEDS),
    }
    out = {"brown": cells, "synthetic": synth, "analysis": analysis}
    write_new_document(Path(OUT_PATH), out)
    print(json.dumps(analysis, indent=2))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
