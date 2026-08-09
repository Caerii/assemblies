"""#151 G-unit: ATTRIBUTE the class-imbalance inversion before fixing it.

The Brown graduation (#150) left one unattributed fact: with scaling OFF
the split still biases toward the rare class, and the bias GROWS with n
(SG 0.513 -> 0.215 from n=3000 to n=10000 while PL rose 0.709 -> 0.872).
The alpha* capacity account was already refuted by intervention. Two
candidate mechanisms, registered with their signatures BEFORE this run:

  H1 BACKGROUND EXTREME VALUE. For a word of class X, the OPPOSING
     area's recall drive is untrained background; its top-k winners are
     extreme order statistics over the candidate pool, which GROW with
     n, while trained drive compounds (1+beta)^exposure independent of
     n. Signature: median drive_other rises n=3k -> 10k; median
     drive_own stays flat. H1 alone predicts BOTH classes degrade with
     n -- it cannot explain PL rising -- so H1 can at most be a
     component.

  H2 WINNER CHURN. The frequent area's per-episode k-WTA runs under
     363 competing forms plus ongoing recruitment, so a given form's
     ~3 episodes reinforce DIFFERENT winner sets (image never coheres);
     the rare area's label attractor is revisited every ~7 episodes and
     stays stable, so PL images cohere. Churn grows with the candidate
     pool. Signature: per-form TRAINING-image consecutive-episode
     overlap SG << PL at both n, gap widening with n; and
     exposure-matched own-drive SG << PL.

REGISTERED BARS (seeds 42-46, arms n in {3000, 10000}, split default,
scaling OFF -- the D/E substrate):
  G1  median(drive_other) ratio (n=1e4 / n=3e3) > 1.3 with the paired
      CI excluding 1.0, AND median(drive_own) ratio inside [0.77, 1.3]
      -> H1's component is REAL. Both flat -> H1 refuted.
  G2  per-form training-image stability (mean |W_i ∩ W_{i+1}|/k over a
      form's consecutive episodes): mean_PL - mean_SG > 0 with paired
      CI excluding 0 at BOTH n -> H2 confirmed; gap(n=1e4) > gap(n=3e3)
      (paired) -> churn explains the n-scaling.
  G3  DECISION DECOMPOSITION on exposure-matched forms (2-4 episodes):
      if own-drive(SG)/own-drive(PL) < 0.77 the asymmetry lives in the
      TRAINED image (H2); if in [0.77, 1.3] and other-drive decides,
      it lives in the BACKGROUND (H1). Recorded either way.

METHOD NOTE (comparability gate, E18 pattern): training is DECOMPOSED
into single-word train_number calls so winners can be snapshotted per
episode. With deferred scaling OFF the per-call phase-end flush is a
no-op, so the decomposition should be operation-identical to the
monolithic call; this is CHECKED (seed 42, n=3000: exam answers must
match the monolithic run) and a mismatch is reported as a finding, not
patched around.

Run: python imbalance_attribution.py
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

SEEDS = [42, 43, 44, 45, 46]
NS = (3000, 10000)
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "imbalance_attribution_results.json")


def make_parser(n, seed, labels):
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
    return p


def episode_stream(sentences, labels):
    """(word, label) episodes in exactly train_number's iteration order."""
    for sent in sentences:
        for w in sent:
            lab = labels.get(w)
            if lab in ("SG", "PL"):
                yield w, lab


def run_cell(n, seed, sentences, labels, exposures, ambiguous):
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        NUMBER, feature_value_area,
    )

    p = make_parser(n, seed, labels)
    area_of = {lab: feature_value_area(NUMBER, lab) for lab in ("SG", "PL")}
    winners_hist: dict = {}
    for w, lab in episode_stream(sentences, labels):
        p.train_number([[w]], labels={w: lab})
        ws = frozenset(p.brain.areas[area_of[lab]].winners)
        winners_hist.setdefault(w, []).append(ws)

    # Per-form training-image stability (H2 instrument).
    stab = {"SG": [], "PL": []}
    for w, hist in winners_hist.items():
        if len(hist) < 2:
            continue
        ovs = [len(a & b) / p.k for a, b in zip(hist, hist[1:])]
        stab[labels[w]].append(st.mean(ovs))

    # Recall drives (H1/G3 instrument) + accuracy (context).
    rows = []
    for w, lab in labels.items():
        if w in ambiguous:
            continue
        got, diag = p.recall_number(w)
        s = diag.get("scores") or {}
        if set(s) != {"SG", "PL"}:
            continue
        other = "PL" if lab == "SG" else "SG"
        rows.append({
            "form": w, "label": lab, "exposure": exposures[w],
            "own": s[lab], "other": s[other],
            "ok": diag.get("mi_answer", got) == lab,
        })

    def med(xs):
        return st.median(xs) if xs else float("nan")

    matched = [r for r in rows if 2 <= r["exposure"] <= 4]
    return {
        "acc_sg": st.mean(r["ok"] for r in rows if r["label"] == "SG"),
        "acc_pl": st.mean(r["ok"] for r in rows if r["label"] == "PL"),
        "med_own": med([r["own"] for r in rows]),
        "med_other": med([r["other"] for r in rows]),
        "med_own_sg": med([r["own"] for r in rows if r["label"] == "SG"]),
        "med_own_pl": med([r["own"] for r in rows if r["label"] == "PL"]),
        "stab_sg": st.mean(stab["SG"]) if stab["SG"] else None,
        "stab_pl": st.mean(stab["PL"]) if stab["PL"] else None,
        "n_stab": {k: len(v) for k, v in stab.items()},
        "matched_own_sg": med([r["own"] for r in matched
                               if r["label"] == "SG"]),
        "matched_own_pl": med([r["own"] for r in matched
                               if r["label"] == "PL"]),
        "matched_other_sg": med([r["other"] for r in matched
                                 if r["label"] == "SG"]),
    }


def equivalence_check(sentences, labels, ambiguous):
    """Decomposed vs monolithic training, seed 42 n=3000: same answers?"""
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        NUMBER, feature_value_area,  # noqa: F401
    )

    answers = {}
    for mode in ("mono", "decomp"):
        p = make_parser(3000, 42, labels)
        if mode == "mono":
            p.train_number(sentences, labels=labels)
        else:
            for w, lab in episode_stream(sentences, labels):
                p.train_number([[w]], labels={w: lab})
        got = {}
        for w in labels:
            if w in ambiguous:
                continue
            g, d = p.recall_number(w)
            got[w] = d.get("mi_answer", g)
        answers[mode] = got
    same = sum(answers["mono"][w] == answers["decomp"][w]
               for w in answers["mono"])
    return {"agree": same, "total": len(answers["mono"]),
            "frac": same / len(answers["mono"])}


def main():
    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)

    eq = equivalence_check(sentences, labels, ambiguous)
    print(f"equivalence gate: {eq['agree']}/{eq['total']} "
          f"({eq['frac']:.4f})", flush=True)

    cells = {}
    for n in NS:
        for seed in SEEDS:
            r = run_cell(n, seed, sentences, labels, exposures, ambiguous)
            cells[f"{n}-{seed}"] = r
            print(f"n={n} seed={seed} acc sg/pl {r['acc_sg']:.3f}/"
                  f"{r['acc_pl']:.3f} med own/other {r['med_own']:.3f}/"
                  f"{r['med_other']:.3f} stab sg/pl "
                  f"{r['stab_sg']:.4f}/{r['stab_pl']:.4f}", flush=True)

    def paired(metric_fn):
        deltas = [metric_fn(cells[f"10000-{s}"]) - metric_fn(cells[f"3000-{s}"])
                  for s in SEEDS]
        return {"mean": st.mean(deltas),
                "ci": 2.776 * st.stdev(deltas) / len(deltas) ** 0.5}

    def ratio(metric):
        rs = [cells[f"10000-{s}"][metric] / cells[f"3000-{s}"][metric]
              for s in SEEDS]
        return {"mean": st.mean(rs),
                "ci": 2.776 * st.stdev(rs) / len(rs) ** 0.5}

    def gap(cell):
        return cell["stab_pl"] - cell["stab_sg"]

    def mci(xs):
        xs = list(xs)
        return {"mean": st.mean(xs),
                "ci": 2.776 * st.stdev(xs) / len(xs) ** 0.5}

    analysis = {
        "G1_other_ratio": ratio("med_other"),
        "G1_own_ratio": ratio("med_own"),
        "G2_gap_3000": mci(gap(cells[f"3000-{s}"]) for s in SEEDS),
        "G2_gap_10000": mci(gap(cells[f"10000-{s}"]) for s in SEEDS),
        "G2_gap_growth": paired(gap),
        "G3_matched_own_sg_over_pl_3000": st.mean(
            cells[f"3000-{s}"]["matched_own_sg"]
            / cells[f"3000-{s}"]["matched_own_pl"] for s in SEEDS),
        "G3_matched_own_sg_over_pl_10000": st.mean(
            cells[f"10000-{s}"]["matched_own_sg"]
            / cells[f"10000-{s}"]["matched_own_pl"] for s in SEEDS),
    }
    out = {"equivalence": eq, "cells": cells, "analysis": analysis}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(analysis, indent=2))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
