"""#151 D-unit: the dead-fiber hunt -- WHY does seed 45's
NOUN_CORE->NUMBER_PL fiber read own-drive EXACTLY 0.0 on 45/46 trained
PL words (one word at 0.679) while eleven sibling seeds read nonzero?

Context (the_gate_passed_and_found_a_dead_fiber.md): the adoption gate
blocked on this. Exact zero on a trained fiber is the silent-no-op /
dead-fiber signature (content-addressed / deferred-init family; five
doors closed, #50 still open). This unit is a CENSUS, not a fix:
measure the fiber, split the door, identify the served word.

REGISTERED QUESTIONS AND SIGNATURES (before the run; seeds 45 defect +
42 control, n=1e5, L0G4, Brown slice -- byte-identical construction to
adoption_gate.brown_cell):
  D0 REPRODUCTION: the seed-45 recall signature must reproduce
     deterministically (own=0.0 on >=40 of 46 PL words, one word >0.5).
     If it does not, the one-seeding-path determinism assumption is
     broken and the unit STOPS there -- that would itself be the
     finding.
  D1 STRUCTURAL CENSUS of NOUN_CORE->{NUMBER_PL, NUMBER_SG}, both
     seeds: shape, nnz, fiber_extent, materialized_count,
     extent_desync, col/row coverage. Pre-registered door split:
       (a) extent_desync > 0 AND the PL label-image columns fall in the
           uncovered tail -> DESYNC DOOR (area grew through a fiber
           this one was not expanded with; last columns are
           uninitialised zeros).
       (b) nnz == 0 or total mass ~ 0 -> ZEROED DOOR (reset/flush
           zeroed the connectome; check flush_synaptic_scaling).
       (c) census structurally healthy but per-word mass concentrated
           on ONE word's rows/cols -> COLLAPSE-OR-COLLISION DOOR
           (single-attractor training collapse, or content-addressed
           init collision); then the recall zeros must come from the
           words' core ROWS carrying no weight into the competition.
       (d) word core rows >= w.shape[0] or image cols >= w.shape[1]
           -> INDEX-SPACE DOOR (rows/cols outside the allocated block;
           two-index-spaces family).
     The door is decided by the MEASUREMENT, and more than one may be
     open (the #145 lesson: three defects masked each other).
  D2 THE ONE SERVED WORD: identified as argmax own-drive.
     PRE-STATED PREDICTION: it is the FIRST PL word in episode order --
     its columns were materialized through this fiber before whatever
     event killed the rest. A different word refutes that account.
  D3 WRITER/READER SPLIT: compare the PL label-image columns (the
     reader's fixed columns) against the columns holding the fiber's
     trained mass (the writer's). Disjoint sets = the
     writer-and-reader-must-share-the-lookup signature even though
     recall drive, not mass, is the failing readout.
  D4 CONTROL: every instrument runs on seed 42 and must read healthy
     there; an instrument that reads defective on BOTH seeds indicts
     the instrument, not the seed.

Run: python dead_fiber_hunt.py           (full, ~2 min)
     DFH_SMOKE=1 python dead_fiber_hunt.py   (API smoke at n=3000,
     seed 42 only -- checks nothing directional)
"""
from __future__ import annotations


import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from research.json_documents import write_new_document

from childes_graduation import build_number_slice, load_corpus  # noqa: E402

SMOKE = os.environ.get("DFH_SMOKE") == "1"
N = 3000 if SMOKE else 100000
SEEDS = [42] if SMOKE else [45, 42]
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "dead_fiber_hunt_results.json")


def build(seed, sentences, labels):
    """Byte-identical to adoption_gate.brown_cell's construction."""
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
    return p


def episode_order(p, sentences, labels):
    """First-episode index per word, replicating train_number's gates."""
    order, ep = {}, 0
    for sent in sentences:
        for word in sent:
            if word not in p.stim_map:
                continue
            g = p.word_grounding.get(word)
            if g is None or g.dominant_modality not in ("visual", "motor"):
                continue
            if labels.get(word) not in ("SG", "PL"):
                continue
            ep += 1
            order.setdefault(word, ep)
    return order


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


def dense(w):
    return np.asarray(w.todense() if hasattr(w, "todense") else w,
                      dtype=float)


def fiber_summary(p, src, dst):
    """Structural census of one fiber: the D1 instruments."""
    eng = p.brain._engine
    conn = getattr(eng, "_area_conns", {}).get(src, {}).get(dst)
    out = {"src": src, "dst": dst, "exists": conn is not None}
    if conn is None:
        return out, None
    w = dense(getattr(conn, "weights", None))
    out["shape"] = list(w.shape)
    out["nnz"] = int((w > 0).sum())
    out["total_mass"] = float(w.sum())
    out["extent"] = _safe(eng, "fiber_extent", src, dst)
    out["materialized_dst"] = _safe(eng, "materialized_count", dst)
    out["materialized_src"] = _safe(eng, "materialized_count", src)
    if out["extent"] is not None and out["materialized_dst"] is not None:
        out["extent_desync"] = out["materialized_dst"] - out["extent"]
    colsum = w.sum(axis=0)
    rowsum = w.sum(axis=1)
    out["cols_with_mass"] = int((colsum > 0).sum())
    out["rows_with_mass"] = int((rowsum > 0).sum())
    if (colsum > 0).any():
        pos = colsum[colsum > 0]
        med = float(np.median(pos))
        out["col_p99_over_median"] = (
            float(np.percentile(pos, 99)) / med if med > 0 else None)
        top = np.sort(colsum)[::-1]
        out["top30_col_mass_share"] = float(top[:30].sum() / colsum.sum())
    return out, w


def _safe(eng, name, *args):
    try:
        v = getattr(eng, name)(*args)
    except Exception:                                    # noqa: BLE001
        return None
    return None if v is None else int(v)


def label_image_cols(p, feature, label):
    rounds = max(1, int(p.rounds))
    area = f"{feature}_{label}"
    return list(p._feature_image_cache.get(
        (area, label, rounds, "compact"), []))


def cell(seed, sentences, labels, exposures, ambiguous):
    p = build(seed, sentences, labels)
    order = episode_order(p, sentences, labels)

    # D0: the recall signature, per PL word (also fills the image cache).
    words = []
    for w, lab in labels.items():
        if lab != "PL" or w in ambiguous:
            continue
        got, diag = p.recall_number(w)
        s = diag.get("scores") or {}
        words.append({"form": w, "exposure": exposures.get(w),
                      "episode": order.get(w),
                      "own": s.get("PL"), "other": s.get("SG"),
                      "ok": diag.get("mi_answer", got) == "PL"})

    pl_img = label_image_cols(p, "NUMBER", "PL")
    sg_img = label_image_cols(p, "NUMBER", "SG")

    # D1: structural census of both value fibers.
    pl_fiber, w_pl = fiber_summary(p, "NOUN_CORE", "NUMBER_PL")
    sg_fiber, _w_sg = fiber_summary(p, "NOUN_CORE", "NUMBER_SG")

    # Per-word mass geometry on the PL fiber (D1c/D1d/D3).
    if w_pl is not None:
        nrows, ncols = w_pl.shape
        img_in = [c for c in pl_img if c < ncols]
        pl_fiber["img_cols"] = len(pl_img)
        pl_fiber["img_cols_out_of_range"] = len(pl_img) - len(img_in)
        pl_fiber["img_col_mass"] = (
            float(w_pl[:, img_in].sum()) if img_in else 0.0)
        for rec in words:
            rows = core_rows(p, rec["form"]) or []
            rin = [r for r in rows if r < nrows]
            rec["rows_out_of_range"] = len(rows) - len(rin)
            rec["row_mass"] = float(w_pl[rin, :].sum()) if rin else 0.0
            rec["mass_into_img"] = (
                float(w_pl[np.ix_(rin, img_in)].sum())
                if rin and img_in else 0.0)

    served = max(words, key=lambda r: (r["own"] or 0.0))
    zeros = sum(1 for r in words if (r["own"] or 0.0) == 0.0)
    return {
        "n_pl_words": len(words),
        "own_exact_zero": zeros,
        "served_word": {k: served[k] for k in
                        ("form", "own", "episode", "exposure")},
        "first_pl_episode_word": min(
            (r for r in words if r["episode"] is not None),
            key=lambda r: r["episode"])["form"],
        "pl_fiber": pl_fiber,
        "sg_fiber": sg_fiber,
        "pl_image_cols": pl_img,
        "sg_image_cols_n": len(sg_img),
        "words": words,
    }


def main():
    utts, _s, _f = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)

    cells = {}
    for seed in SEEDS:
        r = cell(seed, sentences, labels, exposures, ambiguous)
        cells[seed] = r
        pf = r["pl_fiber"]
        print(f"seed={seed} zeros={r['own_exact_zero']}/{r['n_pl_words']} "
              f"served={r['served_word']['form']} "
              f"(own={r['served_word']['own']}, "
              f"ep={r['served_word']['episode']}) "
              f"first_pl={r['first_pl_episode_word']}", flush=True)
        print(f"  PL fiber shape={pf.get('shape')} nnz={pf.get('nnz')} "
              f"extent={pf.get('extent')} "
              f"mat_dst={pf.get('materialized_dst')} "
              f"desync={pf.get('extent_desync')} "
              f"cols_with_mass={pf.get('cols_with_mass')} "
              f"top30_share={pf.get('top30_col_mass_share')}", flush=True)
        print(f"  SG fiber shape={r['sg_fiber'].get('shape')} "
              f"nnz={r['sg_fiber'].get('nnz')} "
              f"desync={r['sg_fiber'].get('extent_desync')}", flush=True)

    write_new_document(Path(OUT_PATH), {"n": N, "cells": cells})
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
