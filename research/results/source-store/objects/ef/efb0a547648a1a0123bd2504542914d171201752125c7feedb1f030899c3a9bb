"""Morph-feature test sets and recall scoring -- the canonical copy.

Promoted from `research/experiments/what_variation_buys.py` (task #129)
after THREE copies existed within a day (two experiments and a test's
private `_attested_forms`) -- the duplication signal this module answers.
Experiments stay thin protocol descriptions; the measurement helpers live
here.

WHY A RAW-DATA SCAN AND NOT `lookup_lexicon_entry`. The test sets must
exclude surfaces that are AMBIGUOUS at form level, and the single-slot
lookup cannot see ambiguity (it resolves homographs to one entry; the
multi-entry index fixed the visibility, but the exclusion decision is the
measurement's, not the index's). The three exclusion classes, each found
by an instrumented dry run that caught the test set lying:

  * NOUN/VERB HOMOGRAPHS ("answers/hopes/loves/fears/surprises"): 3sg verb
    forms in the corpus AND noun plurals in the lexicon. Undecidable at
    form level; excluded from every class.
  * ZERO-DERIVATION PASTS ("put/let/cut/read"): past == lemma, and lemmas
    register unconditionally and train as present. Excluded (w != lemma).
  * SELF-PLURALS ("fish"): plural == lemma. Same exclusion.

Membership is corpus-attested by construction: inflected surfaces only
enter `stim_map` via `SentenceGenerator.register_surface_forms` when they
occur in a sentence.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


def attested_morph_sets(parser) -> Dict[str, List[str]]:
    """Corpus-attested, form-level-unambiguous test items per class.

    Returns {"PAST": [...], "PRESENT": [...], "PL": [...], "SG": [...]}.
    PRESENT is the 3sg finite form (the bare lemma is ambiguous); SG is the
    lemma of exactly the nouns whose plural is attested (paired classes).
    """
    from neural_assemblies.lexicon.data import NOUNS, VERBS

    verb_surfaces, noun_surfaces = set(), set()
    for e in VERBS:
        verb_surfaces.add(e["lemma"])
        verb_surfaces.update(
            v for v in e.get("forms", {}).values() if isinstance(v, str))
    for e in NOUNS:
        noun_surfaces.add(e["lemma"])
        pl = e.get("forms", {}).get("plural")
        if pl:
            noun_surfaces.add(pl)

    past: List[str] = []
    pres: List[str] = []
    plural: List[str] = []
    singular: List[str] = []
    attested = set(parser.stim_map)
    for e in VERBS:
        forms = e.get("forms", {})
        for w, bucket in ((forms.get("past"), past),
                          (forms.get("3sg"), pres)):
            if (w and w in attested and w != e["lemma"]
                    and w not in noun_surfaces):
                bucket.append(w)
    for e in NOUNS:
        pl = e.get("forms", {}).get("plural")
        if (pl and pl in attested and pl != e["lemma"]
                and pl not in verb_surfaces):
            plural.append(pl)
            if (e["lemma"] in attested
                    and e["lemma"] not in verb_surfaces):
                singular.append(e["lemma"])
    return {"PAST": past, "PRESENT": pres, "PL": plural, "SG": singular}


def feature_images_compact(
    parser, feature_area: str, stim_by_label: Dict[str, str],
) -> Dict[str, List[int]]:
    """Label images in COMPACT space (the weight matrix's coordinates).

    The `_recall_morph_feature` image protocol (stimulus projection plus
    ``rounds - 1`` recurrent settles), promoted from E12
    (`drive_decomposition.py`, task #141) on its second use. Compact
    winners -- NOT `_snap` Assemblies -- because the consumer indexes the
    engine's weight matrices, which live in compact coordinates.
    """
    brain = parser.brain
    rounds = max(1, int(parser.rounds))
    images: Dict[str, List[int]] = {}
    with brain.read_only():
        for label, stim in stim_by_label.items():
            if stim not in brain.stimuli:
                continue
            brain.inhibit_areas([feature_area])
            brain.project({stim: [feature_area]}, {})
            for _ in range(rounds - 1):
                brain.project({stim: [feature_area]},
                              {feature_area: [feature_area]})
            images[label] = [int(c) for c in
                             brain.areas[feature_area].winners]
    return images


def item_afferent_mass(
    parser, word: str, feature_area: str,
    images: Dict[str, List[int]],
) -> "dict | None":
    """Summed weight from `word`'s core-assembly rows into each label image.

    The E12 readout (task #141, rho 0.846 against item correctness): rows
    are the recall probe's own core activation in compact space, columns
    are `feature_images_compact` label images. WHERE-structured by
    construction -- it sums over exactly the columns the recall readout
    compares -- which is why it sees the decision the all-candidates
    input_drive probe cannot. Returns {label: mass} or None if the word
    has no phon/route.
    """
    from neural_assemblies.assembly_calculus.ops import (
        project as _ops_project,
    )

    brain = parser.brain
    # The feature connectome is owned by the target feature area.  Reading the
    # primary engine would miss explicit/alternate owners in mixed brains.
    eng = brain.engine_for(feature_area)
    core = parser._word_core_area(word)
    phon = parser.stim_map.get(word)
    conn = eng._area_conns.get(core, {}).get(feature_area)
    w = getattr(conn, "weights", None) if conn is not None else None
    if phon is None or w is None or getattr(w, "ndim", 0) != 2:
        return None
    rounds = max(1, int(parser.rounds))
    with brain.read_only():
        brain.inhibit_areas([feature_area])
        _ops_project(brain, phon, core, rounds=rounds)
        rows = [int(r) for r in brain.areas[core].winners]
    return afferent_mass(w, rows, images)


def afferent_mass(w, rows: List[int],
                  images: Dict[str, List[int]]) -> Dict[str, float]:
    """Summed weight from `rows` into each label image's columns.

    The matrix-sum core of `item_afferent_mass`, exposed on its second
    use (#151): the `morph_readout="mass"` recall path scores areas at
    FIXED label-image columns with the core assembly it has ALREADY
    settled, so it must not re-settle. Rows and columns are COMPACT
    indices and are bounds-filtered against the matrix -- a compact
    index beyond the materialized extent has no trained mass by
    definition.
    """
    rows = [r for r in rows if r < w.shape[0]]
    out = {}
    for label, img in images.items():
        cols = [c for c in img if c < w.shape[1]]
        out[label] = (float(np.asarray(w[np.ix_(rows, cols)]).sum())
                      if rows and cols else 0.0)
    return out


def excess_afferent_mass(w, rows: List[int],
                         images: Dict[str, List[int]]) -> Dict[str, dict]:
    """Word-specific EXCESS mass into each label image: raw mass minus
    the image's base rate for a random row-set of the same size.

    WHY THE CORRECTION EXISTS (#151, mass_readout_gate M1 FAILURE,
    recorded before this was written): raw mass into the frequent
    class's image is inflated by SHARED CORE ROWS -- with hundreds of
    words' core assemblies colliding in one area, any word's rows carry
    other words' boosts, so raw mass reads CLASS TOTAL mass (flipping
    the bias toward SG at n=3000 and back toward PL at n=10000 as
    collisions thin). Subtracting expected mass for |rows| random rows
    (|rows| * colsum / n_rows) leaves the word-SPECIFIC evidence --
    what this word's episodes wrote above what any word would read.

    Returns {label: {"mass": raw, "baseline": expected, "excess": diff}}.
    """
    rows = [r for r in rows if r < w.shape[0]]
    out: Dict[str, dict] = {}
    n_rows_total = max(1, int(w.shape[0]))
    for label, img in images.items():
        cols = [c for c in img if c < w.shape[1]]
        if not rows or not cols:
            out[label] = {"mass": 0.0, "baseline": 0.0, "excess": 0.0}
            continue
        mass = float(np.asarray(w[np.ix_(rows, cols)]).sum())
        colsum = float(np.asarray(w[:, cols]).sum())
        baseline = colsum * (len(rows) / n_rows_total)
        out[label] = {"mass": mass, "baseline": baseline,
                      "excess": mass - baseline}
    return out


def score_recall(
    recall: Callable[[str], Tuple],
    items_by_label: Dict[str, List[str]],
) -> dict:
    """Per-class accuracy, tie rate, margins, and image separation.

    `recall` is `parser.recall_tense` or `parser.recall_number`. A None
    answer is a TIE -- the readout refusing rather than guessing -- and
    counts against accuracy without being a wrong label. `_balanced` is the
    unweighted mean over classes (the metric frequency imbalance cannot
    flatter); `_image_separation` near 1.0 means the label images have
    merged and no accuracy from this readout is trustworthy.

    The np.mean sites here average over ITEMS within one run; seed
    summaries belong in `diagnostics.ensemble`/`paired_delta`.
    """
    res: dict = {}
    seps: List[float] = []
    for label, items in items_by_label.items():
        n_ok = n_tie = 0
        margins: List[float] = []
        for w in items:
            got, diag = recall(w)
            if diag.get("image_separation") is not None:
                seps.append(diag["image_separation"])
            if got is None:
                n_tie += 1
            elif got == label:
                n_ok += 1
            if diag.get("margin") is not None:
                margins.append(diag["margin"])
        res[label] = {
            "n": len(items), "acc": (n_ok / len(items)) if items else None,
            "tie_rate": (n_tie / len(items)) if items else None,
            "mean_margin": float(np.mean(margins)) if margins else None,
        }
    accs = [v["acc"] for v in res.values() if v["acc"] is not None]
    res["_balanced"] = float(np.mean(accs)) if accs else None
    res["_image_separation"] = float(np.mean(seps)) if seps else None
    return res
