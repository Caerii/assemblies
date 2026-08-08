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
