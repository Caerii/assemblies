"""Induce word order from UNAMBIGUOUS sentences, apply it to AMBIGUOUS ones.

THE QUESTION, stated so it can fail
------------------------------------
Word order is recoverable when nouns carry lexical role bias (dog acts, ball is
acted on) -- `grounded_induction.py`, 6/6. It cannot be recovered from
REVERSIBLE sentences, where both participants are animate and either could act:
"boy chases girl" and "girl chases boy" are equally plausible scenes, so no
semantic cue distinguishes them. That is not a defect to engineer around. It is
precisely WHY languages need word order.

So the claim worth testing is not "induce order from ambiguous sentences" -- it
is that order induced from the UNAMBIGUOUS ones TRANSFERS to the ambiguous ones.
That is the semantic-bootstrapping story properly stated, and it is the same
irreversible/reversible split the lesion study used, now on the induction side.

PROTOCOL
--------
* INDUCE on animal-verb-object sentences only (dog chases ball): the lexicon is
  biased, so agreement has evidence to work with.
* EVALUATE on person-verb-person sentences (boy chases girl), held out and never
  used for induction: both nouns are role-BALANCED by construction, so nothing
  but the induced order can assign the roles.
* Repeat for all six true orders. A method that always answers SVO scores 1/6.

PREDICTIONS, recorded before running
-------------------------------------
1. Induction on biased sentences recovers the true order 6/6, as before.
2. The induced order parses held-out BALANCED sentences correctly -- this is
   the transfer claim, and the one that can fail.
3. A WRONG order parses those same balanced sentences at ~0.0 rather than at
   chance, because gating is a deterministic positional template: it does not
   guess, it systematically reverses. (Measured before at 0.000 on irreversible
   items.)
4. Balanced sentences alone CANNOT induce the order -- agreement should tie
   across the orders that share a verb position, since neither noun prefers a
   role. Reported explicitly, because a tie here is the CORRECT answer and not
   a failure.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Dict, List, Sequence

ORDERS = ("SVO", "SOV", "VSO", "VOS", "OSV", "OVS")
BALANCED_WORDS = ("boy", "girl", "man", "woman", "child")
BIASED_AGENTS = ("dog", "cat", "bird", "horse", "mouse", "fox")
BIASED_PATIENTS = ("ball", "book", "food", "table", "car", "cup", "stick", "box")


def _order_words(subject: str, verb: str, obj: str, order: str) -> List[str]:
    slot = {"S": subject, "V": verb, "O": obj}
    return [slot[c] for c in order]


def run(seeds: Sequence[int] = (42,), n_train: int = 1200) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    from rich_corpus import generate
    from word_order_induction import lexical_preference
    from neural_assemblies.assembly_calculus.emergent.core.scene import (
        roles_from_scene,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs,
    )
    from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser

    verbs = ("chases", "finds", "sees", "holds")
    biased_items = [(s, v, o) for s in BIASED_AGENTS[:4] for v in verbs
                    for o in BIASED_PATIENTS[:4]]
    balanced_items = [(s, v, o) for s in BALANCED_WORDS for v in verbs
                      for o in BALANCED_WORDS if s != o]

    print(f"\n  induce on {len(biased_items)} BIASED items "
          f"(animal-verb-object), evaluate on {len(balanced_items)} held-out "
          f"BALANCED items (person-verb-person)\n")
    print(f"  {'true':<6}{'induced':<10}{'transfer to balanced':>22}"
          f"{'wrong-order ctrl':>20}{'balanced-only induction':>26}")

    for seed in seeds:
        for true_order in ORDERS:
            corpus = generate(n_train, seed=seed, word_order=true_order)
            # Roles from PERCEPTION, never from the annotation.
            for sentence in corpus:
                sentence.roles = roles_from_scene(sentence)
            parser = EmergentParser(n=1000, k=50, p=0.05, beta=0.1,
                                    seed=seed, rounds=10)
            parser.train(corpus)
            transitive = infer_transitive_verbs(corpus)

            prefs = {w: lexical_preference(parser, w, transitive)
                     for w in BIASED_AGENTS[:4] + BIASED_PATIENTS[:4]}

            # 1. Induce from the BIASED items. Surface strings are laid out in
            # the TRUE order and read under each HYPOTHESIS -- that asymmetry is
            # the whole test, so it must not be collapsed.
            scores = {}
            for h in ORDERS:
                agree = judged = 0
                for s, v, o in biased_items:
                    words = _order_words(s, v, o, true_order)
                    pred = NemoParser(copy.deepcopy(parser),
                                      transitive_verbs=transitive,
                                      sequential=True, word_order_type=h,
                                      ).parse(words)
                    for w in words:
                        if prefs.get(w) in ("AGENT", "PATIENT"):
                            judged += 1
                            agree += 1 if pred.get(w) == prefs[w] else 0
                scores[h] = agree / max(judged, 1)
            best = [h for h in ORDERS if scores[h] == max(scores.values())]
            induced = best[0] if len(best) == 1 else "/".join(best)

            # 2. TRANSFER: parse held-out BALANCED items with the induced order
            def role_accuracy(hypothesis: str) -> float:
                ok = tot = 0
                for s, v, o in balanced_items:
                    words = _order_words(s, v, o, true_order)
                    pred = NemoParser(copy.deepcopy(parser),
                                      transitive_verbs=transitive,
                                      sequential=True,
                                      word_order_type=hypothesis).parse(words)
                    tot += 2
                    ok += 1 if pred.get(s) == "AGENT" else 0
                    ok += 1 if pred.get(o) == "PATIENT" else 0
                return ok / max(tot, 1)

            transfer = role_accuracy(best[0]) if len(best) == 1 else float("nan")
            wrong = next(h for h in ORDERS if h != true_order)
            control = role_accuracy(wrong)

            # 3. Could BALANCED items alone have induced it? Expect ties.
            bal_scores = {}
            judged = 0
            for h in ORDERS:
                agree = judged = 0
                for s, v, o in balanced_items:
                    words = _order_words(s, v, o, true_order)
                    pred = NemoParser(copy.deepcopy(parser),
                                      transitive_verbs=transitive,
                                      sequential=True, word_order_type=h,
                                      ).parse(words)
                    for w in (s, o):
                        pref = prefs.get(w)
                        if pref in ("AGENT", "PATIENT"):
                            judged += 1
                            agree += 1 if pred.get(w) == pref else 0
                bal_scores[h] = agree / max(judged, 1) if judged else float("nan")
            bal_best = [h for h in ORDERS
                        if bal_scores[h] == max(bal_scores.values())]
            bal_note = ("no preference at all" if judged == 0
                        else "/".join(bal_best))

            flag = "OK" if induced == true_order else "MISS"
            print(f"  {true_order:<6}{induced:<10}{transfer:>22.3f}"
                  f"{control:>20.3f}{bal_note:>26}   {flag}")

    print("\n  transfer = held-out reversible sentences parsed with the INDUCED")
    print("  order. control = the same sentences under a wrong order.")
    print("  balanced-only = what those sentences could have induced alone.")


if __name__ == "__main__":
    run()
