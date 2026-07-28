"""Word-order induction with the role ANNOTATIONS DELETED.

`word_order_induction.py` recovers the order 6/6, but its lexical preferences
came from role-annotated training, so the honest label was semantic
bootstrapping with a supervised lexicon. This removes the annotation.

THE PIPELINE, end to end and annotation-free at every step:

    build corpus with SceneEvent  ->  strip_roles()  ->  roles_from_scene()
    ->  train  ->  induce word order

After `strip_roles` every `roles` entry is None, so anything the parser learns
about roles must have come through PERCEIVED EVENT STRUCTURE. `roles_from_scene`
consults participants and their causal order, never position -- which is what
keeps the induction non-circular.

PREDICTION, recorded before running
------------------------------------
Identical results to the annotated run: derived roles reproduce the annotations
198/198, so the trained connectome should be the SAME and the induction should
again recover 6/6. A DIFFERENCE would mean the derivation is not actually
equivalent and something else leaks in -- so this is a pipeline test, not a
fishing expedition. The value is provenance, not novelty: the same number, now
obtained without any linguistic annotation in the loop.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence


def train_on_perception(seed: int, *, n: int = 1000, k: int = 50):
    """Train with roles DERIVED FROM SCENES, never read from annotations."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from grounded_corpus import build, strip_roles
    from neural_assemblies.assembly_calculus.emergent.core.scene import (
        roles_from_scene,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.parser import EmergentParser

    corpus = strip_roles(build())
    assert all(all(r is None for r in s.roles) for s in corpus), (
        "annotations survived the strip -- the whole claim rests on this"
    )
    for sentence in corpus:
        sentence.roles = roles_from_scene(sentence)
    filled = sum(1 for s in corpus if any(r for r in s.roles))
    assert filled == len(corpus), (
        f"only {filled}/{len(corpus)} sentences got roles from perception"
    )

    parser = EmergentParser(n=n, k=k, p=0.05, beta=0.1, seed=seed, rounds=10)
    parser.train(create_training_sentences() + corpus)
    return parser


def run(seeds: Sequence[int] = (42,)) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    import copy
    from lesion_aphasia import build_corpus
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs,
    )
    from word_order_induction import (
        AGENTY, ORDERS, PATIENTY, VERBS, _permute, lexical_preference,
    )

    transitive = infer_transitive_verbs(create_training_sentences() + build_corpus())
    items = [(s, v, o) for s in AGENTY for v in VERBS for o in PATIENTY]

    print("\n  GROUNDED induction: roles derived from SceneEvent, annotations")
    print("  deleted before training. No linguistic role label in the loop.\n")

    total = hits = 0
    for seed in seeds:
        trained = train_on_perception(seed)
        prefs = {w: lexical_preference(trained, w, transitive)
                 for w in AGENTY + PATIENTY}
        print(f"  seed {seed} preferences learned from perception: {prefs}")
        print(f"\n  {'true order':<14}{'induced':<24}{'ok'}")
        for true_order in ORDERS:
            sentences = [_permute(s, v, o, true_order) for s, v, o in items]
            agreement = {}
            for hypothesis in ORDERS:
                agree = judged = 0
                for words in sentences:
                    parser = copy.deepcopy(trained)
                    pred = NemoParser(parser, transitive_verbs=transitive,
                                      sequential=True,
                                      word_order_type=hypothesis,
                                      ).parse(list(words))
                    for w in words:
                        if prefs.get(w) in ("AGENT", "PATIENT"):
                            judged += 1
                            agree += 1 if pred.get(w) == prefs[w] else 0
                agreement[hypothesis] = agree / max(judged, 1)
            best = [h for h in ORDERS
                    if agreement[h] == max(agreement.values())]
            ok = best == [true_order]
            total += 1
            hits += 1 if ok else 0
            print(f"  {true_order:<14}{','.join(best):<24}{'YES' if ok else 'no'}")

    print(f"\n  recovered {hits}/{total} with NO role annotation anywhere.")
    print("  Compare word_order_induction.py (annotated): 6/6 per seed.")


if __name__ == "__main__":
    run()
