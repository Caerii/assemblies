"""A/B: gating-based role assignment vs the current symbolic path.

The gate for step 4 of `research/PRIMITIVES_AUDIT.md`. The current parser scores
each candidate role with `_role_binding_margin` and blocks used roles with a
Python `inhibited` set, reaching 0.930 +/-0.043 on irreversible items
(20 seeds, `research/experiments/lesion_aphasia.py`). The NEMO path instead
opens fibers and lets the derived project map decide.

Replacing the working path is only worth discussing if the neural one MATCHES.
"It runs" is not the bar, and neither is "it is more principled".

Both paths are measured on the SAME trained parser per seed, so the comparison
is paired -- differences are the decision rule, not the wiring.

Run: .venv/Scripts/python.exe research/experiments/nemo_vs_symbolic.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np


def _mean_ci(xs) -> tuple:
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return (float(a.mean()) if a.size else float("nan")), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(a.size))


def _items():
    """Canonical SVO items -- both paths should get these right.

    Deliberately NOT the lesion study's non-canonical items: those were built so
    position and semantics CONFLICT, which is a separate question from whether
    gating assigns roles at all. Start with the easy case; if the gating path
    cannot do canonical SVO there is no point measuring the hard one.
    """
    out = []
    for subj, verb, obj in (
        ("dog", "chases", "cat"), ("cat", "sees", "dog"),
        ("dog", "finds", "ball"), ("cat", "chases", "bird"),
        ("boy", "sees", "girl"), ("girl", "finds", "book"),
    ):
        out.append(([subj, verb, obj],
                    {subj: "AGENT", verb: "ACTION", obj: "PATIENT"}))
    return out


def _score(pred: Dict[str, object], gold: Dict[str, str]) -> tuple:
    ok = sum(1 for w, r in gold.items() if pred.get(w) == r)
    return ok, len(gold)


def run(seeds: Sequence[int] = (42, 43, 44, 45, 46, 47, 48, 49)) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (
        get_parser_cache,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs,
    )

    corpus = create_training_sentences()
    transitive = infer_transitive_verbs(corpus)
    print("gating vs symbolic role assignment -- paired, same parser per seed")
    print(f"transitive verbs inferred from the corpus: {sorted(transitive)}\n")

    sym_acc: List[float] = []
    nemo_acc: List[float] = []
    per_role = {"AGENT": [0, 0], "ACTION": [0, 0], "PATIENT": [0, 0]}

    for seed in seeds:
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        s_ok = s_tot = n_ok = n_tot = 0
        for words, gold in _items():
            got_sym = parser.parse(list(words))["roles"]
            a, b = _score(got_sym, gold)
            s_ok, s_tot = s_ok + a, s_tot + b

            np_ = NemoParser(parser, transitive_verbs=transitive)
            got_nemo = np_.parse(list(words))
            a, b = _score(got_nemo, gold)
            n_ok, n_tot = n_ok + a, n_tot + b
            for w, r in gold.items():
                per_role[r][1] += 1
                if got_nemo.get(w) == r:
                    per_role[r][0] += 1

        sym_acc.append(s_ok / max(s_tot, 1))
        nemo_acc.append(n_ok / max(n_tot, 1))
        print(f"  seed {seed}: symbolic {sym_acc[-1]:.3f}   gating {nemo_acc[-1]:.3f}",
              flush=True)

    ms, cs = _mean_ci(sym_acc)
    mn, cn = _mean_ci(nemo_acc)
    dm, dc = _mean_ci(np.asarray(nemo_acc) - np.asarray(sym_acc))
    print(f"\n  symbolic (current) : {ms:.3f} +/-{cs:.3f}")
    print(f"  gating   (NEMO)    : {mn:.3f} +/-{cn:.3f}")
    print(f"  paired delta       : {dm:+.3f} +/-{dc:.3f}")
    print("\n  gating accuracy by role:")
    for r, (ok, tot) in per_role.items():
        print(f"    {r:<9}{ok}/{tot}")
    if np.isfinite(dc) and dm - dc > 0.02:
        print("\n  -> gating is BETTER; replacing the symbolic path is on the table.")
    elif np.isfinite(dc) and dm + dc < -0.02:
        print("\n  -> gating is WORSE. The symbolic shortcut is doing real work "
              "the gating does not yet reproduce. That is a finding, not a bug "
              "to tune away -- report it before changing anything.")
    else:
        print("\n  -> indistinguishable at this n; neither replacing nor "
              "rejecting is justified yet.")


if __name__ == "__main__":
    run()
