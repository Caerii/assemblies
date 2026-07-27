"""A/B: gating-based role assignment vs the current symbolic path.

The gate for step 4 of `research/PRIMITIVES_AUDIT.md`. The current parser scores
each candidate role with `_role_binding_margin` and blocks used roles with a
Python `inhibited` set. The NEMO path instead opens fibers and lets the derived
project map decide where a word binds.

Replacing a working path is only worth discussing if the neural one MATCHES.
"It runs" is not the bar, and neither is "it is more principled".

Both paths run on the SAME trained parser per seed, so the comparison is paired:
differences are the decision rule, not the wiring.

WHY THERE ARE TWO ITEM KINDS -- the first version of this experiment was wrong
-----------------------------------------------------------------------------
It measured canonical SVO only and reported gating at 1.000 +/-0.000 against
symbolic 0.889. That was a TAUTOLOGY twice over: the readout read the GATE
rather than the brain, and canonical items are exactly what a positional
template gets right by construction. A perfect score there is evidence of
nothing.

Non-canonical items put position and lexical experience in CONFLICT ("ball
chases dog" -- only what the corpus taught about `ball` makes `dog` the agent).
A pure positional template must score ~0 on them. That contrast is the only
thing separating "binds roles" from "recites the word order".

Run: .venv/Scripts/python.exe research/experiments/nemo_vs_symbolic.py
"""

from __future__ import annotations

import os
import sys
import copy
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


def _items(kind: str = "canonical"):
    """Delegate to the lesion study's VETTED items for this corpus.

    Do not invent items here. Three times now, hand-written test sentences have
    contradicted what the corpus actually taught -- the first A/B scored words
    (`cat`, `chases`) that were not in the SENTENCES vocabulary at all, and the
    second put AGENT-only nouns in patient position, so the symbolic path was
    penalised for correctly refusing the gold. `lesion_aphasia.test_items()` is
    already built against `build_corpus()` and checked:

      reversible   balanced nouns in canonical order. No lexical preference
                   exists by construction, so ONLY word order can assign the
                   roles -- this is what a gating mechanism should be good at.
      irreversible role-exclusive nouns in NON-canonical order, so position says
                   the wrong thing and only lexical experience rescues it --
                   this is what a positional mechanism must fail.

    That is exactly the contrast the A/B needs, already validated.
    """
    from lesion_aphasia import test_items as _ti

    want = "reversible" if kind == "canonical" else "irreversible"
    return [(list(words), gold) for words, gold, k in _ti() if k == want]


#: ACTION is excluded from scoring. VERB_CORE -> ROLE_ACTION is never
#: materialized (measured: shape (0,0), zero weight), so no verb is ever
#: neurally bound to its thematic role -- in EITHER path. The symbolic parser
#: reports ACTION from the word's category, not from a binding. Scoring it would
#: credit both paths for something neither computes.
SCORED_ROLES = ("AGENT", "PATIENT")


def _score(pred: Dict[str, object], gold: Dict[str, str]) -> tuple:
    scored = {w: r for w, r in gold.items() if r in SCORED_ROLES}
    ok = sum(1 for w, r in scored.items() if pred.get(w) == r)
    return ok, len(scored)


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

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lesion_aphasia import build_corpus, train_parser as _train

    corpus = create_training_sentences() + build_corpus()
    transitive = infer_transitive_verbs(corpus)
    print("gating vs symbolic role assignment -- paired, same parser per seed")
    print(f"transitive verbs inferred from corpus: {sorted(transitive)}")
    print()

    results: Dict[str, tuple] = {}
    for kind in ("canonical", "non_canonical"):
        sym_acc: List[float] = []
        nemo_acc: List[float] = []
        for seed in seeds:
            parser = _train(seed)
            s_ok = s_tot = n_ok = n_tot = 0
            for words, gold in _items(kind):
                # SEPARATE COPIES. NemoParser mutates the brain
                # (prepare_targets clears winners, fix/unfix changes state), so
                # sharing one parser let the gating run corrupt every later
                # symbolic parse -- symbolic scored 0.083 on reversible items
                # where lesion_aphasia measures 1.000 for the same parser.
                a, b = _score(
                    copy.deepcopy(parser).parse(list(words))["roles"], gold)
                s_ok, s_tot = s_ok + a, s_tot + b
                got = NemoParser(copy.deepcopy(parser),
                                 transitive_verbs=transitive).parse(list(words))
                a, b = _score(got, gold)
                n_ok, n_tot = n_ok + a, n_tot + b
            sym_acc.append(s_ok / max(s_tot, 1))
            nemo_acc.append(n_ok / max(n_tot, 1))
        results[kind] = (sym_acc, nemo_acc)
        ms, cs = _mean_ci(sym_acc)
        mn, cn = _mean_ci(nemo_acc)
        print(f"  {kind:<14} symbolic {ms:.3f} +/-{cs:.3f}   "
              f"gating {mn:.3f} +/-{cn:.3f}", flush=True)

    cs_, cn_ = results["canonical"]
    ns_, nn_ = results["non_canonical"]
    dm, dc = _mean_ci(np.asarray(cn_) - np.asarray(cs_))
    m_non_n, _ = _mean_ci(nn_)
    m_non_s, _ = _mean_ci(ns_)

    print()
    print(f"  canonical paired delta (gating - symbolic): {dm:+.3f} +/-{dc:.3f}")
    print()
    print("  A pure positional template scores ~1.0 canonical and ~0.0")
    print("  non-canonical. If gating shows that profile it is reciting word")
    print("  order rather than binding roles, and canonical accuracy means")
    print("  nothing.")
    print()

    m_can_n, _ = _mean_ci(cn_)
    if m_can_n < 0.3 and m_non_n < 0.3:
        print("  -> INCONCLUSIVE: the gating path is not functioning, so this")
        print("     comparison says nothing about gating vs symbolic. Diagnose")
        print("     before interpreting. Known cause: the readout needs stored")
        print("     role assemblies, and role_lexicons[ROLE_ACTION] is EMPTY")
        print("     while most words are absent from the other role lexicons.")
        print("     The binding itself works (role areas do get winners).")
    elif m_can_n > 0.7 and m_non_n < 0.2 and m_non_s > 0.4:
        print("  -> GATING IS A POSITIONAL TEMPLATE: high canonical, ~0")
        print("     non-canonical. It has no lexical route, so it cannot")
        print("     override word order. Report; do not tune away.")
    elif np.isfinite(dc) and dm - dc > 0.02 and m_non_n >= m_non_s:
        print("  -> gating matches or beats symbolic on BOTH item kinds;")
        print("     replacing the symbolic path is on the table.")
    elif m_non_n < m_non_s - 0.1:
        print("  -> DO NOT REPLACE, and the reason is informative. Gating does")
        print("     well where word ORDER suffices but sits at chance where")
        print("     LEXICAL experience must override position -- it has no")
        print("     lexical route. The symbolic margin scoring supplies that")
        print("     route. The two are COMPLEMENTARY, not substitutes, which")
        print("     matches the two-route structure the lesion study found.")
    else:
        print("  -> mixed or indistinguishable; not justified either way yet.")


if __name__ == "__main__":
    run()
