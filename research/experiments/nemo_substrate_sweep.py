"""Does the gating result survive more seeds and a bigger brain?

`nemo_vs_symbolic.py` measured, at n=1000/k=50 with 3 seeds:

    reversible   (order suffices)        symbolic 1.000   gating 0.833
    irreversible (needs lexical override) symbolic 1.000   gating 0.500

Two separate doubts, and they dissociate, so both are swept here:

* STATISTICAL n. Three seeds with zero variance is not a measurement. More
  seeds narrows the interval without changing the model.
* SUBSTRATE n. Every number in the lesion study and the A/B comes from
  n=1000/k=50. The reference ERP work ran n=10000/k=100, and this repo's own
  notes put assembly capacity around n^0.29, so small-brain results can be
  substrate artifacts rather than model properties.

PREDICTION, recorded before running so it can be wrong
------------------------------------------------------
The 0.500 on IRREVERSIBLE items is FUNDAMENTAL, not a substrate artifact. The
gating path has no lexical route by construction -- nothing in it can let what
the corpus taught about a word override where the word order puts it -- so more
neurons cannot help. It should sit at chance (0.5 on a binary agent/patient
choice) at every scale.

The 0.833 on REVERSIBLE items is more likely substrate-limited: reading a role
out means overlapping assemblies, and at k=50 with ~30 winners that is noisy.
It should rise toward 1.0 as n grows.

IF THE IRREVERSIBLE SCORE CLIMBS WITH n, the "no lexical route" explanation is
WRONG and something subtler is happening. That is the falsifiable part, and it
is the reason to run this rather than assume.

RESULT -- prediction half confirmed, half REFUTED
------------------------------------------------
       n    k  seeds  kind                symbolic            gating
    1000   50     12  reversible     1.000 +/-0.000    0.833 +/-0.000
    3000   50     12  reversible     1.000 +/-0.000    0.833 +/-0.000
   10000  100      4  reversible     1.000 +/-0.000    0.833 +/-0.000
    1000   50     12  irreversible   1.000 +/-0.000    0.492 +/-0.016
    3000   50     12  irreversible   0.983 +/-0.033    0.200 +/-0.118
   10000  100      4  irreversible   1.000 +/-0.000    0.300 +/-0.196

CONFIRMED, and more strongly than predicted: irreversible never climbs with n.
It goes BELOW CHANCE (0.200 at n=3000) as the substrate improves, which is the
signature of a RELIABLY POSITIONAL mechanism -- these items are built so
position gives the WRONG answer, so the better gating gets at word order the
worse it must score. "No lexical route" survives.

REFUTED: reversible was predicted to RISE toward 1.0 as substrate noise
cleared. It is EXACTLY 0.833 at every substrate with ZERO variance -- 20 of 24
judgements, the same 4 wrong every time, across a 10x increase in neurons. That
is a deterministic structural error, not noise, and the "substrate-limited"
reasoning was simply wrong. 5/6 with no variance is a bug to be found, not an
accuracy to be improved: chase WHICH 4 judgements fail before doing anything
else with this path.

Run: .venv/Scripts/python.exe research/experiments/nemo_substrate_sweep.py
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

#: (n, k, seeds). Cost climbs steeply with n, so the big substrate gets fewer
#: seeds -- stated rather than hidden, since its interval will be wider.
GRID: Tuple[Tuple[int, int, int], ...] = (
    (1000, 50, 12),
    (3000, 50, 12),
    (10000, 100, 4),
)


def _mean_ci(xs) -> tuple:
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return (float(a.mean()) if a.size else float("nan")), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(a.size))


def run(grid: Sequence[Tuple[int, int, int]] = GRID) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lesion_aphasia import build_corpus, test_items, train_parser
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.nemo_parse import (
        NemoParser, infer_transitive_verbs,
    )
    from nemo_vs_symbolic import SCORED_ROLES, _score

    transitive = infer_transitive_verbs(
        create_training_sentences() + build_corpus())
    items = {
        "reversible": [(list(w), g) for w, g, k in test_items() if k == "reversible"],
        "irreversible": [(list(w), g) for w, g, k in test_items()
                         if k == "irreversible"],
    }

    print("substrate + seed sweep of the gating A/B")
    print(f"scoring {SCORED_ROLES} only (ACTION is never neurally bound in "
          f"either path)")
    print()
    print(f"  {'n':>6}{'k':>5}{'seeds':>7}  {'kind':<13}"
          f"{'symbolic':>18}{'gating':>18}")
    print("  " + "-" * 68)

    for n, k, n_seeds in grid:
        seeds = list(range(42, 42 + n_seeds))
        acc = {kind: ([], []) for kind in items}
        for seed in seeds:
            # Train ONCE per seed and score both item kinds off it -- the
            # earlier version retrained per kind, paying double for nothing.
            parser = train_parser(seed, n=n, k=k)
            for kind, rows in items.items():
                s_ok = s_tot = g_ok = g_tot = 0
                for words, gold in rows:
                    a, b = _score(
                        copy.deepcopy(parser).parse(list(words))["roles"], gold)
                    s_ok, s_tot = s_ok + a, s_tot + b
                    got = NemoParser(copy.deepcopy(parser),
                                     transitive_verbs=transitive).parse(
                        list(words))
                    a, b = _score(got, gold)
                    g_ok, g_tot = g_ok + a, g_tot + b
                acc[kind][0].append(s_ok / max(s_tot, 1))
                acc[kind][1].append(g_ok / max(g_tot, 1))
        for kind, (sym, gat) in acc.items():
            ms, cs = _mean_ci(sym)
            mg, cg = _mean_ci(gat)
            print(f"  {n:>6}{k:>5}{n_seeds:>7}  {kind:<13}"
                  f"{ms:>11.3f} +/-{cs:<5.3f}{mg:>11.3f} +/-{cg:<5.3f}",
                  flush=True)

    print()
    print("  PREDICTION UNDER TEST: irreversible gating stays ~0.5 at every n")
    print("  (no lexical route), while reversible gating rises toward 1.0.")
    print("  If irreversible CLIMBS with n, that explanation is wrong.")


if __name__ == "__main__":
    run()
