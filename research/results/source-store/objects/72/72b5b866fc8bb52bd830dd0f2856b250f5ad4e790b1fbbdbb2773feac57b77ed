"""Does re-binding the FAILURES beat re-binding the same number at random?

THE OPPORTUNITY. Role binding works and is weak: rank-1 retrieval is 0.33-0.42
against 1/36 chance, with a rank1/rank2 margin of 1.13-1.29
(`role_binding_works_the_metric_cannot_see_it.md`). So roughly 60% of words
retrieve the WRONG role assembly, and nothing currently notices -- training
binds once per annotated occurrence and never checks whether the binding took.

An error signal makes that fixable: measure `bind_strength` after binding, and
re-bind the items that failed.

WHY TARGETED RATHER THAN GLOBAL, and this is the whole reason the experiment is
interesting. More reinforcement is NOT free here. This repo has already measured
that deep reinforcement produces a strong single attractor but MERGES a
multi-assembly area, and role areas hold 36-46 assemblies. Turning up rounds
globally would trade retrieval for collapse. Re-binding only the failures is how
to buy accuracy without paying that -- IF the guidance is doing any work.

THE CONTROL IS THE EXPERIMENT. A guided arm alone cannot distinguish "targeting
failures helps" from "more binding helps". So:

    guided   re-bind the F words whose rank-1 retrieval is WRONG
    random   re-bind F words chosen at random

Identical budget, identical protocol, identical recency structure. The last one
matters more than it looks: re-binding re-snapshots a word's stored role
assembly, so a re-bound word's target is fresher than everyone else's and would
win a retrieval contest for that reason alone. Both arms inherit that bias
equally, so the DIFFERENCE between them is not explained by it -- which is why
the comparison is guided-vs-random and never guided-vs-baseline.

Arms run on `deepcopy` of one parser, so they share every bit of training state
and differ only in which words were re-bound.

REPORTED, and all four are needed to read the result honestly:

  * retrieval OVERALL
  * retrieval on the RE-BOUND subset -- expected to rise in both arms, since
    those targets are freshest; a rise here alone means nothing
  * retrieval on the UNTOUCHED subset -- the interference check. If re-binding
    some words costs the others, that is the capacity ceiling showing up, and
    an overall gain could be entirely redistribution
  * SPREAD -- the merge check. A gain bought by collapsing the area toward one
    attractor is a loss, and spread is what makes that visible rather than
    inferable.

PRE-REGISTERED:
  * guided > random on OVERALL retrieval, with untouched not falling and spread
    not rising -> the error signal is worth wiring into training.
  * guided ~= random -> targeting adds nothing over extra exposure; report that
    and do not ship a mechanism whose selectivity is decorative.
  * untouched FALLS by roughly what re-bound gains -> capacity, not learning.
    Redistribution, and the honest headline.
"""
import copy
import os
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import numpy as np                                                     # noqa: E402

from _substrate import read, similarity                                # noqa: E402
from neural_assemblies.assembly_calculus.emergent.core.areas import (  # noqa: E402
    NOUN_CORE, ROLE_AGENT, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.assembly_calculus.ops import (                  # noqa: E402
    activate_assembly, bind,
)

SEEDS = [11, 12, 42]
ROLE_AREAS = (ROLE_PATIENT, ROLE_AGENT)
#: `train_roles` uses tail_rounds = _ROLE_BINDING_ROUNDS - 1 = 1. Same protocol
#: for the re-bind, so this measures REPETITION and not a different operation.
TAIL_ROUNDS = 1


def _retrieval(parser, role_area):
    """(accuracy, per-word correctness, margins) for one role area."""
    brain = parser.brain
    core = parser.core_lexicons.get(NOUN_CORE, {})
    lex = parser.role_lexicons.get(role_area, {})
    shared = sorted(w for w in lex if w in core)
    if len(shared) < 3:
        return float("nan"), {}, []
    stored = {w: np.asarray(lex[w].winners, dtype=np.int64) for w in shared}
    ok, margins = {}, []
    for w in shared:
        with brain.probe():
            activate_assembly(brain, core[w])
            brain.project({}, {NOUN_CORE: [role_area]})
            live = read(brain, role_area)
        scores = sorted(((similarity(live, a), o) for o, a in stored.items()),
                        reverse=True)
        ok[w] = scores[0][1] == w
        if len(scores) > 1 and scores[1][0] > 0:
            margins.append(scores[0][0] / scores[1][0])
    return sum(ok.values()) / len(ok), ok, margins


def _spread(parser, role_area, cap=30):
    lex = parser.role_lexicons.get(role_area, {})
    items = [np.asarray(a.winners, dtype=np.int64) for a in lex.values()][:cap]
    pairs = list(combinations(items, 2))
    return float(np.mean([similarity(x, y) for x, y in pairs])) if pairs else float("nan")


def _rebind(parser, role_area, words):
    """Re-apply the SAME binding protocol train_roles uses, and re-snapshot."""
    core = parser.core_lexicons.get(NOUN_CORE, {})
    for w in words:
        stored_core = core.get(w)
        if stored_core is None:
            continue
        asm = bind(parser.brain, NOUN_CORE, role_area, stored_core,
                   tail_rounds=TAIL_ROUNDS)
        parser.role_lexicons[role_area][w] = asm


def main():
    print(f"tail_rounds={TAIL_ROUNDS} (train_roles' own protocol)  seeds={SEEDS}")
    print()
    agg = {}
    for role_area in ROLE_AREAS:
        print(f"########## {role_area} ##########")
        hdr = (f"{'seed':>5} {'arm':<8} {'overall':>9} {'rebound':>9} "
               f"{'untouched':>11} {'spread':>8} {'margin':>8}")
        print(hdr)
        print("-" * len(hdr))
        for seed in SEEDS:
            base = get_parser_cache().fork("SENTENCES", seed=seed)
            acc0, ok0, marg0 = _retrieval(base, role_area)
            if not ok0:
                print(f"{seed:>5} (no measurable retrieval)")
                continue
            failures = sorted(w for w, good in ok0.items() if not good)
            rng = np.random.default_rng(seed)
            pool = sorted(ok0)
            random_pick = sorted(rng.choice(pool, size=len(failures),
                                            replace=False).tolist())
            print(f"{seed:>5} {'baseline':<8} {acc0:>9.3f} {'-':>9} {'-':>11} "
                  f"{_spread(base, role_area):>8.4f} "
                  f"{(statistics.fmean(marg0) if marg0 else float('nan')):>8.4f}")

            for arm, picked in (("guided", failures), ("random", random_pick)):
                p = copy.deepcopy(base)
                _rebind(p, role_area, picked)
                acc, ok, marg = _retrieval(p, role_area)
                sel = set(picked)
                reb = [ok[w] for w in ok if w in sel]
                unt = [ok[w] for w in ok if w not in sel]
                row = (
                    acc,
                    statistics.fmean(reb) if reb else float("nan"),
                    statistics.fmean(unt) if unt else float("nan"),
                    _spread(p, role_area),
                    statistics.fmean(marg) if marg else float("nan"),
                )
                agg.setdefault((role_area, arm), []).append(row)
                print(f"{seed:>5} {arm:<8} {row[0]:>9.3f} {row[1]:>9.3f} "
                      f"{row[2]:>11.3f} {row[3]:>8.4f} {row[4]:>8.4f}")
            print()

    print("=" * 62)
    print("MEANS ACROSS SEEDS")
    for role_area in ROLE_AREAS:
        for arm in ("guided", "random"):
            rows = agg.get((role_area, arm), [])
            if not rows:
                continue
            m = [statistics.fmean(r[i] for r in rows if r[i] == r[i])
                 for i in range(5)]
            print(f"  {role_area:<14} {arm:<8} overall={m[0]:.3f} "
                  f"rebound={m[1]:.3f} untouched={m[2]:.3f} "
                  f"spread={m[3]:.4f} margin={m[4]:.4f}")
    print()
    print("READ THE UNTOUCHED COLUMN FIRST. If it falls by about what `rebound`")
    print("gains, the area is at capacity and this is redistribution, not")
    print("learning -- and `overall` will look like progress anyway. Then read")
    print("guided vs random: only that difference is attributable to the error")
    print("signal, because both arms share the re-snapshot recency bias.")


if __name__ == "__main__":
    main()
