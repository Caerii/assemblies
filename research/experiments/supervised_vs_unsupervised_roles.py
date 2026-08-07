"""Does the SUPERVISED bind() protocol write more retrievable roles than the
unsupervised route -- and which route actually runs at each depth?

WHY. Everything measured in this arc (retrieval 0.33-0.42 at 12-19x chance,
re-binding redistributing, the alpha ~ 0.36 ceiling) was measured on a
`SENTENCES` parser. And on the curriculum path `build_stage_schedule` hands every
sentence `roles=[None] * len(sent)`, so the "roles" phase calls
`train_unsupervised`, not `train_roles`. If that reading is right, every number
in this arc describes UNSUPERVISED role induction, and the careful `bind()`
protocol -- the one three drifted copies were unified into -- never runs there.

`FULL_TRAIN` goes through `parser.train(create_training_sentences())`, which
DOES carry role annotations, so it should take the supervised path. That makes
the A/B available with no code change.

STEP 1 IS VERIFICATION, NOT MEASUREMENT, and it is first on purpose. Twice in
this session a result turned out to be about something other than what the call
graph implied -- `forced_category` was a projection target, `Area.w` was an
alias. So rather than infer which route runs, this COUNTS it: every writer of
`role_lexicons` is wrapped and reports how many entries it wrote, per depth.
There are exactly three (roles.py:98, unsupervised.py, training/batch.py) and
all three are instrumented.

If the counts contradict the call-graph reading, the counts win and step 2 is
reinterpreted rather than the counts explained away.

STEP 2 IS THE COMPARISON, and it needs a control to be worth anything.
Retrieval depends on M through capacity -- measured, the role area saturates
around alpha = Mk/n ~ 0.36 -- and the two depths train on different corpora, so
they will not have the same M. A bare accuracy difference between them would
confound ROUTE with LOAD. So M, k, n and alpha are reported alongside retrieval
for every area, and any claim has to survive that table.

READING:
  * supervised retrieval >> unsupervised AT COMPARABLE alpha
        -> the curriculum has been running a degraded route, and #116 stops
           being a footnote about six `goal` annotations.
  * comparable at comparable alpha
        -> the route is not the constraint; capacity is, and #122 is the main
           line.
  * alphas too far apart to compare
        -> report that and say so. It is a real outcome of this design, not a
           failure to be papered over with a ratio.
"""
import os
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
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    train_parser_to_depth,
)
from neural_assemblies.assembly_calculus.ops import activate_assembly  # noqa: E402

DEPTHS = ["SENTENCES", "FULL_TRAIN"]
SEEDS = [11, 42]
ROLE_AREAS = (ROLE_PATIENT, ROLE_AGENT)


class RouteCounter:
    """Counts role-lexicon writes per route. All three writers are wrapped."""

    def __init__(self):
        self.writes = {}

    def install(self):
        from neural_assemblies.assembly_calculus.emergent.parser_mixins import (
            roles as roles_mod, unsupervised as unsup_mod,
        )
        undo = []

        def wrap(owner, name, route):
            orig = getattr(owner, name)

            def wrapped(self_, *a, **kw):
                before = sum(len(v) for v in
                             (getattr(self_, "role_lexicons", {}) or {}).values())
                out = orig(self_, *a, **kw)
                after = sum(len(v) for v in
                            (getattr(self_, "role_lexicons", {}) or {}).values())
                self.writes[route] = self.writes.get(route, 0) + (after - before)
                return out

            setattr(owner, name, wrapped)
            undo.append(lambda: setattr(owner, name, orig))

        wrap(roles_mod.RoleBindingMixin, "train_roles", "train_roles (SUPERVISED)")
        for cand in ("train_unsupervised",):
            for cls_name in dir(unsup_mod):
                cls = getattr(unsup_mod, cls_name)
                if isinstance(cls, type) and hasattr(cls, cand):
                    wrap(cls, cand, f"{cand} (UNSUPERVISED)")
                    break
        return lambda: [f() for f in reversed(undo)]


def _retrieval_at_candidate_size(parser, role_area, size, trials=40, seed=0):
    """Retrieval when the candidate set is subsampled to `size`.

    THE LOAD CONFOUND MADE PARTLY TRACTABLE. FULL_TRAIN stores M=6 role
    assemblies and SENTENCES stores M=36, so their raw retrieval numbers are not
    comparable: discriminating among 6 is a different task from discriminating
    among 36, and chance differs 7x. Restricting the CANDIDATE SET to the same
    size equalises the READOUT difficulty.

    What it does NOT equalise is the WRITING load -- the SENTENCES bindings were
    still written with 36 assemblies competing for the area. So this compares
    "how discriminable are the bindings as written, judged at matched
    difficulty", which is closer to a route comparison than the raw numbers and
    is still not one. The clean version needs the same corpus at the same M.
    """
    brain = parser.brain
    core = parser.core_lexicons.get(NOUN_CORE, {})
    lex = parser.role_lexicons.get(role_area, {})
    shared = sorted(w for w in lex if w in core)
    if len(shared) < size or size < 2:
        return float("nan")
    stored = {w: np.asarray(lex[w].winners, dtype=np.int64) for w in shared}
    live_cache = {}
    for w in shared:
        with brain.probe():
            activate_assembly(brain, core[w])
            brain.project({}, {NOUN_CORE: [role_area]})
            live_cache[w] = read(brain, role_area)
    rng = np.random.default_rng(seed)
    hits = total = 0
    for _ in range(trials):
        subset = list(rng.choice(shared, size=size, replace=False))
        for w in subset:
            sc = max(((similarity(live_cache[w], stored[o]), o) for o in subset))
            hits += int(sc[1] == w)
            total += 1
    return hits / total if total else float("nan")


def _retrieval(parser, role_area):
    brain = parser.brain
    core = parser.core_lexicons.get(NOUN_CORE, {})
    lex = parser.role_lexicons.get(role_area, {})
    shared = sorted(w for w in lex if w in core)
    if len(shared) < 3:
        return float("nan"), float("nan"), len(shared)
    stored = {w: np.asarray(lex[w].winners, dtype=np.int64) for w in shared}
    hits, margins = 0, []
    for w in shared:
        with brain.probe():
            activate_assembly(brain, core[w])
            brain.project({}, {NOUN_CORE: [role_area]})
            live = read(brain, role_area)
        sc = sorted(((similarity(live, a), o) for o, a in stored.items()),
                    reverse=True)
        hits += int(sc[0][1] == w)
        if len(sc) > 1 and sc[1][0] > 0:
            margins.append(sc[0][0] / sc[1][0])
    return (hits / len(shared),
            float(np.mean(margins)) if margins else float("nan"),
            len(shared))


def _spread(parser, area, cap=30):
    lex = parser.role_lexicons.get(area, {})
    items = [np.asarray(a.winners, dtype=np.int64) for a in lex.values()][:cap]
    pairs = list(combinations(items, 2))
    return float(np.mean([similarity(x, y) for x, y in pairs])) if pairs else float("nan")


def main():
    rows = []
    parsers = {}
    for depth in DEPTHS:
        for seed in SEEDS:
            counter = RouteCounter()
            restore = counter.install()
            try:
                parser = train_parser_to_depth(depth, seed=seed)
            finally:
                restore()

            parsers.setdefault(depth, parser)
            print(f"=== depth={depth} seed={seed} ===")
            if not counter.writes:
                print("  NO ROUTE WROTE role_lexicons -- the instrumentation "
                      "found nothing, so the depth comparison below is not "
                      "attributable to a route")
            for route, n in sorted(counter.writes.items()):
                print(f"  {route:<34} wrote {n:>4} role-lexicon entries")

            for area_name in ROLE_AREAS:
                area = parser.brain.areas.get(area_name)
                k = getattr(area, "k", 0) or 0
                n_ = getattr(area, "n", 0) or 0
                m = len(parser.role_lexicons.get(area_name, {}))
                alpha = (m * k / n_) if n_ else float("nan")
                acc, margin, shared = _retrieval(parser, area_name)
                sp = _spread(parser, area_name)
                rows.append((depth, seed, area_name, m, alpha, acc, margin, sp))
                print(f"  {area_name:<14} M={m:<4} k={k:<4} n={n_:<6} "
                      f"alpha={alpha:<6.3f} retrieval={acc:<6.3f} "
                      f"(chance {1/shared if shared else float('nan'):.3f}) "
                      f"margin={margin:<6.3f} spread={sp:.4f}")
            print()

    # Matched-candidate-set readout, so the two depths are judged at the same
    # task difficulty even though they were written under different loads.
    print("=" * 78)
    print("MATCHED CANDIDATE SET -- readout difficulty equalised")
    print(f"{'depth':<12} {'area':<14} {'M stored':>9} "
          f"{'ret@6':>8} {'ret@11':>8}  (chance 0.167 / 0.091)")
    print("-" * 78)
    for depth, parser in sorted(parsers.items()):
        for area_name in ROLE_AREAS:
            m = len(parser.role_lexicons.get(area_name, {}))
            r6 = _retrieval_at_candidate_size(parser, area_name, 6)
            r11 = _retrieval_at_candidate_size(parser, area_name, 11)
            print(f"{depth:<12} {area_name:<14} {m:>9} {r6:>8.3f} {r11:>8.3f}")
    print()

    print("=" * 78)
    print("ROUTE vs LOAD -- retrieval is only comparable at comparable alpha")
    print(f"{'depth':<12} {'area':<14} {'M':>4} {'alpha':>7} {'retrieval':>10} "
          f"{'margin':>8} {'spread':>8}")
    print("-" * 78)
    for depth in DEPTHS:
        for area_name in ROLE_AREAS:
            sel = [r for r in rows if r[0] == depth and r[2] == area_name]
            if not sel:
                continue
            f = lambda i: float(np.nanmean([s[i] for s in sel]))   # noqa: E731
            print(f"{depth:<12} {area_name:<14} {f(3):>4.0f} {f(4):>7.3f} "
                  f"{f(5):>10.3f} {f(6):>8.3f} {f(7):>8.4f}")


if __name__ == "__main__":
    main()
