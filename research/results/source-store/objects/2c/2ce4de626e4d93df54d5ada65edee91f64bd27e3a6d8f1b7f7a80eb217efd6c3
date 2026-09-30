"""Is SOURCE SEPARATION the variable that costs the parser its retrieval?

WHERE THIS SITS. Eight mechanisms are excluded: the readout instrument, the
training route, `norm_init`, the index space, consolidation, target capacity,
and -- by arithmetic rather than experiment -- packing density. The toy is 2.7x
MORE densely packed than the parser (Mk/n 1.92 vs 0.71) and 8x LESS overlapping
(source spread 0.017 vs 0.13), so the parser's overlap is not forced by how many
assemblies share the area. It comes from how they are built.

What is left is separation itself, and this is the last localization: if it
confirms, the question becomes "how do we build better-separated core
assemblies", which is engineering with a metric. If it does not, the honest
report is that the parser's assemblies are measurably worse than a toy's and we
do not know why -- which beats a ninth hypothesis.

THE KNOB, and why this one. Each word is built from its private stimulus PLUS a
shared pool that EVERY word also fires. More shared drive pulls all the
assemblies toward one attractor, so `n_shared` raises source spread
monotonically with a single scalar.

Chosen over the two alternatives on purpose:

THE FIRST KNOB WAS BINARY AND THE VERDICT LOGIC ACCEPTED COLLAPSE. Firing a
FULL-SIZE shared stimulus alongside the private one took source spread from
0.0171 (n_shared=0) straight to 0.9261 (n_shared=1) -- one shared stimulus is
enough to pull every assembly onto a single attractor. There is no intermediate
regime that way, and nothing was measured anywhere near the parser's 0.13.

Worse, the auto-verdict FIRED. It looked for the first row with spread >= 0.13,
found 0.9261, and accepted ret@6 = 0.168 as "matching the parser's 0.725". But
0.167 IS CHANCE at six candidates -- the assemblies had collapsed entirely. A
degenerate arm satisfied the acceptance test, which is the exact trap this repo
has recorded twice: before believing a criterion, ask what ELSE passes it. The
guard also watched only for the knob UNDERSHOOTING and said nothing when it
overshot.

Both are fixed below: the shared stimulus now has a SIZE, which is the graded
version of the same idea, and the verdict requires landing in a WINDOW around
the parser's spread AND retrieval staying above chance.

  * explicitly MIXING neuron IDs would control overlap exactly, but the results
    would not be attractors of the dynamics -- binding into a hand-built
    assembly is a different operation, and the artificial-assembly confound
    would sit underneath every number;
  * varying BUILD ROUNDS changes convergence, not separation, and conflates the
    two -- which matters because under-convergence is the competing hypothesis
    this sweep should not quietly assume.

Shared drive is also what the parser actually has: words share grounding
features by design, so some source overlap there is INTENDED. The question is
not how to reach zero overlap, it is how much is affordable.

EVERYTHING ELSE IS HELD FIXED at the parser's regime -- M = 24 bindings,
alpha = 0.48, retrieval scored among 6 candidates so chance stays 1/6. The only
moving part is how separable the sources are.

READING:
  * ret@6 falls toward 0.73 as source spread approaches 0.13
        -> separation is the causal variable. The diagnosis moves upstream to
           the lexicon and this arc ends.
  * ret@6 stays high at spread 0.13
        -> spread is a correlate, not a cause. Stop localizing and report that
           the parser loses something none of the excluded mechanisms covers.
  * spread never reaches 0.13
        -> the knob is too weak to answer the question; say so rather than
           extrapolating a curve past its data.
"""
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                                     # noqa: E402

from _substrate import read, similarity                                # noqa: E402
from neural_assemblies.assembly_calculus.assembly import Assembly      # noqa: E402
from neural_assemblies.core.brain import Brain                         # noqa: E402
from neural_assemblies.assembly_calculus.ops import activate_assembly  # noqa: E402

SRC, DST = "SRC", "DST"
N, K, P, BETA = 2000, 40, 0.05, 0.10
M = 24                     # alpha = Mk/n = 0.48, the parser's regime
BUILD_ROUNDS, BIND_ROUNDS = 6, 10
CANDIDATES, SUBSETS = 6, 40
#: SIZE of the single shared stimulus, not how many. Size is the graded knob;
#: count is binary (one full-size shared stimulus already collapses the area).
SHARED_SIZES = [0, 1, 2, 4, 8, 16, 40]
SEEDS = [42, 7, 123]

#: The parser's numbers, so the table can be read against them directly.
PARSER_SRC_SPREAD = 0.13
PARSER_RET_AT_6 = 0.725


def _spread(items, cap=30):
    vals = list(items)[:cap]
    pairs = list(combinations(vals, 2))
    return statistics.fmean(similarity(x, y) for x, y in pairs) if pairs else float("nan")


def trial(shared_size, seed, build_rounds=BUILD_ROUNDS):
    # Engine PINNED: auto resolves to numpy_sparse at these sizes, so this is
    # behaviour-preserving, but an implicit engine lets a default change
    # silently reinterpret recorded numbers ([[pin-backend-not-global]]).
    brain = Brain(p=P, seed=seed, norm_init=True, engine="numpy_sparse")
    brain.add_area(SRC, N, K, beta=BETA)
    brain.add_area(DST, N, K, beta=BETA)
    names = [f"a{i}" for i in range(M)]
    for nm in names:
        brain.add_stimulus(nm, K)
    shared = []
    if shared_size > 0:
        brain.add_stimulus("g0", shared_size)
        shared = ["g0"]

    # Private stimulus PLUS the shared pool every word also fires.
    sources = {}
    for nm in names:
        stim = {nm: [SRC]}
        stim.update({g: [SRC] for g in shared})
        for _ in range(build_rounds):
            brain.project(stim, {})
        sources[nm] = read(brain, SRC)

    targets = {}
    for nm in names:
        for _ in range(BIND_ROUNDS):
            brain.project({nm: [SRC]}, {SRC: [DST]})
        targets[nm] = read(brain, DST)

    live = {}
    for nm in names:
        with brain.probe():
            activate_assembly(brain, Assembly(SRC, sources[nm]))
            brain.project({}, {SRC: [DST]})
            live[nm] = read(brain, DST)

    rng = np.random.default_rng(seed)
    hits = total = 0
    margins = []
    for _ in range(SUBSETS):
        subset = list(rng.choice(names, size=CANDIDATES, replace=False))
        for w in subset:
            sc = sorted(((similarity(live[w], targets[o]), o) for o in subset),
                        reverse=True)
            hits += int(sc[0][1] == w)
            total += 1
            if len(sc) > 1 and sc[1][0] > 0:
                margins.append(sc[0][0] / sc[1][0])
    return (_spread(sources.values()), _spread(targets.values()),
            hits / total if total else float("nan"),
            statistics.fmean(margins) if margins else float("nan"))


def main():
    print(f"n={N} k={K} M={M} (alpha={M*K/N:.2f}) candidates={CANDIDATES} "
          f"(chance {1/CANDIDATES:.3f}) seeds={SEEDS}")
    print(f"parser reference: source spread {PARSER_SRC_SPREAD}, "
          f"ret@6 {PARSER_RET_AT_6}")
    print()
    hdr = (f"{'shared k':>9} {'src spread':>11} {'tgt spread':>11} "
           f"{'ret@6':>8} {'margin':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for g in SHARED_SIZES:
        per_seed = [trial(g, s) for s in SEEDS]
        srcs = statistics.fmean(x[0] for x in per_seed)
        tgts = statistics.fmean(x[1] for x in per_seed)
        acc = statistics.fmean(x[2] for x in per_seed)
        mg = statistics.fmean(x[3] for x in per_seed)
        rows.append((g, srcs, tgts, acc, mg))
        print(f"{g:>9} {srcs:>11.4f} {tgts:>11.4f} {acc:>8.3f} {mg:>9.3f}")

    print()
    # SECOND KNOB: CONVERGENCE. The shared-drive knob turned out to have a
    # PHASE TRANSITION -- spread 0.015 at shared k=2 and 0.496 at k=4, with
    # nothing in between -- so it cannot reach the parser's 0.13 at all. That is
    # itself informative: shared drive does not produce intermediate overlap in
    # this substrate, it produces separation or collapse.
    #
    # Under-convergence is the competing hypothesis, deliberately excluded from
    # the knob above so it would not be quietly assumed. It now has independent
    # support: the parser's stored core assemblies overlap what their own
    # stimulus currently produces by only 0.27, which is what an assembly that
    # never settled looks like. Fewer build rounds is the direct test.
    print()
    print("BUILD ROUNDS (shared k = 0) -- does UNDER-CONVERGENCE produce the")
    print("intermediate overlap that shared drive cannot?")
    hdr2 = (f"{'rounds':>7} {'src spread':>11} {'tgt spread':>11} "
            f"{'ret@6':>8} {'margin':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in [1, 2, 3, 6, 12, 24]:
        per_seed = [trial(0, s_, build_rounds=r) for s_ in SEEDS]
        srcs = statistics.fmean(x[0] for x in per_seed)
        tgts = statistics.fmean(x[1] for x in per_seed)
        acc = statistics.fmean(x[2] for x in per_seed)
        mg = statistics.fmean(x[3] for x in per_seed)
        rows.append((f"r{r}", srcs, tgts, acc, mg))
        print(f"{r:>7} {srcs:>11.4f} {tgts:>11.4f} {acc:>8.3f} {mg:>9.3f}")
    print()

    # A WINDOW, not a threshold, and retrieval must be ABOVE CHANCE. The first
    # version took any row with spread >= 0.13 and accepted any retrieval at or
    # below the parser's -- which a fully collapsed area satisfies at chance.
    chance = 1.0 / CANDIDATES
    window = [r for r in rows
              if abs(r[1] - PARSER_SRC_SPREAD) <= 0.05 and r[3] > chance + 0.05]
    if not window:
        near = min(rows, key=lambda r: abs(r[1] - PARSER_SRC_SPREAD))
        collapsed = [r for r in rows if r[3] <= chance + 0.05]
        print(f"** NO USABLE POINT NEAR THE PARSER'S SPREAD. Closest row is "
              f"spread {near[1]:.4f} with ret@6 {near[3]:.3f} "
              f"(chance {chance:.3f}).")
        if collapsed:
            print(f"   {len(collapsed)} row(s) are AT CHANCE -- the knob "
                  f"overshoots into total collapse rather than passing through "
                  f"the regime of interest.")
        print("   The curve must NOT be extrapolated into the gap. Report the "
              "range covered and pick a knob with resolution there. **")
        return

    at = min(window, key=lambda r: abs(r[1] - PARSER_SRC_SPREAD))
    print(f"At source spread {at[1]:.4f} (parser: {PARSER_SRC_SPREAD}): "
          f"ret@6 = {at[3]:.3f}, parser reads {PARSER_RET_AT_6}")
    print()
    if at[3] <= PARSER_RET_AT_6 + 0.08:
        print("SEPARATION IS THE CAUSAL VARIABLE. Matching the parser's source")
        print("overlap reproduces its retrieval on a substrate with nothing")
        print("else wrong. The diagnosis moves upstream to how core assemblies")
        print("are built -- a LEXICON question -- and this arc ends.")
    else:
        print("SEPARATION IS NOT SUFFICIENT. At the parser's source overlap the")
        print("toy still retrieves far better, so spread is a correlate rather")
        print("than the cause. Per the stopping rule: stop localizing, and")
        print("report that the parser loses something the excluded mechanisms")
        print("do not cover.")


if __name__ == "__main__":
    main()
