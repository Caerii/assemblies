"""Is the parser's source overlap DEGRADATION, or is it STRUCTURE?

THE ARITHMETIC THAT REOPENED THIS. The whole arc compared source spread RAW
across two substrates: the toy reads 0.017 and the parser's NOUN_CORE reads
0.13, quoted as "the parser is 8-12x more overlapping". But `overlap` normalises
by k, so the random-pair floor is k/n, and THE TWO SUBSTRATES HAVE DIFFERENT
FLOORS:

    toy      k=40, n=2000   -> floor 0.0200,  observed 0.017  ->  0.85x floor
    parser   k=30, n=3000   -> floor 0.0100,  observed 0.125  -> 12.5x floor

(The parser's n/k are MEASURED off the built area, not read from
`EmergentParser.__init__`'s 10000/100 defaults -- `train_parser_to_depth`
overrides both. The floor happens to come out the same either way, but the class
default is not what runs and this file should not have quoted it.)

The toy is not "well separated". It is PINNED AT THE FLOOR -- its assemblies are
as unrelated as random sets can be -- which is exactly why the separation sweep
found only two regimes with nothing between them. A substrate sitting on its
floor has no room to express intermediate overlap. The parser's 0.13 is the only
one of the two numbers that is doing anything.

So "why is the parser worse" may be the wrong question. It is at least equally
consistent with the parser representing SIMILARITY -- words that share grounding
sharing neurons -- which the toy cannot do by construction, because its words
are independent random stimuli with nothing to share.

THE IDENTITY THIS TURNS ON, and the reason no sweep is needed. Let d_x be how
many of the M assemblies contain neuron x. Then

    mean pairwise overlap  =  sum_x C(d_x, 2) / ( C(M, 2) * k )

exactly -- every shared neuron contributes to exactly C(d_x, 2) pairs, and every
pair is normalised by k. So THE MEAN IS A FUNCTION OF THE NEURON DEGREE
DISTRIBUTION ALONE. It cannot distinguish "every pair shares a little" from "a
few hub neurons are in everything", and 13x the floor is a statement about
degree over-dispersion, not about which words are related.

That has a direct consequence for what to measure, and it is why this file does
not ablate anything: the mean spread we have been quoting for four experiments
is the wrong statistic to have been quoting. It is computed here as a SELF-CHECK
(the identity must hold to floating point, or the readout is wrong) and then set
aside.

THREE QUESTIONS, in order of what they would license:

  1. HUBS OR BREADTH. Print the degree histogram. Under the uniform null every
     degree is ~Binomial(M, k/n) with mean Mk/n < 1, so essentially no neuron
     sits in more than two assemblies. If a handful of neurons are in most of
     them, the overlap is a HUB artifact -- a known failure mode here
     (norm_init exists because recurrence collapses through high-degree hubs) --
     and it names a concrete fix. If the excess is spread broadly, it is not.

  2. RELATIONAL STRUCTURE BEYOND DEGREE. A degree-preserving shuffle of the
     word x neuron incidence matrix keeps every d_x and every k, so BY THE
     IDENTITY it reproduces the mean overlap exactly -- which is a free
     correctness check on the shuffle. What it destroys is WHICH words share
     WHICH neurons. If the observed pairwise overlaps are more dispersed than
     the shuffled ones, specific pairs share more than their degrees require,
     and that residue is relational structure. If they match, the overlap
     carries no information about word identity at all.

  3. DOES IT EXPLAIN THE ERRORS. The parser retrieves 0.73 at 6 candidates.
     For every failure, ask where the winning impostor sits in the SOURCE
     overlap ordering of that trial's competitors. If confusions land on the
     nearest sources, source overlap is causal and the diagnosis is settled
     without a single ablation. If the impostor's rank is uniform, source
     overlap is a correlate -- the same verdict the sweep reached, but reached
     on the system that actually exhibits the phenomenon.

THE DEGENERATE ARM, named in advance because three criteria this session were
satisfied by one. If NOUN_CORE had collapsed, question 1 would show one giant
degree class, question 2 would show observed == shuffled (nothing left to
destroy), and question 3 would read uniform -- and the honest reading of that
combination is "the area is dead", not "overlap is not causal". So distinctness
is asserted first, and the degenerate signature is printed as a named outcome
rather than left to be misread as a negative.

STATISTIC FOR Q3, and why not "fraction of errors on the top competitor".
Overlaps at k=100 are multiples of 0.01, so ties at the top are common and a
top-1 count inflates with them. The reported number is a tie-safe probability of
superiority: for each error, the fraction of competitors the impostor
out-overlaps (ties counted as half). Uniform null is exactly 0.5 regardless of
how many ties there are.
"""
import math
import os
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import numpy as np                                                     # noqa: E402

from _substrate import read, similarity, check_distinct, report_rate   # noqa: E402
from neural_assemblies.assembly_calculus.emergent.core.areas import (  # noqa: E402
    NOUN_CORE, ROLE_AGENT, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    train_parser_to_depth,
)
from neural_assemblies.assembly_calculus.ops import activate_assembly  # noqa: E402

DEPTH = "SENTENCES"
SEEDS = [11, 42]
ROLE_AREAS = (ROLE_PATIENT, ROLE_AGENT)
CANDIDATES = 6
SUBSETS = 60
SHUFFLES = 20
SWAP_FACTOR = 20


# --------------------------------------------------------------------------
# Q1/Q2 machinery
# --------------------------------------------------------------------------

def _mean_overlap_from_degrees(degrees, m, k):
    """The identity: mean pairwise overlap from the degree sequence alone."""
    if m < 2:
        return float("nan")
    return sum(d * (d - 1) / 2 for d in degrees) / (m * (m - 1) / 2 * k)


def _pairwise(assemblies):
    sets = [set(int(x) for x in a) for a in assemblies]
    k = statistics.fmean(len(s) for s in sets)
    return [len(a & b) / k for a, b in combinations(sets, 2)]


def _degree_preserving_shuffle(sets, rng, swaps):
    """Bipartite double-edge swap: keeps every d_x and every |A_w| exactly.

    BY THE IDENTITY this must reproduce the mean overlap to floating point. That
    is not a nicety -- it is the only check that the shuffle is a valid null and
    not quietly changing the thing it is supposed to hold fixed.
    """
    work = [set(s) for s in sets]
    edges = [(w, x) for w, s in enumerate(work) for x in s]
    done = 0
    for _ in range(swaps * 4):
        if done >= swaps:
            break
        i, j = rng.integers(0, len(edges), size=2)
        if i == j:
            continue
        w1, x1 = edges[i]
        w2, x2 = edges[j]
        if w1 == w2 or x1 == x2:
            continue
        if x2 in work[w1] or x1 in work[w2]:
            continue
        work[w1].discard(x1); work[w1].add(x2)
        work[w2].discard(x2); work[w2].add(x1)
        edges[i] = (w1, x2)
        edges[j] = (w2, x1)
        done += 1
    return work, done


# --------------------------------------------------------------------------
# Q3 machinery
# --------------------------------------------------------------------------

def _superiority(value, others):
    """Fraction of `others` that `value` exceeds; ties count half. Null 0.5."""
    if not others:
        return float("nan")
    lo = sum(1 for o in others if value > o)
    tie = sum(1 for o in others if value == o)
    return (lo + 0.5 * tie) / len(others)


def _errors_vs_source_overlap(parser, role_area, rng):
    """Retrieve at fixed difficulty; for each error, locate the impostor in the
    SOURCE overlap ordering of that trial's competitors."""
    brain = parser.brain
    core = parser.core_lexicons.get(NOUN_CORE, {})
    lex = parser.role_lexicons.get(role_area, {})
    shared = sorted(w for w in lex if w in core)
    if len(shared) < CANDIDATES + 1:
        return None

    stored = {w: np.asarray(lex[w].winners, dtype=np.int64) for w in shared}
    src = {w: np.asarray(core[w].winners, dtype=np.int64) for w in shared}
    live = {}
    for w in shared:
        with brain.probe():
            activate_assembly(brain, core[w])
            brain.project({}, {NOUN_CORE: [role_area]})
            live[w] = read(brain, role_area)

    src_ov = {(a, b): similarity(src[a], src[b])
              for a, b in combinations(shared, 2)}

    def so(a, b):
        return src_ov[(a, b)] if (a, b) in src_ov else src_ov[(b, a)]

    hits = total = 0
    err_sup, hit_sup = [], []
    for _ in range(SUBSETS):
        subset = list(rng.choice(shared, size=CANDIDATES, replace=False))
        for w in subset:
            comp = [o for o in subset if o != w]
            best = max(((similarity(live[w], stored[o]), o) for o in subset))[1]
            total += 1
            if best == w:
                hits += 1
                # The runner-up on a CORRECT trial: the same question asked of
                # the near miss, so a null result on errors alone cannot be
                # blamed on there being too few errors.
                ru = max(((similarity(live[w], stored[o]), o) for o in comp))[1]
                hit_sup.append(_superiority(so(w, ru),
                                            [so(w, o) for o in comp if o != ru]))
            else:
                err_sup.append(_superiority(so(w, best),
                                            [so(w, o) for o in comp if o != best]))
    return {
        "shared": shared, "hits": hits, "total": total,
        "err_sup": err_sup, "hit_sup": hit_sup,
    }


# --------------------------------------------------------------------------

def analyse(parser, seed):
    brain = parser.brain
    core = parser.core_lexicons.get(NOUN_CORE, {})
    area = brain.areas[NOUN_CORE]
    n, k = int(area.n), int(area.k)
    floor = k / n

    words = sorted(core)
    asms = [np.asarray(core[w].winners, dtype=np.int64) for w in words]
    m = len(asms)
    print(f"  NOUN_CORE  n={n}  k={k}  M={m}  random-pair floor k/n={floor:.4f}")
    if m < 3:
        print("  too few core assemblies to analyse")
        return None

    spr, note = check_distinct(asms, n, k, where="NOUN_CORE")
    print(f"  observed mean pairwise overlap {spr:.4f} "
          f"= {spr / floor:.1f}x floor   {note}")

    sets = [set(int(x) for x in a) for a in asms]
    deg = Counter()
    for s in sets:
        deg.update(s)
    degrees = list(deg.values())

    # SELF-CHECK: the identity must reproduce the measured mean.
    ident = _mean_overlap_from_degrees(degrees, m, statistics.fmean(len(s) for s in sets))
    print(f"  identity from degree sequence   {ident:.4f}   "
          f"(must match the line above; delta {abs(ident - spr):.2e})")

    # ---- Q1: hubs or breadth ------------------------------------------
    print()
    print("  Q1  DEGREE DISTRIBUTION -- is the excess carried by hubs?")
    exp_deg = m * k / n
    print(f"      neurons used {len(degrees)} of {n}   "
          f"mean degree {statistics.fmean(degrees):.3f}  "
          f"(uniform null would be {exp_deg:.3f})")
    hist = Counter(degrees)
    for d in sorted(hist)[:12]:
        share = d * (d - 1) / 2 * hist[d]
        print(f"      d={d:<3} {hist[d]:>6} neurons   "
              f"carry {share / max(sum(x * (x - 1) / 2 for x in degrees), 1e-9):>6.1%} "
              f"of all shared-pair mass")
    top = sorted(degrees, reverse=True)
    n_top = max(1, len(degrees) // 100)
    mass_top = sum(d * (d - 1) / 2 for d in top[:n_top])
    mass_all = sum(d * (d - 1) / 2 for d in degrees)
    print(f"      max degree {top[0]} of M={m}   "
          f"top 1% of used neurons carry {mass_top / max(mass_all, 1e-9):.1%} "
          f"of the overlap")

    # ---- Q2: relational structure beyond degree -----------------------
    print()
    print("  Q2  DEGREE-PRESERVING SHUFFLE -- is there structure beyond degree?")
    obs = _pairwise(asms)
    rng = np.random.default_rng(seed)
    n_edges = sum(len(s) for s in sets)
    null_means, null_sds, null_maxes, accepted = [], [], [], []
    for _ in range(SHUFFLES):
        sh, done = _degree_preserving_shuffle(sets, rng, n_edges * SWAP_FACTOR)
        vals = _pairwise([np.array(sorted(s)) for s in sh])
        null_means.append(statistics.fmean(vals))
        null_sds.append(statistics.pstdev(vals))
        null_maxes.append(max(vals))
        accepted.append(done)
    # A REJECTED SWAP LEAVES THE MATRIX UNCHANGED. If most proposals bounce,
    # the "null" is largely the observed data and any agreement below is
    # circular -- so the acceptance count is reported, not assumed.
    want = n_edges * SWAP_FACTOR
    print(f"      swaps    {statistics.fmean(accepted):.0f} of {want} accepted "
          f"({statistics.fmean(accepted) / max(want, 1):.1%}), "
          f"{statistics.fmean(accepted) / max(n_edges, 1):.1f} per edge")
    print(f"      mean     observed {statistics.fmean(obs):.4f}   "
          f"shuffled {statistics.fmean(null_means):.4f}   "
          f"<- MUST match; the shuffle preserves degrees, so the identity fixes it")
    print(f"      sd       observed {statistics.pstdev(obs):.4f}   "
          f"shuffled {statistics.fmean(null_sds):.4f}   "
          f"ratio {statistics.pstdev(obs) / max(statistics.fmean(null_sds), 1e-9):.2f}x")
    print(f"      max pair observed {max(obs):.4f}   "
          f"shuffled {statistics.fmean(null_maxes):.4f}")

    # ---- Q3: do the errors follow source overlap? ---------------------
    print()
    print("  Q3  DO RETRIEVAL ERRORS LAND ON THE NEAREST SOURCES?")
    out = {}
    for ra in ROLE_AREAS:
        res = _errors_vs_source_overlap(parser, ra, np.random.default_rng(seed))
        if res is None:
            print(f"      {ra:<14} fewer than {CANDIDATES + 1} shared words -- skipped")
            continue
        print(report_rate(f"{ra} ret@{CANDIDATES}", res["hits"], res["total"],
                          1.0 / CANDIDATES))
        for label, vals in (("impostor on ERRORS", res["err_sup"]),
                            ("runner-up on HITS", res["hit_sup"])):
            vals = [v for v in vals if v == v]
            if len(vals) < 20:
                print(f"      {label:<22} n={len(vals)} -- too few to read")
                continue
            mu = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            se = sd / math.sqrt(len(vals))
            print(f"      {label:<22} superiority {mu:.3f} +- {1.96 * se:.3f}"
                  f"   (null 0.500, n={len(vals)})")
        out[ra] = res
    return {"spread": spr, "floor": floor, "degrees": degrees, "m": m,
            "obs": obs, "null_sds": null_sds, "q3": out}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Is the parser's source overlap DEGRADATION or STRUCTURE?")
    print("The mean spread is a function of the DEGREE SEQUENCE alone "
          "(identity checked below),")
    print("so it cannot answer that on its own -- Q1/Q2/Q3 can.")
    print()
    results = {}
    for seed in SEEDS:
        print(f"=== depth={DEPTH} seed={seed} ===")
        parser = train_parser_to_depth(DEPTH, seed=seed)
        results[seed] = analyse(parser, seed)
        print()

    print("=" * 72)
    print("READING THIS TABLE")
    print("  Q1 hub-dominated  -> the fix is degree, not 'better separation'.")
    print("  Q2 sd > shuffled  -> specific word pairs share more than their")
    print("                       degrees require: the overlap is RELATIONAL.")
    print("  Q3 superiority >> 0.5 -> confusions land on the nearest sources,")
    print("                       so source overlap is CAUSAL for the 0.73.")
    print("  Q3 ~ 0.5          -> source overlap is a correlate; the sweep's")
    print("                       verdict, now reached on the real system.")
    print("  all three flat AND spread near 1.0 -> the area is DEGENERATE and")
    print("                       none of the above may be read as a negative.")


if __name__ == "__main__":
    main()
