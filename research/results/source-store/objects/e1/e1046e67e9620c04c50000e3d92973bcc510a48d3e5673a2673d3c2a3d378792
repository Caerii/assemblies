"""Is compositional structure a fact about LANGUAGE, or about the substrate?

WHY THIS EXISTS
---------------
`constituent_structure.py` measured that a merged constituent's assembly tracks
its parts: overlap is graded by how many parents two constituents share
(+0.2492 share-subject, +0.1213 share-verb, both intervals excluding zero over
5 seeds), and `held_out_recombination.py` measured that never-trained
combinations get the same gradient (+0.2298). Those are the strongest results
in the project.

They were both measured on subject-verb constituents, so on their own they
support a narrow reading: THIS grammar, in THIS parser, composes. The
interesting claim is the wide one -- that composition is a property of the
assembly substrate plus the merge operation, and language is one instance of it
rather than a special case. That claim is worth much more, and it is also much
easier to fool yourself about.

WHY NOT THROUGH EmergentParser
------------------------------
The obvious test -- feed a non-linguistic corpus to the parser -- cannot work,
and the reason is worth stating because it would have produced a confident
wrong answer. `classify_distributional` names its categories in the source
("VERB", "NOUN", "PRON") and scores them from verb-anchored geometry
(`word_as_pre_verb`, `word_as_post_verb`, `word_as_action`); `GROUNDING_TO_CORE`
maps a hand-authored modality table onto POS core areas. A non-linguistic domain
pushed through that pipeline either fails for reasons that have nothing to do
with composition, or "succeeds" because the answer was written into the code.

So the test runs at the SUBSTRATE: Brain, project, merge, overlap. That is the
same level the language claim was actually measured at -- assembly overlap
binned by shared parents, which never referenced a part of speech -- so the two
numbers are directly comparable.

WHAT IS MANIPULATED
-------------------
Abstract symbols with no linguistic identity, no grounding table, no categories.
Slot-A symbols a0..a5 and slot-B symbols b0..b5, composed by merge into a
composite area. The structure is factorial and that is ALL it is.

PRE-REGISTERED PREDICTIONS
--------------------------
H1 UNIVERSALITY. The graded profile appears here too: overlap(share a) and
   overlap(share b) both exceed overlap(share nothing). If it does not, the
   language result is about the parser, not the substrate, and the wide claim
   dies.

H2 NOT A SUBSTRATE ARTIFACT. The gradient must come from the sharing, not from
   the merge target having a strong attractor that pulls everything together.
   Tested by PERMUTATION: keep every assembly and every overlap exactly as
   measured, and only shuffle which (a, b) label each composite is filed under,
   then re-bin. Predict the gradient vanishes. This is the control that decides
   whether H1 means anything -- identical numbers, identical merges, only the
   bookkeeping permuted.

H3 IT EXTENDS PAST ARITY 2, WHICH LANGUAGE NEVER TESTED HERE. Compose three
   symbols hierarchically, merge(merge(a, b), c), and bin pairs by how many of
   the three parents they share. Predict overlap is MONOTONE in shared-parent
   count: 2 > 1 > 0. A flat or non-monotone profile would say arity 2 was a
   special case.

H3b DEPTH ASYMMETRY. In merge(merge(a, b), c), the symbol c attaches one level
   shallower than a and b. Predict sharing the SHALLOW parent c moves overlap
   MORE than sharing a deep parent, because c's contribution is not filtered
   through an intermediate k-WTA. This is a genuine prediction of hierarchical
   composition and it is the one most likely to be wrong; recorded either way.

H4 PRODUCTIVITY. Combinations held out of the composition phase entirely, then
   composed fresh, show the same gradient as the ones built during it.

Every prediction above was written before the first run. Chance overlap is
k/n = 0.05 at n=1000, k=50, and is printed on every table so "above chance" is
never taken on trust.

RESULT (2026-07-28), 3 seeds, mean +/- 95% half-width over seeds
----------------------------------------------------------------
    H1  arity 2      share a        0.3460 +/- 0.0490    +0.2349
                     share b        0.3717 +/- 0.0178    +0.2606
                     share nothing  0.1111 +/- 0.0240

    H2  permuted     share a        0.1967 +/- 0.0598    +0.0210
                     share b        0.1819 +/- 0.0085    +0.0062
                     share nothing  0.1757 +/- 0.0173

    H4  held out     share a        0.3951 +/- 0.0460    +0.2321
                     share b        0.3951 +/- 0.0631    +0.2321
                     share nothing  0.1631 +/- 0.0726

    H3  arity 3      share 2        0.7612 +/- 0.0405    +0.1561
                     share 1 shallow 0.6921 +/- 0.0480   +0.0870
                     share 1 deep   0.6500 +/- 0.0901    +0.0449
                     share 0        0.6050 +/- 0.0880

All five predictions hold. The number that matters is the comparison to the
language result, which was measured on a different corpus, a different pipeline
and a different set of areas:

    share-first-parent    language +0.2492      abstract symbols +0.2349
    share-second-parent   language +0.1213      abstract symbols +0.2606
    held-out combination  language +0.2298      abstract symbols +0.2321

The productivity numbers agree to three decimal places, which is closer than
the seed-to-seed spread of either and should be read as coincidence in its
precision, not in its direction. Composition here is a property of the
substrate and the merge operation. Language is one instance of it.

H2 is what licenses reading any of that. The permuted arm uses the SAME
assemblies and the SAME overlaps, shuffling only which label each composite is
filed under, and the gradient drops from +0.2349 to +0.0210 -- an order of
magnitude, into its own error bar. So the gradient tracks actual parent
sharing, not a merge target that pulls everything together. Note that "share
nothing" sits at 0.1111, still above chance 0.05: there IS residual attractor
pull, and the permutation control is what separates it from the signal rather
than pretending it is absent.

H3b was the prediction most likely to be wrong and it held: sharing the
SHALLOW parent c moves overlap more than sharing a deep parent a or b
(+0.0870 vs +0.0449). Composition is not flat in depth -- a parent attached
closer to the surface contributes more to the composite's identity, which is
what hierarchical structure predicts and what a bag-of-parents account does
not. Language never tested past arity 2 here, so this is new.

TWO HARNESS DEFECTS FOUND WHILE GETTING HERE, both of which first produced a
confident wrong answer:
  1. rounds=10 collapsed the merge target and every bin read exactly 1.0000.
     See MERGE_ROUNDS below.
  2. The held-out cells were chosen by striding the flattened grid, which
     selected the entire j=0 column. No trained composite then had j=0, the
     "share b" bin was empty, an empty bin scored as 0.0, and H4 reported
     FAILS at -0.1672 while the bin that did have data showed +0.3102. Fixed
     by holding out a Latin-square diagonal, and `grad` now returns nan for an
     empty bin instead of treating absent data as a measured zero.
"""

from __future__ import annotations

import itertools
import os
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TRAIN_PROGRESS", "0")

N, K, P = 1000, 50, 0.05
BETA = 0.10
SEEDS = (42, 7, 123)

#: Merge depth, and the single most important parameter here.
#:
#: The first run of this file used rounds=10 and every bin came back at overlap
#: EXACTLY 1.0000. `universality_diagnose.py` split that: the parents were fine
#: (mean pairwise 0.029, at chance, and a symbol re-projects to itself at 0.98)
#: and the MERGE TARGET had collapsed -- 36 merges into one area, mean pairwise
#: 0.9989. Same pathology as task #31's 94 VPs.
#:
#: The parser already knew. `parser_mixins/phrases.py` sets MERGE_ROUNDS = 2 and
#: records why: "recurrence merges assemblies that share an area... 94 merges
#: into an n=1000 area give mean pairwise overlap 0.752 at rounds=10 but 0.051
#: at rounds=1".
#:
#: This matters for what the file is claiming. The thing that makes composition
#: work is a SUBSTRATE parameter -- how much recurrence runs before the target
#: settles -- not a linguistic prior. Had the fix been something about verbs,
#: the universality claim would have been dead on arrival. Instead the parser
#: and this file need the same number for the same reason, which is a point in
#: favour of H1 before H1 is even measured.
MERGE_ROUNDS = 2

#: Slot inventories. Six per slot gives 36 arity-2 composites and 630 pairs,
#: enough that every bin has hundreds of members at three seeds.
M2 = 6
#: Arity 3 uses four per slot: 64 composites, 2016 pairs, and the merge count
#: stays affordable because each composite costs two merges rather than one.
M3 = 4

SRC_A, SRC_B, SRC_C = "SRC_A", "SRC_B", "SRC_C"
COMP, COMP2 = "COMP", "COMP2"

_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}


def ci(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n < 2:
        return values[0], float("nan")
    t = _T95.get(n, 1.96)
    return statistics.mean(values), t * statistics.stdev(values) / (n ** 0.5)


def build_brain(seed: int, areas: Sequence[str], stims: Sequence[str]):
    """A bare Brain: the areas and stimuli, nothing language-specific."""
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P, seed=seed)
    for a in areas:
        brain.add_area(a, N, K, beta=BETA)
    for s in stims:
        brain.add_stimulus(s, K)
    return brain


def learn_symbols(brain, pairs: Sequence[Tuple[str, str]], rounds: int = 12):
    """Give each symbol a stable assembly in its slot area.

    Without this the "parents" would be whatever the first merge happened to
    settle on, and shared-parent bins would be comparing noise to noise.
    """
    from neural_assemblies.assembly_calculus.ops import project

    for stim, area in pairs:
        project(brain, stim, area, rounds=rounds)


def compose2(brain, i: int, j: int, rounds: int = MERGE_ROUNDS):
    """merge(a_i, b_j) -> COMP, with the parent stimuli driven.

    Stimuli are passed deliberately: `merge`'s own docstring records that the
    stimulus-less path cannot write the back-projection synapses (measured
    potentiation 3.47 driven vs none fixed), and the back-projection is what
    makes merge a binding operation rather than a downstream readout.
    """
    from neural_assemblies.assembly_calculus.ops import merge

    return merge(brain, SRC_A, SRC_B, COMP,
                 stim_a=f"a{i}", stim_b=f"b{j}", rounds=rounds)


def compose3(brain, i: int, j: int, l: int, rounds: int = MERGE_ROUNDS):
    """merge(merge(a_i, b_j) -> COMP, c_l) -> COMP2, hierarchically.

    `merge` is binary, so arity 3 is necessarily a two-step build, and that is
    the point of H3b: c attaches one level shallower than a and b.
    """
    from neural_assemblies.assembly_calculus.ops import merge

    merge(brain, SRC_A, SRC_B, COMP,
          stim_a=f"a{i}", stim_b=f"b{j}", rounds=rounds)
    return merge(brain, COMP, SRC_C, COMP2, stim_b=f"c{l}", rounds=rounds,
                 unstimulated_source_mode="evolving")


# --------------------------------------------------------------------------
# Arity 2
# --------------------------------------------------------------------------

BINS2 = ("share a", "share b", "share nothing")


def bin2(ka, kb) -> str:
    if ka[0] == kb[0]:
        return "share a"
    if ka[1] == kb[1]:
        return "share b"
    return "share nothing"


def run_arity2(seed: int, holdout: int = 6):
    """Compose the grid minus a holdout, then compose the holdout fresh.

    Returns (trained_bins, heldout_bins, permuted_bins).
    """
    from neural_assemblies.assembly_calculus.assembly import overlap

    stims = [f"a{i}" for i in range(M2)] + [f"b{j}" for j in range(M2)]
    brain = build_brain(seed, (SRC_A, SRC_B, COMP), stims)
    learn_symbols(brain,
                  [(f"a{i}", SRC_A) for i in range(M2)]
                  + [(f"b{j}", SRC_B) for j in range(M2)])

    grid = [(i, j) for i in range(M2) for j in range(M2)]
    # Hold out a LATIN-SQUARE DIAGONAL: exactly one cell per row and exactly
    # one per column.
    #
    # The first version strided the flattened grid, which silently selected the
    # entire j=0 column -- so no TRAINED composite had j=0, the "share b" bin
    # was empty, and H4 scored as failed on a bin that had no data rather than
    # on a bin that showed no effect. The measured gradient in the bin that DID
    # have data was +0.3102, i.e. the opposite of the reported verdict.
    #
    # A diagonal guarantees every held-out cell has trained neighbours sharing
    # its a-parent AND trained neighbours sharing its b-parent, which is what
    # "compose something never composed, from parts seen elsewhere" requires.
    held = [(i, (i + 3) % M2) for i in range(min(holdout, M2))]
    held_set = set(held)
    trained = [g for g in grid if g not in held_set]

    asm: Dict[Tuple[int, int], object] = {}
    for i, j in trained:
        asm[(i, j)] = compose2(brain, i, j)
    heldout_asm = {(i, j): compose2(brain, i, j) for i, j in held}

    def bins_over(keys, table):
        out: Dict[str, List[float]] = {b: [] for b in BINS2}
        for ka, kb in itertools.combinations(keys, 2):
            out[bin2(ka, kb)].append(overlap(table[ka], table[kb]))
        return out

    trained_bins = bins_over(trained, asm)

    # Held-out composites are compared against the TRAINED ones, not each
    # other: the question is whether a fresh combination lands in the right
    # place relative to the established structure.
    held_bins: Dict[str, List[float]] = {b: [] for b in BINS2}
    for kh in held:
        for kt in trained:
            held_bins[bin2(kh, kt)].append(overlap(heldout_asm[kh], asm[kt]))

    # PERMUTATION CONTROL. Same assemblies, same overlaps -- only the label
    # each assembly is filed under is shuffled, so the bin structure is
    # destroyed while every measured number survives. A gradient that persists
    # here is an artifact of the merge target, not of sharing.
    import random

    rng = random.Random(seed)
    shuffled = list(trained)
    rng.shuffle(shuffled)
    relabel = dict(zip(trained, shuffled))
    perm_bins: Dict[str, List[float]] = {b: [] for b in BINS2}
    for ka, kb in itertools.combinations(trained, 2):
        perm_bins[bin2(relabel[ka], relabel[kb])].append(
            overlap(asm[ka], asm[kb]))

    return trained_bins, held_bins, perm_bins


# --------------------------------------------------------------------------
# Arity 3
# --------------------------------------------------------------------------

BINS3 = ("share 2", "share 1 deep", "share 1 shallow", "share 0")


def bin3(ka, kb) -> str:
    """Bin by how many of (a, b, c) match, splitting the 1-shared case.

    'deep' means the single shared parent is a or b -- attached through the
    intermediate COMP -- and 'shallow' means it is c, attached directly to the
    final area. H3b is the prediction that these two differ.
    """
    same = [ka[t] == kb[t] for t in range(3)]
    total = sum(same)
    if total >= 2:
        return "share 2"
    if total == 0:
        return "share 0"
    return "share 1 shallow" if same[2] else "share 1 deep"


def run_arity3(seed: int):
    from neural_assemblies.assembly_calculus.assembly import overlap

    stims = ([f"a{i}" for i in range(M3)] + [f"b{j}" for j in range(M3)]
             + [f"c{l}" for l in range(M3)])
    brain = build_brain(seed, (SRC_A, SRC_B, SRC_C, COMP, COMP2), stims)
    learn_symbols(brain,
                  [(f"a{i}", SRC_A) for i in range(M3)]
                  + [(f"b{j}", SRC_B) for j in range(M3)]
                  + [(f"c{l}", SRC_C) for l in range(M3)])

    keys = [(i, j, l) for i in range(M3) for j in range(M3) for l in range(M3)]
    asm = {kk: compose3(brain, *kk) for kk in keys}

    out: Dict[str, List[float]] = {b: [] for b in BINS3}
    for ka, kb in itertools.combinations(keys, 2):
        out[bin3(ka, kb)].append(overlap(asm[ka], asm[kb]))
    return out


# --------------------------------------------------------------------------

def table(title: str, per_seed: Dict[str, List[float]], order: Sequence[str],
          baseline: str):
    print(f"\n  {title}")
    print(f"    {'bin':<18} {'mean overlap':>14} {'95% half':>10} "
          f"{'vs ' + baseline:>12}")
    base = statistics.mean(per_seed[baseline]) if per_seed[baseline] else 0.0
    means = {}
    for b in order:
        vals = per_seed[b]
        if not vals:
            continue
        m, h = ci(vals)
        means[b] = m
        delta = "" if b == baseline else f"{m - base:+.4f}"
        print(f"    {b:<18} {m:>14.4f} {h:>10.4f} {delta:>12}")
    print(f"    {'chance (k/n)':<18} {K / N:>14.4f}")
    return means


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  substrate only: n={N} k={K} p={P} beta={BETA}, "
          f"seeds {list(SEEDS)}")
    print("  no lexicon, no grounding table, no categories -- abstract symbols")

    tr = {b: [] for b in BINS2}
    hd = {b: [] for b in BINS2}
    pm = {b: [] for b in BINS2}
    a3 = {b: [] for b in BINS3}

    for seed in SEEDS:
        t, h, p = run_arity2(seed)
        for b in BINS2:
            # Per-seed MEANS, so each seed contributes one observation and the
            # interval is over seeds rather than over pairs. Pooling pairs
            # would treat 630 correlated comparisons as independent and give a
            # CI roughly an order of magnitude too tight.
            if t[b]:
                tr[b].append(statistics.mean(t[b]))
            if h[b]:
                hd[b].append(statistics.mean(h[b]))
            if p[b]:
                pm[b].append(statistics.mean(p[b]))
        r3 = run_arity3(seed)
        for b in BINS3:
            if r3[b]:
                a3[b].append(statistics.mean(r3[b]))

    m_tr = table("H1 arity 2, composed during training", tr, BINS2,
                 "share nothing")
    m_pm = table("H2 permutation control (same overlaps, labels shuffled)", pm,
                 BINS2, "share nothing")
    m_hd = table("H4 held-out combinations, composed fresh", hd, BINS2,
                 "share nothing")
    m_a3 = table("H3 arity 3, hierarchical merge(merge(a,b),c)", a3, BINS3,
                 "share 0")

    print("\n  verdicts")

    def grad(m, hi, lo):
        """Gradient of bin *hi* over bin *lo*, or nan if either bin is EMPTY.

        A missing bin used to read as 0.0 and score as a failure, which is how
        H4 first came back "FAILS" off a bin that had no members. An absent
        measurement is not a measurement of absence.
        """
        if hi not in m or lo not in m:
            return float("nan")
        return m[hi] - m[lo]

    g_tr = min(grad(m_tr, "share a", "share nothing"),
               grad(m_tr, "share b", "share nothing"))
    print(f"    H1 universality      : smallest gradient {g_tr:+.4f} "
          f"-> {'HOLDS' if g_tr > 0 else 'FAILS'}")

    g_pm = max(abs(grad(m_pm, "share a", "share nothing")),
               abs(grad(m_pm, "share b", "share nothing")))
    print(f"    H2 permutation flat  : largest |gradient| {g_pm:+.4f} "
          f"-> {'HOLDS' if g_pm < g_tr / 2 else 'FAILS -- H1 is an artifact'}")

    mono = (m_a3.get("share 2", 0) > max(m_a3.get("share 1 deep", 0),
                                         m_a3.get("share 1 shallow", 0))
            > m_a3.get("share 0", 0))
    print(f"    H3 arity-3 monotone  : "
          f"2={m_a3.get('share 2', 0):.4f} "
          f"1d={m_a3.get('share 1 deep', 0):.4f} "
          f"1s={m_a3.get('share 1 shallow', 0):.4f} "
          f"0={m_a3.get('share 0', 0):.4f} -> {'HOLDS' if mono else 'FAILS'}")

    shallow = m_a3.get("share 1 shallow", 0) - m_a3.get("share 1 deep", 0)
    print(f"    H3b shallow > deep   : {shallow:+.4f} "
          f"-> {'HOLDS' if shallow > 0 else 'REFUTED (deep >= shallow)'}")

    g_hd = min(grad(m_hd, "share a", "share nothing"),
               grad(m_hd, "share b", "share nothing"))
    print(f"    H4 productivity      : smallest gradient {g_hd:+.4f} "
          f"-> {'HOLDS' if g_hd > 0 else 'FAILS'}")
    print(f"\n    language reference (constituent_structure.py, 5 seeds): "
          f"share-subject +0.2492, share-verb +0.1213")


if __name__ == "__main__":
    main()
