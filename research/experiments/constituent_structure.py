"""Does a merged constituent's representation TRACK ITS PARTS?

THE QUESTION, stated so it can fail
------------------------------------
Removing `reset_area_connections(VP)` from `train_phrases` made VP constituents
distinct and retrievable (mean pairwise overlap 1.000 -> 0.492, rank-1 recall
from one parent 0.156 -> 0.781). But 0.492 is still far above chance (k/n =
0.05 at n=1000), and that residual admits two OPPOSITE readings:

  COLLAPSE   the merge target still has too strong an attractor and all VPs are
             being pulled toward a common assembly. The retrievability measured
             is then weaker than it looks and recursion should not be built on
             it.

  STRUCTURE  `dog_chases` and `dog_sees` SHARE A PARENT. If merge is genuinely a
             conjunctive bind, they are supposed to overlap. High mean overlap
             would then be evidence FOR compositionality, not against it.

These make opposite predictions about how overlap is DISTRIBUTED, so the two
can be separated without any new machinery.

PROTOCOL
--------
Partition every pair of stored subj_verb constituents by how many parents they
share, and compare mean overlap per bin:

    share subject   (dog_chases, dog_sees)
    share verb      (dog_chases, cat_chases)
    share nothing   (dog_chases, cat_sees)

Distinct keys cannot share both, so there are exactly three bins.

PREDICTIONS, recorded before running
-------------------------------------
1. STRUCTURE: overlap(share subject) > overlap(share nothing) and
   overlap(share verb) > overlap(share nothing). A graded profile is the
   signature of conjunctive binding.
2. COLLAPSE is the flat alternative: all three bins equal. That is a real
   possible outcome and would say the 0.492 is residual attractor pull, not
   composition. It would also mean the VP fix is not yet enough to build on.
3. share-nothing sits ABOVE chance regardless. Two constituents in one area
   with a shared recurrent history are not independent draws, so the honest
   baseline is the share-nothing bin itself, not k/n.
4. No prediction is registered for subject-vs-verb asymmetry. Retrieval was
   better from the subject (0.781) than the verb (0.646), so an asymmetry
   would be consistent with that, but the mechanism is not understood well
   enough to call a direction in advance.

THE CONFOUND, and why it is measured rather than assumed away
--------------------------------------------------------------
VPs formed close together in training share connectome state, so temporal
proximity alone can produce overlap. If the corpus happens to group sentences
sharing a subject, the bin effect would be a training-order artifact wearing a
compositional costume. Two things are reported to catch this: mean training-
index distance per bin (are the bins balanced?), and the bin profile recomputed
on DISTANT pairs only. A structure effect that survives on distant pairs is not
proximity.
"""

from __future__ import annotations

import itertools
import os
import statistics
import sys
from typing import Dict, List, NamedTuple, Sequence, Tuple

# 95% t multipliers for small seed counts (two-sided).
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262}

BINS = ("share subject", "share verb", "share nothing")


def _ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and 95% half-width across seeds; half-width nan for n < 2."""
    n = len(values)
    if n < 2:
        return (values[0] if values else float("nan")), float("nan")
    t = _T95.get(n, 1.96)
    return statistics.mean(values), t * statistics.stdev(values) / (n ** 0.5)


def _bin_of(key_a: str, key_b: str) -> str:
    """Which bin an unordered pair of `subj_verb` keys falls into."""
    sa, va = key_a.split("_", 1)
    sb, vb = key_b.split("_", 1)
    if sa == sb:
        return "share subject"
    if va == vb:
        return "share verb"
    return "share nothing"


class Analysis(NamedTuple):
    n_constituents: int
    by_bin: Dict[str, List[float]]
    dist_by_bin: Dict[str, List[int]]
    distant: Dict[str, List[float]]
    cut: int


def _pairs(parser):
    """The subj_verb constituents, in training order."""
    return {k: v for k, v in parser.vp_assemblies.items() if k.count("_") == 1}


def analyse(parser) -> Analysis:
    """Bin every constituent pair by shared parents; return per-bin overlaps."""
    from neural_assemblies.assembly_calculus.assembly import overlap

    pair = _pairs(parser)
    keys = list(pair)
    order = {k: i for i, k in enumerate(keys)}

    by_bin: Dict[str, List[float]] = {b: [] for b in BINS}
    dist_by_bin: Dict[str, List[int]] = {b: [] for b in BINS}
    distant: Dict[str, List[float]] = {b: [] for b in BINS}

    records = []
    for a, b in itertools.combinations(keys, 2):
        ov = overlap(pair[a], pair[b])
        d = abs(order[a] - order[b])
        bin_name = _bin_of(a, b)
        by_bin[bin_name].append(ov)
        dist_by_bin[bin_name].append(d)
        records.append((bin_name, ov, d))

    # "Distant" = training-index distance above the median, so proximity cannot
    # be doing the work. Computed over ALL pairs so the cut is bin-independent.
    all_d = sorted(d for _, _, d in records)
    cut = all_d[len(all_d) // 2] if all_d else 0
    for bin_name, ov, d in records:
        if d > cut:
            distant[bin_name].append(ov)

    return Analysis(len(keys), by_bin, dist_by_bin, distant, cut)


def run(seeds: Sequence[int] = (42, 7, 123, 2024, 5),
        n: int = 1000, k: int = 50) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
    os.environ.setdefault("TRAIN_PROGRESS", "0")

    from lesion_aphasia import train_parser

    per_seed: Dict[str, List[float]] = {b: [] for b in BINS}
    per_seed_distant: Dict[str, List[float]] = {b: [] for b in BINS}
    per_seed_dist: Dict[str, List[float]] = {b: [] for b in BINS}
    counts: Dict[str, int] = {b: 0 for b in BINS}
    n_const = 0

    for seed in seeds:
        parser = train_parser(seed, n=n, k=k)
        res = analyse(parser)
        n_const = res.n_constituents
        for b in BINS:
            vals = res.by_bin[b]
            if vals:
                per_seed[b].append(statistics.mean(vals))
                per_seed_dist[b].append(statistics.mean(res.dist_by_bin[b]))
                counts[b] = len(vals)
            dvals = res.distant[b]
            if dvals:
                per_seed_distant[b].append(statistics.mean(dvals))

    chance = k / n
    print(f"\n  {len(seeds)} seeds, n={n} k={k}, {n_const} constituents/seed")
    print(f"  chance overlap k/n = {chance:.3f}\n")
    print(f"  {'bin':<16}{'pairs':>7}{'overlap (95% CI)':>22}"
          f"{'mean train dist':>18}")
    for b in BINS:
        m, h = _ci(per_seed[b])
        dm, _ = _ci(per_seed_dist[b])
        print(f"  {b:<16}{counts[b]:>7}{m:>14.3f} +-{h:>5.3f}{dm:>18.1f}")

    print(f"\n  DISTANT PAIRS ONLY (training-index distance > median)")
    print(f"  {'bin':<16}{'overlap (95% CI)':>22}")
    for b in BINS:
        m, h = _ci(per_seed_distant[b])
        print(f"  {b:<16}{m:>14.3f} +-{h:>5.3f}")

    # PAIRED differences. Seeds are matched -- the same brain contributes to
    # every bin -- so the per-seed difference removes between-brain variance
    # that the unpaired CIs above carry. This is the test the claim rests on;
    # a paired CI excluding 0 is the evidence, not two intervals that happen
    # not to touch.
    def paired(store, bin_name):
        diffs = [a - b for a, b in zip(store[bin_name], store["share nothing"])]
        return _ci(diffs)

    print("\n  PAIRED vs share-nothing (same seed; CI excluding 0 = real)")
    print(f"  {'contrast':<28}{'all pairs':>20}{'distant only':>22}")
    for b in ("share subject", "share verb"):
        m, h = paired(per_seed, b)
        dm, dh = paired(per_seed_distant, b)
        print(f"  {b + ' - nothing':<28}{m:>12.3f} +-{h:>5.3f}"
              f"{dm:>14.3f} +-{dh:>5.3f}")

    base, _ = _ci(per_seed["share nothing"])
    sm, sh = paired(per_seed, "share subject")
    vm, vh = paired(per_seed, "share verb")
    dsm, dsh = paired(per_seed_distant, "share subject")
    dvm, dvh = paired(per_seed_distant, "share verb")

    print("\n  READING")
    graded = (sm - sh) > 0 and (vm - vh) > 0
    still = (dsm - dsh) > 0 and (dvm - dvh) > 0
    print(f"    sharing a parent raises overlap:      {graded}")
    print(f"    survives on distant pairs:            {still}")
    print(f"    share-nothing above chance:           "
          f"{base > chance}   ({base:.3f} vs {chance:.3f})")
    if graded and still:
        print("    -> STRUCTURE: overlap tracks shared parts, and it is not")
        print("       explained by training proximity.")
    elif graded:
        print("    -> AMBIGUOUS: graded overall but not on distant pairs, so")
        print("       training proximity is a live explanation.")
    else:
        print("    -> COLLAPSE: flat profile. The residual overlap is attractor")
        print("       pull, not composition. Do NOT build recursion on this.")


if __name__ == "__main__":
    run()
