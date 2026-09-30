"""24% of the core lexicon are EXACT DUPLICATES. Which words, and do they own
the retrieval failure?

WHAT CAME BEFORE. `source_overlap_structure_and_errors.py` asked whether the
parser's source overlap is degradation or similarity structure, and the answer
was neither of the two things I had framed:

  * the mean spread 0.125 is 12.5x the k/n floor, but the mean is a function of
    the DEGREE SEQUENCE alone (identity verified to 1e-17), so it never could
    have distinguished "every pair shares a little" from "some pairs are the
    same assembly";
  * a degree-preserving shuffle reproduces the mean exactly and the SD not at
    all -- observed 0.2598 against 0.0567, and observed max pair 1.0000 against
    a shuffled 0.3533. The distribution is BIMODAL: most pairs near the floor,
    a tail at identity;
  * `check_distinct` said so outright -- PARTIAL-COLLAPSE(distinct=0.761). 24%
    of the 71 NOUN_CORE assemblies are exact duplicates of another word's.

And the errors follow it: the impostor that wins a failed retrieval out-overlaps
its competitors at superiority 0.914-0.968 against a 0.500 null.

WHY FOUR EXPERIMENTS MISSED IT. `_substrate.check_distinct` carries a
partial-collapse detector precisely because mean overlap is nearly blind to
duplicates -- 256 items on 58 distinct assemblies still reads 0.0129 against a
0.0125 floor. Four experiments in this arc hand-rolled a local `_spread` instead
of calling it, so the detector never ran and the mean was quoted as if it were
the whole story. One canonical way, again: the hand-rolled path is the one
without the alarm.

WHAT THIS FILE ADDS. Correlation is not mechanism. Three questions:

  1. WHICH WORDS. Cluster the core assemblies by exact identity and print the
     members. A cluster of synonyms means something very different from a
     cluster of arbitrary words.

  2. WHY THOSE. Corpus frequency per word, from the same modules the trainer
     imports. If the duplicated words are the rare ones, the mechanism is
     under-training -- a word seen once or never does not get an assembly of
     its own, and k-WTA still returns k winners, so it silently lands on
     whatever the tie-break gives. That is the totalizing substrate: no bottom,
     so the failure returns a plausible assembly instead of nothing.

  3. DO THEY OWN THE ERRORS. The decisive arm. Retrieval at a FIXED candidate
     set size of 6 -- so difficulty is identical across arms -- over three
     pools: every word, only words whose core assembly is unique, and only
     words in a duplicate cluster. If singletons retrieve near 1.000 while
     duplicates sit at chance, the 0.73 is not a capacity limit or a separation
     limit; it is a MIXTURE of a working substrate and a set of words that were
     never given distinct representations.

THE ARM THAT WOULD FALSIFY THIS. If singleton-only retrieval stays near 0.73,
duplicates are a visible symptom of something broader and removing them buys
nothing -- in which case the honest report is that the 0.73 survives the most
obvious explanation. That arm is run and printed either way.

WHAT WOULD MAKE THE THIRD ARM DEGENERATE, named in advance: if fewer than
CANDIDATES + 1 words survive a filter, its retrieval is undefined rather than
bad, and it is printed as `n/a` with the pool size rather than as a number.
"""
import os
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import numpy as np                                                     # noqa: E402

from _substrate import read, similarity, report_rate                   # noqa: E402
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
NEAR = 0.5


def _corpus_counts():
    """Frequencies from the same modules `CurriculumTrainer` imports."""
    from neural_assemblies.lexicon.curriculum.stage4_sentences import (
        STAGE4_CORPUS,
    )
    from neural_assemblies.lexicon.curriculum.stage3_two_word import (
        STAGE3_CORPUS,
    )
    return Counter(
        w for line in list(STAGE4_CORPUS) + list(STAGE3_CORPUS)
        for w in line.split()
    )


def _clusters(lex):
    """word -> assembly, grouped by EXACT identity of the neuron set."""
    by_key = defaultdict(list)
    for w, asm in lex.items():
        key = tuple(sorted(int(x) for x in np.asarray(asm.winners)))
        by_key[key].append(w)
    return by_key


def _retrieval_over_pool(parser, role_area, pool, rng):
    """Rank-1 among a random CANDIDATES of `pool`. Difficulty is constant."""
    if len(pool) < CANDIDATES + 1:
        return None
    brain = parser.brain
    core = parser.core_lexicons.get(NOUN_CORE, {})
    lex = parser.role_lexicons.get(role_area, {})
    stored = {w: np.asarray(lex[w].winners, dtype=np.int64) for w in pool}
    live = {}
    for w in pool:
        with brain.probe():
            activate_assembly(brain, core[w])
            brain.project({}, {NOUN_CORE: [role_area]})
            live[w] = read(brain, role_area)
    hits = total = 0
    for _ in range(SUBSETS):
        subset = list(rng.choice(list(pool), size=CANDIDATES, replace=False))
        for w in subset:
            best = max(((similarity(live[w], stored[o]), o) for o in subset))[1]
            hits += int(best == w)
            total += 1
    return hits, total


def analyse(parser, seed, counts):
    core = parser.core_lexicons.get(NOUN_CORE, {})
    area = parser.brain.areas[NOUN_CORE]
    n, k = int(area.n), int(area.k)

    # ---- Q1: which words -----------------------------------------------
    groups = _clusters(core)
    dup_groups = {kk: ws for kk, ws in groups.items() if len(ws) > 1}
    dup_words = {w for ws in dup_groups.values() for w in ws}
    print(f"  NOUN_CORE n={n} k={k}  M={len(core)}  "
          f"distinct assemblies {len(groups)} "
          f"({len(groups) / max(len(core), 1):.1%})")
    print(f"  words sharing an assembly with another word: {len(dup_words)} "
          f"({len(dup_words) / max(len(core), 1):.1%}) in "
          f"{len(dup_groups)} clusters")
    print()
    print("  Q1  EXACT-DUPLICATE CLUSTERS (corpus count in brackets)")
    for _key, ws in sorted(dup_groups.items(), key=lambda kv: -len(kv[1])):
        shown = ", ".join(f"{w}[{counts.get(w, 0)}]" for w in sorted(ws))
        print(f"      x{len(ws):<3} {shown}")
    if not dup_groups:
        print("      none")

    # ---- Q1b: are the INPUTS identical too? -----------------------------
    # THE QUESTION THAT DECIDES WHOSE DEFECT THIS IS. `train_lexicon` projects
    # phon + grounding SIMULTANEOUSLY, so two words with the same total input
    # must produce the same winners -- that is k-WTA working correctly, not
    # failing. If the grounding sets collide, the substrate is exonerated and
    # the defect is in the featural code upstream of it. If the inputs DIFFER
    # and the assemblies still collide, the substrate is losing information
    # that was handed to it, which is a completely different bug.
    def _ground_key(w):
        ctx = getattr(parser, "word_grounding", {}).get(w)
        if ctx is None:
            return ("<missing>",)
        if isinstance(ctx, dict):
            return tuple(sorted(map(str, ctx.keys())))
        if isinstance(ctx, (list, tuple, set, frozenset)):
            return tuple(sorted(map(str, ctx)))
        return (str(ctx),)

    words = sorted(core)
    gkeys = {w: _ground_key(w) for w in words}
    print()
    print("  Q1b IS THE COLLISION ALREADY PRESENT IN THE INPUT?")
    print(f"      {len(words)} words -> {len(set(gkeys.values()))} distinct "
          f"grounding sets -> {len(groups)} distinct assemblies")
    same_in = same_out = 0
    for _key, ws in dup_groups.items():
        ident = len({gkeys[w] for w in ws}) == 1
        same_out += 1
        same_in += int(ident)
        mark = "SAME INPUT" if ident else "DIFFERENT INPUTS"
        print(f"      cluster of {len(ws):<3} {mark:<17} "
              f"grounding {sorted({gkeys[w] for w in ws})[0]}")
    if same_out:
        print(f"      {same_in}/{same_out} duplicate clusters were ALREADY "
              f"identical at the input")

    # near-duplicates that are not exact
    asms = {w: np.asarray(core[w].winners, dtype=np.int64) for w in words}
    near = [(a, b) for a, b in combinations(words, 2)
            if NEAR <= similarity(asms[a], asms[b]) < 1.0]
    print(f"      plus {len(near)} NON-identical pairs at overlap in "
          f"[{NEAR}, 1.0)  -- near-collisions the exact test does not catch")

    # ---- Q2: why those --------------------------------------------------
    print()
    print("  Q2  IS DUPLICATION EXPLAINED BY CORPUS FREQUENCY?")
    print("      CAVEAT, and it is fatal to reading a null here: these counts")
    print("      are from the SENTENCE corpus (stage3+stage4), which is NOT")
    print("      what trains the core lexicon -- `train_lexicon` walks the")
    print("      VOCABULARY PRESET. Most words read 0 in BOTH groups because")
    print("      the wrong corpus is being counted, which is the same error")
    print("      that put `chases` in the ERP frames. Treat a flat result as")
    print("      'not measured', not as 'frequency does not matter'.")
    dup_c = [counts.get(w, 0) for w in words if w in dup_words]
    uni_c = [counts.get(w, 0) for w in words if w not in dup_words]
    for label, vals in (("duplicated", dup_c), ("unique", uni_c)):
        if not vals:
            print(f"      {label:<12} (empty)")
            continue
        zero = sum(1 for v in vals if v == 0)
        print(f"      {label:<12} n={len(vals):<4} median count "
              f"{statistics.median(vals):>6.1f}  mean {statistics.fmean(vals):>7.2f}"
              f"   zero-count {zero}/{len(vals)} ({zero / len(vals):.0%})")

    # ---- Q3: do they own the errors -------------------------------------
    print()
    print(f"  Q3  RETRIEVAL AT A FIXED CANDIDATE SET OF {CANDIDATES} "
          f"(chance {1 / CANDIDATES:.3f})")
    for ra in ROLE_AREAS:
        rl = parser.role_lexicons.get(ra, {})
        shared = sorted(w for w in rl if w in core)
        pools = {
            "all words": shared,
            "UNIQUE core only": [w for w in shared if w not in dup_words],
            "DUPLICATED core only": [w for w in shared if w in dup_words],
        }
        print(f"      {ra}")
        for label, pool in pools.items():
            res = _retrieval_over_pool(parser, ra, pool,
                                       np.random.default_rng(seed))
            if res is None:
                print(f"        {label:<24} n/a -- pool has {len(pool)} words, "
                      f"needs > {CANDIDATES}")
                continue
            hits, total = res
            print(report_rate(f"  {label} (pool {len(pool)})", hits, total,
                              1.0 / CANDIDATES))
    return {"dup_words": dup_words, "n_core": len(core),
            "n_distinct": len(groups)}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("EXACT DUPLICATES IN THE CORE LEXICON -- which words, why, and do")
    print("they own the retrieval failure?")
    print()
    counts = _corpus_counts()
    for seed in SEEDS:
        print(f"=== depth={DEPTH} seed={seed} ===")
        parser = train_parser_to_depth(DEPTH, seed=seed)
        analyse(parser, seed, counts)
        print()

    print("=" * 72)
    print("READING THIS")
    print("  UNIQUE >> all >> DUPLICATED  -> the 0.73 is a MIXTURE. The")
    print("      substrate works; a subset of words was never given a distinct")
    print("      representation, and every capacity/separation number in this")
    print("      arc was averaged over both populations.")
    print("  UNIQUE ~ all                 -> duplicates are a symptom, not the")
    print("      cause, and the 0.73 survives its most obvious explanation.")


if __name__ == "__main__":
    main()
