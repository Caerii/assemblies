"""Does norm_init make recurrence safe? Only for very few items.

THE CLAIM UNDER TEST IS IN THE ENGINE
-------------------------------------
`core/brain.py:project_rounds` carries a long comment justifying why the fast
path drops target self-recurrence (`a != target`). It reports the collapse that
motivated it -- TWO independent stimuli projected into one area reach overlap
0.240 at 5 rounds and 0.940 at 15, against chance 0.025 -- then diagnoses the
mechanism as degree bias, shows `norm_init` removes it, and concludes:

    "Self-recurrence is therefore enabled only when that normalization is
     active ... after which recurrence is safe"

`lexicon_capacity_law.py` measured something that does not fit that conclusion.
Its brains were built with `Brain(p=..., seed=...)`, i.e. norm_init at the
PRODUCTION DEFAULT of True, and the recurrent build still collapsed: capacity
ceilings of M=32 / 64 / 256 words at n=1000 / 2000 / 4000, with pairwise
overlap 0.6869 at M=256, n=1000 against a floor of 0.0500.

Both can be true. The engine's measurement used M=2. If norm_init's rescue
degrades with the NUMBER of items sharing the area, then "recurrence is safe"
holds exactly where it was tested and fails everywhere a lexicon lives. That is
worth knowing precisely, because the comment is load-bearing: it is the stated
justification for `recurrent_projection`, and anyone who reads it as a blanket
guarantee and flips that flag will silently collapse the production lexicon --
which trains through this very code path, with its self-fiber stripped by the
default, and is protected only by that accident.

THE MEASUREMENT
---------------
Sweep M from 2 upward with recurrence ON, crossing norm_init on/off, and find
the M at which each stops working. M=2 reproduces the engine's own numbers and
anchors the comparison; everything above it is new.

`acc` (rank-1 identity across the M stored items) is the reported quantity, not
`spread` alone -- at M=2 an overlap of 0.24 still leaves two items trivially
distinguishable, so overlap alone overstates the damage at small M and
understates it at large M where a small mean hides a collapsed subset.

PRE-REGISTERED
--------------
Y1 At M=2, norm_init holds overlap near chance with recurrence on, reproducing
   the engine comment. If this fails, the comment describes a regime this file
   is not reproducing and nothing below applies.
Y2 norm_init's rescue DEGRADES with M: there is an M* above which recurrence
   collapses the area even with norm_init on. The claim is that M* exists and
   is small enough to matter for a lexicon.
Y3 norm_init is still strictly better than no norm_init at every M -- the
   degree-bias fix is real, just not sufficient. If norm_init is NOT better,
   the original diagnosis is wrong, which would be a much larger finding.
Y4 Feed-forward has no ceiling in this same sweep, confirming that what
   norm_init partially fixes is a problem recurrence creates.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _substrate import MIN_TRIALS, probe, rank1, read, similarity, spread  # noqa: E402

N, K_, P_, BETA = 1000, 50, 0.05, 0.10
BUILD_ROUNDS = 6
AREA = "L"
M_SWEEP = (2, 4, 8, 16, 32, 64, 128)
SEEDS = (42, 7, 123, 2024, 5, 99)


def build(brain, stim, rounds, recurrent):
    if recurrent:
        from neural_assemblies.assembly_calculus.ops import project
        project(brain, stim, AREA, rounds=rounds, recurrent=True)
    else:
        for _ in range(rounds):
            brain.project({stim: [AREA]}, {})
    return read(brain, AREA)


def trial(m_words, recurrent, norm_init, seed, engine="numpy_sparse"):
    from neural_assemblies.core.brain import Brain

    brain = Brain(p=P_, seed=seed, norm_init=norm_init, engine=engine)
    brain.add_area(AREA, N, K_, beta=BETA)
    for m in range(m_words):
        brain.add_stimulus(f"w{m}", K_)

    stored = {m: build(brain, f"w{m}", BUILD_ROUNDS, recurrent)
              for m in range(m_words)}

    hits, ident = 0, []
    for m in range(m_words):
        with probe(brain):
            live = build(brain, f"w{m}", BUILD_ROUNDS, recurrent)
        hits += rank1(live, stored) == m
        ident.append(similarity(live, stored[m]))
    return hits, m_words, statistics.mean(ident), spread(stored.values())


def run(m_words, recurrent, norm_init, engine="numpy_sparse"):
    res = [trial(m_words, recurrent, norm_init, s, engine) for s in SEEDS]
    tot = sum(x[1] for x in res)
    return (sum(x[0] for x in res) / tot,
            statistics.mean(x[2] for x in res),
            statistics.mean(x[3] for x in res), tot)


ARMS = (("rec", True, True), ("rec", True, False), ("ff", False, True))
LABEL = {("rec", True): "rec norm", ("rec", False): "rec RAW",
         ("ff", True): "ff  norm"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n  n={N} k={K_} beta={BETA} p={P_}, buildT={BUILD_ROUNDS}, ONE "
          f"area, no merges")
    print(f"  rec = area->area self-recurrence on;  ff = feed-forward only")
    print(f"  norm = norm_init True (production default);  RAW = norm_init "
          f"False")
    print(f"  engine comment's regime is M=2, rec, and it is the first row\n")
    print(f"  {'M':>5}{'arm':>10}{'chance':>9}{'acc':>8}{'ident':>8}"
          f"{'spread':>9}{'floor':>8}{'trials':>8}")

    table = {}
    for m_words in M_SWEEP:
        for name, rec, norm in ARMS:
            r = run(m_words, rec, norm)
            table[(m_words, name, norm)] = r
            flag = "" if r[3] >= MIN_TRIALS else "  [UNDER-POWERED]"
            print(f"  {m_words:>5}{LABEL[(name, norm)]:>10}"
                  f"{1 / m_words:>9.4f}{r[0]:>8.4f}{r[1]:>8.4f}"
                  f"{r[2]:>9.4f}{K_ / N:>8.4f}{r[3]:>8}{flag}")
        print()

    def ceiling(name, norm):
        ok = [m for m in M_SWEEP if table[(m, name, norm)][0] > 0.90]
        return max(ok) if ok else 0

    print("  READING")
    m2 = table[(2, "rec", True)]
    y1 = m2[2] < 0.10
    print(f"    Y1 engine's M=2 regime reproduces:   {y1}   "
          f"spread {m2[2]:.4f} vs floor {K_ / N:.4f}, acc {m2[0]:.4f}")

    c_rec = ceiling("rec", True)
    y2 = c_rec < max(M_SWEEP)
    print(f"    Y2 norm_init's rescue degrades with M: {y2}   "
          f"rec+norm holds to M={c_rec}, fails at M="
          f"{min([m for m in M_SWEEP if m > c_rec], default='-')}")
    y3 = all(table[(m, "rec", True)][0] >= table[(m, "rec", False)][0]
             for m in M_SWEEP)
    print(f"    Y3 norm_init still strictly helps:   {y3}   "
          f"raw ceiling M={ceiling('rec', False)} vs norm M={c_rec}")
    c_ff = ceiling("ff", True)
    y4 = c_ff == max(M_SWEEP)
    print(f"    Y4 feed-forward has no ceiling here: {y4}   "
          f"ff ceiling M={c_ff} (sweep top {max(M_SWEEP)})")

    print()
    if y1 and y2:
        print(f"    THE ENGINE COMMENT IS RIGHT WHERE IT WAS MEASURED AND WRONG")
        print(f"    AS A GENERALISATION. norm_init removes the degree bias, and")
        print(f"    at M=2 that is enough. It is not enough at M>{c_rec}: the")
        print(f"    competitor for a new item's k-WTA is not a hub, it is the")
        print(f"    ALREADY-POTENTIATED assemblies of the items before it, and")
        print(f"    normalising initial weights does nothing about those.")
        print(f"    'after which recurrence is safe' should read 'after which")
        print(f"    recurrence is safe for a SINGLE assembly per area'.")
        print(f"    Consequence: `recurrent_projection` must not be flipped on")
        print(f"    globally. The production lexicon trains through this path")
        print(f"    with dozens of words in one core area.")
    elif not y1:
        print(f"    Y1 FAILED -- M=2 does not reproduce the engine's numbers, so")
        print(f"    this file is not in the regime the comment describes and")
        print(f"    cannot be used to correct it. Find the difference first.")
    else:
        print(f"    norm_init holds across the whole sweep. The engine comment")
        print(f"    generalises after all, and the lexicon collapse measured in")
        print(f"    lexicon_capacity_law.py must come from something else --")
        print(f"    that discrepancy is now the thing to chase.")


if __name__ == "__main__":
    main()
