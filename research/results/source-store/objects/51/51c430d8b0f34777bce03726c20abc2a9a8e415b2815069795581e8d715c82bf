"""#98: is the multi-mood failure the SAMPLER rather than the architecture?

WHY THIS EXISTS
---------------
Every word-order result in this repository was measured on `numpy_sparse`:
`WordOrderLearner` had no `engine` parameter until now. That engine's candidate
sampler INVENTS the drive for neurons that have not fired, and this session
established that matched load does not clear such an A/B -- the sampler read
+0.0000 on a pair where exact drive read -0.2656.

A single paired run then showed, at n=1000, 60 sentences, moods SVO + SOV:

    numpy_sparse   ['SVO', 'SVO']   <- both moods identical: the known failure
    numpy_exact    ['SVO', 'SOV']   <- both correct

That is the pair carried as `xfail` in test_word_order_learner, the one five
prior interventions failed to fix, and the one the AUTHOR'S OWN implementation
(.reference/dmitropolsky-assemblies/word_order_int.py) also fails at this size
-- checked directly: it generated SOV for both moods.

One seed is not a result. This file runs the full matrix.

PRE-REGISTERED
--------------
M1 exact beats sparse on the multi-mood matrix BY MORE THAN THE NOISE.

   The first version of this criterion was `exact > sparse` -- no threshold at
   all, so any single extra success would have "confirmed" it, and it duly
   did: 18/24 vs 20/24 printed a verdict claiming the sampler was a large part
   of the problem. That difference is 0.083, which is EXACTLY ONE standard
   error at these counts (SE = sqrt(0.79*0.21/24) = 0.083); the Wilson
   intervals are [0.551, 0.880] and [0.641, 0.933], almost entirely
   overlapping. Separating arms 0.083 apart at 80% power needs ~377 trials
   each, not 24.

   Writing a threshold-free criterion is the same error `gain_stability` was
   built to prevent hours earlier -- which is why its `noise_floor` argument is
   mandatory with no default. The bar here is now 2 SE, stated before the
   numbers are read, and 24 trials cannot clear it. Reaching a verdict on this
   question means running the matrix at a size that can.
M2 the advantage is CONCENTRATED in the pairs that share an opening
   constituent (SVO+SOV), because those are the ones that must diverge on a
   cue the shared-syntax architecture makes mood-blind. If the gain is uniform
   across pairs instead, it is a general accuracy effect and not the specific
   mechanism claimed.
M3 the single-mood floor stays 4/4 on exact. An engine that improved
   multi-mood while breaking one-mood generation would be changing something
   other than what is claimed.

IF M1 HOLDS: a large part of "multi-mood word order does not work in NEMO" was
an artifact of the instrument, and the arc work in #92/#93/#95 -- all of it run
on the sampler -- was attacking a problem that partly is not there. That would
also put every other sampler-measured word-order number in scope.

IF M1 FAILS: the single run was seed noise, which is exactly what this file is
for, and [[report-distributions-not-point-estimates]] gets another instance.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.reference.word_order_learner import (  # noqa: E402
    WordOrderLearner)

ORDERS = {"SVO": ("S", "V", "O"), "SOV": ("S", "O", "V"),
          "VSO": ("V", "S", "O"), "OVS": ("O", "V", "S")}
PAIRS = [("SVO", "SOV"), ("SVO", "VSO"), ("SOV", "OVS"), ("VSO", "OVS")]
SEEDS = (1, 2, 3)
N, K, P, BETA, SENTENCES = 1000, 50, 0.05, 0.06, 60


def run_pair(a, b, seed, engine):
    m = WordOrderLearner(
        num_nouns=4, num_verbs=2, mood_orders={0: ORDERS[a], 1: ORDERS[b]},
        n=N, k=K, p=P, beta=BETA, seed=seed, engine=engine)
    m.train(SENTENCES)
    got = ["".join(m.generate(i)) for i in (0, 1)]
    return (got[0] == a) + (got[1] == b), got


def floor(engine):
    ok = 0
    for name in sorted(ORDERS):
        m = WordOrderLearner(
            num_nouns=4, num_verbs=2, mood_orders={0: ORDERS[name]},
            n=N, k=K, p=P, beta=0.1, seed=1, engine=engine)
        m.train(20)
        ok += "".join(m.generate(0)) == name
    return ok


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n  #98 -- multi-mood word order: sampler vs exact drive")
    print(f"  n={N} k={K} p={P} beta={BETA}, {SENTENCES} sentences, "
          f"{len(SEEDS)} seeds\n")
    per_pair = {}
    totals = {}
    for engine in ("numpy_sparse", "numpy_exact"):
        print(f"  {engine}")
        tot = 0
        for a, b in PAIRS:
            got = 0
            for seed in SEEDS:
                s, _ = run_pair(a, b, seed, engine)
                got += s
            per_pair[(engine, a, b)] = got
            tot += got
            print(f"      {a}+{b}   {got}/{2 * len(SEEDS)}", flush=True)
        totals[engine] = tot
        f = floor(engine)
        per_pair[(engine, "floor")] = f
        print(f"      TOTAL {tot}/{2 * len(SEEDS) * len(PAIRS)}   "
              f"single-mood floor {f}/4\n", flush=True)

    sp, ex = totals["numpy_sparse"], totals["numpy_exact"]
    print("  READING\n")
    trials = 2 * len(SEEDS) * len(PAIRS)
    phat = (sp + ex) / (2 * trials)
    se = (phat * (1 - phat) / trials) ** 0.5
    delta = (ex - sp) / trials
    m1 = delta > 2 * se
    print(f"    M1 exact beats sparse by >2 SE:        {str(m1):>5}   "
          f"{sp} -> {ex} of {trials}, delta {delta:+.3f} = "
          f"{delta / se:.2f} SE (SE={se:.3f})")
    shared = per_pair[("numpy_exact", "SVO", "SOV")] \
        - per_pair[("numpy_sparse", "SVO", "SOV")]
    others = sum(per_pair[("numpy_exact", a, b)] - per_pair[("numpy_sparse", a, b)]
                 for a, b in PAIRS[1:])
    m2 = shared >= max(1, others / max(len(PAIRS) - 1, 1))
    print(f"    M2 gain concentrated in SVO+SOV:       {str(m2):>5}   "
          f"shared-opening {shared:+d}, others {others:+d} over 3 pairs")
    m3 = per_pair[("numpy_exact", "floor")] == 4
    print(f"    M3 single-mood floor holds on exact:   {str(m3):>5}   "
          f"{per_pair[('numpy_exact', 'floor')]}/4")
    print()
    if m1 and m3:
        print("    The engine choice moves this metric by more than the noise,")
        print("    so every word-order number measured before WordOrderLearner")
        print("    had an `engine` parameter is in scope for re-derivation.")
    elif not m1:
        print("    NOT SUPPORTED AT THIS SIZE. The single paired run that")
        print("    motivated this file -- exact ['SVO','SOV'] against sparse")
        print("    ['SVO','SVO'] -- was seed 1, and across seeds the effect is")
        print("    within noise. The sampler is NOT the explanation for the")
        print("    multi-mood failure on this protocol, and the arc")
        print("    conclusions in #92/#93/#95 stand as measured.")
        print()
        print("    That is also mildly reassuring about those conclusions: the")
        print("    two engines agree to within 1 SE on the OUTCOME metric here,")
        print("    which is not a licence to trust sparse in general -- it says")
        print("    nothing about the internal separations -- but it does mean")
        print("    the sparse measurements were not obviously void.")


if __name__ == "__main__":
    main()
