"""#95 x #92: is the constituent competition decided by learning or by size?

THE DEFECT
----------
`WordOrderLearner` scored the "which constituent comes next" competition with
`input_drive(metric="pre_kwta")` -- a mean over every candidate neuron, hence a
function of how much substrate each helper area recruited. Measured on a trained
model (research/notes/conjunctive_arc_measured.md):

                      agent  action  patient   picks  wanted
    mood0 (SVO)        8.26    5.72     9.34       O       V
    mood1 (SOV)        7.07    5.42     9.09       O       O

    helper recruited w:   S 432   V 658   O 484

Mood-dependent signal ~15%, constant per-helper spread ~63%. Patient wins in
both moods regardless of the trained order.

The port had justified `pre_kwta` as "one deliberate improvement over the
reference", claiming the reference's winner-sum "biases the comparison toward
whichever area recruited more neurons". That is backwards. The reference computes

    sum over w in from.winners, u in to.winners of connectome[w, u]

which has k_from * k_to terms -- both always k -- and therefore CANNOT depend on
recruited size. `pre_kwta` can and does. The "improvement" introduced the
confound it claimed to remove, and every word-order number produced through it
was scored through a nuisance term ~4x larger than the signal.

WHAT THIS FILE MEASURES
-----------------------
A 2x2: scoring (winners | pre_kwta) x arc (off | on), on the four mood pairs
that the multi-mood tests use, over several seeds. Reporting both factors
together is the point -- #92's arc arm read 5/24 against an 18/24 baseline, but
BOTH were scored through the confounded readout, so that comparison was between
nuisance terms and cannot be interpreted.

PRE-REGISTERED
--------------
R1 `winners` beats `pre_kwta` at arc=off. If the readout was the binding
   constraint on the EXISTING architecture, fixing it should move the baseline
   on its own. If it does not, the confound is real but not load-bearing, and
   that is worth knowing before more is built on top of it.
R2 The arc's sign flips. Under `pre_kwta` the arc scored WORSE (5/24 vs 18/24).
   The claim being tested is that this was the readout, not the arc: under
   `winners` the arc should be no worse, and the mechanism section of #92
   predicts better, since the arc is the only configuration where the mood
   distinction is present in an assembly (separation 0.46-0.78 against HELPER's
   1.00).
R3 The single-mood floor stays 4/4 in every arm. Already checked for both
   scorings before this run; repeated here because an arm that loses the floor
   has broken something basic and its multi-mood number means nothing.

NOT PREDICTED, AND LEFT OPEN: whether `winners` is itself unbiased. It removes
the `w` dependence by construction, but it is a sum over k*k SPECIFIC synapses
and could carry some other structure. The check reported below is the ratio of
the mood-dependent part of the score to the constant per-helper part; if that
ratio is still well under 1 in the winning arm, the readout is still the
binding constraint and R1/R2 are being read off a floor.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_assemblies.reference.word_order_learner import (  # noqa: E402
    HELPER, MOOD, WordOrderLearner)

ORDERS = {"SVO": ("S", "V", "O"), "SOV": ("S", "O", "V"),
          "VSO": ("V", "S", "O"), "OVS": ("O", "V", "S")}
PAIRS = [("SVO", "SOV"), ("SVO", "VSO"), ("SOV", "OVS"), ("VSO", "OVS")]
SEEDS = (1, 2, 3, 4)
N, K, P, BETA, SENTENCES = 1000, 50, 0.05, 0.06, 60


def _build(a, b, seed, scoring, arc):
    return WordOrderLearner(
        num_nouns=4, num_verbs=2, mood_orders={0: ORDERS[a], 1: ORDERS[b]},
        n=N, k=K, p=P, beta=BETA, seed=seed,
        scoring=scoring, conjunctive_arc=arc)


def single_mood_floor(scoring, arc):
    ok = 0
    for name in sorted(ORDERS):
        m = WordOrderLearner(
            num_nouns=4, num_verbs=2, mood_orders={0: ORDERS[name]},
            n=N, k=K, p=P, beta=0.1, seed=1,
            scoring=scoring, conjunctive_arc=arc)
        m.train(20)
        ok += "".join(m.generate(0)) == name
    return ok


def signal_to_bias(m):
    """mood-dependent share of the score vs the constant per-helper share.

    For the transition out of S, take each candidate helper's score under mood 0
    and mood 1. The MOOD-DEPENDENT part is the mean |score(m0) - score(m1)|; the
    CONSTANT part is the spread across helpers of their mean score. A readout
    that decides on learning has a ratio above 1; the confounded one measured
    0.15 / 0.63 = 0.24.
    """
    from neural_assemblies.assembly_calculus.binding import input_drive
    import copy
    per_mood = []
    live = m.brain
    try:
        for mi in (0, 1):
            m.brain = copy.deepcopy(live)
            with m.brain.frozen():
                m._mood_now = mi
                m.brain.activate(MOOD, mi)
                m._activate_role(0, "S", firings=3)
                syn = m._syn("S")
                m.brain.project({}, {HELPER["S"]: [syn], MOOD: [syn]})
                if m.conjunctive_arc:
                    m._form_arc("S")
                    src = "ARC"
                else:
                    src = syn
                d = input_drive(m.brain, sources=[src],
                                target_areas=[HELPER[c] for c in ("S", "V", "O")],
                                metric=m.scoring)
                per_mood.append([d[HELPER[c]] for c in ("S", "V", "O")])
    finally:
        m.brain = live
    mood_part = statistics.mean(
        abs(a - b) for a, b in zip(per_mood[0], per_mood[1]))
    means = [(a + b) / 2 for a, b in zip(per_mood[0], per_mood[1])]
    const_part = max(means) - min(means)
    return mood_part / const_part if const_part else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n  #95 -- does the readout decide the competition, or does size?")
    print(f"  n={N} k={K} p={P} beta={BETA}, {SENTENCES} sentences, "
          f"{len(PAIRS)} mood pairs x {len(SEEDS)} seeds = "
          f"{len(PAIRS) * len(SEEDS) * 2} generations per arm\n")
    print(f"  {'scoring':>9} {'arc':>5} {'multi-mood':>12} {'1-mood':>8} "
          f"{'signal/bias':>12}")

    results = {}
    for scoring in ("pre_kwta", "winners"):
        for arc in (False, True):
            ok = tot = 0
            ratios = []
            for a, b in PAIRS:
                for seed in SEEDS:
                    m = _build(a, b, seed, scoring, arc)
                    m.train(SENTENCES)
                    got = ["".join(m.generate(i)) for i in (0, 1)]
                    ok += (got[0] == a) + (got[1] == b)
                    tot += 2
                    if seed == SEEDS[0]:
                        ratios.append(signal_to_bias(m))
            floor = single_mood_floor(scoring, arc)
            r = statistics.mean(ratios)
            results[(scoring, arc)] = (ok, tot, floor, r)
            print(f"  {scoring:>9} {str(arc):>5} {ok:>7}/{tot:<4} "
                  f"{floor:>6}/4 {r:>12.2f}", flush=True)

    print("\n  READING\n")
    base_old = results[("pre_kwta", False)]
    base_new = results[("winners", False)]
    arc_old = results[("pre_kwta", True)]
    arc_new = results[("winners", True)]

    r1 = base_new[0] > base_old[0]
    print(f"    R1 `winners` beats `pre_kwta` at arc=off:  {str(r1):>5}   "
          f"{base_old[0]}/{base_old[1]} -> {base_new[0]}/{base_new[1]}")
    d_old = arc_old[0] - base_old[0]
    d_new = arc_new[0] - base_new[0]
    r2 = d_new > d_old
    print(f"    R2 the arc's effect improves:              {str(r2):>5}   "
          f"{d_old:+d} under pre_kwta, {d_new:+d} under winners")
    r3 = all(v[2] == 4 for v in results.values())
    print(f"    R3 single-mood floor is 4/4 everywhere:    {str(r3):>5}   "
          f"{[v[2] for v in results.values()]}")

    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"\n    best arm: scoring={best[0][0]} arc={best[0][1]} -> "
          f"{best[1][0]}/{best[1][1]}, signal/bias {best[1][3]:.2f}")
    if best[1][3] < 1.0:
        print("    signal/bias is still BELOW 1 in the best arm: the score is")
        print("    still dominated by a constant per-helper term, so these")
        print("    numbers are read off a floor and the readout is STILL the")
        print("    binding constraint. Do not treat the mechanism comparison")
        print("    as settled.")
    else:
        print("    signal/bias is above 1: the competition is decided by what")
        print("    was learned, and the mechanism comparison is interpretable")
        print("    for the first time.")


if __name__ == "__main__":
    main()
