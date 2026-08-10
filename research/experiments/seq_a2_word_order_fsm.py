"""A2: #92's word-order conjunction, re-run through the fixed organ.

PRE-REGISTERED. Committed before the run; no Brain-side numbers were read first.

WHAT #92 FOUND. A conjunctive arc conditioning word order on MOOD collapsed:
arc overlap across STATES within one mood read 0.90-0.99 (should be low) and
across MOODS for one state 0.48-0.97. Sweeping MOOD->ARC plasticity found no
interior optimum and a strictly losing trade, ~2:1 against, so
`arc_conjunction_has_no_operating_point.md` concluded the arc has none. It also
failed the SINGLE-mood floor, where there is no mood to condition on and the
arc should behave like the direct order synapse it replaced -- an arm that
fails that floor was never producing an interpretable multi-mood number.

WHY RE-RUN IT. That note names three violations, and all three are now fixable:
p=0.05 against the paper's 0.2 (per-fiber p now exists), beta too high, and no
per-round homeostasis -- "since the failure here is one input out-accumulating
the other, per-round renormalisation is not a detail, it is the thing that
would stop the accumulation". Refraction IS that force, and it is now
drive-proportional and accumulates across transitions
([[ARC-CONJUNCT-EXPOSURE]], [[REFRACTION-PROPORTIONAL]]).

THE TASK, kept faithful to what broke. Each mood is a symbol; the state is the
constituent just emitted, so the STATE TRAJECTORY IS the word order:

    SVO mood:  START -> S -> V -> O
    VSO mood:  START -> V -> S -> O
    OSV mood:  START -> O -> S -> V

The exposure asymmetry that caused #92 is PRESERVED and reported: a mood fires
on every step of its sentence while any given state fires once, so the mood
conjunct accumulates several times the potentiation. If refraction is the
anti-swamping force, this is where it has to show.

BARS.

  P-ORDER   >= 8/10 seeds emit the correct order for EVERY mood, read from the
            state assembly. #92's arc arms scored 7/32 and 6/32.
  P-CONJ2   arc overlap across STATES (fixed mood) AND across MOODS (fixed
            state) both < 0.15, on NEURON IDs. #92 read 0.90-0.99 and
            0.48-0.97. Both directions, per [[ARC-CONJUNCT-EXPOSURE]].
  P-FLOOR   the SINGLE-mood arm also >= 8/10. This is R3, the floor #92's arc
            arms failed (4/4, 1/4, 4/4, 0/4). An arm failing it is
            uninterpretable on multi-mood.
  P-NULL    refraction off must FAIL P-ORDER (<= 2/10), or the result is not
            attributable to the mechanism under test.

INTERPRETATION, stated first. P-CONJ2 passing overturns "the arc has no
operating point" IN ITS OWN TASK, not merely on mod-3. P-ORDER passing while
P-CONJ2 fails would mean the order is carried by something other than the
conjunction, and P-ORDER would then NOT be evidence for the organ. P-FLOOR
failing while P-ORDER passes means the multi-mood number is uninterpretable,
exactly as in #92.

Index space: every overlap is measured on NEURON IDs via `_snap`, because
compact indices produced a *published* wrong claim in #92
([[two-index-spaces-compact-vs-neuron-id]]).
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

import numpy as np

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import format_report, regime_audit
from neural_assemblies.programs.nemo_fsm import NemoArcFSM

SEEDS = tuple(range(1, 11))
N_ARC, N_STATE, K = 5000, 500, 70
AMBIENT_P, ORGAN_P, BETA = 0.05, 0.40, 0.1
PRESENTATIONS = 15

ORDERS = {"svo": ("S", "V", "O"),
          "vso": ("V", "S", "O"),
          "osv": ("O", "S", "V")}
STATES = ["START", "S", "V", "O"]


def transitions_for(moods):
    """(from_state, mood, to_state), so the state trajectory IS the order."""
    out = []
    for mood in moods:
        order = ORDERS[mood]
        out.append(("START", mood, order[0]))
        out.append((order[0], mood, order[1]))
        out.append((order[1], mood, order[2]))
    return out


def exposure(moods):
    """How often each conjunct fires, which is what #92's failure was made of."""
    trans = transitions_for(moods)
    return Counter(m for _, m, _ in trans), Counter(q for q, _, _ in trans)


def build(seed, moods, *, refraction=True):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=AMBIENT_P, save_winners=True, seed=seed,
                  engine="numpy_sparse", norm_init=False)
    fsm = NemoArcFSM(
        brain, states=list(STATES), symbols=list(moods),
        transitions=transitions_for(moods),
        n=N_ARC, k=K, n_state=N_STATE, beta=BETA, organ_p=ORGAN_P,
        refracted_strength=0.1 if refraction else 0.0, prefix="_a2",
    )
    if not refraction:
        brain.set_refracted(fsm.arc_area, False)
    fsm.train_from_list([(m, q, r) for q, m, r in transitions_for(moods)],
                        presentations=PRESENTATIONS)
    return brain, fsm


def order_correct(fsm, mood):
    """Emit three constituents from START; the trajectory must be the order."""
    return fsm.run([mood] * 3, start_state="START") == list(ORDERS[mood])


def arc_overlaps(brain, fsm, moods):
    """(across-STATE at fixed mood, across-MOOD at fixed state), neuron IDs."""
    def arc(state, mood):
        with brain.probe():
            brain.inhibit_areas([fsm.arc_area, fsm.state_area])
            fsm._cue_state(state)
            brain.project({fsm._sym_stim[mood]: [fsm.arc_area]},
                          {fsm.state_area: [fsm.arc_area]})
            return _snap(brain, fsm.arc_area)

    across_state = []
    for mood in moods:
        a = [arc(q, mood) for q in STATES]
        across_state += [overlap(a[i], a[j])
                         for i in range(len(a)) for j in range(i + 1, len(a))]
    across_mood = []
    if len(moods) > 1:
        for q in STATES:
            a = [arc(q, m) for m in moods]
            across_mood += [overlap(a[i], a[j])
                            for i in range(len(a)) for j in range(i + 1, len(a))]
    return (float(np.mean(across_state)),
            float(np.mean(across_mood)) if across_mood else float("nan"))


def arm(name, moods, *, refraction=True, seeds=SEEDS, show_audit=False):
    rows = []
    for seed in seeds:
        brain, fsm = build(seed, moods, refraction=refraction)
        ok = all(order_correct(fsm, m) for m in moods)
        a_state, a_mood = arc_overlaps(brain, fsm, moods)
        rows.append({"seed": seed, "all_moods_correct": bool(ok),
                     "across_state": a_state, "across_mood": a_mood})
        print(f"    seed {seed:2d}: order {'ok' if ok else ' .'}   "
              f"arc across-state {a_state:.3f}  across-mood {a_mood:.3f}",
              flush=True)
        if show_audit and seed == seeds[0]:
            driven = {fsm.arc_area: [fsm.state_area, fsm._sym_stim[moods[0]]],
                      fsm.state_area: [fsm.arc_area]}
            print(format_report([r for r in regime_audit(brain, driven)
                                 if "_a2" in r.area]))
    correct = sum(r["all_moods_correct"] for r in rows)
    print(f"    -> {correct}/{len(rows)} seeds with every mood correct")
    return {"arm": name, "moods": list(moods), "refraction": refraction,
            "correct": correct, "n": len(rows),
            "across_state": float(np.mean([r["across_state"] for r in rows])),
            "across_mood": float(np.nanmean([r["across_mood"] for r in rows])),
            "rows": rows}


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    multi = list(ORDERS)
    mood_exp, state_exp = exposure(multi)
    print("=== A2: #92's word-order conjunction through the fixed organ ===")
    print(f"    ambient p={AMBIENT_P}, organ p={ORGAN_P}, k={K}, "
          f"presentations={PRESENTATIONS}")
    print(f"    exposure per presentation -- mood {dict(mood_exp)}, "
          f"state {dict(state_exp)}")
    print(f"    the asymmetry #92 failed on is PRESERVED: a mood fires "
          f"{max(mood_exp.values())}x where a state fires "
          f"{min(state_exp.values())}x")

    arms = []
    print("\n  [multi-mood] refraction on")
    arms.append(arm("multi_mood", multi, seeds=seeds, show_audit=True))
    print("\n  [single-mood] the floor #92's arc arms failed (R3)")
    arms.append(arm("single_mood", ["svo"], seeds=seeds))
    print("\n  [null] refraction off -- must fail")
    arms.append(arm("null_no_refraction", multi, refraction=False, seeds=seeds))

    n = len(seeds)
    m, single, null = arms[0], arms[1], arms[2]
    verdicts = {
        "P-ORDER": m["correct"] >= 0.8 * n,
        "P-CONJ2": m["across_state"] < 0.15 and m["across_mood"] < 0.15,
        "P-FLOOR": single["correct"] >= 0.8 * n,
        "P-NULL": null["correct"] <= 0.2 * n,
    }
    print("\n=== SUMMARY ===")
    print(f"  {'arm':<20s} {'correct':>9s} {'across-state':>13s} {'across-mood':>12s}")
    for a in arms:
        print(f"  {a['arm']:<20s} {a['correct']:>6d}/{a['n']:<2d} "
              f"{a['across_state']:>13.3f} {a['across_mood']:>12.3f}")
    print("\n  #92 measured across-state 0.90-0.99 and across-mood 0.48-0.97")
    print("\n=== BARS ===")
    for bar, ok in verdicts.items():
        print(f"  {bar:<9s} {'PASS' if ok else 'FAIL'}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "seq_a2_word_order_results.json")
    with open(out, "w") as fh:
        json.dump({"arms": arms, "verdicts": verdicts,
                   "params": {"ambient_p": AMBIENT_P, "organ_p": ORGAN_P,
                              "n_arc": N_ARC, "n_state": N_STATE, "k": K,
                              "beta": BETA, "presentations": PRESENTATIONS}},
                  fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
