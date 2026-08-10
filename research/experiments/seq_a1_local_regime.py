"""Does the organ work at its OWN regime inside a sparser brain? [[SEQ-ORGAN-EMBEDS]]

A1 closed at a global p=0.4, which the whole brain shared. That is not how the
organ has to live: the parser runs p=0.05, and an organ that only works when
everything around it is dense is not embeddable. Per-fiber density
(`Brain.add_connectivity`, now implemented on numpy_sparse) is the mechanism;
this is the measurement.

PREDICTION, STATED BEFORE THE NUMBERS.

  * ambient p=0.05, organ p=0.4  -> 10/10 trajectories, matching the uniform
    p=0.4 result, IF the regime condition is genuinely local to the fibers the
    organ drives.
  * ambient p=0.05, no override  -> fails, since every organ fiber then sits at
    kp = 70*0.05 = 3.5 against floors of 18.6 and 25.6.

A failure of the first arm would mean the pooled candidate draw does not
survive heterogeneity -- it is moment-matched across fibers, exact in the first
two moments and an approximation beyond them -- and would send the organ to
numpy_exact, which computes drive without sampling at all.

The ambient brain is not decorative: unrelated dense areas and stimuli are
added and driven, so the organ is embedded in traffic rather than alone in a
brain that merely has a low `p` attribute.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.diagnostics import format_report, regime_audit
from neural_assemblies.programs.mod3_fsm import (
    END_SYMBOL, build_mod3_fsm, train_mod3_fsm,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_a1_fsm_parity import (
    BETA, K, N_ARC, N_STATE, NEGATIVE, POSITIVE, PRESENTATIONS, SEEDS,
)
from seq_a1_drift import true_trajectory

AMBIENT_P = 0.05
ORGAN_P = 0.40


def build(seed, *, organ_p, ambient_p=AMBIENT_P, with_neighbours=True):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(engine="numpy_sparse", p=ambient_p, save_winners=True,
                  seed=seed, norm_init=False)
    fsm = build_mod3_fsm(brain, n=N_ARC, k=K, n_state=N_STATE, beta=BETA,
                         organ_p=organ_p)
    if with_neighbours:
        # Unrelated traffic at the AMBIENT density, so the organ is embedded in
        # a working brain rather than alone in one that merely has a low `p`.
        for name in ("NEIGHBOUR_A", "NEIGHBOUR_B"):
            brain.add_area(name, 3000, K, BETA)
        brain.add_stimulus("neighbour_stim", K)
        for _ in range(5):
            brain.project({"neighbour_stim": ["NEIGHBOUR_A"]}, {})
            brain.project({}, {"NEIGHBOUR_A": ["NEIGHBOUR_B"]})
    train_mod3_fsm(fsm, presentations=PRESENTATIONS)
    return brain, fsm


def trajectory_correct(fsm, digits):
    symbols = [END_SYMBOL if d == 10 else str(d) for d in digits]
    return fsm.run(symbols, start_state="0") == true_trajectory(digits)


def arm(name, *, organ_p, seeds):
    rows = []
    for seed in seeds:
        brain, fsm = build(seed, organ_p=organ_p)
        pos = trajectory_correct(fsm, list(POSITIVE))
        neg = trajectory_correct(fsm, list(NEGATIVE))
        rows.append({"seed": seed, "positive": pos, "negative": neg,
                     "both": bool(pos and neg)})
        print(f"    seed {seed:2d}: positive {'ok' if pos else ' .'}  "
              f"negative {'ok' if neg else ' .'}", flush=True)
        if seed == seeds[0]:
            driven = {fsm.arc_area: [fsm.state_area, fsm._sym_stim["4"]],
                      fsm.state_area: [fsm.arc_area]}
            print(format_report([r for r in regime_audit(brain, driven)
                                 if "mod3" in r.area]))
    ok = sum(r["both"] for r in rows)
    print(f"    -> {ok}/{len(rows)} fully correct trajectories")
    return {"arm": name, "organ_p": organ_p, "correct": ok,
            "n": len(rows), "rows": rows}


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print(f"=== organ at its own regime inside a sparse brain ===")
    print(f"    ambient p = {AMBIENT_P}, organ p = {ORGAN_P}, k = {K}")
    print(f"    organ floors: arc 3 ln {N_ARC} = {3 * np.log(N_ARC):.1f}, "
          f"state 3 ln {N_STATE} = {3 * np.log(N_STATE):.1f}")

    arms = []
    print(f"\n  [local regime] organ fibers at p={ORGAN_P}")
    arms.append(arm("local_regime", organ_p=ORGAN_P, seeds=seeds))
    print(f"\n  [ambient only] no override -- every organ fiber at p={AMBIENT_P}")
    arms.append(arm("ambient_only", organ_p=None, seeds=seeds))

    print("\n=== SUMMARY ===")
    for a in arms:
        print(f"  {a['arm']:<14s} organ_p={str(a['organ_p']):<5s} "
              f"{a['correct']}/{a['n']} correct trajectories")
    verdict = (arms[0]["correct"] >= 0.8 * arms[0]["n"]
               and arms[1]["correct"] <= 0.2 * arms[1]["n"])
    print(f"\n  SEQ-ORGAN-EMBEDS: {'SUPPORTED' if verdict else 'NOT SUPPORTED'}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "seq_a1_local_regime_results.json")
    with open(out, "w") as fh:
        json.dump({"ambient_p": AMBIENT_P, "organ_p": ORGAN_P, "arms": arms},
                  fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
