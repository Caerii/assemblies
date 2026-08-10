"""A1 follow-up 2: does the state assembly DRIFT along a sequence?

Established so far: the arc is a perfect conjunction (across-state and
across-symbol overlap both 0.000) and every one of the 33 transitions is
correct in isolation (330/330 over 10 seeds). Yet only 5/10 seeds decide a
6-symbol sequence. Single-step tests always start from the EXACT stored state
assembly; step 2 of a sequence starts from whatever the arc recovered.

PREDICTION, STATED BEFORE THE NUMBERS. The readout labels a state by nearest
overlap, so it tolerates a partly-correct assembly; the arc does not, because a
conjunction with 0.000 between-assembly overlap is maximally sensitive to its
input. If that is the mechanism:

  * overlap between the state area's assembly and the CORRECT stored assembly
    should fall along the sequence rather than staying at 1.000, and
  * seeds that fail should show the drop at or before the step where their
    trajectory diverges from the true one.

If instead overlap stays at 1.000 while the trajectory still goes wrong, drift
is NOT the mechanism and the fault is in the readout or in the cue.

Reports the true trajectory alongside the observed one, so a divergence can be
located rather than inferred.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.programs.mod3_fsm import END_SYMBOL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_a1_fsm_parity import NEGATIVE, POSITIVE, PRESENTATIONS, SEEDS, build


def true_trajectory(digits):
    """Ground truth for the mod-3 machine."""
    state, out = 0, []
    for d in digits:
        if d == 10:
            state = 3 if state == 0 else 4
        else:
            state = (state + d) % 3
        out.append({0: "0", 1: "1", 2: "2", 3: "accept", 4: "reject"}[state])
    return out


def traced_run(brain, fsm, digits):
    """Run a digit string, recording overlap with the CORRECT state each step."""
    symbols = [END_SYMBOL if d == 10 else str(d) for d in digits]
    truth = true_trajectory(digits)
    observed, overlaps = [], []
    with brain.probe():
        brain.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state("0")
        fsm._unfix_state()
        for symbol, want in zip(symbols, truth):
            got = fsm.step(symbol)
            observed.append(got)
            current = _snap(brain, fsm.state_area)
            overlaps.append(float(overlap(current, fsm.state_assembly(want))))
    return observed, truth, overlaps


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    records = []
    for seed in seeds:
        brain, fsm = build(seed, presentations=PRESENTATIONS)
        pos_obs, pos_true, pos_ov = traced_run(brain, fsm, list(POSITIVE))
        neg_obs, neg_true, neg_ov = traced_run(brain, fsm, list(NEGATIVE))
        decided = (pos_obs[-1] == "accept" and neg_obs[-1] == "reject")
        records.append({"seed": seed, "decided": decided,
                        "pos_observed": pos_obs, "pos_true": pos_true,
                        "pos_overlap": pos_ov,
                        "neg_observed": neg_obs, "neg_true": neg_true,
                        "neg_overlap": neg_ov})
        print(f"  seed {seed:2d} {'DECIDED' if decided else '   .   '}  "
              f"positive overlap with correct state per step: "
              + " ".join(f"{v:.2f}" for v in pos_ov), flush=True)
        if pos_obs != pos_true:
            first = next(i for i, (a, b) in enumerate(zip(pos_obs, pos_true))
                         if a != b)
            print(f"            diverges at step {first + 1}: "
                  f"got {pos_obs[first]!r}, want {pos_true[first]!r} "
                  f"(overlap there {pos_ov[first]:.2f})")

    step1 = float(np.mean([r["pos_overlap"][0] for r in records]))
    last = float(np.mean([r["pos_overlap"][-1] for r in records]))
    ok = [r for r in records if r["decided"]]
    bad = [r for r in records if not r["decided"]]
    print(f"\n  mean overlap with the correct state: step 1 {step1:.3f} "
          f"-> final step {last:.3f}")
    if ok and bad:
        print(f"  deciding seeds  (n={len(ok)}): mean overlap "
              f"{np.mean([v for r in ok for v in r['pos_overlap']]):.3f}")
        print(f"  failing seeds   (n={len(bad)}): mean overlap "
              f"{np.mean([v for r in bad for v in r['pos_overlap']]):.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "seq_a1_drift_results.json")
    with open(out, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
