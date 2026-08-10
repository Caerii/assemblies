"""Is the running machine a limit cycle? Constant input, measured in ASSEMBLY space.

The organ is a DRIVEN map: the state transition depends on the input symbol, so
a random digit stream wanders aperiodically among discrete attractors. Hold the
input constant and mod-3 arithmetic makes it periodic -- digit d should give
period 3/gcd(d, 3): a fixed point for d in {0,3,6,9}, a 3-cycle otherwise.

The claim worth testing is stronger than the label sequence repeating. A1 found
that the arc AMPLIFIES a state perturbation ~8x while recovery is nonetheless
exact, which is only consistent if k-WTA acts as a QUANTIZER: inside a basin it
maps a whole neighbourhood onto exactly one point in a single step
(superattracting), which is also why failure is a cliff rather than a decay.

PREDICTION, STATED BEFORE THE NUMBERS. If that is right the orbit closes in
ASSEMBLY space, not merely in the readout: the winner set at step t is
BIT-IDENTICAL to the winner set at step t + period, giving assembly overlap
exactly 1.000 and 0.000 between different phases of the cycle. A system that
merely labels correctly while drifting inside each basin would show high but
sub-1.000 return overlap, and would degrade over many revolutions.

Run 60 revolutions so any slow drift has room to show.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.mod3_fsm import build_mod3_fsm, train_mod3_fsm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_a1_fsm_parity import BETA, K, N_ARC, N_STATE, PRESENTATIONS

STEPS = 60
SEEDS = (1, 2, 3)
P = 0.4


def detect_period(assemblies):
    """Smallest q such that assembly t and t+q are identical for all t."""
    for q in range(1, len(assemblies) // 2 + 1):
        if all(overlap(assemblies[t], assemblies[t + q]) >= 1.0
               for t in range(len(assemblies) - q)):
            return q
    return None


def run_constant(seed, digit):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=P, save_winners=True, seed=seed, engine="numpy_sparse",
                  norm_init=False)
    fsm = build_mod3_fsm(brain, n=N_ARC, k=K, n_state=N_STATE, beta=BETA)
    train_mod3_fsm(fsm, presentations=PRESENTATIONS)

    labels, assemblies = [], []
    with brain.probe():
        brain.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state("0")
        fsm._unfix_state()
        for _ in range(STEPS):
            labels.append(fsm.step(str(digit)))
            assemblies.append(_snap(brain, fsm.state_area))

    expected = 3 // math.gcd(digit, 3)
    truth = [str((digit * (i + 1)) % 3) for i in range(STEPS)]
    period = detect_period(assemblies)
    # how exactly does the orbit close, and how distinct are its phases?
    ret = ([float(overlap(assemblies[t], assemblies[t + expected]))
            for t in range(STEPS - expected)])
    cross = ([float(overlap(assemblies[0], assemblies[j]))
              for j in range(1, expected)] if expected > 1 else [])
    return {"seed": seed, "digit": digit, "expected_period": expected,
            "measured_period": period, "labels_correct": labels == truth,
            "return_overlap_min": min(ret), "return_overlap_mean": float(np.mean(ret)),
            "cross_phase_overlap": cross, "labels": labels[:9]}


def main():
    print(f"=== constant input: is it a limit cycle, in ASSEMBLY space? ===")
    print(f"    p={P}, {STEPS} steps, period should be 3/gcd(d,3)\n")
    print(f"  {'seed':>4s} {'digit':>5s} {'expect':>7s} {'measured':>9s} "
          f"{'labels':>7s} {'return overlap':>15s} {'cross-phase':>12s}")
    rows = []
    for seed in SEEDS:
        for digit in range(10):
            r = run_constant(seed, digit)
            rows.append(r)
            cross = (f"{max(r['cross_phase_overlap']):.3f}"
                     if r["cross_phase_overlap"] else "n/a")
            print(f"  {seed:>4d} {digit:>5d} {r['expected_period']:>7d} "
                  f"{str(r['measured_period']):>9s} "
                  f"{'ok' if r['labels_correct'] else 'WRONG':>7s} "
                  f"{r['return_overlap_min']:>15.3f} {cross:>12s}", flush=True)

    ok = sum(r["measured_period"] == r["expected_period"] for r in rows)
    exact = sum(r["return_overlap_min"] >= 1.0 for r in rows)
    print(f"\n  period as predicted: {ok}/{len(rows)}")
    print(f"  orbit closes EXACTLY (return overlap 1.000 every revolution): "
          f"{exact}/{len(rows)}")
    cross_all = [v for r in rows for v in r["cross_phase_overlap"]]
    if cross_all:
        print(f"  max overlap between DIFFERENT phases of a cycle: "
              f"{max(cross_all):.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "seq_a1_limit_cycle_results.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
