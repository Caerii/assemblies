"""How long a sequence can the transition organ run?

A1 closed with the mechanism: the state area is a DISCRETE attractor, the arc
amplifies any error ~8x per step, and above the regime floor recovery is exact
(100/100 steps at p=0.4). That makes a sharp, falsifiable prediction about
sequence length.

PREDICTION, STATED BEFORE THE NUMBERS.

  * If recovery is exactly exact, there is no error to amplify and the horizon
    is UNBOUNDED -- accuracy stays 1.000 at any length, and the first-error
    index is "never" within whatever we can afford to run.
  * If a residual per-step error rate eps > 0 remains, the trajectory survives
    about 1/eps steps and then fails, because the amplification turns the first
    wrong neuron into a wrong state within a few steps. p=0.3 recovers exactly
    on 80/100 steps in the sweep, so it should show a SHORT horizon; p=0.4
    recovers 100/100 and should not fail at all.

So the interesting number is not mean accuracy but the index of the FIRST
divergence from ground truth. One long run yields every prefix, so a single
2000-step sequence per seed measures the whole curve.

Only digits are used. States `accept` and `reject` are absorbing -- they never
appear as a from-state in the mod-3 table -- so a long string must stay in the
residue states and end (if at all) with `end`.

Refraction cannot confound this: `run` executes under `brain.probe()`, so no
bias is charged and step t cannot alter step t+1 through that channel.
"""
from __future__ import annotations

import json
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

LENGTH = 2000
SEEDS = (1, 2, 3, 4, 5)
P_VALUES = (0.3, 0.4)
CHECKPOINTS = (10, 50, 100, 500, 1000, 2000)


def run_long(seed, p, length=LENGTH):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse",
                  norm_init=False)
    fsm = build_mod3_fsm(brain, n=N_ARC, k=K, n_state=N_STATE, beta=BETA)
    train_mod3_fsm(fsm, presentations=PRESENTATIONS)

    rng = random.Random(seed * 7919)
    digits = [rng.randrange(10) for _ in range(length)]

    residue, correct, first_error, exact_steps = 0, [], None, 0
    with brain.probe():
        brain.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state("0")
        fsm._unfix_state()
        for i, d in enumerate(digits):
            residue = (residue + d) % 3
            got = fsm.step(str(d))
            ok = (got == str(residue))
            correct.append(ok)
            if overlap(_snap(brain, fsm.state_area),
                       fsm.state_assembly(str(residue))) >= 1.0:
                exact_steps += 1
            if not ok and first_error is None:
                first_error = i + 1
    return {
        "seed": seed, "p": p, "length": length,
        "first_error": first_error,
        "accuracy": float(np.mean(correct)),
        "exact_fraction": exact_steps / length,
        "prefix_correct": {c: bool(all(correct[:c])) for c in CHECKPOINTS
                           if c <= length},
    }


def main():
    print(f"=== how far does the machine run? {LENGTH} random digits, "
          f"{len(SEEDS)} seeds ===")
    print(f"    state floor 3 ln {N_STATE} = {3 * np.log(N_STATE):.1f} "
          f"-> p = {3 * np.log(N_STATE) / K:.3f}")
    rows = []
    for p in P_VALUES:
        print(f"\n  p = {p}  (state kp = {K * p:.0f})")
        got = []
        for seed in SEEDS:
            r = run_long(seed, p)
            got.append(r)
            rows.append(r)
            fe = r["first_error"]
            print(f"    seed {seed}: first error at step "
                  f"{fe if fe is not None else 'NEVER':>6}   "
                  f"accuracy {r['accuracy']:.4f}   "
                  f"exact {r['exact_fraction']:.3f}", flush=True)
        survived = [r for r in got if r["first_error"] is None]
        print(f"    -> {len(survived)}/{len(got)} seeds ran all {LENGTH} steps "
              f"without a single error")
        for c in CHECKPOINTS:
            n_ok = sum(r["prefix_correct"].get(c, False) for r in got)
            print(f"       first {c:>5} steps perfect: {n_ok}/{len(got)}")

    from _results import results_path
    out = results_path("sequence", "seq_a1_horizon_results.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
