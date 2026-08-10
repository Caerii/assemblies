"""A1 follow-up 3: WHICH link drifts -- the arc, or arc->state recovery?

A1 established: single transitions 330/330 correct, but state overlap decays
0.987 -> 0.327 along a 6-step sequence. Two links can be responsible and they
imply different fixes:

  ARC SENSITIVITY. The arc is a 0.000-overlap conjunction, so it may have no
    tolerance for a state input that is 99% right. Then the arc assembly is
    already wrong at step 2 and everything downstream follows. Fix direction:
    tolerance -- a softer conjunction, or clean-up before the arc reads.

  LOSSY RECOVERY. The arc picks the right assembly but arc->state returns an
    imperfect copy of the taught state. Fix direction: the state area's
    recovery -- regime, or extra settling rounds.

Measured by comparing, at every step of the sequence, the assembly that
actually fired against the one that WOULD have fired from an exact cue:

  arc_ref[i]   = arc assembly for (true_state[i], symbol[i]), exact cue
  state_ref[i] = the stored assembly for true_state[i]

PREDICTION, STATED BEFORE THE NUMBERS. If arc sensitivity dominates, arc
overlap falls first or faster than state overlap. If recovery is lossy, state
overlap falls while arc overlap stays high. The reference runs its state area
at the SAME kp = 14 against the same 3 ln 500 = 18.6 floor and decides 3/3, so
the regime is not obviously the binding constraint and the substrate is the
better-supported suspect -- but neither of those is what this script measures.

The reference's own per-step overlap is measured too, as the control that says
what "no drift" looks like on this task.
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
from seq_a1_fsm_parity import (
    POSITIVE, PRESENTATIONS, SEEDS, arc_assembly, build,
)
from seq_a1_drift import true_trajectory

LABELS = {0: "0", 1: "1", 2: "2", 3: "accept", 4: "reject"}


def ours(seed):
    """Per-step arc and state overlap against what an exact cue would give."""
    brain, fsm = build(seed, presentations=PRESENTATIONS)
    digits = list(POSITIVE)
    symbols = [END_SYMBOL if d == 10 else str(d) for d in digits]
    truth = true_trajectory(digits)
    prior = ["0"] + truth[:-1]          # the state the arc SHOULD read at step i

    # Reference assemblies, taken from exact cues before the sequence runs.
    arc_ref = [arc_assembly(brain, fsm, q, s) for q, s in zip(prior, symbols)]

    arc_ov, state_ov, observed = [], [], []
    with brain.probe():
        brain.inhibit_areas([fsm.arc_area, fsm.state_area])
        fsm._cue_state("0")
        fsm._unfix_state()
        for i, symbol in enumerate(symbols):
            brain.project({fsm._sym_stim[symbol]: [fsm.arc_area]},
                          {fsm.state_area: [fsm.arc_area]})
            arc_ov.append(float(overlap(_snap(brain, fsm.arc_area), arc_ref[i])))
            brain.project({}, {fsm.arc_area: [fsm.state_area]})
            state_ov.append(float(overlap(_snap(brain, fsm.state_area),
                                          fsm.state_assembly(truth[i]))))
            observed.append(fsm.read_state())
    return {"seed": seed, "arc_overlap": arc_ov, "state_overlap": state_ov,
            "observed": observed, "truth": truth,
            "trajectory_correct": observed == truth}


def reference(seed):
    """The control: the vendored reference's own per-step state overlap."""
    from neural_assemblies.reference.nemo_numpy.fsm_network import (
        FSMNetwork, build_mod3_symbols_states, mod3_transition_list,
    )
    rng = np.random.default_rng(seed)
    fsm = FSMNetwork(1000, 500, 5000, 70, 0.2, 0.1, rng)
    symbols, states = build_mod3_symbols_states(70, rng)
    for _ in range(PRESENTATIONS):
        for fr, sym, to in mod3_transition_list():
            fsm.train(symbols[sym], states[fr], states[to])

    digits = list(POSITIVE)
    truth = true_trajectory(digits)
    idx = {"0": 0, "1": 1, "2": 2, "accept": 3, "reject": 4}
    fsm.inhibit()
    fsm.state_area.fire(states[0], update=False)
    state_ov, observed = [], []
    for d, want in zip(digits, truth):
        fsm.forward(symbols[d], update=False)
        read = fsm.read()
        state_ov.append(len(np.intersect1d(read, states[idx[want]])) / 70)
        best = max(range(5), key=lambda i:
                   len(np.intersect1d(read, states[i])) / max(len(read), 1))
        observed.append(LABELS[best])
    return {"seed": seed, "state_overlap": state_ov,
            "observed": observed, "truth": truth,
            "trajectory_correct": observed == truth}


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    print("=== ours (numpy_sparse, p=0.2) ===")
    print("    arc   = overlap with the arc assembly an EXACT cue would select")
    print("    state = overlap with the stored assembly for the CORRECT state")
    mine = []
    for seed in seeds:
        r = ours(seed)
        mine.append(r)
        print(f"  seed {seed:2d} {'traj OK' if r['trajectory_correct'] else '  .    '}")
        print(f"      arc   " + " ".join(f"{v:.2f}" for v in r["arc_overlap"]))
        print(f"      state " + " ".join(f"{v:.2f}" for v in r["state_overlap"]),
              flush=True)

    print("\n=== reference control (same task, same regime) ===")
    refs = []
    for seed in (42, 7, 13):
        r = reference(seed)
        refs.append(r)
        print(f"  seed {seed:2d} {'traj OK' if r['trajectory_correct'] else '  .    '}"
              f"  state " + " ".join(f"{v:.2f}" for v in r["state_overlap"]),
              flush=True)

    arc = np.array([r["arc_overlap"] for r in mine])
    state = np.array([r["state_overlap"] for r in mine])
    print(f"\n  mean per step (ours)      arc   "
          + " ".join(f"{v:.2f}" for v in arc.mean(axis=0)))
    print(f"                            state "
          + " ".join(f"{v:.2f}" for v in state.mean(axis=0)))
    print(f"  mean per step (reference) state "
          + " ".join(f"{v:.2f}" for v in
                     np.array([r["state_overlap"] for r in refs]).mean(axis=0)))
    print(f"\n  trajectory correct: ours {sum(r['trajectory_correct'] for r in mine)}"
          f"/{len(mine)}   reference "
          f"{sum(r['trajectory_correct'] for r in refs)}/{len(refs)}")

    # Which link breaks FIRST? Compare the step at which each crosses 0.9.
    def first_below(row, thr=0.9):
        below = [i for i, v in enumerate(row) if v < thr]
        return below[0] if below else len(row)
    arc_first = np.mean([first_below(r["arc_overlap"]) for r in mine])
    state_first = np.mean([first_below(r["state_overlap"]) for r in mine])
    print(f"\n  mean step index where overlap first drops below 0.9: "
          f"arc {arc_first:.1f}, state {state_first:.1f}")
    print("  arc earlier -> arc sensitivity dominates; "
          "state earlier -> recovery is lossy")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "seq_a1_where_drift_starts_results.json")
    with open(out, "w") as fh:
        json.dump({"ours": mine, "reference": refs}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
