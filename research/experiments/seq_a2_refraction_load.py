"""A2 follow-up: refraction needs LOAD to converge.

A2's single-mood arm failed P-FLOOR at 2/10 while its arc was a clean
conjunction (across-state overlap 0.021) -- the same floor #92's arc arms
failed. Two hypotheses were ruled out before this one:

  * refraction itself -- ablating it makes the arm WORSE (0/10, arc collapsed
    to 0.988), so refraction is not the problem;
  * conjunct exposure -- lowering the over-exposed mood fiber's beta, which is
    the lever #92 swept, rescues nothing (0-2/10 at every value including 0).

What is left is convergence. Measured directly: the arc assembly for one
(state, symbol) pair NEVER SETTLES in the single-mood arm -- overlap 0.514
between presentations 5 and 10, 0.600 between 10 and 15, and 0.286 between 5
and 15 -- while the multi-mood arm converges (0.957 between 10 and 15). A write
smeared over assemblies that keep moving cannot dominate at test.

THE MECHANISM. Refraction separates by pushing winners off neurons that have
already fired. That needs somewhere to push FROM and something to push
AGAINST: with 3 conjunctions in a 5000-neuron area there is always fresh unused
space, so the assembly wanders into it instead of tiling. With enough
conjunctions competing, the assemblies fill the space and lock.

This sweeps arc SIZE at fixed content, so load M*k/n is the only thing moving.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

from neural_assemblies.core.brain import Brain
from neural_assemblies.programs.nemo_fsm import NemoArcFSM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_a2_word_order_fsm import (
    AMBIENT_P, BETA, K, N_STATE, ORDERS, ORGAN_P, PRESENTATIONS, SEEDS, STATES,
    order_correct, transitions_for,
)

N_ARCS = (5000, 2000, 1000, 500, 350)


def build(seed, moods, n_arc):
    random.seed(seed)
    np.random.seed(seed)
    brain = Brain(engine="numpy_sparse", p=AMBIENT_P, save_winners=True,
                  seed=seed, norm_init=False)
    fsm = NemoArcFSM(brain, states=list(STATES), symbols=list(moods),
                     transitions=transitions_for(moods), n=n_arc, k=K,
                     n_state=N_STATE, beta=BETA, organ_p=ORGAN_P,
                     prefix="_a2load")
    fsm.train_from_list([(m, q, r) for q, m, r in transitions_for(moods)],
                        presentations=PRESENTATIONS)
    return brain, fsm


def main():
    seeds = SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else SEEDS
    rows = []
    print("=== does refraction need LOAD to converge? ===")
    print(f"    sweeping arc size at fixed content, so load M*k/n is the only "
          f"thing moving\n")
    for label, moods in (("single-mood (3 conjunctions)", ["svo"]),
                         ("multi-mood (9 conjunctions)", list(ORDERS))):
        m_count = len(transitions_for(moods))
        print(f"  {label}")
        for n_arc in N_ARCS:
            ok = sum(order_correct(build(s, moods, n_arc)[1], moods[0])
                     if len(moods) == 1
                     else all(order_correct(build(s, moods, n_arc)[1], mm)
                              for mm in moods)
                     for s in seeds)
            load = m_count * K / n_arc
            rows.append({"arm": label, "n_arc": n_arc, "load": load,
                         "correct": ok, "n": len(seeds)})
            print(f"    n_arc {n_arc:5d}  load {load:5.2f}  "
                  f"{ok}/{len(seeds)} correct", flush=True)
        print()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "seq_a2_refraction_load_results.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
