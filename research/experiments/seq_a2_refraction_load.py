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

import random
from pathlib import Path
from typing import Any

import numpy as np

from neural_assemblies import Brain, describe_brain_model
from neural_assemblies.programs.nemo_fsm import NemoArcFSM

from research.experiments.seq_a2_word_order_fsm import (
    AMBIENT_P, BETA, K, N_STATE, ORDERS, ORGAN_P, PRESENTATIONS, SEEDS, STATES,
    order_correct, transitions_for,
)
from research.runner import (
    experiment_parser, run_experiment, validate_registered_seeds,
    validate_seed_identities,
)

N_ARCS = (5000, 2000, 1000, 500, 350)


def build(seed, moods, n_arc, *, materialized=False, presentations=PRESENTATIONS):
    random.seed(seed)
    legacy_random: Any = np.random
    legacy_random.seed(seed)
    brain = Brain(engine="numpy_sparse", p=AMBIENT_P, save_winners=True,
                  seed=seed, norm_init=False)
    fsm = NemoArcFSM(brain, states=list(STATES), symbols=list(moods),
                     transitions=transitions_for(moods), n=n_arc, k=K,
                     n_state=N_STATE, beta=BETA, organ_p=ORGAN_P,
                     prefix="_a2load")
    if materialized:                                # PREREG_sampler_audit.md
        brain.materialize_area(fsm.arc_area)
    fsm.train_from_list([(m, q, r) for q, m, r in transitions_for(moods)],
                        presentations=presentations)
    return brain, fsm


def experiment(record):
    if record.get("mode", "study") == "study":
        validate_seed_identities(record["seeds"], SEEDS)
    parameters = record["parameters"]
    seeds = record["seeds"]
    n_arcs = parameters["n_arcs"]
    materialized = parameters["materialized"]
    presentations = parameters["presentations"]
    rows = []
    for label, moods in (("single-mood (3 conjunctions)", ["svo"]),
                         ("multi-mood (9 conjunctions)", list(ORDERS))):
        m_count = len(transitions_for(moods))
        for n_arc in n_arcs:
            ok = sum(order_correct(build(
                            s, moods, n_arc, materialized=materialized,
                            presentations=presentations)[1], moods[0])
                     if len(moods) == 1
                     else all(order_correct(build(
                            s, moods, n_arc, materialized=materialized,
                            presentations=presentations)[1], mm)
                              for mm in moods)
                     for s in seeds)
            load = m_count * K / n_arc
            rows.append({"arm": label, "n_arc": n_arc, "load": load,
                         "correct": ok, "n": len(seeds)})
            rows[-1]["materialized"] = materialized
    return {"verdict": "VOID" if record["mode"] == "smoke" else "UNADOPTED",
            "rows": rows, "scope": "refraction convergence versus arc load"}


def main(argv=None):
    parser = experiment_parser(
        __doc__ or "A2 refraction load sweep", engines=("numpy_sparse",),
        default_seeds=tuple(SEEDS),
    )
    parser.add_argument("--materialized", action="store_true")
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, tuple(SEEDS))
    parameters = {"n_arcs": list(N_ARCS[:2] if args.smoke else N_ARCS),
                  "ambient_p": AMBIENT_P, "organ_p": ORGAN_P, "k": K,
                  "n_state": N_STATE, "beta": BETA,
                  "presentations": 2 if args.smoke else PRESENTATIONS,
                  "materialized": args.materialized, "norm_init": False}
    path = run_experiment(
        script=Path(__file__), protocol="sequence.a2-refraction-load",
        protocol_version="2", registration="research/notes/sequence/PREREG_sampler_audit.md",
        engine=args.engine, seeds=args.seeds, tag=args.tag, smoke=args.smoke,
        parameters=parameters,
        model_semantics=describe_brain_model("numpy_sparse", p=AMBIENT_P,
                                             norm_init=False),
        measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
