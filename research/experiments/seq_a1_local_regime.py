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

import random
from pathlib import Path
from typing import Any

import numpy as np

from neural_assemblies import Brain, describe_brain_model
from neural_assemblies.diagnostics import format_report, regime_audit
from neural_assemblies.programs.mod3_fsm import (
    END_SYMBOL, build_mod3_fsm, train_mod3_fsm,
)

from research.experiments.seq_a1_fsm_parity import (
    BETA, K, N_ARC, N_STATE, NEGATIVE, POSITIVE, PRESENTATIONS, SEEDS,
)
from research.experiments.seq_a1_drift import true_trajectory
from research.runner import (
    experiment_parser, run_experiment, validate_registered_seeds,
    validate_seed_identities,
)

AMBIENT_P = 0.05
ORGAN_P = 0.40


def build(seed, *, organ_p, ambient_p=AMBIENT_P, with_neighbours=True,
          materialized=False, presentations=PRESENTATIONS):
    random.seed(seed)
    legacy_random: Any = np.random
    legacy_random.seed(seed)
    brain = Brain(engine="numpy_sparse", p=ambient_p, save_winners=True,
                  seed=seed, norm_init=False)
    fsm = build_mod3_fsm(brain, n=N_ARC, k=K, n_state=N_STATE, beta=BETA,
                         organ_p=organ_p)
    if materialized:                                # PREREG_sampler_audit.md
        brain.materialize_area(fsm.arc_area)
    if with_neighbours:
        # Unrelated traffic at the AMBIENT density, so the organ is embedded in
        # a working brain rather than alone in one that merely has a low `p`.
        for name in ("NEIGHBOUR_A", "NEIGHBOUR_B"):
            brain.add_area(name, 3000, K, BETA)
        brain.add_stimulus("neighbour_stim", K)
        for _ in range(5):
            brain.project({"neighbour_stim": ["NEIGHBOUR_A"]}, {})
            brain.project({}, {"NEIGHBOUR_A": ["NEIGHBOUR_B"]})
    train_mod3_fsm(fsm, presentations=presentations)
    return brain, fsm


def trajectory_correct(fsm, digits):
    symbols = [END_SYMBOL if d == 10 else str(d) for d in digits]
    return fsm.run(symbols, start_state="0") == true_trajectory(digits)


def arm(name, *, organ_p, seeds, materialized=False, presentations=PRESENTATIONS):
    rows = []
    for seed in seeds:
        brain, fsm = build(seed, organ_p=organ_p, materialized=materialized,
                           presentations=presentations)
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


def experiment(record):
    if record.get("mode", "study") == "study":
        validate_seed_identities(record["seeds"], SEEDS)
    parameters = record["parameters"]
    seeds = record["seeds"]
    materialized = parameters["materialized"]
    presentations = parameters["presentations"]
    arms = []
    arms.append(arm("local_regime", organ_p=ORGAN_P, seeds=seeds,
                    materialized=materialized, presentations=presentations))
    arms.append(arm("ambient_only", organ_p=None, seeds=seeds,
                    materialized=materialized, presentations=presentations))
    verdict = (arms[0]["correct"] >= 0.8 * arms[0]["n"]
               and arms[1]["correct"] <= 0.2 * arms[1]["n"])
    return {"verdict": "VOID" if record["mode"] == "smoke" else (
                "PASS" if verdict else "FAIL"),
            "ambient_p": AMBIENT_P, "organ_p": ORGAN_P,
            "materialized": materialized, "arms": arms,
            "scope": "local organ regime embedded in ambient sparse traffic"}


def main(argv=None):
    parser = experiment_parser(
        __doc__ or "A1 local organ regime study", engines=("numpy_sparse",),
        default_seeds=tuple(SEEDS),
    )
    parser.add_argument("--materialized", action="store_true",
                        help="draw the full arc connectome before training")
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, tuple(SEEDS))
    parameters = {"ambient_p": AMBIENT_P, "organ_p": ORGAN_P, "k": K,
                  "n_arc": N_ARC, "n_state": N_STATE, "beta": BETA,
                  "presentations": 2 if args.smoke else PRESENTATIONS,
                  "materialized": args.materialized, "norm_init": False}
    path = run_experiment(
        script=Path(__file__), protocol="sequence.a1-local-regime",
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
