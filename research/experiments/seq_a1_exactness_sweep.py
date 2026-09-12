"""A1 follow-up 5: what makes arc->state recovery EXACT?

Established: the state area is a discrete attractor. Both arcs amplify a state
error about 8-10x per step, so recovery that returns 69 of 70 neurons is not a
near-miss -- 1/70 = 0.014 reaches order 1 within five steps. The reference
succeeds by recovering EXACTLY (seed 7 holds overlap 1.00 at all six steps),
not by tolerating error.

So the target metric is not mean overlap. It is the fraction of steps where
recovery is exact, and the lever is the margin at the k-th winner.

`p` is the direct lever: it sets how many afferents an active assembly delivers,
so it sets the gap between the k-th and (k+1)-th candidate. At p=0.2 the state
area receives k*p = 14 synapses from the arc's assembly, against a floor of
3 ln 500 = 18.6 -- below it, though the reference works there too, which is why
this is a sweep and not a fix.

PREDICTION, STATED BEFORE THE NUMBERS. Exactness rises with p, and trajectory
correctness follows it closely -- much more closely than it follows MEAN
overlap, because the mechanism is all-or-nothing. Crossing the floor
(p >= 18.6/70 = 0.266) should be where it turns, if the floor is meaningful
here. If exactness stays low at every p, the margin is not the binding
constraint and the residual is substrate noise, which is the pre-registered
numpy_exact experiment.

Reports exactness, trajectory correctness and mean overlap side by side, so a
metric that moves without the others is visible rather than averaged in.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.assembly_calculus.ops import _snap
from neural_assemblies import Brain, describe_brain_model
from neural_assemblies.programs.mod3_fsm import (
    END_SYMBOL, build_mod3_fsm, train_mod3_fsm,
)

from research.experiments.seq_a1_fsm_parity import (
    BETA, K, N_ARC, N_STATE, NEGATIVE, POSITIVE, PRESENTATIONS, SEEDS,
)
from research.experiments.seq_a1_drift import true_trajectory
from research.runner import experiment_parser, run_experiment

P_VALUES = (0.2, 0.3, 0.4, 0.5)


def trial(seed, p, *, materialized=False, presentations=PRESENTATIONS):
    random.seed(seed)
    legacy_random: Any = np.random
    legacy_random.seed(seed)
    brain = Brain(p=p, save_winners=True, seed=seed, engine="numpy_sparse",
                  norm_init=False)
    fsm = build_mod3_fsm(brain, n=N_ARC, k=K, n_state=N_STATE, beta=BETA)
    if materialized:                                # PREREG_sampler_audit.md
        brain.materialize_area(fsm.arc_area)
    train_mod3_fsm(fsm, presentations=presentations)

    out = {}
    for name, digits in (("pos", list(POSITIVE)), ("neg", list(NEGATIVE))):
        symbols = [END_SYMBOL if d == 10 else str(d) for d in digits]
        truth = true_trajectory(digits)
        observed, overlaps = [], []
        with brain.probe():
            brain.inhibit_areas([fsm.arc_area, fsm.state_area])
            fsm._cue_state("0")
            fsm._unfix_state()
            for symbol, want in zip(symbols, truth):
                observed.append(fsm.step(symbol))
                overlaps.append(float(overlap(_snap(brain, fsm.state_area),
                                              fsm.state_assembly(want))))
        out[name] = {"observed": observed, "truth": truth, "overlap": overlaps}

    steps = out["pos"]["overlap"] + out["neg"]["overlap"]
    return {
        "seed": seed, "p": p,
        "trajectory_correct": (out["pos"]["observed"] == out["pos"]["truth"]
                               and out["neg"]["observed"] == out["neg"]["truth"]),
        "decided": (out["pos"]["observed"][-1] == "accept"
                    and out["neg"]["observed"][-1] == "reject"),
        "exact_steps": sum(1 for v in steps if v >= 1.0),
        "total_steps": len(steps),
        "mean_overlap": float(np.mean(steps)),
        "first_step_exact": bool(out["pos"]["overlap"][0] >= 1.0),
    }


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    presentations = parameters["presentations"]
    materialized = parameters["materialized"]
    rows, summary = [], []
    for p in P_VALUES:
        got = [trial(s, p, materialized=materialized,
                     presentations=presentations) for s in seeds]
        rows.extend(got)
        exact = sum(r["exact_steps"] for r in got)
        total = sum(r["total_steps"] for r in got)
        s = {"p": p, "exact_frac": exact / total,
             "trajectory_correct": sum(r["trajectory_correct"] for r in got),
             "decided": sum(r["decided"] for r in got),
             "mean_overlap": float(np.mean([r["mean_overlap"] for r in got])),
             "n": len(got)}
        summary.append(s)
    return {"verdict": "VOID" if record["mode"] == "smoke" else "UNADOPTED",
            "summary": summary, "rows": rows, "materialized": materialized,
            "scope": "A1 exact-step recovery versus afferent probability"}


def main(argv=None):
    parser = experiment_parser(
        __doc__ or "A1 exact recovery sweep", engines=("numpy_sparse",),
        default_seeds=tuple(SEEDS),
    )
    parser.add_argument("--materialized", action="store_true",
                        help="draw the full arc connectome before training")
    args = parser.parse_args(argv)
    parameters = {"p_values": list(P_VALUES), "n_arc": N_ARC,
                  "n_state": N_STATE, "k": K, "beta": BETA,
                  "presentations": 2 if args.smoke else PRESENTATIONS,
                  "materialized": args.materialized, "norm_init": False}
    path = run_experiment(
        script=Path(__file__), protocol="sequence.a1-exactness-sweep",
        protocol_version="2", registration="research/notes/sequence/PREREG_sampler_audit.md",
        engine=args.engine, seeds=args.seeds, tag=args.tag, smoke=args.smoke,
        parameters=parameters,
        model_semantics=describe_brain_model("numpy_sparse", p=P_VALUES[0], norm_init=False),
        measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
