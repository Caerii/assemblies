"""Does the calculus's sequence operation advance when each element is written
as a transition rather than an attractor?

Registered in research/notes/sequence/PREREG_ordered_recall_reproduction.md,
Amendment 2.

`sequence_memorize` + `ordered_recall` advance zero steps after the cue in
this package (`test_ordered_recall_advances.py`, strict xfail). The memory
line's Amendments 25 and 26 found why an area stores attractors or
sequences: whether its activity holds still while the write runs. The
xfail's construction projects each element for eight rounds, so each
element becomes an attractor (within-element weights reach the clip) while
the bridge to the next one is written in the one round the two coincide.
Written with ONE stimulus-and-recurrence round per element
(`rounds_per_step=1, phase_b_ratio=1.0`), every round writes a transition and
none an attractor. This study measures the registration's own bars OR-1 to
OR-6 for that construction at the xfail's cell, with the write at the
convergence threshold theta, and two contrasts: the xfail's construction at
the same write (OR-7) and one round per element at the xfail's write (OR-8).

    python -m research.runner ordered_recall_repair --tag NAME [--smoke]
"""
from __future__ import annotations

import copy
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies import Brain, describe_brain_model                # noqa: E402
from neural_assemblies.assembly_calculus.assembly import overlap         # noqa: E402
from neural_assemblies.assembly_calculus.ops import (                    # noqa: E402
    ordered_recall, sequence_memorize,
)
from research.runner import (                                            # noqa: E402
    experiment_parser, run_experiment, validate_registered_seeds,
)

PROTOCOL = "sequence.ordered-recall-repair"
VERSION = "1"
REGISTRATION = "research/notes/sequence/PREREG_ordered_recall_reproduction.md"
#: the strict xfail's cell, sequence and recall
N, K, P, W_MAX, LENGTH = 4000, 50, 0.05, 20.0, 8
PERIOD, STRENGTH, MATCH = 3, 100.0, 0.3
#: the write at the convergence threshold theta = sqrt((1 - p) ln n / (p k))
THETA = math.sqrt((1 - P) * math.log(N) / (P * K))
BETA = round(THETA, 4)
#: the xfail's own write and construction
XFAIL_BETA, XFAIL_ROUNDS, XFAIL_RATIO, XFAIL_REPS = 0.10, 8, 0.5, 3
REGISTERED_SEEDS = tuple(range(500, 520))
#: arm -> (beta, rounds_per_step, phase_b_ratio, repetitions, materialize, inhibition)
ARMS = {
    "one_round": (BETA, 1, 1.0, 1, True, STRENGTH),
    "one_round_no_inhibition": (BETA, 1, 1.0, 1, True, 0.0),
    "one_round_sampled": (BETA, 1, 1.0, 1, False, STRENGTH),
    "xfail_construction": (BETA, XFAIL_ROUNDS, XFAIL_RATIO, XFAIL_REPS, True, STRENGTH),
    "one_round_xfail_beta": (XFAIL_BETA, 1, 1.0, 1, True, STRENGTH),
}


def _memorized(seed, beta, rounds, ratio, reps, materialize):
    brain = Brain(p=P, seed=seed, engine="numpy_sparse", w_max=W_MAX,
                  sampled_recurrence_policy="acknowledged")
    brain.add_area("SEQ", N, K, beta)
    if materialize:
        brain.materialize_area("SEQ")
    names = [f"s{i}" for i in range(LENGTH)]
    for name in names:
        brain.add_stimulus(name, K)
    stored = list(sequence_memorize(brain, names, "SEQ", rounds_per_step=rounds,
                                    repetitions=reps, phase_b_ratio=ratio))
    return brain, names, stored


def _recall(brain, names, stored, inhibition):
    brain.set_lri("SEQ", refractory_period=PERIOD, inhibition_strength=inhibition)
    return list(ordered_recall(brain, "SEQ", names[0], max_steps=LENGTH + 4,
                               known_assemblies=stored))


def score(stored, recalled):
    """Steps after the cue (consecutive matches strictly after index 0, the
    xfail's measure), whether every matched step is nearest its OWN index,
    and the cue's retrieval."""
    steps = 0
    for i in range(1, min(len(recalled), len(stored))):
        if overlap(recalled[i], stored[i]) >= MATCH:
            steps += 1
        else:
            break
    in_order = all(max(range(len(stored)), key=lambda j: overlap(recalled[i], stored[j])) == i
                   for i in range(1, steps + 1))
    return {"steps": steps, "in_order": in_order, "cue": overlap(recalled[0], stored[0])}


def run_seed(seed, arms):
    out = {}
    trained = {}
    for name in arms:
        beta, rounds, ratio, reps, materialize, inhibition = ARMS[name]
        key = (beta, rounds, ratio, reps, materialize)
        if key not in trained:
            trained[key] = _memorized(seed, beta, rounds, ratio, reps, materialize)
        brain, names, stored = trained[key]
        # OR-4's null reads the SAME trained brain: every recall gets a copy
        recalled = _recall(copy.deepcopy(brain), names, stored, inhibition)
        out[name] = score(stored, recalled)
    return out


def experiment(record):
    arms = record["parameters"]["arms"]
    rows = {}
    for seed in record["seeds"]:
        rows[str(seed)] = run_seed(seed, arms)
        print(f"seed {seed}: " + ", ".join(f"{a} {r['steps']}" for a, r in rows[str(seed)].items()),
              flush=True)
    return {"rows": rows, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


def evaluate(observations):
    """OR-1 to OR-8 (Amendment 2)."""
    rows = list(observations["rows"].values())
    n = len(rows)
    need = math.ceil(0.9 * n)              # "at least 18 of 20"
    col = lambda arm, key: [r[arm][key] for r in rows]
    one = col("one_round", "steps")
    bars = {
        "OR-1": sum(s >= 1 for s in one) >= need,
        "OR-2": statistics.median(one) >= 3,
        "OR-3": all(col("one_round", "in_order")),
        "OR-4": all(s == 0 for s in col("one_round_no_inhibition", "steps")),
        "OR-5": sum(abs(a - b) <= 1 for a, b in zip(one, col("one_round_sampled", "steps"))) >= need,
        "OR-6": sum(c >= 0.7 for c in col("one_round", "cue")) >= need,
        "OR-7": sum(s == 0 for s in col("xfail_construction", "steps")) >= need,
        "OR-8": sum(s == 0 for s in col("one_round_xfail_beta", "steps")) >= need,
    }
    summary = {arm: {"steps": col(arm, "steps"), "median": statistics.median(col(arm, "steps")),
                     "cue_median": statistics.median(col(arm, "cue"))}
               for arm in ARMS if all(arm in r for r in rows)}
    return {"bars": bars, "summary": summary}


def main(argv=None):
    parser = experiment_parser(
        "ordered_recall with one stimulus-and-recurrence round per element",
        engines=("numpy_sparse",), default_seeds=REGISTERED_SEEDS,
    )
    args = parser.parse_args(argv)
    if not args.smoke:
        validate_registered_seeds(parser, args, REGISTERED_SEEDS)
    parameters = {"n": N, "k": K, "p": P, "w_max": W_MAX, "length": LENGTH,
                  "refractory_period": PERIOD, "inhibition_strength": STRENGTH,
                  "match": MATCH, "theta": THETA, "beta": BETA,
                  "arms": {name: dict(zip(("beta", "rounds_per_step", "phase_b_ratio",
                                           "repetitions", "materialize", "inhibition"), spec))
                           for name, spec in ARMS.items()}}
    path = run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATION, engine=args.engine, seeds=args.seeds,
        tag=args.tag, smoke=args.smoke, parameters=parameters,
        # ordered_recall projects with plasticity on, as the calculus defines it
        observation_policy="plastic",
        model_semantics=describe_brain_model("numpy_sparse", p=P, w_max=W_MAX),
        measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
