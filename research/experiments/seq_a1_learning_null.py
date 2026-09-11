"""Registered paired instrument control using the existing horizon execution path."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from neural_assemblies.diagnostics import ensemble_from_values, paired_delta
from neural_assemblies import describe_hashed_arc_fsm
from research.experiments.seq_a1_horizon_hashed import HorizonProtocol, run_width
from research.runner import ROOT, experiment_parser, run_experiment

REFERENCE = "research/results/runs/sequence.a1-horizon/horizon-record-consumed-20260910/run.json"
BARS = {"trained_accuracy_low": .99, "null_accuracy_high": .9,
        "accuracy_delta_low": .1, "exact_delta_low": .1}


def score_pair(rows, seeds, bars):
    """Specification: research/notes/sequence/PREREG_a1_learning_null.md"""
    summaries = {}
    for metric in ("accuracy", "exact_fraction"):
        ensembles = {}
        for arm in ("trained", "null"):
            if [row["seed"] for row in rows[arm]] != seeds:
                raise ValueError("arm observations must preserve unique seed order")
            if any(type(row[metric]) not in (int, float) or not 0 <= row[metric] <= 1
                   for row in rows[arm]):
                raise ValueError("accuracy and exact fraction must be probabilities")
            ensembles[arm] = ensemble_from_values(
                [row[metric] for row in rows[arm]], label=arm, keys=seeds)
        ensembles["delta"] = paired_delta(ensembles["trained"], ensembles["null"])
        summaries[metric] = {name: {**asdict(value), "low": value.low, "high": value.high}
                             for name, value in ensembles.items()}
    accuracy, exact = summaries["accuracy"], summaries["exact_fraction"]
    checks = {
        "trained_accuracy_low": accuracy["trained"]["low"] > bars["trained_accuracy_low"],
        "null_accuracy_high": accuracy["null"]["high"] < bars["null_accuracy_high"],
        "accuracy_delta_low": accuracy["delta"]["low"] > bars["accuracy_delta_low"],
        "exact_delta_low": exact["delta"]["low"] > bars["exact_delta_low"],
    }
    return {"ensembles": summaries, "checks": checks, "passed": all(checks.values())}


def experiment(record):
    parameters, seeds = record["parameters"], record["seeds"]
    protocols = {arm: HorizonProtocol.from_parameters(values)
                 for arm, values in parameters["protocols"].items()}
    if set(protocols) != {"trained", "null"}:
        raise ValueError("exactly the trained and null arms are required")
    if ([cell["p"] for cell in parameters["schedule"]] != list(protocols["trained"].p_values)
            or any(sorted(cell["order"]) != ["null", "trained"] for cell in parameters["schedule"])):
        raise ValueError("schedule must cover each probability and both arms exactly once")
    if protocols["null"] != replace(protocols["trained"], beta=0., strength=0.):
        raise ValueError("null must differ only by disabling beta and strength")
    cells = []
    for cell in parameters["schedule"]:
        p = cell["p"]
        rows = {
            arm: run_width(
                seeds, p, protocols[arm],
                record["execution_semantics"]["profiles"][arm],
            )
            for arm in cell["order"]
        }
        scored = score_pair(rows, seeds, parameters["bars"])
        cells.append({"p": p, "order": cell["order"], "rows": rows, **scored})
        print(f"p={p}: {scored['checks']}", flush=True)
    passed = bool(cells) and all(cell["passed"] for cell in cells)
    return {"cells": cells, "verdict": "VOID" if record["mode"] == "smoke" else
            ("PASS" if passed else "FAIL"), "scope": "registered instrument sensitivity only"}


def main(argv=None):
    parser = experiment_parser(__doc__, engines=("hashed_arc_fsm",),
                               default_seeds=tuple(range(1, 21)))
    args = parser.parse_args(argv)
    if not args.smoke and args.seeds != list(range(1, 21)):
        parser.error("this registration requires seed identities 1..20 in order")
    reference = json.loads((ROOT / REFERENCE).read_text(encoding="utf-8"))
    trained = HorizonProtocol.from_parameters(reference["parameters"])
    if args.smoke:
        trained = replace(trained, length=50)
    null = replace(trained, beta=0., strength=0.)
    path = run_experiment(
        script=Path(__file__), protocol="sequence.a1-learning-null", protocol_version="1",
        registration="research/notes/sequence/PREREG_a1_learning_null.md",
        engine=args.engine, seeds=args.seeds, tag=args.tag, smoke=args.smoke,
        minimum_study_seeds=20, input_artifacts=(REFERENCE,),
        organ_semantics={
            arm: describe_hashed_arc_fsm(
                w_max=protocol.w_max, norm_init=protocol.norm_init,
                refracted_strength=protocol.strength,
            )
            for arm, protocol in {"trained": trained, "null": null}.items()
        },
        parameters={"protocols": {"trained": asdict(trained), "null": asdict(null)},
                    "bars": BARS, "schedule": [
                        {"p": p, "order": ["trained", "null"] if i % 2 == 0 else ["null", "trained"]}
                        for i, p in enumerate(trained.p_values)]}, measure=experiment)
    print(path)


if __name__ == "__main__":
    main()
