"""Is the lexicon's plasticity regime set by n_LEX p?

Registered in research/notes/aligner/PREREG_word_capacity.md, Amendment 6.

Amendment 5's record, read after its bars, split the lexicon into two
regimes: below n_LEX p ~ 100 (the number of lexical neurons each feature
neuron hears from) the best plasticity sat at or above the grid's top (0.4
to 0.8) and more connections did not buy vocabulary; from n_LEX p ~ 100 the
best plasticity was a weak interior optimum (0.025 to 0.1) and vocabulary
grew with p. This tests that on new brains, with two new lexicon sizes so
that equal n_LEX p is reached by different (n, p):

    cells A (1000 x 50), F (1400 x 50), B (2000 x 50), G (2800 x 50),
    C (4000 x 50), each at p = 0.025, 0.05, 0.1 (n_LEX p = 25 to 400),

with FEAT = 4000 x 100 and the plasticity grid 0.0125 to 0.8 (factor 2)
of Amendment 5, one run per p, seeds 222 to 241.

    python -m research.runner word-capacity-regime --p 0.05 --tag NAME [--smoke]
"""
from dataclasses import replace
from pathlib import Path

from neural_assemblies import ExecutionKind, ExecutionSemantics, describe_hashed_aligner
from research.experiments import word_capacity as capacity
from research.experiments import word_capacity_synapses_run as synapses
from research.experiments.word_capacity_protocol import (
    REGISTERED_PROTOCOL, CapacityCell, WordCapacityProtocol,
)
from research.runner import experiment_parser, run_experiment, validate_registered_seeds

PROTOCOL = "aligner.word-capacity-regime"
VERSION = "1"
REGISTRATION = "research/notes/aligner/PREREG_word_capacity.md"
SEEDS = tuple(range(222, 242))
CELLS = ("A", "F", "B", "G", "C")
NEW_CELLS = (CapacityCell("F", 1400, 50, 50), CapacityCell("G", 2800, 50, 50))
PROBABILITIES = (0.025, 0.05, 0.1)
BETAS = synapses.OPT_BETAS
#: the bars' thresholds
STRONG_BELOW, WEAK_ABOVE = 50, 200
STRONG_BETA, WEAK_BETA = 0.4, 0.1
#: Amendment 5's seed-tight instrument cells at p = 0.05, beta = 0.1
A5 = {"B": 154.24, "C": 333.91}


def protocol_for(p, *, smoke=False):
    base = replace(REGISTERED_PROTOCOL,
                   cell_definitions=REGISTERED_PROTOCOL.cell_definitions + NEW_CELLS)
    selected = base.select(
        cells=("A",) if smoke else CELLS,
        vocabulary_sizes=(8, 16) if smoke else None,
        feature_area=(1000, 50) if smoke else (4000, 100),
    )
    return replace(selected, connection_probability=p)


def measure(record):
    if record.get("protocol") != PROTOCOL or record.get("protocol_version") != VERSION:
        raise ValueError("word-capacity-regime protocol identity mismatch")
    execution = ExecutionSemantics.normalize(record.get("execution_semantics"))
    if execution.kind is not ExecutionKind.ALIGNMENT:
        raise ValueError("word capacity requires alignment execution semantics")
    if record.get("engine") != "scheduled_aligner":
        raise ValueError("this protocol requires scheduled_aligner")
    profile = execution.profiles["default"].to_dict()
    protocol = WordCapacityProtocol.from_parameters(record["parameters"]["protocol"])
    betas = record["parameters"]["betas"]
    selected_for = {beta: replace(protocol, plasticity=beta) for beta in betas}
    all_curves = capacity.run_cells_concurrent({
        (beta, name): (name, record["seeds"], selected_for[beta], profile)
        for beta in betas for name in protocol.cells})
    sweep = {}
    for beta in betas:
        selected = selected_for[beta]
        curves = {name: all_curves[(beta, name)] for name in selected.cells}
        report = capacity.capacity_report(curves, record["seeds"], protocol=selected)
        sweep[f"{beta:g}"] = {
            "beta": beta,
            "curves": {name: {str(v): values for v, values in curve.items()}
                       for name, curve in curves.items()},
            "cells": report["cells"],
        }
        print(f"p={protocol.connection_probability:g} beta={beta:g}: " + ", ".join(
            f"{name} {report['cells'][name]['ceiling']['mean']:.1f}"
            f" ({report['cells'][name]['censored_seeds']} censored)"
            for name in selected.cells), flush=True)
    return {"verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED",
            "p": protocol.connection_probability, "sweep": sweep}


# ----------------------------------------------------------------- evaluation
def n_of(name):
    return {c.name: c.n for c in REGISTERED_PROTOCOL.cell_definitions + NEW_CELLS}[name]


def evaluate(observations_by_p):
    """Amendment 6's bars from the three runs, keyed by p."""
    rows = {}
    for p, obs in observations_by_p.items():
        for name in CELLS:
            values = [obs["sweep"][f"{b:g}"]["cells"][name]["ceiling"]["mean"] for b in BETAS]
            i = max(range(len(values)), key=lambda j: values[j])
            rows[(name, p)] = {"np": n_of(name) * p, "best": values[i],
                               "best_beta": BETAS[i], "curve": dict(zip(BETAS, values))}
    out = {"cells": {f"{n} p={p:g}": r for (n, p), r in rows.items()}, "bars": {}}
    inst = observations_by_p.get(0.05)
    out["bars"]["EV"] = inst is not None and all(
        abs(inst["sweep"]["0.1"]["cells"][c]["ceiling"]["mean"] / v - 1) <= 0.15
        for c, v in A5.items())
    strong = [r for r in rows.values() if r["np"] <= STRONG_BELOW]
    weak = [r for r in rows.values() if r["np"] >= WEAK_ABOVE]
    out["bars"]["E1"] = (bool(strong) and bool(weak)
                         and all(r["best_beta"] >= STRONG_BETA for r in strong)
                         and all(r["best_beta"] <= WEAK_BETA for r in weak))
    by_np = {}
    for r in rows.values():
        by_np.setdefault(round(r["np"], 6), []).append(r["best_beta"])
    groups = {k: v for k, v in by_np.items() if len(v) > 1}
    out["equal_np"] = {f"{k:g}": v for k, v in groups.items()}
    out["bars"]["E2"] = bool(groups) and all(max(v) / min(v) <= 2.0 + 1e-9
                                             for v in groups.values())
    return out


def main(argv=None):
    parser = experiment_parser(
        "Word capacity: the plasticity regime against n_LEX p",
        engines=("scheduled_aligner",), default_seeds=SEEDS,
    )
    parser.add_argument("--p", type=float, required=True)
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, SEEDS)
    if not args.smoke and args.p not in PROBABILITIES:
        parser.error(f"--p must be one of {PROBABILITIES}")
    protocol = protocol_for(args.p, smoke=args.smoke)
    betas = [0.1] if args.smoke else list(BETAS)
    semantics = describe_hashed_aligner(
        p=protocol.connection_probability, beta=protocol.plasticity,
        rounds_word=protocol.rounds_per_pair, norm_init=True, scaling=True,
        w_max=None, stim_beta=0.0, stim_gain=None, store="present",
    )
    return run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATION, engine=args.engine, seeds=args.seeds,
        tag=args.tag, smoke=args.smoke,
        parameters={"protocol": protocol.to_parameters(), "betas": betas},
        measure=measure, aligner_semantics=semantics,
    )


if __name__ == "__main__":
    main()
