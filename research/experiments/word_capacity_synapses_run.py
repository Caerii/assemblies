"""Is vocabulary capacity synapse-limited, and does the lexicon's best
learning rate scale with fan-in?

Registered in research/notes/aligner/PREREG_word_capacity.md, Amendment 4.

The registered word-capacity learner (Amendment 3, Part 2: FEAT = 4000 x 100,
twenty seeds) holds about n / 12 word types per lexical area: V* is
proportional to n_LEX and does not move with k_LEX. A hetero-association
signal-to-noise argument gives exactly that for a SYNAPSE-limited fiber:
V* proportional to n_LEX p (the LEX -> FEAT synapses a word's assembly can
use), independent of k_LEX. This varies what that argument says matters and
the registered study held fixed:

    the connection probability p (0.025, 0.05, 0.1), one run per value --
        every fiber's p, the stimulus anchor gain following as 1/p, as the
        registered learner does;
    the plasticity beta (0.025, 0.05, 0.1, 0.2, 0.4), swept inside a run --
        the LEX -> FEAT fiber is the only one that learns.

Cells A (1000 x 50), B (2000 x 50), C (4000 x 50); everything else is the
registered protocol 3.3. At p = 0.05, beta = 0.1 the run replays Amendment 3
Part 2 exactly (the instrument).

    python -m research.runner word-capacity-synapses --p 0.05 --tag NAME [--smoke]
"""
from dataclasses import replace
import math
from pathlib import Path

from neural_assemblies import ExecutionKind, ExecutionSemantics, describe_hashed_aligner
from research.experiments import word_capacity as capacity
from research.experiments.word_capacity_protocol import (
    REGISTERED_PROTOCOL, WordCapacityProtocol,
)
from research.runner import experiment_parser, run_experiment, validate_registered_seeds

PROTOCOL = "aligner.word-capacity-synapses"
VERSION = "1"
REGISTRATION = "research/notes/aligner/PREREG_word_capacity.md"
STUDY_SEEDS = (42, *range(1, 20))
CELLS = ("A", "B", "C")
PROBABILITIES = (0.025, 0.05, 0.1)
BETAS = (0.025, 0.05, 0.1, 0.2, 0.4)
#: Amendment 3 Part 2's ceilings at p = 0.05, beta = 0.1 (L0)
PART2 = {"A": 73.826, "B": 166.4, "C": 324.4}


def base_protocol(p, *, smoke=False):
    selected = REGISTERED_PROTOCOL.select(
        cells=("A",) if smoke else CELLS,
        vocabulary_sizes=(8, 16) if smoke else None,
        feature_area=(1000, 50) if smoke else (4000, 100),
    )
    return replace(selected, connection_probability=p)


def _validate(raw):
    protocol = WordCapacityProtocol.from_parameters(raw["protocol"])
    registered = REGISTERED_PROTOCOL.to_parameters()
    variable = {"cells", "vocabulary_sizes", "feature_area", "connection_probability"}
    if any(raw["protocol"][key] != value for key, value in registered.items()
           if key not in variable):
        raise ValueError("this protocol varies only p, beta and the cells")
    if protocol.connection_probability not in PROBABILITIES and raw.get("mode") != "smoke":
        raise ValueError("p must be one of the registered probabilities")
    betas = raw["betas"]
    if not betas or any(type(b) is not float or b <= 0 for b in betas):
        raise ValueError("betas must be positive floats")
    return protocol, betas


def measure(record):
    if record.get("protocol") != PROTOCOL or record.get("protocol_version") != VERSION:
        raise ValueError("word-capacity-synapses protocol identity mismatch")
    execution = ExecutionSemantics.normalize(record.get("execution_semantics"))
    if execution.kind is not ExecutionKind.ALIGNMENT:
        raise ValueError("word capacity requires alignment execution semantics")
    if record.get("engine") != "scheduled_aligner":
        raise ValueError("this protocol requires scheduled_aligner")
    profile = execution.profiles["default"].to_dict()
    protocol, betas = _validate({**record["parameters"], "mode": record["mode"]})
    sweep = {}
    for beta in betas:
        selected = replace(protocol, plasticity=beta)
        curves = {name: capacity.run_cell(name, record["seeds"], selected.vocabulary_sizes,
                                          engine="scheduled", protocol=selected,
                                          aligner_semantics=profile)
                  for name in selected.cells}
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
def optimum(betas, values):
    """beta* by a parabola in log beta through the best grid point and its
    neighbours; None at a grid end."""
    i = max(range(len(values)), key=lambda j: values[j])
    if i == 0 or i == len(values) - 1:
        return None
    x = [math.log(b) for b in betas[i - 1:i + 2]]
    y = values[i - 1:i + 2]
    denom = y[0] - 2 * y[1] + y[2]
    if denom >= 0:
        return betas[i]
    h = x[1] - x[0]
    return math.exp(x[1] + h * (y[0] - y[2]) / (2 * denom))


def evaluate(observations_by_p):
    """Amendment 4's bars from the three runs, keyed by p."""
    out = {"bars": {}, "table": {}}

    def vstar(p, beta, cell):
        c = observations_by_p[p]["sweep"][f"{beta:g}"]["cells"][cell]
        return c["ceiling"]["mean"], c["censored_seeds"]

    ps = sorted(observations_by_p)
    for p in ps:
        for beta in BETAS:
            for cell in CELLS:
                out["table"][f"p={p:g} beta={beta:g} {cell}"] = vstar(p, beta, cell)
    # L0: the instrument replays Amendment 3 Part 2
    out["bars"]["L0"] = all(abs(vstar(0.05, 0.1, c)[0] / PART2[c] - 1) <= 0.01 for c in CELLS)
    # FEAT-bound flag: within 15% of C at the same (p, beta)

    def bound(p, beta, cell):
        return vstar(p, beta, cell)[0] >= 0.85 * vstar(p, beta, "C")[0]

    def usable(p, beta, cell):
        v, censored = vstar(p, beta, cell)
        return censored == 0 and not bound(p, beta, cell)
    l1 = True
    for cell in ("A", "B"):
        for lo, hi in zip(ps, ps[1:]):
            if not (usable(lo, 0.1, cell) and usable(hi, 0.1, cell)):
                out.setdefault("excluded", []).append(f"L1 {cell} {lo}->{hi}")
                continue
            ratio = vstar(hi, 0.1, cell)[0] / vstar(lo, 0.1, cell)[0]
            out["table"][f"L1 {cell} {lo:g}->{hi:g}"] = ratio
            l1 &= 1.6 <= ratio <= 2.5
    out["bars"]["L1"] = l1
    l2 = True
    for p in ps:
        if not (usable(p, 0.1, "A") and usable(p, 0.1, "B")):
            continue
        a = vstar(p, 0.1, "A")[0] / (1000 * p)
        b = vstar(p, 0.1, "B")[0] / (2000 * p)
        out["table"][f"L2 p={p:g} A, B per n p"] = (a, b)
        l2 &= abs(a / b - 1) <= 0.25
    out["bars"]["L2"] = l2
    stars = {}
    l3 = True
    for cell in ("A", "B"):
        for p in ps:
            values = [vstar(p, b, cell)[0] for b in BETAS]
            star = optimum(list(BETAS), values)
            stars[(cell, p)] = star
            l3 &= star is not None
    out["beta_star"] = {f"{c} p={p:g}": s for (c, p), s in stars.items()}
    out["bars"]["L3"] = l3
    l4 = l3
    if l3:
        for cell in ("A", "B"):
            s = [stars[(cell, p)] for p in ps]
            l4 &= all(a > b for a, b in zip(s, s[1:])) and 1.4 <= s[0] / s[-1] <= 2.8
    out["bars"]["L4"] = l4
    return out


def main(argv=None):
    parser = experiment_parser(
        "Word capacity against connection probability and plasticity",
        engines=("scheduled_aligner",), default_seeds=STUDY_SEEDS,
    )
    parser.add_argument("--p", type=float, required=True)
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, STUDY_SEEDS)
    if not args.smoke and args.p not in PROBABILITIES:
        parser.error(f"--p must be one of {PROBABILITIES}")
    protocol = base_protocol(args.p, smoke=args.smoke)
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
