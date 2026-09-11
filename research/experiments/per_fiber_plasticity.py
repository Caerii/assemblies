"""Causal materialized measurement of the public per-fiber beta route."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from neural_assemblies import Brain
from neural_assemblies.diagnostics import ensemble_from_values
from research.runner import ExperimentOutput, experiment_parser, run_experiment

REGISTRATION = "research/notes/memory/PREREG_per_fiber_plasticity.md"
SEEDS = list(range(201, 221))
TARGETS = ("forward", "swapped", "equal")
SOURCES = ("A", "B")
DEFAULTS = {
    "n": 80,
    "k": 10,
    "p": 0.2,
    "default_beta": 0.005,
    "fast_beta": 0.06,
    "slow_beta": 0.005,
    "rounds": 50,
    "w_max": 20.0,
    "ratio_low": 10.0,
    "equal_tolerance": 1e-6,
    "expected_relative_tolerance": 2e-5,
}


def parameters() -> dict:
    return {**DEFAULTS, "sources": list(SOURCES), "targets": list(TARGETS),
            "rates": {
                "forward": {"A": DEFAULTS["fast_beta"], "B": DEFAULTS["slow_beta"]},
                "swapped": {"A": DEFAULTS["slow_beta"], "B": DEFAULTS["fast_beta"]},
                "equal": {"A": DEFAULTS["slow_beta"], "B": DEFAULTS["slow_beta"]},
            }}


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _geometric_mean(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("geometric mean requires finite positive present-edge weights")
    return float(np.exp(np.mean(np.log(values.astype(np.float64)))))


def run_seed(seed: int, config: dict) -> tuple[dict, dict]:
    """Specification: research/notes/memory/PREREG_per_fiber_plasticity.md"""
    brain = Brain(p=config["p"], seed=seed, engine="numpy_explicit",
                  norm_init=False, w_max=config["w_max"])
    for name in [*config["sources"], *config["targets"]]:
        brain.add_area(name, config["n"], config["k"], config["default_beta"])

    base = np.array(brain.connectomes["A"]["forward"].weights, copy=True)
    for target in config["targets"]:
        for source in config["sources"]:
            brain.connectomes[source][target].weights = np.array(base, copy=True)
    for target, rates in config["rates"].items():
        for source, beta in rates.items():
            brain.update_plasticity(source, target, beta)

    for target in config["targets"]:
        first = brain.connectomes["A"][target].weights
        second = brain.connectomes["B"][target].weights
        if first is second or not np.array_equal(first, second):
            raise RuntimeError(f"{target} base fibers must be equal independent arrays")

    pre = np.arange(config["k"], dtype=np.uint32)
    histories = {target: [] for target in config["targets"]}
    projections = {source: list(config["targets"]) for source in config["sources"]}
    for _ in range(config["rounds"]):
        brain.project(external_inputs={source: pre for source in config["sources"]},
                      projections=projections)
        for target in config["targets"]:
            winners = np.asarray(brain.areas[target].winners, dtype=np.uint32)
            if winners.shape != (config["k"],):
                raise RuntimeError(f"{target} returned {len(winners)} winners")
            histories[target].append(winners.copy())
        if any(not np.array_equal(np.asarray(brain.areas[source].winners), pre)
               for source in config["sources"]):
            raise RuntimeError("source activity changed after identical reinjection")

    histories_equal = all(
        np.array_equal(forward, swapped)
        for forward, swapped in zip(histories["forward"], histories["swapped"])
    )
    cells, raw_cells = {}, {}
    for target in config["targets"]:
        post = histories[target][-1]
        initial_block = base[np.ix_(pre, post)]
        present = initial_block > 0
        if not present.any():
            raise RuntimeError(f"{target} final co-active block has no present edge")
        weights = {}
        means = {}
        for source in config["sources"]:
            block = np.asarray(brain.connectomes[source][target].weights)[np.ix_(pre, post)]
            if not np.isfinite(block).all():
                raise RuntimeError(f"{source}->{target} contains a nonfinite weight")
            weights[source] = block
            means[source] = _geometric_mean(block[present])
        rates = config["rates"][target]
        if target == "forward":
            fast, slow = means["A"], means["B"]
        elif target == "swapped":
            fast, slow = means["B"], means["A"]
        else:
            fast, slow = means["A"], means["B"]
        cells[target] = {
            "present_edges": int(present.sum()),
            "means": means,
            "ratio": fast / slow,
            "maximum_selected_weight": max(float(values.max()) for values in weights.values()),
            "initial_block_sha256": _digest(initial_block),
        }
        raw_cells[target] = {
            "winner_history": [row.tolist() for row in histories[target]],
            "initial_block": initial_block.tolist(),
            "final_blocks": {name: values.tolist() for name, values in weights.items()},
        }
    return ({"seed": seed, "histories_equal": histories_equal, "cells": cells},
            {"seed": seed, "cells": raw_cells})


def _summary(values: list[float], label: str, seeds: list[int]) -> dict:
    result = ensemble_from_values(values, label, keys=seeds)
    return {**asdict(result), "low": result.low, "high": result.high}


def judge(rows: list[dict], config: dict) -> dict[str, bool]:
    expected = {
        "fast": (1 + config["fast_beta"]) ** config["rounds"],
        "slow": (1 + config["slow_beta"]) ** config["rounds"],
    }
    tolerance = config["expected_relative_tolerance"]
    return {
        "forward-ratio": all(row["cells"]["forward"]["ratio"] > config["ratio_low"] for row in rows),
        "swapped-ratio": all(row["cells"]["swapped"]["ratio"] > config["ratio_low"] for row in rows),
        "equal-null": all(abs(row["cells"]["equal"]["ratio"] - 1) <= config["equal_tolerance"] for row in rows),
        "fast-formula": all(
            abs(row["cells"][target]["means"][source] - expected["fast"]) / expected["fast"] <= tolerance
            for row in rows for target, source in (("forward", "A"), ("swapped", "B"))),
        "slow-formula": all(
            abs(row["cells"][target]["means"][source] - expected["slow"]) / expected["slow"] <= tolerance
            for row in rows for target, source in (("forward", "B"), ("swapped", "A"),
                                                    ("equal", "A"), ("equal", "B"))),
        "unclipped": all(row["cells"][target]["maximum_selected_weight"] < config["w_max"]
                         for row in rows for target in config["targets"]),
        "winner-history-symmetry": all(row["histories_equal"] for row in rows),
    }


def experiment(record: dict) -> ExperimentOutput:
    config, seeds = record["parameters"], record["seeds"]
    rows, raw_rows = [], []
    for seed in seeds:
        row, raw = run_seed(seed, config)
        rows.append(row)
        raw_rows.append(raw)
        print(seed, row["cells"]["forward"]["ratio"], flush=True)
    summaries = {
        target: {
            "ratio": _summary([row["cells"][target]["ratio"] for row in rows],
                              f"{target}-ratio", seeds),
            **{f"{source}_mean": _summary(
                [row["cells"][target]["means"][source] for row in rows],
                f"{source}-{target}-mean", seeds) for source in config["sources"]},
        } for target in config["targets"]
    }
    checks = {} if record["mode"] == "smoke" else judge(rows, config)
    verdict = "VOID" if record["mode"] == "smoke" else ("PASS" if all(checks.values()) else "FAIL")
    return ExperimentOutput(
        observations={"rows": rows, "summaries": summaries, "checks": checks,
                      "verdict": verdict,
                      "scope": "materialized NumPy per-fiber beta routing only"},
        json_attachments={"raw-fibers.json.gz": {"rows": raw_rows}},
    )


def main(argv=None):
    parser = experiment_parser(__doc__, engines=("numpy_explicit",),
                               default_seeds=tuple(SEEDS))
    args = parser.parse_args(argv)
    if not args.smoke and args.seeds != SEEDS:
        parser.error("study requires brain seeds 201..220 in order")
    if args.smoke and len(args.seeds) != 3:
        parser.error("smoke requires exactly three explicit seeds")
    print(run_experiment(
        script=Path(__file__), protocol="mechanism.per-fiber-plasticity",
        protocol_version="1", registration=REGISTRATION, engine=args.engine,
        seeds=args.seeds, tag=args.tag, smoke=args.smoke,
        minimum_study_seeds=20, parameters=parameters(), measure=experiment,
    ))


if __name__ == "__main__":
    main()
