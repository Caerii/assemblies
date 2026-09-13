"""Is the relocation period the clip arithmetic, or a number that landed once?

Registration: research/notes/memory/PREREG_refraction_period_law.md.

`PREREG_refraction_convergence.md` Amendment 2 measured a refracted recurrent
assembly relocating every 41.52 rounds against a predicted 40.93, at ONE
operating point (w_max = 20, beta = 0.10). The prediction

    period = ln(w_max) / ln(1 + beta) + (1 - 1 / w_max) / beta

is close to `(ln w_max + 1) / beta` for small beta, so it says the period is
LINEAR in 1/beta and LOGARITHMIC in w_max. This sweeps each parameter through
the measured point and checks both.

The unrefracted control is the true negative: its weights clip on the same
schedule, so if relocation were the clip alone it would relocate too. It must
not, because without a bias nothing erodes the member's margin.

The event definition, the stability threshold and the formula are imported
from `refraction_convergence`, which owns them.

Run:  python -m research.runner refraction-period-law --tag UNIQUE
      (smoke: --smoke --seeds 1 2 3; VOID)
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from neural_assemblies import describe_assembly_memory
from neural_assemblies.core.numpy_engine import _seeding
from neural_assemblies.diagnostics import ensemble_from_values
from research.experiments.refraction_convergence import (
    FORMATION_ROUND, RELOCATED, STABLE, STABLE_FROM, clip_period,
    relocation_events,
)
from research.runner import experiment_parser, run_experiment, validate_registered_seeds

PROTOCOL = "memory.refraction-period-law"
VERSION = "1"
REGISTRATION = "research/notes/memory/PREREG_refraction_period_law.md"
REGISTERED_SEEDS = tuple(range(42, 62))

N, K, P = 4000, 100, 0.5
STRENGTH_RATIO = 0.5               # of beta: the arm that relocates
MIN_ROUNDS, PERIODS_WANTED = 240, 5.5
#: (w_max, beta). Each parameter swept separately through the measured point.
CELLS = ((20.0, 0.20), (20.0, 0.10), (20.0, 0.05), (5.0, 0.10), (100.0, 0.10))
SMOKE_CELLS = ((20.0, 0.20), (5.0, 0.10))
SMOKE_ROUNDS = 60
ARMS = ("refracted", "control")


def to_i32(value):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def cell_name(w_max, beta):
    return f"w{w_max:g}b{beta:g}"


def rounds_for(w_max, beta):
    """Enough rounds for PERIODS_WANTED relocations, from the PREDICTION.

    A declared function of the formula, so the run length is never read off
    the data it is about to measure.
    """
    return max(MIN_ROUNDS, math.ceil(PERIODS_WANTED * clip_period(w_max, beta)))


def profile_for(w_max, beta, ratio):
    """The organ profile of one cell and arm. `ratio` is a fraction of beta."""
    return describe_assembly_memory(w_max=w_max, norm_init=True,
                                    synaptic_scaling=False, strength=ratio,
                                    beta=beta, gate=False, inference="trajectory")


def run_cell(seeds, w_max, beta, ratio, *, rounds, device, organ_semantics):
    """One cell and arm: per-round winners, consecutive overlap, events."""
    from neural_assemblies.core._torch_ops import torch_ops
    from neural_assemblies.core.semantics import OrganSemantics
    from neural_assemblies.core.torch_engine._hashed import (
        AreaFiber, HashedArea, StimulusFiber)

    expected = profile_for(w_max, beta, ratio)
    if OrganSemantics.normalize(organ_semantics).mismatch(expected):
        raise ValueError("recorded organ profile disagrees with the cell's construction")
    nbrain = len(seeds)
    sd = [to_i32(_seeding.fnv1a_pair_seed(seed, "A", "A")) for seed in seeds]
    ss = [to_i32(_seeding.fnv1a_pair_seed(seed, "s0", "A")) for seed in seeds]
    area = HashedArea(N, K, sd, device=device, refracted_strength=ratio * beta)
    fiber = AreaFiber(sd, N, N, P, beta=beta, w_max=w_max, norm_init=True,
                      synaptic_scaling=False, max_rounds=rounds, device=device)
    stim = StimulusFiber(ss, K, N, P, beta=beta, w_max=w_max, norm_init=True,
                         max_rounds=rounds, device=device)
    consec, prev = [], None
    for _ in range(rounds):
        w = cast(Any, area.project(1, [fiber, stim]))
        if prev is not None:
            pm = torch_ops.zeros(nbrain, N, dtype=torch_ops.bool, device=device)
            pm.scatter_(1, prev, True)
            consec.append((torch_ops.gather(pm, 1, w).sum(1).float() / K).cpu().numpy())
        prev = w
    cs = np.stack(consec)                       # [rounds - 1, B]
    rows = []
    for b, seed in enumerate(seeds):
        curve = [round(float(v), 4) for v in cs[:, b]]
        starts, lengths = relocation_events(curve)
        after = (starts[1:], lengths[1:]) if starts and starts[0] <= FORMATION_ROUND else (starts, lengths)
        counted = cs[STABLE_FROM - 2:, b]
        rows.append({
            "seed": seed, "n_events": len(starts), "n_relocations": len(after[0]),
            "event_starts": starts, "event_lengths": lengths,
            "relocation_starts": after[0], "relocation_lengths": after[1],
            "spacings": [y - x for x, y in zip(starts, starts[1:])],
            "stable_fraction": float((counted >= STABLE).mean()) if len(counted) else None,
            "relocated_fraction": float((counted < RELOCATED).mean()) if len(counted) else None,
            "fill": float(area.fill.cpu().numpy()[b]),
        })
    del area, fiber, stim
    torch_ops.cuda.empty_cache()
    return rows


def _ens(values, label, seeds):
    e = ensemble_from_values([float(v) for v in values], label, keys=list(seeds))
    return {**asdict(e), "low": e.low, "high": e.high}


def mean_spacing(rows):
    """Pooled mean spacing between event starts, or None when nothing recurred."""
    pooled = [s for r in rows for s in r["spacings"]]
    return float(np.mean(pooled)) if pooled else None


def experiment(record):
    p = record["parameters"]
    seeds = list(record["seeds"])
    smoke = record["mode"] == "smoke"
    cells = {}
    for w_max, beta in (tuple(c) for c in p["cells"]):
        name = cell_name(w_max, beta)
        rounds = int(p["rounds_by_cell"][name])
        predicted = clip_period(w_max, beta)
        arms = {}
        for arm in ARMS:
            ratio = STRENGTH_RATIO if arm == "refracted" else 0.0
            rows = run_cell(seeds, w_max, beta, ratio, rounds=rounds,
                            device=p["device"],
                            organ_semantics=record["execution_semantics"]["profiles"][f"{name}.{arm}"])
            pooled = mean_spacing(rows)
            arms[arm] = {
                "rows": rows, "mean_spacing": pooled,
                "relocations": _ens([r["n_relocations"] for r in rows], f"{name}:{arm}:reloc", seeds),
                "stable_fraction": _ens([r["stable_fraction"] for r in rows], f"{name}:{arm}:stable", seeds),
            }
            err = (None if pooled is None else abs(pooled - predicted) / predicted)
            print(f"  {name:<10s} {arm:<10s} rounds {rounds:>4d} relocations/brain "
                  f"{np.mean([r['n_relocations'] for r in rows]):>5.1f}  mean spacing "
                  f"{'n/a' if pooled is None else f'{pooled:7.2f}'}  predicted {predicted:7.2f}"
                  f"{'' if err is None else f'  error {err * 100:5.1f}%'}", flush=True)
        cells[name] = {"w_max": w_max, "beta": beta, "rounds": rounds,
                       "predicted_period": predicted, "arms": arms}

    bars, comparisons = {}, {}
    if not smoke:
        def spacing(name):
            return cells[name]["arms"]["refracted"]["mean_spacing"]
        beta_cells = [cell_name(20.0, b) for b in (0.20, 0.10, 0.05)]
        wmax_cells = [cell_name(w, 0.10) for w in (5.0, 20.0, 100.0)]
        beta_ratio = (None if spacing(beta_cells[0]) in (None, 0)
                      else spacing(beta_cells[2]) / spacing(beta_cells[0]))
        wmax_ratio = (None if spacing(wmax_cells[0]) in (None, 0)
                      else spacing(wmax_cells[2]) / spacing(wmax_cells[0]))
        comparisons = {
            "measured_spacing": {key: spacing(key) for key in cells},
            "predicted_period": {key: cells[key]["predicted_period"] for key in cells},
            "relative_error": {key: (None if spacing(key) is None else
                                     abs(spacing(key) - cells[key]["predicted_period"])
                                     / cells[key]["predicted_period"]) for key in cells},
            "beta_ratio_0.05_over_0.20": beta_ratio,
            "wmax_ratio_100_over_5": wmax_ratio,
        }
        errors = comparisons["relative_error"]
        bars = {
            "PL-1 the law holds in every cell: mean spacing within 15% of the predicted period":
                all(v is not None and v <= 0.15 for v in errors.values()),
            "PL-2 beta is inverse: spacing strictly decreasing in beta and the 0.05/0.20 ratio >= 3.0":
                all(spacing(a) is not None and spacing(b) is not None
                    and spacing(a) < spacing(b)
                    for a, b in zip(beta_cells, beta_cells[1:]))
                and beta_ratio is not None and beta_ratio >= 3.0,
            "PL-3 w_max is logarithmic and weak: spacing strictly increasing in w_max and the 100/5 ratio <= 3.0":
                all(spacing(a) is not None and spacing(b) is not None
                    and spacing(a) < spacing(b)
                    for a, b in zip(wmax_cells, wmax_cells[1:]))
                and wmax_ratio is not None and wmax_ratio <= 3.0,
            "PL-4 the levers are ordered: the beta ratio exceeds the w_max ratio despite w_max moving further":
                beta_ratio is not None and wmax_ratio is not None and beta_ratio > wmax_ratio,
            "PL-5 relocation is refraction, not the clip: no control relocation anywhere, >= 3 on every refracted brain":
                all(r["n_relocations"] == 0 and r["stable_fraction"] >= 0.95
                    for c in cells.values() for r in c["arms"]["control"]["rows"])
                and all(r["n_relocations"] >= 3
                        for c in cells.values() for r in c["arms"]["refracted"]["rows"]),
        }
        bars = {name: bool(ok) for name, ok in bars.items()}
        for name, ok in bars.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"  beta ratio (0.05 / 0.20) {beta_ratio}; w_max ratio (100 / 5) {wmax_ratio}")
    verdict = "VOID" if smoke else ("PASS" if all(bars.values()) else "FAIL")
    return {"verdict": verdict, "bars": bars, "cells": cells,
            "comparisons": comparisons,
            "scope": "the relocation period of one refracted recurrent assembly "
                     "against w_max and beta, hashed substrate, n = 4000, k = 100, p = 0.5"}


def main(argv=None):
    parser = experiment_parser(
        "Does the relocation period follow the clip arithmetic across w_max and beta?",
        engines=("hashed_assembly_memory",), default_seeds=REGISTERED_SEEDS,
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, REGISTERED_SEEDS)
    cells = SMOKE_CELLS if args.smoke else CELLS
    rounds_by_cell = {cell_name(w, b): (SMOKE_ROUNDS if args.smoke else rounds_for(w, b))
                      for w, b in cells}
    parameters = {"n": N, "k": K, "p": P, "cells": [list(c) for c in cells],
                  "rounds_by_cell": rounds_by_cell, "arms": list(ARMS),
                  "strength_ratio": STRENGTH_RATIO, "stable": STABLE,
                  "relocated": RELOCATED, "stable_from": STABLE_FROM,
                  "formation_round": FORMATION_ROUND,
                  "predicted_period": {cell_name(w, b): clip_period(w, b) for w, b in cells},
                  "device": args.device}
    profiles = {f"{cell_name(w, b)}.{arm}":
                profile_for(w, b, STRENGTH_RATIO if arm == "refracted" else 0.0)
                for w, b in cells for arm in ARMS}
    path = run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATION, engine=args.engine, seeds=args.seeds,
        tag=args.tag, smoke=args.smoke, minimum_study_seeds=20,
        parameters=parameters, organ_semantics=profiles, measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
