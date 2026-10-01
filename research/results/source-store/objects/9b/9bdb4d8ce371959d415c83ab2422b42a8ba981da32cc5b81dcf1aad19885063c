"""Is the best learning rate a fixed fraction of the projection-convergence
threshold, above and below the regime floor?

Registered in PREREG_refraction_memory.md, Amendment 17.

Amendments 13 and 16 fixed the completion-optimal learning rate of the
refracted memory above the regime floor (k p >= 3 ln n). Read against the
literature (research/notes/LITERATURE_SYNTHESIS.md), both sets of optima are
a near-constant fraction, about 0.20, of the form the assembly calculus's
projection-convergence thresholds scale as,

    theta(n, k, p) = sqrt((1 - p) ln n / (p k)),

seen after the data and not registered. This registers it, and carries it
below the floor, into the fan-in range every published simulation runs at
(k p = 1 to 10): two families and the PNAS 2020 cell,

    p = 0.5, n = 4000:     k = 10, 20, 40, 60, 80, 160   (k p = 5 to 80)
    p = 0.05, n = 8000:    k = 100, 200, 400             (k p = 5, 10, 20)
    PNAS 2020:             n = 10000, k = 100, p = 0.01  (k p = 1)

The refracted memory (0.5 beta, T = 8, w_max 20, arm B, ungated) at each
cell is swept over 0.2 theta x 2^(j/4), j = -4..4, at or above 0.025; distinct
completion and capacity as Amendments 13-16, checkpoints from M = 2, no stop
decision before M = 32.

    python -m research.runner threshold-law --registration PATH --tag NAME [--smoke]
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from neural_assemblies.diagnostics import ensemble_from_values          # noqa: E402
from research.experiments import memory_learning_rate as lr             # noqa: E402
from research.experiments import memory_pattern_efficiency as pe        # noqa: E402
from research.experiments import memory_recall_law as rl                # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

FRACTION = 0.20
CELLS = ((4000, 10, 0.5), (4000, 20, 0.5), (4000, 40, 0.5), (4000, 60, 0.5),
         (4000, 80, 0.5), (4000, 160, 0.5),
         (8000, 100, 0.05), (8000, 200, 0.05), (8000, 400, 0.05),
         (10000, 100, 0.01))
MIN_BETA = 0.025
STOP_FROM = 32
SEEDS = tuple(range(142, 162))
#: Amendment 14's best capacity at the shared cell (TV)
A14 = {(4000, 60, 0.5): 1616.0}


def theta(n, k, p):
    """The convergence-threshold form sqrt((1 - p) ln n / (p k))."""
    return math.sqrt((1 - p) * math.log(n) / (p * k))


def above_floor(n, k, p):
    return k * p >= 3 * math.log(n)


def beta_pred(n, k, p):
    return FRACTION * theta(n, k, p)


def betas(n, k, p):
    grid = {round(beta_pred(n, k, p) * 2 ** (j / 4), 4) for j in range(-4, 5)}
    return sorted(b for b in grid if b >= MIN_BETA)


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        guess = max(rl.predicted(n, k, p), 64.0)
        out.append({"n": n, "k": k, "p": p, "above_floor": above_floor(n, k, p),
                    "theta": theta(n, k, p),
                    "betas": [round(beta_pred(n, k, p), 4)] if smoke else betas(n, k, p),
                    "cap": 32 if smoke else int(min(8 * guess, 40000)),
                    # a cell whose completion never rises stops here
                    "give_up": 16 if smoke else int(min(4 * guess + 64, 8192))})
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    profiles = record["execution_semantics"]["profiles"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        sweep = {}
        for beta in spec["betas"]:
            c, cache = lr.run_beta(n, k, beta, seeds, spec["cap"], device,
                                   profiles[lr.profile_name(beta)],
                                   stop_on=("complete_distinct",), p=p, grid_start=2,
                                   give_up=spec["give_up"], stop_from=STOP_FROM)
            windows = {m: ws.edges([(M, ensemble_from_values(r[m]).mean)
                                    for M, r in cache.items()])
                       for m in ("rank1", "complete", "complete_distinct")}
            sweep[f"{beta:g}"] = {
                "beta": beta, "c_first_item": c, "windows": windows,
                "ensembles": {M: {nm: asdict(ensemble_from_values(v, keys=seeds, label=nm))
                                  for nm, v in r.items()} for M, r in sorted(cache.items())},
            }
            print(f"({n}, {k}, {p}) beta={beta:g} c={c}: distinct "
                  f"{windows['complete_distinct'].get('upper')}, rank-1 "
                  f"{windows['rank1'].get('upper')}", flush=True)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p,
                                 "above_floor": spec["above_floor"],
                                 "theta": spec["theta"], "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations, *, a14=None):
    """Amendment 17's bars."""
    a14 = A14 if a14 is None else a14
    rows = {}
    for cell in observations["cells"].values():
        items = sorted(cell["sweep"].values(), key=lambda s: s["beta"])
        bs = [s["beta"] for s in items]
        caps = [lr.capacity(s["windows"]["complete_distinct"]) for s in items]
        star = lr.optimum(bs, caps)
        n, k, p = cell["n"], cell["k"], cell["p"]
        rows[(n, k, p)] = {
            "kp": k * p, "above_floor": cell["above_floor"], "betas": bs, "capacity": caps,
            "best": max(caps), "beta_star": star,
            "fraction": None if star is None else star / theta(n, k, p),
            "C": max(caps) / rl.synapse_scale(n, k, p),
        }
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    out["bars"]["TV"] = all(abs(rows[c]["best"] / v - 1) <= 0.15 for c, v in a14.items()
                            if c in rows)
    above = [r for r in rows.values() if r["above_floor"]]
    below = [r for r in rows.values() if not r["above_floor"]]
    out["bars"]["T1"] = bool(above) and all(
        r["fraction"] is not None and abs(r["fraction"] / FRACTION - 1) <= 0.15 for r in above)
    resolved_below = [r for r in below if r["fraction"] is not None]
    out["below_floor_resolved"] = len(resolved_below)
    out["bars"]["T2"] = bool(resolved_below) and all(
        abs(r["fraction"] / FRACTION - 1) <= 0.25 for r in resolved_below)
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Threshold law", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: the ten cells)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 17 is registered on seeds 142..161")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(b): lr.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.threshold-law", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "fraction": FRACTION, "min_beta": MIN_BETA,
                    "stop_from": STOP_FROM, "strength": lr.STRENGTH, "w_max": pe.W_MAX,
                    "rounds": pe.T, "half_bar": pe.HALF_BAR, "complete": ws.COMPLETE,
                    "recall_sample": pe.RECALL_SAMPLE,
                    "measurement_seed": pe.MEASUREMENT_SEED, "grid_start": 2,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
