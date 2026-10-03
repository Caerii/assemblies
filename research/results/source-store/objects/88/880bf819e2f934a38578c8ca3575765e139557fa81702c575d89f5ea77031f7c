"""Is the refracted memory's capacity at its best write set by the in-degree
d = n p alone?

Registered in PREREG_refraction_memory.md, Amendment 21.

Read across Amendments 13-19 after their bars were judged, the best
distinct-completion capacity of 46 cells with d = n p >= 1000 (n 1333 to
8000, k 10 to 240, p 0.125 to 0.75) follows

    C = 0.0135 d^1.51

with a residual of 15% (log-scale sd 0.143); adding n/k barely helps (its
exponent 0.08). The number of synapses a neuron receives from its area, not
the assembly size, the area size or the connection probability separately,
sets how many items it stores. This tests that on nine cells never
measured, six of them at in-degrees never measured (1500, 3000, 6000).
Within each in-degree n/k varies two- to fourfold, so a law in n/k (such as
(n/k)^2) and the in-degree law predict different things at equal d:

    d = 1000:  (2500, 25, 0.4)   n/k 100   (5000, 100, 0.2)   n/k 50
    d = 1500:  (3000, 15, 0.5)   n/k 200   (6000, 60, 0.25)   n/k 100
               (12000, 240, 0.125) n/k 50
    d = 3000:  (6000, 30, 0.5)   n/k 200   (12000, 240, 0.25) n/k 50
    d = 6000:  (12000, 60, 0.5)  n/k 200
    the instrument: (4000, 60, 0.5)         (Amendment 18: 1597)

each swept over theta x 0.15 x 2^(j/8) up to 0.30 theta, theta =
sqrt((1 - p) ln n / (p k)) -- Amendment 18 put every onset at 0.16-0.18 theta
and every dense optimum within a step of it.

    python -m research.runner degree-law --registration PATH --tag NAME [--smoke]
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
from research.experiments import memory_threshold_law as tl             # noqa: E402
from research.experiments import memory_write_strength as ws            # noqa: E402
from research.runner import experiment_parser, run_experiment           # noqa: E402

CELLS = ((2500, 25, 0.4), (5000, 100, 0.2),
         (3000, 15, 0.5), (6000, 60, 0.25), (12000, 240, 0.125),
         (6000, 30, 0.5), (12000, 240, 0.25),
         (12000, 60, 0.5),
         (4000, 60, 0.5))
INSTRUMENT = {(4000, 60, 0.5): 1597.0}
#: the law fitted after the data (Amendments 13-19, 46 cells, d >= 1000)
C0, GAMMA = 0.0135, 1.51
F_LO, F_HI, STEPS_PER_OCTAVE = 0.15, 0.30, 8
MIN_BETA = tl.MIN_BETA
STOP_FROM = 32
SEEDS = tuple(range(202, 222))
PREDICTION_BAND, SAME_D_BAND, GAMMA_BAND = 0.25, 0.20, (1.35, 1.65)


def degree(n, p):
    return n * p


def predicted(n, k, p):
    return C0 * degree(n, p) ** GAMMA


def betas(n, k, p):
    theta = tl.theta(n, k, p)
    out, j = [], 0
    while F_LO * 2 ** (j / STEPS_PER_OCTAVE) <= F_HI + 1e-12:
        b = round(theta * F_LO * 2 ** (j / STEPS_PER_OCTAVE), 4)
        if b >= MIN_BETA:
            out.append(b)
        j += 1
    return sorted(set(out))


def plan(cells, *, smoke=False):
    out = []
    for n, k, p in cells:
        guess = predicted(n, k, p)
        grid = betas(n, k, p)
        out.append({"n": n, "k": k, "p": p, "d": degree(n, p), "predicted": guess,
                    "theta": tl.theta(n, k, p),
                    "betas": grid[2:4] if smoke else grid,
                    "cap": 32 if smoke else int(3 * guess + 256),
                    # a rate below the onset opens its window late (Amendment
                    # 18: up to ~1100 items at n = 8000); it stores this far
                    "give_up": 16 if smoke else int(max(8192, 2 * guess))})
    return out


def experiment(record):
    parameters = record["parameters"]
    seeds = record["seeds"]
    device = parameters["device"]
    profiles = record["execution_semantics"]["profiles"]
    out = {}
    for spec in parameters["cells"]:
        n, k, p = spec["n"], spec["k"], spec["p"]
        grid = spec["betas"]
        results = lr.run_betas(n, k, grid, seeds, spec["cap"], device,
                               {b: profiles[lr.profile_name(b)] for b in grid},
                               stop_on=("complete_distinct",), p=p, grid_start=2,
                               give_up=spec["give_up"], stop_from=STOP_FROM)
        sweep = {}
        for beta, (c, cache) in zip(grid, results):
            windows = {m: ws.edges([(M, ensemble_from_values(r[m]).mean)
                                    for M, r in cache.items()])
                       for m in ("rank1", "complete", "complete_distinct")}
            sweep[f"{beta:g}"] = {
                "beta": beta, "c_first_item": c, "windows": windows,
                "ensembles": {M: {nm: asdict(ensemble_from_values(v, keys=seeds, label=nm))
                                  for nm, v in r.items()} for M, r in sorted(cache.items())},
            }
            w = windows["complete_distinct"]
            print(f"({n}, {k}, {p}) d={spec['d']:g} beta={beta:g} "
                  f"({beta / spec['theta']:.3f} theta): distinct [{w.get('lower')}, "
                  f"{w.get('upper')}] (predicted {spec['predicted']:.0f})", flush=True)
        out[f"{n}/{k}/{p:g}"] = {"n": n, "k": k, "p": p, "d": spec["d"],
                                 "predicted": spec["predicted"], "theta": spec["theta"],
                                 "sweep": sweep}
    return {"cells": out, "verdict": "VOID" if record["mode"] == "smoke" else "UNJUDGED"}


# ----------------------------------------------------------------- evaluation
def evaluate(observations, *, instrument=None):
    """Amendment 21's bars."""
    instrument = INSTRUMENT if instrument is None else instrument
    rows = {}
    for cell in observations["cells"].values():
        key = (cell["n"], cell["k"], cell["p"])
        best = max(lr.capacity(s["windows"]["complete_distinct"])
                   for s in cell["sweep"].values())
        rows[key] = {"d": cell["d"], "best": best, "predicted": predicted(*key),
                     "ratio": best / predicted(*key)}
    out = {"cells": {f"{n}/{k}/{p:g}": r for (n, k, p), r in rows.items()}, "bars": {}}
    out["bars"]["DV"] = all(abs(rows[c]["best"] / v - 1) <= 0.15
                            for c, v in instrument.items() if c in rows)
    new = {c: r for c, r in rows.items() if c not in instrument}
    out["bars"]["D1"] = bool(new) and all(abs(r["ratio"] - 1) <= PREDICTION_BAND
                                          for r in new.values())
    same = {}
    for c, r in new.items():
        same.setdefault(r["d"], []).append(r["best"])
    groups = {d: v for d, v in same.items() if len(v) > 1}
    out["same_d"] = {f"{d:g}": v for d, v in groups.items()}
    out["bars"]["D2"] = bool(groups) and all(
        all(abs(x / (sum(v) / len(v)) - 1) <= SAME_D_BAND for x in v) for v in groups.values())
    pts = [(math.log(r["d"]), math.log(r["best"])) for r in new.values() if r["best"] > 0]
    if len(pts) > 1:
        mx = sum(x for x, _ in pts) / len(pts)
        my = sum(y for _, y in pts) / len(pts)
        slope = (sum((x - mx) * (y - my) for x, y in pts)
                 / sum((x - mx) ** 2 for x, _ in pts))
    else:
        slope = None
    out["gamma"] = slope
    out["bars"]["D3"] = slope is not None and GAMMA_BAND[0] <= slope <= GAMMA_BAND[1]
    return out


def main(argv=None):
    ap = experiment_parser(__doc__ or "Degree law", engines=("hashed_assembly_memory",),
                           default_seeds=SEEDS)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--cells", help="n:k:p triples (default: all nine)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cells = ([tuple(float(v) if i == 2 else int(v) for i, v in enumerate(t.split(":")))
              for t in args.cells.split(",")] if args.cells else list(CELLS))
    if any(c not in CELLS for c in cells) or len(set(cells)) != len(cells):
        ap.error(f"cells must be distinct members of {CELLS}")
    if not args.smoke and list(args.seeds) != list(SEEDS):
        ap.error("Amendment 21 is registered on seeds 202..221")
    specs = plan(cells, smoke=args.smoke)
    profiles = {lr.profile_name(b): lr.profile(b) for s in specs for b in s["betas"]}
    path = run_experiment(
        script=__file__, protocol="memory.degree-law", protocol_version="1",
        registration=args.registration, engine=args.engine, seeds=args.seeds, tag=args.tag,
        smoke=args.smoke, measure=experiment, organ_semantics=profiles,
        parameters={"cells": specs, "c0": C0, "gamma": GAMMA, "f_lo": F_LO, "f_hi": F_HI,
                    "steps_per_octave": STEPS_PER_OCTAVE, "min_beta": MIN_BETA,
                    "stop_from": STOP_FROM, "strength": lr.STRENGTH, "w_max": pe.W_MAX,
                    "rounds": pe.T, "half_bar": pe.HALF_BAR, "complete": ws.COMPLETE,
                    "recall_sample": pe.RECALL_SAMPLE,
                    "measurement_seed": pe.MEASUREMENT_SEED, "grid_start": 2,
                    "device": args.device},
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
